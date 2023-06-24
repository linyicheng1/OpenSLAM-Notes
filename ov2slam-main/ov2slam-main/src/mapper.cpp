/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/

#include <thread>

#include "mapper.hpp"
#include "opencv2/video/tracking.hpp"

/**
 * @brief 地图构建类的构造函数 
 * @param pslamstate slam系统的状态
 * @param pmap 地图管理类
 * @param pframe 当前帧指针
*/
Mapper::Mapper(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap, 
            std::shared_ptr<Frame> pframe)
    : pslamstate_(pslamstate), pmap_(pmap), pcurframe_(pframe)
    , pestimator_( new Estimator(pslamstate_, pmap_) )
    , ploopcloser_( new LoopCloser(pslamstate_, pmap_) )
{
    // 创建地图构建线程 
    std::thread mapper_thread(&Mapper::run, this);
    mapper_thread.detach();

    std::cout << "\nMapper Object is created!\n";
}

// 地图构建线程 
void Mapper::run()
{
    std::cout << "\nMapper is ready to process Keyframes!\n";
    
    Keyframe kf; // 当前关键帧 
    
    // 状态估计线程 
    std::thread estimator_thread(&Estimator::run, pestimator_);
    // 回环检测线程 
    std::thread lc_thread(&LoopCloser::run, ploopcloser_);

    while( !bexit_required_ ) {
        // 判断是否有新的关键帧需要处理 
        if( getNewKf(kf) ) 
        {
            if( pslamstate_->debug_ )
                std::cout << "\n\n - [Mapper (back-End)]: New KF to process : KF #" 
                    << kf.kfid_ << "\n";

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                Profiler::Start("0.Keyframe-Processing_Mapper");

            // Get new KF ptr
            // 获取新添加的关键帧指针 
            std::shared_ptr<Frame> pnewkf = pmap_->getKeyframe(kf.kfid_);
            assert( pnewkf );

            // Triangulate stereo
            if( pslamstate_->stereo_ )// 双目  
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n\n - [Mapper (back-End)]: Applying stereo matching!\n";

                cv::Mat imright;
                if( pslamstate_->use_clahe_ ) {
                    // 复制右目图像 
                    pmap_->ptracker_->pclahe_->apply(kf.imrightraw_, imright);
                } else {
                    imright = kf.imrightraw_;
                }
                // 构建右目金字塔
                std::vector<cv::Mat> vpyr_imright;
                cv::buildOpticalFlowPyramid(imright, vpyr_imright, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
                // 双目匹配 
                pmap_->stereoMatching(*pnewkf, kf.vpyr_imleft_, vpyr_imright);
                
                if( pnewkf->nb2dkps_ > 0 && pnewkf->nb_stereo_kps_ > 0 ) {
                    
                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n - [Mapper (back-End)]: Stereo Triangulation!\n";

                        std::cout << "\n\n  \t >>> (BEFORE STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ 
                            << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }

                    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
                    // 双目三角化 
                    triangulateStereo(*pnewkf);

                    if( pslamstate_->debug_ ) {
                        std::cout << "\n\n  \t >>> (AFTER STEREO TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                        std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ 
                            << " / " << pnewkf->nb_stereo_kps_ << "\n";
                    }
                }
            }

            // Triangulate temporal
            if( pnewkf->nb2dkps_ > 0 && pnewkf->kfid_ > 0 ) 
            {
                if( pslamstate_->debug_ ) {
                    std::cout << "\n\n - [Mapper (back-End)]: Temporal Triangulation!\n";

                    std::cout << "\n\n  \t >>> (BEFORE TEMPORAL TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                    std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ 
                        << " / " << pnewkf->nb_stereo_kps_ << "\n";
                }

                std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
                // 时序三角化,基于上一帧关键帧 
                triangulateTemporal(*pnewkf);
                
                if( pslamstate_->debug_ ) {
                    std::cout << "\n\n  \t >>> (AFTER TEMPORAL TRIANGULATION) New KF nb 2d kps / 3d kps / stereokps : ";
                    std::cout << pnewkf->nb2dkps_ << " / " << pnewkf->nb3dkps_ << " / " << pnewkf->nb_stereo_kps_ << "\n";
                }
            }

            // If Mono mode, check if reset is required
            if( pslamstate_->mono_ && pslamstate_->bvision_init_ ) 
            {// 单目模式，检查初始化质量是否足够高 
                if( kf.kfid_ == 1 && pnewkf->nb3dkps_ < 30 ) {
                    std::cout << "\n Bad initialization detected! Resetting\n";
                    pslamstate_->breset_req_ = true;
                    reset();
                    continue;
                } 
                else if( kf.kfid_ < 10 && pnewkf->nb3dkps_ < 3 ) {
                    std::cout << "\n Reset required : Nb 3D kps #" 
                            << pnewkf->nb3dkps_;
                    pslamstate_->breset_req_ = true;
                    reset();
                    continue;
                }
            }

            // Update the MPs and the covisilbe graph between KFs
            // (done here for real-time performance reason)
            // 跟新关键帧之间的共视图 
            pmap_->updateFrameCovisibility(*pnewkf);

            // Dirty but useful for visualization
            pcurframe_->map_covkfs_ = pnewkf->map_covkfs_;

            if( pslamstate_->use_brief_ && kf.kfid_ > 0 
                && !bnewkfavailable_ ) 
            {
                if( pslamstate_->bdo_track_localmap_ )
                {
                    if( pslamstate_->debug_ )
                        std::cout << "\n\n - [Mapper (back-End)]: matchingToLocalMap()!\n";
                    // 关键帧与局部地图匹配，只有使用breif描述子才会使用
                    matchingToLocalMap(*pnewkf);
                }
            }

            // Send new KF to estimator for BA
            // 估计器线程，添加新的关键帧 
            pestimator_->addNewKf(pnewkf);

            // Send KF along with left image to LC thread
            if( pslamstate_->buse_loop_closer_ ) {// 是否使用回环检测 
                // 回环检测线程，添加新的关键帧 
                ploopcloser_->addNewKf(pnewkf, kf.imleftraw_);
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Keyframe-Processing_Mapper");

        } else {
            std::chrono::microseconds dura(100);
            std::this_thread::sleep_for(dura);
        }
    }

    pestimator_->bexit_required_ = true;
    ploopcloser_->bexit_required_ = true;

    estimator_thread.join();
    lc_thread.join();
    
    std::cout << "\nMapper is stopping!\n";
}

/**
 * @brief 三角化
*/
void Mapper::triangulateTemporal(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateTemporal");

    // Get New KF kps / pose
    // 当前帧的所有特征点 
    std::vector<Keypoint> vkps = frame.getKeypoints2d();
    // 当前帧的位姿
    Sophus::SE3d Twcj = frame.getTwc();

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to temporal triangulate...\n";
        return;
    }

    // Setup triangulatation for OpenGV-based mapping
    size_t nbkps = vkps.size();// 当前的特征点数量 
    // 转换成Eigen::Vector3d类型
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;
    vleftbvs.reserve(nbkps);
    vrightbvs.reserve(nbkps);

    // Init a pkf object that will point to the prev KF to use
    // for triangulation
    std::shared_ptr<Frame> pkf;// 新建一个帧对象 
    pkf.reset( new Frame() );
    pkf->kfid_ = -1;

    // Relative motions between new KF and prev. KFs
    int relkfid = -1;
    Sophus::SE3d Tcicj, Tcjci;
    Eigen::Matrix3d Rcicj;

    // New 3D MPs projections
    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, right_pt, wpt;

    int good = 0, candidates = 0;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // We go through all the 2D kps in new KF
    for( size_t i = 0 ; i < nbkps ; i++ )
    {// 遍历所有的特征点 
        // Get the related MP and check if it is ready 
        // to be triangulated 
        // 获取特征点对应的地图点 
        std::shared_ptr<MapPoint> plm = pmap_->getMapPoint(vkps.at(i).lmid_);

        if( plm == nullptr ) {
            pmap_->removeMapPointObs(vkps.at(i).lmid_, frame.kfid_);
            continue;
        }

        // If MP is already 3D continue (should not happen)
        if( plm->is3d_ ) {// 确保没有被三角化过 
            continue;
        }

        // Get the set of KFs sharing observation of this 2D MP
        std::set<int> co_kf_ids = plm->getKfObsSet();// 获取共视图 

        // Continue if new KF is the only one observing it
        if( co_kf_ids.size() < 2 ) {// 至少被看到两次 
            continue;
        }

        int kfid = *co_kf_ids.begin();

        if( frame.kfid_ == kfid ) {
            continue;
        }

        // Get the 1st KF observation of the related MP
        pkf = pmap_->getKeyframe(kfid);// 获取第一帧观测到该地图点的帧 
        
        if( pkf == nullptr ) {
            continue;
        }

        // Compute relative motion between new KF and selected KF
        // (only if req.)
        if( relkfid != kfid ) {
            // 计算相对的位置 
            Sophus::SE3d Tciw = pkf->getTcw();
            Tcicj = Tciw * Twcj;
            Tcjci = Tcicj.inverse();
            Rcicj = Tcicj.rotationMatrix();

            relkfid = kfid;
        }

        // If no motion between both KF, skip
        // 保证两帧之间有足够的运动 
        if( pslamstate_->stereo_ && Tcicj.translation().norm() < 0.01 ) {
            continue;
        }
        
        // Get obs kp
        // 获取第一帧观测到该地图点的特征点 
        Keypoint kfkp = pkf->getKeypointById(vkps.at(i).lmid_);
        if( kfkp.lmid_ != vkps.at(i).lmid_ ) {
            continue;
        }

        // Check rotation-compensated parallax
        // 计算视差 
        cv::Point2f rotpx = frame.projCamToImage(Rcicj * vkps.at(i).bv_);
        double parallax = cv::norm(kfkp.unpx_ - rotpx);

        candidates++;

        // Compute 3D pos and check if its good or not
        // 三角化
        left_pt = computeTriangulation(Tcicj, kfkp.bv_, vkps.at(i).bv_);

        // Project into right cam (new KF)
        right_pt = Tcjci * left_pt;

        // Ensure that the 3D MP is in front of both camera
        if( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
            if( parallax > 20. ) {
                pmap_->removeMapPointObs(kfkp.lmid_, frame.kfid_);
            }
            continue;
        }   
        // 计算重投影误差
        // Remove MP with high reprojection error
        left_px_proj = pkf->projCamToImage(left_pt);
        right_px_proj = frame.projCamToImage(right_pt);
        ldist = cv::norm(left_px_proj - kfkp.unpx_);
        rdist = cv::norm(right_px_proj - vkps.at(i).unpx_);
        // 重投影误差过大，且视差足够，删除该地图点，重新三角化 
        if( ldist > pslamstate_->fmax_reproj_err_ 
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            if( parallax > 20. ) {
                pmap_->removeMapPointObs(kfkp.lmid_, frame.kfid_);
            }
            continue;
        }
        // 投影到世界坐标系 
        // The 3D pos is good, update SLAM MP and related KF / Frame
        wpt = pkf->projCamToWorld(left_pt);
        // 更新地图点 
        pmap_->updateMapPoint(vkps.at(i).lmid_, wpt, 1./left_pt.z());

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Temporal Mapping : " << good << " 3D MPs out of " 
            << candidates << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateTemporal");
}

/**
 * @brief 双目关键帧的三角化 
*/
void Mapper::triangulateStereo(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_TriangulateStereo");

    // INIT STEREO TRIANGULATE
    std::vector<Keypoint> vkps;

    // Get the new KF stereo kps
    vkps = frame.getKeypointsStereo();// 获取双目特征点 

    if( vkps.empty() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t >>> No kps to stereo triangulate...\n";
        return;
    }

    // Store the stereo kps along with their idx
    std::vector<int> vstereoidx;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;

    size_t nbkps = vkps.size();

    vleftbvs.reserve(nbkps);
    vrightbvs.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
    std::vector<int> vlmids;
    vwpts.reserve(nbkps);
    vlmids.reserve(nbkps);

    // Get the extrinsic transformation
    Sophus::SE3d Tlr = frame.pcalib_rightcam_->getExtrinsic();
    Sophus::SE3d Trl = Tlr.inverse();

    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        if( !vkps.at(i).is3d_ && vkps.at(i).is_stereo_ )
        {
            vstereoidx.push_back(i);
            vleftbvs.push_back(vkps.at(i).bv_);
            vrightbvs.push_back(vkps.at(i).rbv_);
        }
    }

    if( vstereoidx.empty() ) {
        return;
    }

    size_t nbstereo = vstereoidx.size();

    cv::Point2f left_px_proj, right_px_proj;
    float ldist, rdist;
    Eigen::Vector3d left_pt, right_pt, wpt;

    int kpidx;

    int good = 0;

    // For each stereo kp
    for( size_t i = 0 ; i < nbstereo ; i++ ) 
    {
        kpidx = vstereoidx.at(i);

        if( pslamstate_->bdo_stereo_rect_ ) {
            float disp = vkps.at(kpidx).unpx_.x - vkps.at(kpidx).runpx_.x;

            if( disp < 0. ) {
                frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
                continue;
            }

            float z = frame.pcalib_leftcam_->fx_ * frame.pcalib_rightcam_->Tcic0_.translation().norm() / fabs(disp);

            left_pt << vkps.at(kpidx).unpx_.x, vkps.at(kpidx).unpx_.y, 1.;
            left_pt = z * frame.pcalib_leftcam_->iK_ * left_pt.eval();
        } else {
            // Triangulate in left cam frame
            left_pt = computeTriangulation(Tlr, vleftbvs.at(i), vrightbvs.at(i));
        }

        // Project into right cam frame
        right_pt = Trl * left_pt;

        if( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Remove MP with high reprojection error
        left_px_proj = frame.projCamToImage(left_pt);
        right_px_proj = frame.projCamToRightImage(left_pt);
        ldist = cv::norm(left_px_proj - vkps.at(kpidx).unpx_);
        rdist = cv::norm(right_px_proj - vkps.at(kpidx).runpx_);

        if( ldist > pslamstate_->fmax_reproj_err_
            || rdist > pslamstate_->fmax_reproj_err_ ) {
            frame.removeStereoKeypointById(vkps.at(kpidx).lmid_);
            continue;
        }

        // Project MP in world frame
        wpt = frame.projCamToWorld(left_pt);

        pmap_->updateMapPoint(vkps.at(kpidx).lmid_, wpt, 1./left_pt.z());

        good++;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> Stereo Mapping : " << good << " 3D MPs out of " 
            << nbstereo << " kps !\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_TriangulateStereo");
}

/**
 * @brief   三角化计算公式 
*/
inline Eigen::Vector3d Mapper::computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    // OpenGV Triangulate
    return MultiViewGeometry::triangulate(T, bvl, bvr);
}

/**
 * @brief   指定帧和局部地图匹配 
 * @param   frame 指定帧 
*/
bool Mapper::matchingToLocalMap(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_MatchingToLocalMap");

    // Maximum number of MPs to track
    // 局部地图中跟踪的特征点数量最大值  
    const size_t nmax_localplms = pslamstate_->nbmaxkps_ * 10;

    // If room for more kps, get the local map  of the oldest co KF
    // and add it to the set of MPs to search for
    auto cov_map = frame.getCovisibleKfMap();
    // 遍历当前帧的共视关键帧 
    if( frame.set_local_mapids_.size() < nmax_localplms ) 
    {
        // 获取共视帧 
        int kfid = cov_map.begin()->first;
        auto pkf = pmap_->getKeyframe(kfid);
        while( pkf == nullptr  && kfid > 0 && !bnewkfavailable_ ) {
            // 拿到共视帧 
            kfid--;
            pkf = pmap_->getKeyframe(kfid);
        }

        // Skip if no time
        if( bnewkfavailable_ ) {// 如果没有时间，就不进行这一步
            return false;
        }
        
        if( pkf != nullptr ) {
            // 将共视帧的特征点加入到当前帧的局部地图中
            frame.set_local_mapids_.insert( pkf->set_local_mapids_.begin(), pkf->set_local_mapids_.end() );
        }

        // If still far not enough, go for another round
        if( pkf->kfid_ > 0 && frame.set_local_mapids_.size() < 0.5 * nmax_localplms )
        {// 如果特征点数量不够
            // 在得到一帧共视帧 
            pkf = pmap_->getKeyframe(pkf->kfid_);
            while( pkf == nullptr  && kfid > 0 && !bnewkfavailable_ ) {
                kfid--;
                pkf = pmap_->getKeyframe(kfid);
            }

            // Skip if no time
            if( bnewkfavailable_ ) {
                return false;
            }
            // 将共视帧的特征点加入到当前帧的局部地图中 
            if( pkf != nullptr ) {
                frame.set_local_mapids_.insert( pkf->set_local_mapids_.begin(), pkf->set_local_mapids_.end() );
            }
        }
    }

    // Skip if no time
    if( bnewkfavailable_ ) {
        return false;
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> matchToLocalMap() --> Number of local MPs selected : " 
            << frame.set_local_mapids_.size() << "\n";

    // Track local map
    // 当前帧和地图点的匹配
    std::map<int,int> map_previd_newid = matchToMap(
                                            frame, pslamstate_->fmax_proj_pxdist_, 
                                            pslamstate_->fmax_desc_dist_, frame.set_local_mapids_
                                            );

    size_t nbmatches = map_previd_newid.size();

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> matchToLocalMap() --> Match To Local Map found #" 
            << nbmatches << " matches \n"; 

    // Return if no matches
    if( nbmatches == 0 ) {
        return false;
    }

    // Merge in a thread to avoid waiting for BA to finish
    // mergeMatches(frame, map_previd_newid);
    // 合并匹配结果 
    std::thread thread(&Mapper::mergeMatches, this, std::ref(frame), map_previd_newid);
    thread.detach();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_MatchingToLocalMap");
        
    return true;
}

/**
 * @brief  合并匹配结果
 * @param frame 当前帧 
 * @param map_kpids_lmids 匹配结果 
*/
void Mapper::mergeMatches(const Frame &frame, const std::map<int,int> &map_kpids_lmids)
{
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);

    // Merge the matches
    for( const auto &ids : map_kpids_lmids )
    {// 遍历匹配结果 
        int prevlmid = ids.first;
        int newlmid = ids.second;
        // 合并地图点 
        pmap_->mergeMapPoints(prevlmid, newlmid);
    }

    if( pslamstate_->debug_ )
        std::cout << "\n >>> matchToLocalMap() / mergeMatches() --> Number of merges : " 
            << map_kpids_lmids.size() << "\n";
}

/**
 * @brief 当前帧和地图匹配 
 * @param frame 当前帧
 * @param fmaxprojerr 最大投影误差
 * @param fdistratio 最大描述子距离比
 * @param set_local_lmids 地图中的地图点
 * 
*/
std::map<int,int> Mapper::matchToMap(const Frame &frame, const float fmaxprojerr, const float fdistratio, std::unordered_set<int> &set_local_lmids)
{
    std::map<int,int> map_previd_newid;

    // Leave if local map is empty
    if( set_local_lmids.empty() ) {
        // 地图点为空，则返回空 
        return map_previd_newid;
    }

    // Compute max field of view
    // 当前帧的视野范围 
    const float vfov = 0.5 * frame.pcalib_leftcam_->img_h_ / frame.pcalib_leftcam_->fy_;
    const float hfov = 0.5 * frame.pcalib_leftcam_->img_w_ / frame.pcalib_leftcam_->fx_;
    // 视野范围的 atan 
    float maxradfov = 0.;
    if( hfov > vfov ) {
        maxradfov = std::atan(hfov);
    } else {
        maxradfov = std::atan(vfov);
    }
    // 视野范围的 cos 
    const float view_th = std::cos(maxradfov);

    // Define max distance from projection
    float dmaxpxdist = fmaxprojerr;// 最大投影误差 
    if( frame.nb3dkps_ < 30 ) {
        dmaxpxdist *= 2.;
    }

    std::map<int, std::vector<std::pair<int, float>>> map_kpids_vlmidsdist;

    // Go through all MP from the local map
    for( const int lmid : set_local_lmids )
    {// 遍历地图中所有的地图点 
        if( bnewkfavailable_ ) {
            break;
        }
        // 地图点是否被观测到，已经被观测到的不需要再次观测 
        if( frame.isObservingKp(lmid) ) {
            continue;
        }
        // 地图点 
        auto plm = pmap_->getMapPoint(lmid);

        if( plm == nullptr ) {
            continue;
        } else if( !plm->is3d_ || plm->desc_.empty() ) {
            continue;
        }
        // 世界坐标 
        Eigen::Vector3d wpt = plm->getPoint();

        //Project 3D MP into KF's image
        // 投影到当前帧的图像上作为初始值
        Eigen::Vector3d campt = frame.projWorldToCam(wpt);

        if( campt.z() < 0.1 ) {
            continue;
        }
        // 视角变换 
        float view_angle = campt.z() / campt.norm();

        if( fabs(view_angle) < view_th ) {
            continue;
        }
        // 投影到图像上 
        cv::Point2f projpx = frame.projCamToImageDist(campt);

        if( !frame.isInImage(projpx) ) {
            continue;
        }

        // Get all the kps around the MP's projection
        // 获取投影点附近的关键点 
        auto vnearkps = frame.getSurroundingKeypoints(projpx);

        // Find two best matches
        float mindist = plm->desc_.cols * fdistratio * 8.; // * 8 to get bits size
        int bestid = -1;
        int secid = -1;

        float bestdist = mindist;
        float secdist = mindist;

        std::vector<int> vkpids;
        std::vector<float> vpxdist;
        cv::Mat descs;

        for( const auto &kp : vnearkps )
        {// 遍历附近的关键点 
            if( kp.lmid_ < 0 ) {
                continue;
            }
            // 距离不能太大 
            float pxdist = cv::norm(projpx - kp.px_);

            if( pxdist > dmaxpxdist ) {
                continue;
            }

            // Check that this kp and the MP are indeed
            // candidates for matching (by ensuring that they
            // are never both observed in a given KF)
            auto pkplm = pmap_->getMapPoint(kp.lmid_);

            if( pkplm == nullptr ) {
                pmap_->removeMapPointObs(kp.lmid_,frame.kfid_);
                continue;
            }

            if( pkplm->desc_.empty() ) {
                continue;
            }
            bool is_candidate = true;
            auto set_plmkfs = plm->getKfObsSet();
            for( const auto &kfid : pkplm->getKfObsSet() ) {
                if( set_plmkfs.count(kfid) ) {
                    is_candidate = false;
                    break;
                }
            }
            if( !is_candidate ) {
                continue;
            }

            float coprojpx = 0.;
            size_t nbcokp = 0;

            for( const auto &kfid : pkplm->getKfObsSet() ) {
                auto pcokf = pmap_->getKeyframe(kfid);
                if( pcokf != nullptr ) {
                    auto cokp = pcokf->getKeypointById(kp.lmid_);
                    if( cokp.lmid_ == kp.lmid_ ) {
                        coprojpx += cv::norm(cokp.px_ - pcokf->projWorldToImageDist(wpt));
                        nbcokp++;
                    } else {
                        pmap_->removeMapPointObs(kp.lmid_, kfid);
                    }
                } else {
                    pmap_->removeMapPointObs(kp.lmid_, kfid);
                }
            }

            if( coprojpx / nbcokp > dmaxpxdist ) {
                continue;
            }
            
            float dist = plm->computeMinDescDist(*pkplm);

            if( dist <= bestdist ) {
                secdist = bestdist; // Will stay at mindist 1st time
                secid = bestid; // Will stay at -1 1st time

                bestdist = dist;
                bestid = kp.lmid_;
            }
            else if( dist <= secdist ) {
                secdist = dist;
                secid = kp.lmid_;
            }
        }

        if( bestid != -1 && secid != -1 ) {
            if( 0.9 * secdist < bestdist ) {
                bestid = -1;
            }
        }

        if( bestid < 0 ) {
            continue;
        }

        std::pair<int, float> lmid_dist(lmid, bestdist);
        if( !map_kpids_vlmidsdist.count(bestid) ) {
            std::vector<std::pair<int, float>> v(1,lmid_dist);
            map_kpids_vlmidsdist.emplace(bestid, v);
        } else {
            map_kpids_vlmidsdist.at(bestid).push_back(lmid_dist);
        }
    }

    for( const auto &kpid_vlmidsdist : map_kpids_vlmidsdist )
    {
        int kpid = kpid_vlmidsdist.first;

        float bestdist = 1024;
        int bestlmid = -1;

        for( const auto &lmid_dist : kpid_vlmidsdist.second ) {
            if( lmid_dist.second <= bestdist ) {
                bestdist = lmid_dist.second;
                bestlmid = lmid_dist.first;
            }
        }

        if( bestlmid >= 0 ) {
            map_previd_newid.emplace(kpid, bestlmid);
        }
    }

    return map_previd_newid;
}

/**
 * @brief 运行全局BA算法
*/
void Mapper::runFullBA()
{
    bool use_robust_cost = true;
    pestimator_->poptimizer_->fullBA(use_robust_cost);
}

/**
 * @brief 从队列中获取新的关键帧
 * @param kf 新的关键帧 
 * @return 是否成功获取 
*/
bool Mapper::getNewKf(Keyframe &kf)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

    // Check if new KF is available
    if( qkfs_.empty() ) {// 队列中没有关键帧 
        bnewkfavailable_ = false;
        return false;
    }

    // Get new KF and signal BA to stop if
    // it is still processing the previous KF
    kf = qkfs_.front();// 获取队列中的第一个关键帧 
    qkfs_.pop();// 弹出第一个关键帧

    // Setting bnewkfavailable_ to true will limit
    // the processing of the KF to triangulation and postpone
    // other costly tasks to next KF as we are running late!
    // 判断是否还存在关键帧 
    if( qkfs_.empty() ) {
        bnewkfavailable_ = false;
    } else {
        bnewkfavailable_ = true;
    }

    return true;
}


/**
 * @brief 添加新的关键帧，外部调用，添加关键帧数据 
 * @param kf 关键帧数据 
*/
void Mapper::addNewKf(const Keyframe &kf)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);
    // 添加关键帧数据 
    qkfs_.push(kf);
    // 设置新的关键帧可用
    bnewkfavailable_ = true;
}

/**
 * @brief 重置Mapper 线程
*/
void Mapper::reset()
{
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    bnewkfavailable_ = false;
    bwaiting_for_lc_ = false;
    bexit_required_ = false; 

    std::queue<Keyframe> empty;
    std::swap(qkfs_, empty);
}
