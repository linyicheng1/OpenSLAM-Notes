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

#include <opencv2/highgui.hpp>

#include "multi_view_geometry.hpp"

#include "map_manager.hpp"

/**
 * @brief 构造函数 
 * @param pstate slam状态指针
 * @param pframe 当前帧指针 
 * @param pfeatextract 特征提取器指针 
 * @param ptracker 特征跟踪器指针
*/
MapManager::MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker)
    : nlmid_(0), nkfid_(0), nblms_(0), nbkfs_(0), pslamstate_(pstate), pfeatextract_(pfeatextract), ptracker_(ptracker), pcurframe_(pframe)
{
    pcloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcloud_->points.reserve(1e5);
}


// This function turn the current frame into a Keyframe.
// Keypoints extraction is performed and the related MPs and
// the new KF are added to the map.

/**
 * @brief 如果判定当前帧为关键帧，则调用该函数，创建新的关键帧
*/
void MapManager::createKeyframe(const cv::Mat &im, const cv::Mat &imraw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_createKeyframe");

    // Prepare Frame to become a KF
    // (Update observations between MPs / KFs)
    // 当前帧成为关键帧的准备工作 
    prepareFrame();

    // Detect in im and describe in imraw
    // 提取当前帧的特征点 
    extractKeypoints(im, imraw);

    // Add KF to the map
    // 将当前帧添加到地图中 
    addKeyframe();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_createKeyframe");
}

// Prepare Frame to become a KF
// (Update observations between MPs / KFs)
// 当前帧成为关键帧的准备工作 
void MapManager::prepareFrame()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_prepareFrame");

    // Update new KF id
    pcurframe_->kfid_ = nkfid_; // 当前帧的关键帧id 

    // Filter if too many kps
    // 过滤掉当前帧的特征点数目过多的情况 
    if( (int)pcurframe_->nbkps_ > pslamstate_->nbmaxkps_ ) {
        // 遍历所有栅格
        for( const auto &vkpids : pcurframe_->vgridkps_ ) {
            // 如果栅格中的特征点数目大于阈值，则删除栅格中的特征点 
            if( vkpids.size() > 2 ) {
                int lmid2remove = -1;
                size_t minnbobs = std::numeric_limits<size_t>::max();
                for( const auto &lmid : vkpids ) {
                    auto plmit = map_plms_.find(lmid);
                    if( plmit != map_plms_.end() ) {
                        size_t nbobs = plmit->second->getKfObsSet().size();
                        if( nbobs < minnbobs ) {
                            lmid2remove = lmid;
                            minnbobs = nbobs;
                        }
                    } else {
                        removeObsFromCurFrameById(lmid);
                        break;
                    }
                }
                if( lmid2remove >= 0 ) {
                    removeObsFromCurFrameById(lmid2remove);
                }
            }
        }
    }

    for( const auto &kp : pcurframe_->getKeypoints() ) {

        // Get the related MP
        // 获取当前帧中特征点对应的地图点 
        auto plmit = map_plms_.find(kp.lmid_);
        // 如果地图点不存在，则删除当前帧中的特征点 
        if( plmit == map_plms_.end() ) {
            removeObsFromCurFrameById(kp.lmid_);
            continue;
        }
        // 如果地图点存在，则将当前帧中的特征点添加到地图点的观测中 
        // Relate new KF id to the MP
        plmit->second->addKfObs(nkfid_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_prepareFrame");
}

/**
 * @brief 更新关键帧的共视图
 * @param frame 当前帧
*/
void MapManager::updateFrameCovisibility(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_updateFrameCovisilbity");

    // Update the MPs and the covisilbe graph between KFs
    std::map<int,int> map_covkfs;// 地图共视关系图，<关键帧id，共视关系数目>
    std::unordered_set<int> set_local_mapids;// 局部地图点id集合

    for( const auto &kp : frame.getKeypoints() ) {
        // 遍历当前帧中的特征点 
        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);// 获取当前帧中特征点对应的地图点 
        // 如果地图点不存在，则删除当前帧中的特征点 
        if( plmit == map_plms_.end() ) {
            // 删除地图点的观测
            removeMapPointObs(kp.lmid_, frame.kfid_);
            // 删除当前帧中的特征点 
            removeObsFromCurFrameById(kp.lmid_);
            continue;
        }

        // Get the set of KFs observing this KF to update 
        // covisible KFs
        for( const auto &kfid : plmit->second->getKfObsSet() ) 
        {// 遍历地图点中的观测帧
            if( kfid != frame.kfid_ ) 
            {// 不等于当前帧
                auto it = map_covkfs.find(kfid);// 在共视关系图中查找该帧 
                if( it != map_covkfs.end() ) {// 找到了，共视关系数目加1 
                    it->second += 1;
                } else {// 没找到，添加到共视关系图中 
                    map_covkfs.emplace(kfid, 1);
                }
            }
        }
    }

    // Update covisibility for covisible KFs
    std::set<int> set_badkfids;
    for( const auto &kfid_cov : map_covkfs ) // 遍历共视关系图 
    {
        int kfid = kfid_cov.first;// 共视关键帧id 
        int covscore = kfid_cov.second;// 共视关系数目 
        
        auto pkfit = map_pkfs_.find(kfid);// 在关键帧中查找该帧 
        if( pkfit != map_pkfs_.end() ) 
        {// 找到了 
            // Will emplace or update covisiblity
            // 更新共视关系 
            pkfit->second->map_covkfs_[frame.kfid_] = covscore;

            // Set the unobserved local map for future tracking
            // 设置未观察到的局部地图，以便将来追踪
            for( const auto &kp : pkfit->second->getKeypoints3d() ) {
                // 共视帧中的地图点，如果当前帧没有观察到，则添加到未观察到的地图点集合中 
                if( !frame.isObservingKp(kp.lmid_) ) {
                    set_local_mapids.insert(kp.lmid_);
                }
            }
        } else {// 没找到，添加到坏帧集合中 
            set_badkfids.insert(kfid);
        }
    }
    // 对于坏帧，删除共视关系图中的坏帧 
    for( const auto &kfid : set_badkfids ) {
        map_covkfs.erase(kfid);
    }
    
    // Update the set of covisible KFs
    // 更新当前帧的共视关键帧集合
    frame.map_covkfs_.swap(map_covkfs);

    // Update local map of unobserved MPs
    // 更新未观察到的地图点 
    if( set_local_mapids.size() > 0.5 * frame.set_local_mapids_.size() ) {
        frame.set_local_mapids_.swap(set_local_mapids);
    } else {
        frame.set_local_mapids_.insert(set_local_mapids.begin(), set_local_mapids.end());
    }
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_updateFrameCovisilbity");
}

/**
 * @brief 添加特征点到当前帧中 
 * @param im 当前帧图像
 * @param vpts 当前帧图像中的特征点
 * @param frame 当前帧
*/
void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create MPs
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ ) {
        // 特征点添加到当前帧中 
        // Add keypoint to current frame
        frame.addKeypoint(vpts.at(i), nlmid_);

        // Create landmark with same id
        // 特征点的颜色
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
    
        // 创建地图点颜色
        addMapPoint(col);
    }
}

/**
 * @brief 添加特征点到当前帧中，并设置尺度 
 * @param im 当前帧图像
 * @param vpts 当前帧图像中的特征点
 * @param vscales 当前帧图像中的特征点尺度
 * @param frame 当前帧
*/
void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
    const std::vector<int> &vscales, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        // Add keypoint to current frame
        frame.addKeypoint(vpts.at(i), nlmid_, vscales.at(i));

        // Create landmark with same id
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
        addMapPoint(col);
    }
}

/**
 * @brief 添加特征点到当前帧中，并设置描述子
 * @param im 当前帧图像
 * @param vpts 当前帧图像中的特征点
 * @param vdescs 当前帧图像中的特征点描述子
 * @param frame 当前帧
*/

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_, vdescs.at(i));

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(vdescs.at(i), col);
        } 
        else {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_);

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}

/**
 * @brief 添加特征点到当前帧中，并设置尺度和描述子
 * @param im 当前帧图像
 * @param vpts 当前帧图像中的特征点
 * @param vscales 当前帧图像中的特征点尺度
 * @param vdescs 当前帧图像中的特征点描述子
 * @param frame 当前帧
*/
void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<int> &vscales, const std::vector<float> &vangles, 
                        const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();// 特征点数量 
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {// 特征点描述子不为空 
            // 添加特征点到当前帧中 
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_, vdescs.at(i), vscales.at(i), vangles.at(i));

            // Create landmark with same id
            // 创建地图点 
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(vdescs.at(i), col);
        } 
        else {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_);

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}

// Extract new kps into provided image and update cur. Frame
/**
 * @brief 提取新的特征点到当前帧中
 * @param im 当前帧图像
 * @param imraw 当前帧图像
*/
void MapManager::extractKeypoints(const cv::Mat &im, const cv::Mat &imraw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_extractKeypoints");
    // 通过光流法已经获得了的特征点 
    std::vector<Keypoint> vkps = pcurframe_->getKeypoints();
    // 转换成cv::Point2f类型 
    std::vector<cv::Point2f> vpts;
    std::vector<int> vscales;
    std::vector<float> vangles;

    for( auto &kp : vkps ) {
        vpts.push_back(kp.px_);
    }
    // 如果使用brief描述子, 则需要计算角度和尺度 
    if( pslamstate_->use_brief_ ) {
        describeKeypoints(imraw, vkps, vpts);
    }
    // 目标提取的特征点数目， 目标数量 - 已经填充的特征点数目 
    int nb2detect = pslamstate_->nbmaxkps_ - pcurframe_->noccupcells_;

    if( nb2detect > 0 ) {// 如果需要提取的特征点数目大于0 
        // Detect kps in the provided images
        // using the cur kps and img roi to set a mask
        std::vector<cv::Point2f> vnewpts;
        // 如果使用的是gftt角点检测器 
        if( pslamstate_->use_shi_tomasi_ ) {
            vnewpts = pfeatextract_->detectGFTT(im, vpts, pcurframe_->pcalib_leftcam_->roi_mask_, nb2detect);
        } 
        else if( pslamstate_->use_fast_ ) {// 如果使用的是fast角点检测器 
            vnewpts = pfeatextract_->detectGridFAST(im, pslamstate_->nmaxdist_, vpts, pcurframe_->pcalib_leftcam_->roi_rect_);
        } 
        else if ( pslamstate_->use_singlescale_detector_ ) {// 如果使用的是单尺度角点检测器
            vnewpts = pfeatextract_->detectSingleScale(im, pslamstate_->nmaxdist_, vpts, pcurframe_->pcalib_leftcam_->roi_rect_);
        } else {
            std::cerr << "\n Choose a detector between : gftt / FAST / SingleScale detector!";
            exit(-1);
        }

        if( !vnewpts.empty() ) {
            // 对提取的特征点进行描述子计算 
            if( pslamstate_->use_brief_ ) {
                std::vector<cv::Mat> vdescs;
                vdescs = pfeatextract_->describeBRIEF(imraw, vnewpts);
                // 将提取的特征点添加到当前帧中 
                addKeypointsToFrame(im, vnewpts, vdescs, *pcurframe_);
            } 
            else if( pslamstate_->use_shi_tomasi_ || pslamstate_->use_fast_ 
                || pslamstate_->use_singlescale_detector_ ) 
            {
                // 不使用描述子， 直接将特征点添加到当前帧中
                addKeypointsToFrame(im, vnewpts, *pcurframe_);
            }
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_extractKeypoints");
}


// Describe cur frame kps in cur image
/**
 * @brief 计算当前帧中特征点的描述子
 * @param im 当前帧图像
 * @param vkps 当前帧中的特征点
 * @param vpts 当前帧中的特征点
 * @param pvscales 当前帧中的特征点尺度
 * @param pvangles 当前帧中的特征点角度
*/
void MapManager::describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, const std::vector<cv::Point2f> &vpts, const std::vector<int> *pvscales, std::vector<float> *pvangles)
{
    size_t nbkps = vkps.size();// 当前帧中的特征点数目
    std::vector<cv::Mat> vdescs;// 当前帧中的特征点描述子
    // 使用brief描述子 
    if( pslamstate_->use_brief_ ) {
        vdescs = pfeatextract_->describeBRIEF(im, vpts);
    }

    assert( vkps.size() == vdescs.size() );
    // 更新当前帧中的特征点描述子
    for( size_t i = 0 ; i < nbkps ; i++ ) {
        if( !vdescs.at(i).empty() ) {
            // 更新当前帧中的特征点描述子
            pcurframe_->updateKeypointDesc(vkps.at(i).lmid_, vdescs.at(i));
            // 更新地图中的特征点描述子 
            map_plms_.at(vkps.at(i).lmid_)->addDesc(pcurframe_->kfid_, vdescs.at(i));
        }
    }
}


// This function is responsible for performing stereo matching operations
// for the means of triangulation
/**
 * @brief 双目匹配 
 * @param frame 当前帧
 * @param vleftpyr 左图金字塔
 * @param vrightpyr 右图金字塔
*/
void MapManager::stereoMatching(Frame &frame, const std::vector<cv::Mat> &vleftpyr, const std::vector<cv::Mat> &vrightpyr) 
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_stereoMatching");

    // Find stereo correspondances with left kps
    auto vleftkps = frame.getKeypoints();// 获取当前帧的特征点 
    size_t nbkps = vleftkps.size();// 特征点的数量 

    // ZNCC Parameters
    size_t nmaxpyrlvl = pslamstate_->nklt_pyr_lvl_*2;// 金字塔层数 
    int winsize = 7;// 窗口大小 

    float uppyrcoef = std::pow(2,pslamstate_->nklt_pyr_lvl_);
    float downpyrcoef = 1. / uppyrcoef;
    
    std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(frame.nb3dkps_);
    v3dkps.reserve(frame.nb3dkps_);
    v3dpriors.reserve(frame.nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(nbkps);
    vkps.reserve(nbkps);
    vpriors.reserve(nbkps);

    for( size_t i = 0 ; i < nbkps ; i++ )
    {// 遍历当前帧中的特征点 
        // Set left kp
        auto &kp = vleftkps.at(i);// 特征点

        // Set prior right kp
        cv::Point2f priorpt = kp.px_;// 先验位置

        // If 3D, check if we can find a prior in right image
        if( kp.is3d_ ) {// 如果特征点是3D点 
            auto plm = getMapPoint(kp.lmid_);
            if( plm != nullptr ) {
                // 3D点投影到右图像上的位置 
                cv::Point2f projpt = frame.projWorldToRightImageDist(plm->getPoint());
                if( frame.isInRightImage(projpt) ) {
                    v3dkps.push_back(kp.px_);// 左图像上的特征点
                    v3dpriors.push_back(projpt);// 右图像上的特征点先验位置 
                    v3dkpids.push_back(kp.lmid_);
                    continue;
                } 
            } else {
                removeMapPointObs(kp.lmid_, frame.kfid_);
                continue;
            }
        } 
        
        // If stereo rect images, prior from SAD
        if( pslamstate_->bdo_stereo_rect_ ) {

            float xprior = -1.;
            float l1err;

            cv::Point2f pyrleftpt = kp.px_ * downpyrcoef;
            // 线搜索作为先验数据
            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrleftpt, winsize, xprior, l1err, true);

            xprior *= uppyrcoef;

            if( xprior >= 0 && xprior <= kp.px_.x ) {
                priorpt.x = xprior;
            }

        }
        else { // Generate prior from 3d neighbors
            // 从3D点的邻域中生成先验数据
            const size_t nbmin3dcokps = 1;
            // 获取周围的特征点 
            auto vnearkps = frame.getSurroundingKeypoints(kp);
            if( vnearkps.size() >= nbmin3dcokps ) 
            {
                std::vector<Keypoint> vnear3dkps;
                vnear3dkps.reserve(vnearkps.size());
                for( const auto &cokp : vnearkps ) {
                    if( cokp.is3d_ ) {
                        vnear3dkps.push_back(cokp);
                    }
                }

                if( vnear3dkps.size() >= nbmin3dcokps ) {
                
                    size_t nb3dkp = 0;
                    double mean_z = 0.;
                    double weights = 0.;

                    for( const auto &cokp : vnear3dkps ) {
                        auto plm = getMapPoint(cokp.lmid_);
                        if( plm != nullptr ) {
                            nb3dkp++;
                            double coef = 1. / cv::norm(cokp.unpx_ - kp.unpx_);
                            weights += coef;
                            mean_z += coef * frame.projWorldToCam(plm->getPoint()).z();
                        }
                    }

                    if( nb3dkp >= nbmin3dcokps ) {
                        mean_z /= weights;
                        Eigen::Vector3d predcampt = mean_z * ( kp.bv_ / kp.bv_.z() );

                        cv::Point2f projpt = frame.projCamToRightImageDist(predcampt);

                        if( frame.isInRightImage(projpt) ) 
                        {
                            v3dkps.push_back(kp.px_);
                            v3dpriors.push_back(projpt);
                            v3dkpids.push_back(kp.lmid_);
                            continue;
                        }
                    }
                }
            }
        }
        // 待匹配特征点位置 
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(priorpt);
    }

    // Storing good tracks   
    std::vector<cv::Point2f> vgoodrkps;
    std::vector<int> vgoodids;
    vgoodrkps.reserve(nbkps);
    vgoodids.reserve(nbkps);

    // 1st track 3d kps if using prior
    if( !v3dpriors.empty() ) 
    {
        size_t nbpyrlvl = 1;
        int nwinsize = pslamstate_->nklt_win_size_; // What about a smaller window here?

        if( vleftpyr.size() < 2*(nbpyrlvl+1) ) {
            nbpyrlvl = vleftpyr.size() / 2 - 1;
        }

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    vleftpyr, 
                    vrightpyr, 
                    nwinsize, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nb3dkps = v3dkps.size();
        
        for(size_t i = 0 ; i < nb3dkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(v3dpriors.at(i));
                vgoodids.push_back(v3dkpids.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                // without prior for 2d kps
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) 
            std::cout << "\n >>> Stereo KLT Tracking on priors : " << nbgood 
                << " out of " << nb3dkps << " kps tracked!\n";
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    vleftpyr, 
                    vrightpyr, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nb2dkps = vkps.size();

        for(size_t i = 0 ; i < nb2dkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(vpriors.at(i));
                vgoodids.push_back(vkpids.at(i));
                nbgood++;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking w. no priors : " << nbgood
                << " out of " << nb2dkps << " kps tracked!\n";
    }

    nbkps = vgoodids.size();
    size_t nbgood = 0;

    float epi_err = 0.;

    for( size_t i = 0; i < nbkps ; i++ ) 
    {
        cv::Point2f lunpx = frame.getKeypointById(vgoodids.at(i)).unpx_;
        cv::Point2f runpx = frame.pcalib_rightcam_->undistortImagePoint(vgoodrkps.at(i));

        // Check epipolar consistency (same row for rectified images)
        if( pslamstate_->bdo_stereo_rect_ ) {
            epi_err = fabs(lunpx.y - runpx.y);
            // Correct right kp to be on the same row
            vgoodrkps.at(i).y = lunpx.y;
        }
        else {
            epi_err = MultiViewGeometry::computeSampsonDistance(frame.Frl_, lunpx, runpx);
        }
        
        if( epi_err <= 2. ) 
        {
            frame.updateKeypointStereo(vgoodids.at(i), vgoodrkps.at(i));
            nbgood++;
        }
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Nb of stereo tracks: " << nbgood
            << " out of " << nbkps << "\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_stereoMatching");
}


/**
 * @brief 双目三角化 
 * @param T 两帧之间的变换矩阵 
 * @param bvl 左目归一化坐标 
 * @param bvr 右目归一化坐标 
*/
Eigen::Vector3d MapManager::computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    // OpenGV Triangulate， 三角化算法
    return MultiViewGeometry::triangulate(T, bvl, bvr);
}

// This function copies cur. Frame to add it to the KF map
// 将当前帧复制一份，添加到关键帧地图中
void MapManager::addKeyframe()
{
    // Create a copy of Cur. Frame shared_ptr for creating an 
    // independant KF to add to the map
    std::shared_ptr<Frame> pkf = std::allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), *pcurframe_);

    std::lock_guard<std::mutex> lock(kf_mutex_);

    // Add KF to the unordered map and update id/nb
    map_pkfs_.emplace(nkfid_, pkf);
    nbkfs_++;
    nkfid_++;
}

// This function adds a new MP to the map
/**
 * @brief 添加地图点
 * @param color 地图点颜色 
*/
void MapManager::addMapPoint(const cv::Scalar &color)
{
    // Create a new MP with a unique lmid and a KF id obs
    std::shared_ptr<MapPoint> plm = std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), nlmid_, nkfid_, color);

    // Add new MP to the map and update id/nb
    map_plms_.emplace(nlmid_, plm);// 添加到地图点集合中
    nlmid_++;
    nblms_++;

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if( plm->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plm->color_[0] 
                                    , plm->color_[0]
                                    , plm->color_[0]
                                    );
    }
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.push_back(colored_pt);
}


// This function adds a new MP to the map with desc
/**
 * @brief 添加地图点
 * @param desc 地图点描述子 
 * @param color 地图点颜色
*/
void MapManager::addMapPoint(const cv::Mat &desc, const cv::Scalar &color)
{
    // Create a new MP with a unique lmid and a KF id obs
    std::shared_ptr<MapPoint> plm = std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), nlmid_, nkfid_, desc, color);

    // Add new MP to the map and update id/nb
    map_plms_.emplace(nlmid_, plm); // 添加到地图点集合中
    nlmid_++;// 地图点id自增
    nblms_++;// 地图点数量自增

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if( plm->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plm->color_[0] 
                                    , plm->color_[0]
                                    , plm->color_[0]
                                    );
    }
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.push_back(colored_pt);
}

// Returns a shared_ptr of the req. KF
/**
 * @brief 获取关键帧 
 * @param kfid 关键帧id 
 * @return 关键帧 
*/
std::shared_ptr<Frame> MapManager::getKeyframe(const int kfid) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    // 查找关键帧
    auto it = map_pkfs_.find(kfid);
    if( it == map_pkfs_.end() ) {
        return nullptr;// 没有找到 
    }
    return it->second;// 找到了，返回关键帧
}

// Returns a shared_ptr of the req. MP
/**
 * @brief 获取地图点 
 * @param lmid 地图点id 
 * @return 地图点
*/
std::shared_ptr<MapPoint> MapManager::getMapPoint(const int lmid) const
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    // 查找地图点 
    auto it = map_plms_.find(lmid);
    if( it == map_plms_.end() ) {
        return nullptr;// 没有找到 
    }
    // 找到了，返回地图点 
    return it->second;
}

// Update a MP world pos.
/**
 * @brief 更新地图点的世界坐标
 * @param lmid 地图点id 
 * @param wpt 世界坐标 
 * @param kfanch_invdepth 关键帧逆深度
*/
void MapManager::updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, const double kfanch_invdepth)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);
    // 获取地图点
    auto plmit = map_plms_.find(lmid);

    if( plmit == map_plms_.end() ) {
        return;
    }

    if( plmit->second == nullptr ) {
        return;
    }

    // If MP 2D -> 3D => Notif. KFs 
    if( !plmit->second->is3d_ ) {// 如果地图点是2D点 
        for( const auto &kfid : plmit->second->getKfObsSet() ) {
            auto pkfit = map_pkfs_.find(kfid);// 获取关键帧
            if( pkfit != map_pkfs_.end() ) {
                // 将地图点转换为3D点 
                pkfit->second->turnKeypoint3d(lmid);
            } else {
                plmit->second->removeKfObs(kfid);
            }
        }
        // 如果当前帧中被观测到了
        if( plmit->second->isobs_ ) {
            pcurframe_->turnKeypoint3d(lmid);
        }
    }

    // Update MP world pos.
    if( kfanch_invdepth >= 0. ) {
        // 如果逆深度大于0，那么就是关键帧逆深度初始化的地图点
        plmit->second->setPoint(wpt, kfanch_invdepth);
    } else {
        plmit->second->setPoint(wpt);
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if(plmit->second->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plmit->second->color_[0] 
                                    , plmit->second->color_[0]
                                    , plmit->second->color_[0]
                                    );
    }
    colored_pt.x = wpt.x();
    colored_pt.y = wpt.y();
    colored_pt.z = wpt.z();
    pcloud_->points.at(lmid) = colored_pt;
}

// Add a new KF obs to provided MP (lmid)
/**
 * @brief 在lmid对应的MapPoint中添加一个新的观测帧kfid
 * @param lmid 地图点id
 * @param kfid 关键帧id
*/
void MapManager::addMapPointKfObs(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);
    // 关键帧 
    auto pkfit = map_pkfs_.find(kfid);
    // 地图点 
    auto plmit = map_plms_.find(lmid);

    if( pkfit == map_pkfs_.end() ) {
        return;
    }

    if( plmit == map_plms_.end() ) {
        return;
    }
    // 地图点添加观测帧 
    plmit->second->addKfObs(kfid);
    // 关键帧添加地图点
    for( const auto &cokfid : plmit->second->getKfObsSet() ) {
        if( cokfid != kfid ) {
            auto pcokfit =  map_pkfs_.find(cokfid);
            // 更新关键帧的共视关系 
            if( pcokfit != map_pkfs_.end() ) {
                pcokfit->second->addCovisibleKf(kfid);
                pkfit->second->addCovisibleKf(cokfid);
            } else {
                plmit->second->removeKfObs(cokfid);
            }
        }
    }
}

// Merge two MapPoints
/**
 * @brief 合并两个MapPoints 
 * @param prevlmid 旧的MapPoint的id 
 * @param newlmid 新的MapPoint的id
*/
void MapManager::mergeMapPoints(const int prevlmid, const int newlmid)
{
    // 1. Get Kf obs + descs from prev MP
    // 2. Remove prev MP
    // 3. Update new MP and related KF / cur Frame

    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get prev MP to merge into new MP

    auto pprevlmit = map_plms_.find(prevlmid);// 找到旧的MapPoint
    auto pnewlmit = map_plms_.find(newlmid);// 找到新的MapPoint

    if( pprevlmit == map_plms_.end() ) {// 如果旧的MapPoint不存在 
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as prevlm is null\n";
        return;
    } else if( pnewlmit == map_plms_.end() ) {// 如果新的MapPoint不存在
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is null\n";
        return;
    } else if ( !pnewlmit->second->is3d_ ) {// 如果新的MapPoint不是3D点
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is not 3d\n";
        return;
    }

    // 1. Get Kf obs + descs from prev MP
    // 获取新地图点的观测帧
    std::set<int> setnewkfids = pnewlmit->second->getKfObsSet();
    // 获取旧地图点的观测帧
    std::set<int> setprevkfids = pprevlmit->second->getKfObsSet();
    // 获取旧地图点的描述子
    std::unordered_map<int, cv::Mat> map_prev_kf_desc_ = pprevlmit->second->map_kf_desc_;

    // 3. Update new MP and related KF / cur Frame
    for( const auto &pkfid : setprevkfids ) 
    {// 遍历旧地图点的观测帧
        // Get prev KF and update keypoint
        auto pkfit =  map_pkfs_.find(pkfid);// 观测帧 
        if( pkfit != map_pkfs_.end() ) {// 找到了关键帧 
            // 替换旧地图点的观测帧为新地图点
            if( pkfit->second->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
            {
                // 新地图点中添加观测帧 
                pnewlmit->second->addKfObs(pkfid);
                // 观测帧中添加新的共视关系 
                for( const auto &nkfid : setnewkfids ) {
                    auto pcokfit = map_pkfs_.find(nkfid);
                    if( pcokfit != map_pkfs_.end() ) {
                        pkfit->second->addCovisibleKf(nkfid);
                        pcokfit->second->addCovisibleKf(pkfid);
                    }
                }
            }
        }
    }
    // 新的地图点中添加旧地图点的描述子 
    for( const auto &kfid_desc : map_prev_kf_desc_ ) {
        pnewlmit->second->addDesc(kfid_desc.first, kfid_desc.second);
    }

    // Turn new MP observed by cur Frame if prev MP
    // was + update cur Frame's kp ref to new MP
    if( pcurframe_->isObservingKp(prevlmid) ) // 更新当前帧的观测
    {
        if( pcurframe_->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
        {
            setMapPointObs(newlmid);
        }
    }
    // 如果原来的地图点是3D点，那么地图点的数量减一 
    if( pprevlmit->second->is3d_ ) {
        nblms_--; 
    }
    // 删掉被合并的旧地图点
    // Erase MP and update nb MPs
    map_plms_.erase( pprevlmit );
    // 更新颜色 
    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points[prevlmid] = colored_pt;
}

// Remove a KF from the map
/**
 * @brief 地图中删除关键帧
 * @param kfid 关键帧id
*/
void MapManager::removeKeyframe(const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get KF to remove
    auto pkfit = map_pkfs_.find(kfid);// 关键帧
    // Skip if KF does not exist
    if( pkfit == map_pkfs_.end() ) {// 如果关键帧不存在 
        return;
    }

    // Remove the KF obs from all observed MP
    // 删除关键帧中对该地图点的观测
    for( const auto &kp : pkfit->second->getKeypoints() ) {
        // Get MP and remove KF obs
        auto plmit = map_plms_.find(kp.lmid_);
        if( plmit == map_plms_.end() ) {
            continue;
        }
        plmit->second->removeKfObs(kfid);
    }
    // 更新共视关系 
    for( const auto &kfid_cov : pkfit->second->getCovisibleKfMap() ) {
        auto pcokfit = map_pkfs_.find(kfid_cov.first);
        if( pcokfit != map_pkfs_.end() ) {
            pcokfit->second->removeCovisibleKf(kfid);
        }
    }

    // Remove KF and update nb KFs
    map_pkfs_.erase( pkfit );// 删除关键帧
    nbkfs_--;// 更新关键帧数量

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> removeKeyframe() --> Removed KF #" << kfid;
}

// Remove a MP from the map
/**
 * @brief 删除地图中的一个地图点 
 * @param lmid 地图点的id
*/
void MapManager::removeMapPoint(const int lmid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get related MP
    auto plmit = map_plms_.find(lmid);// 地图中找到这个地图点 
    // Skip if MP does not exist
    if( plmit != map_plms_.end() ) {// 找到了这个地图点
        // Remove all observations from KFs
        for( const auto &kfid : plmit->second->getKfObsSet() ) 
        {// 关联的关键帧 
            auto pkfit = map_pkfs_.find(kfid);
            if( pkfit == map_pkfs_.end() ) {// 关键帧不存在
                continue;
            }
            // 删除特征点 
            pkfit->second->removeKeypointById(lmid);
            // 减少共视关系
            for( const auto &cokfid : plmit->second->getKfObsSet() ) {
                if( cokfid != kfid ) {
                    pkfit->second->decreaseCovisibleKf(cokfid);
                }
            }
        }

        // If obs in cur Frame, remove cur obs
        if( plmit->second->isobs_ ) {// 如果这个地图点在当前帧中 
            // 当前帧删除该地图点
            pcurframe_->removeKeypointById(lmid);
        }
        // 如果是3d点，减少地图点数量 
        if( plmit->second->is3d_ ) {
            nblms_--; 
        }

        // Erase MP and update nb MPs
        // 最后删除地图点
        map_plms_.erase( plmit );
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.at(lmid) = colored_pt;
}

// Remove a KF obs from a MP
/**
 * @brief 删除地图点的观测
 * @param lmid 地图点id 
 * @param kfid 关键帧id 
*/
void MapManager::removeMapPointObs(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Remove MP obs from KF
    auto pkfit = map_pkfs_.find(kfid);// 获取关键点
    if( pkfit != map_pkfs_.end() ) {// 如果关键点存在 
        // 删除 lmid 帧中的关键点观测
        pkfit->second->removeKeypointById(lmid);
    }

    // Remove KF obs from MP
    auto plmit = map_plms_.find(lmid);// 获取关键帧 

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        return;
    }
    // 删除关键帧中的地图点观测 
    plmit->second->removeKfObs(kfid);

    // 更新共视关键帧
    if( pkfit != map_pkfs_.end() ) {
        for( const auto &cokfid : plmit->second->getKfObsSet() ) {
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                pkfit->second->decreaseCovisibleKf(cokfid);
                pcokfit->second->decreaseCovisibleKf(kfid);
            }
        }
    }
}

/**
 * @brief 删除当前帧中的某个地图点观测 
 * @param lm 地图点 
 * @param frame 当前帧
*/
void MapManager::removeMapPointObs(MapPoint &lm, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);
    // 当前帧删除地图点观测 
    frame.removeKeypointById(lm.lmid_);
    // 地图点删除当前帧观测 
    lm.removeKfObs(frame.kfid_);

    for( const auto &cokfid : lm.getKfObsSet() ) {// 获取地图点的所有观测帧
        if( cokfid != frame.kfid_ ) {// 不是当前帧 
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {// 找到观测帧
                frame.decreaseCovisibleKf(cokfid);// 当前帧减少观测帧
                // 观测帧减少当前帧
                pcokfit->second->decreaseCovisibleKf(frame.kfid_);
            }
        }
    }
}

// Remove a MP obs from cur Frame
/**
 * @brief 删除当前帧中的观测点
 * @param lmid 地图点 id
*/
void MapManager::removeObsFromCurFrameById(const int lmid)
{
    // Remove cur obs
    // 当前帧中删除观测点 
    pcurframe_->removeKeypointById(lmid);
    
    // Set MP as not obs
    // 找到地图点
    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {// 没有地图点 
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return;
    }
    // 设置地图点为非观测点 
    plmit->second->isobs_ = false;

    // 更新地图点颜色
    // Update MP color
    colored_pt = pcl::PointXYZRGB(plmit->second->color_[0] 
                                , plmit->second->color_[0]
                                , plmit->second->color_[0]
                                );
                                
    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;
}

/**
 * @brief 设置地图点的观测 
*/
bool MapManager::setMapPointObs(const int lmid) 
{
    // 不存在该地图点 
    if( lmid >= (int)pcloud_->points.size() ) {
        return false;
    }   
    // 找到该地图点 
    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {// 不存在该地图点 
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return false;
    }

    plmit->second->isobs_ = true;// 设置该地图点为当前帧的观测点 

    // Update MP color
    // 设置该地图点的颜色 
    colored_pt = pcl::PointXYZRGB(200, 0, 0);
    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;

    return true;
}

// Reset MapManager
/**
 * @brief Reset MapManager
*/
void MapManager::reset()
{
    nlmid_ = 0;
    nkfid_ = 0;
    nblms_ = 0;
    nbkfs_ = 0;

    map_pkfs_.clear();
    map_plms_.clear();

    pcloud_->points.clear();
}
