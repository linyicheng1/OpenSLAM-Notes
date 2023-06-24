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

#include <opencv2/video/tracking.hpp>

#include "visual_front_end.hpp"
#include "multi_view_geometry.hpp"

#include <opencv2/highgui.hpp>

// 视觉前端类 
VisualFrontEnd::VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, 
        std::shared_ptr<MapManager> pmap, std::shared_ptr<FeatureTracker> ptracker)
    : pslamstate_(pstate), pcurframe_(pframe), pmap_(pmap), ptracker_(ptracker)
{}

// 视觉前端跟踪函数
// trackMono 
// createKeyframe 
bool VisualFrontEnd::visualTracking(cv::Mat &iml, double time)
{
    std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("0.Full-Front_End");
    // 基于单目的跟踪 
    bool iskfreq = trackMono(iml, time);

    // 是否需要关键帧 
    if( iskfreq ) {
        // 创建关键帧 
        pmap_->createKeyframe(cur_img_, iml);
        // 构建关键帧的光流金字塔
        if( pslamstate_->btrack_keyframetoframe_ ) {
            cv::buildOpticalFlowPyramid(cur_img_, kf_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "0.Full-Front_End");

    return iskfreq;
}


// Perform tracking in one image, update kps and MP obs, return true if a new KF is req.
/**
 * @brief 单目跟踪函数
 * @param im 输入图像
 * @param time 时间戳 
*/
bool VisualFrontEnd::trackMono(cv::Mat &im, double time)
{
    if( pslamstate_->debug_ )
        std::cout << "\n\n - [Visual-Front-End]: Track Mono Image\n";
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_Track-Mono");

    // Preprocess the new image
    // 对图像进行预处理 
    preprocessImage(im);

    // Create KF if 1st frame processed
    if( pcurframe_->id_ == 0 ) {
        // 第一张图像不需要跟踪，直接返回作为关键帧
        return true;
    }
    
    // Apply Motion model to predict cur Frame pose
    // 运动学模型预测当前帧的位姿 
    Sophus::SE3d Twc = pcurframe_->getTwc();
    motion_model_.applyMotionModel(Twc, time);
    // 更新当前帧的位姿
    pcurframe_->setTwc(Twc);
    
    // Track the new image
    if( pslamstate_->btrack_keyframetoframe_ ) {
        // 当前帧与关键帧之间的光流跟踪 
        kltTrackingFromKF();
    } else {
        // 当前帧与上一帧之间的光流跟踪 
        kltTracking();
    }

    if( pslamstate_->doepipolar_ ) {
        // Check2d2dOutliers
        // 2d-2d极线匹配外点剔除 
        epipolar2d2dFiltering();
    }

    if( pslamstate_->mono_ && !pslamstate_->bvision_init_ ) 
    {// 如果是单目，且还未初始化，则需要检查是否可以初始化 
        if( pcurframe_->nb2dkps_ < 50 ) {
            pslamstate_->breset_req_ = true;
            return false;
        } 
        else if( checkReadyForInit() ) {
            std::cout << "\n\n - [Visual-Front-End]: Mono Visual SLAM ready for initialization!";
            pslamstate_->bvision_init_ = true;
            return true;
        } 
        else {
            std::cout << "\n\n - [Visual-Front-End]: Not ready to init yet!";
            return false;
        }
    }

    // Compute Pose (2D-3D)
    // 2d-3d位姿估计 
    computePose();

    // Update Motion model from estimated pose
    // 从估计的位姿更新运动学模型 
    motion_model_.updateMotionModel(pcurframe_->Twc_, time);

    // Check if New KF req.
    // 检查是否需要新的关键帧 
    bool is_kf_req = checkNewKfReq();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_Track-Mono");

    return is_kf_req;
}


// KLT Tracking with motion prior
// 当前帧和上一帧之间的光流跟踪 
void VisualFrontEnd::kltTracking()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;// 3d kps id, 2d kps id 
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;// 3d 点位置，3d点的先验位置，2d点位置，2d点的先验位置
    std::vector<bool> vkpis3d;// 2d点是否有3d点对应

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(pcurframe_->nb3dkps_);
    v3dkps.reserve(pcurframe_->nb3dkps_);
    v3dpriors.reserve(pcurframe_->nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(pcurframe_->nbkps_);
    vkps.reserve(pcurframe_->nbkps_);
    vpriors.reserve(pcurframe_->nbkps_);

    vkpis3d.reserve(pcurframe_->nbkps_);


    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pcurframe_->mapkps_ ) 
    {// 遍历当前帧的所有特征点 

        auto &kp = it.second;

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {// 使用运动学模型初始化特征点的先验位置
            if( kp.is3d_ ) 
            {
                // 将3d点投影到图像上 
                cv::Point2f projpx = pcurframe_->projWorldToImageDist(pmap_->map_plms_.at(kp.lmid_)->getPoint());

                // Add prior if projected into image
                // 如果投影到图像内，则使用投影像素位置作为先验位置
                if( pcurframe_->isInImage(projpx) ) 
                {
                    // 记录3d点位置
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }
        // 对于2d点，没有先验位置，直接使用当前的像素位置作为先验位置
        // For other kps init prior with prev px pos.
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(kp.px_);
    }

    // 1st track 3d kps if using prior
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() ) 
    {// 先用3d点的先验位置进行光流跟踪 
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;
        // 只采用具有先验位置的3d点进行光流跟踪 
        ptracker_->fbKltTracking(
                    prev_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {// 遍历跟踪结果
            if( vkpstatus.at(i) ) {
                // 成功的更新当前帧的特征点  
                pcurframe_->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
                nbgood++;
            } else {
                // 跟踪失败的，放到没有先验位置的特征点中，后面再进行跟踪
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
        // 如果没有足够的3d点进行光流跟踪，估计的运动模型可能不太准确，建议使用P3P进行跟踪
        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            bp3preq_ = true;
            vpriors = vkps;
        }
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {// 对于没有先验位置的特征点，使用当前的像素位置进行光流跟踪 
        // Good / bad kps vector
        std::vector<bool> vkpstatus;
        // 使用当前帧的金字塔图像进行光流跟踪
        ptracker_->fbKltTracking(
                    prev_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);
        
        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {// 遍历跟踪结果
            // 如果跟踪成功，更新当前帧的特征点 
            if( vkpstatus.at(i) ) {
                pcurframe_->updateKeypoint(vkpids.at(i), vpriors.at(i));
                nbgood++;
            } else {
                // 如果跟踪失败，将特征点从当前帧中移除
                // MapManager is responsible for all the removing operations
                pmap_->removeObsFromCurFrameById(vkpids.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    } 
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}

// 当前帧和关键帧跟踪
void VisualFrontEnd::kltTrackingFromKF()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_KLT-Tracking-from-KF");

    // Get current kps and init priors for tracking
    std::vector<int> v3dkpids, vkpids;// 3d点id和2d点id
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;// 3d 点位置，3d点的先验位置，2d点位置，2d点的先验位置
    std::vector<bool> vkpis3d;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(pcurframe_->nb3dkps_);
    v3dkps.reserve(pcurframe_->nb3dkps_);
    v3dpriors.reserve(pcurframe_->nb3dkps_);

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(pcurframe_->nbkps_);
    vkps.reserve(pcurframe_->nbkps_);
    vpriors.reserve(pcurframe_->nbkps_);

    vkpis3d.reserve(pcurframe_->nbkps_);

    // Get prev KF
    // 从当前帧的关键帧中获取上一关键帧
    auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);

    if( pkf == nullptr ) {
        return;
    }

    std::vector<int> vbadids;
    vbadids.reserve(pcurframe_->nbkps_ * 0.2);


    // Front-End is thread-safe so we can direclty access curframe's kps
    for( const auto &it : pcurframe_->mapkps_ ) 
    {// 遍历当前帧的特征点
        auto &kp = it.second;
        // 在上一关键帧中查找当前帧的特征点 
        auto kfkpit = pkf->mapkps_.find(kp.lmid_);
        if( kfkpit == pkf->mapkps_.end() ) {
            vbadids.push_back(kp.lmid_);
            continue;
        }

        // Init prior px pos. from motion model
        if( pslamstate_->klt_use_prior_ )
        {// 使用运动模型初始化先验位置
            if( kp.is3d_ ) 
            {// 3d点
                // 投影到当前帧的像素坐标系中 
                cv::Point2f projpx = pcurframe_->projWorldToImageDist(pmap_->map_plms_.at(kp.lmid_)->getPoint());

                // Add prior if projected into image
                // 确保投影位置在图像内部 
                if( pcurframe_->isInImage(projpx) ) 
                {
                    // 保存3d点的id，3d点的位置，3d点的先验位置
                    v3dkps.push_back(kfkpit->second.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(kp.lmid_);

                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }

        // For other kps init prior with prev px pos.
        // 没有使用先验位置，直接使用上一关键帧的像素坐标作为先验位置 
        vkpids.push_back(kp.lmid_);
        vkps.push_back(kfkpit->second.px_);
        vpriors.push_back(kp.px_);
    }
    // 在关键帧中没有的点，不可能被匹配上，直接移除
    for( const auto &badid : vbadids ) {
        // MapManager is responsible for all the removing operations
        pmap_->removeObsFromCurFrameById(badid);
    }

    // 1st track 3d kps if using prior
    // 先用3d点的先验位置进行光流跟踪 
    if( pslamstate_->klt_use_prior_ && !v3dpriors.empty() ) 
    {
        int nbpyrlvl = 1;

        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        auto vprior = v3dpriors;
        // 光流法跟踪3d点
        ptracker_->fbKltTracking(
                    kf_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nbkps = v3dkps.size();
        // 遍历跟踪结果
        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                // 更新当前帧的匹配点
                pcurframe_->updateKeypoint(v3dkpids.at(i), v3dpriors.at(i));
                nbgood++;
            } else {
                // 跟踪失败的点，使用2d点的先验位置进行跟踪
                // If tracking failed, gonna try on full pyramid size
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(pcurframe_->mapkps_.at(v3dkpids.at(i)).px_);
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking w. priors : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
        // 如果跟踪成功的点的数量小于总的点的数量的1/3，说明运动模型可能不太准确，建议使用P3P算法进行跟踪
        if( nbgood < 0.33 * nbkps ) {
            // Motion model might be quite wrong, P3P is recommended next
            // and not using any prior
            bp3preq_ = true;
            vpriors = vkps;
        }
    }

    // 2nd track other kps if any
    // 第二次针对没有使用先验位置的点进行光流跟踪
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;
        // 光流法跟踪2d点
        ptracker_->fbKltTracking(
                    kf_pyr_, 
                    cur_pyr_, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);
        
        size_t nbgood = 0;
        size_t nbkps = vkps.size();

        for(size_t i = 0 ; i < nbkps  ; i++ ) 
        {// 遍历跟踪结果
            if( vkpstatus.at(i) ) {
                // 更新当前帧的匹配点
                pcurframe_->updateKeypoint(vkpids.at(i), vpriors.at(i));
                nbgood++;
            } else {
                // 跟踪失败的点，移除
                // MapManager is responsible for all the removing operations
                pmap_->removeObsFromCurFrameById(vkpids.at(i));
            }
        }

        if( pslamstate_->debug_ ) {
            std::cout << "\n >>> KLT Tracking no prior : " << nbgood;
            std::cout << " out of " << nbkps << " kps tracked!\n";
        }
    } 
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_KLT-Tracking");
}


// This function apply a 2d-2d based outliers filtering
/**
 * @brief 基础矩阵筛选离群点匹配
*/
void VisualFrontEnd::epipolar2d2dFiltering()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_EpipolarFiltering");
    
    // Get prev. KF (direct access as Front-End is thread safe)
    // 当前帧关联的上一关键帧
    auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);

    if( pkf == nullptr ) {
        std::cerr << "\nERROR! Previous Kf does not exist yet (epipolar2d2d()).\n";
        exit(-1);
    }

    // Get cur. Frame nb kps
    size_t nbkps = pcurframe_->nbkps_;// 当前帧的特征点数量

    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Setup Essential Matrix computation for OpenGV-based filtering
    std::vector<int> vkpsids, voutliersidx;
    vkpsids.reserve(nbkps);
    voutliersidx.reserve(nbkps);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
    vkfbvs.reserve(nbkps);
    vcurbvs.reserve(nbkps);
    
    size_t nbparallax = 0;
    float avg_parallax = 0.;

    // In stereo mode, we consider 3d kps as better tracks and therefore
    // use only them for computing E with RANSAC, 2d kps are then removed based
    // on the resulting Fundamental Mat.
    // 如果是双目只是用3d点来计算基础矩阵
    bool epifrom3dkps = false;
    if( pslamstate_->stereo_ && pcurframe_->nb3dkps_ > 30 ) {
        epifrom3dkps = true;
    }

    // Compute rotation compensated parallax
    // 当前帧到关键帧的旋转矩阵 
    Eigen::Matrix3d Rkfcur = pkf->getRcw() * pcurframe_->getRwc();

    // Init bearing vectors and check parallax
    for( const auto &it : pcurframe_->mapkps_ ) {
        // 遍历当前帧的所有点
        if( epifrom3dkps ) {
            // 只使用3d点 
            if( !it.second.is3d_ ) {
                continue;
            }
        }

        auto &kp = it.second;
        // 获取关键帧中对应的点
        // Get the prev. KF related kp if it exists
        auto kfkp = pkf->getKeypointById(kp.lmid_);

        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Store the bvs and their ids
        // 保存特征点数据
        vkfbvs.push_back(kfkp.bv_);
        vcurbvs.push_back(kp.bv_);
        vkpsids.push_back(kp.lmid_);
        // 关键帧中的特征点，投影到当前帧，计算视差
        cv::Point2f rotpx = pkf->projCamToImage(Rkfcur * kp.bv_);

        // Compute parallax
        // 视差
        avg_parallax += cv::norm(rotpx - kfkp.unpx_);
        nbparallax++;
    }
    // 匹配点大于8个
    if( nbkps < 8 ) {
        if( pslamstate_->debug_ )
            std::cout << "\nNot enough kps to compute Essential Matrix\n";
        return;
    }

    // Average parallax
    avg_parallax /= nbparallax;
    // 平均视差足够，否则不计算基础矩阵
    if( avg_parallax < 2. * pslamstate_->fransac_err_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough parallax (" << avg_parallax 
                << " px) to compute 5-pt Essential Matrix\n";
        return;
    }

    bool do_optimize = false;

    // In monocular case, we'll use the resulting motion if tracking is poor
    if( pslamstate_->mono_ && pmap_->nbkfs_ > 2 
        && pcurframe_->nb3dkps_ < 30 ) 
    {// 如果单目模式下跟踪指标较差，使用优化的方法估计相机运动
        do_optimize = true;
    }

    Eigen::Matrix3d Rkfc;
    Eigen::Vector3d tkfc;

    if( pslamstate_->debug_ ) {
        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> only on 3d kps : " << epifrom3dkps;
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << "\n\n";
    }
    // 5点法计算基础矩阵 
    bool success = 
        MultiViewGeometry::compute5ptEssentialMatrix(
                    vkfbvs, vcurbvs, 
                    pslamstate_->nransac_iter_, 
                    pslamstate_->fransac_err_, 
                    do_optimize, 
                    pslamstate_->bdo_random, 
                    pcurframe_->pcalib_leftcam_->fx_, 
                    pcurframe_->pcalib_leftcam_->fy_, 
                    Rkfc, tkfc, 
                    voutliersidx);

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();
    // 失败
    if( !success) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
        return;
    }
    // 大部分的点都被认为是离群点，也失败
    if( voutliersidx.size() > 0.5 * vkfbvs.size() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Too many outliers, skipping as might be degenerate case\n";
        return;
    }

    // Remove outliers
    // 删除离群点 
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all the removing operations.
        pmap_->removeObsFromCurFrameById(vkpsids.at(idx));
    }

    // In case we wanted to use the resulting motion 
    // (mono mode - can help when tracking is poor)
    if( do_optimize && pmap_->nbkfs_ > 2 ) 
    {// 单目模式下，更新当前位姿
        // 当前的相机位姿
        // Get motion model translation scale from last KF
        Sophus::SE3d Tkfw = pkf->getTcw();
        Sophus::SE3d Tkfcur = Tkfw * pcurframe_->getTwc();
        // 平移向量的模长
        double scale = Tkfcur.translation().norm();
        tkfc.normalize();

        // Update current pose with Essential Mat. relative motion
        // and current trans. scale
        // 更新当前帧的位姿,基于基础矩阵的相对运动和当前的平移向量的模长
        Sophus::SE3d Tkfc(Rkfc, scale * tkfc);
        pcurframe_->setTwc(pkf->getTwc() * Tkfc);
    }

    // In case we only used 3d kps for computing E (stereo mode)
    if( epifrom3dkps ) {// 双目模式下

        if( pslamstate_->debug_ )
            std::cout << "\n Applying found Essential Mat to 2D kps!\n";

        Sophus::SE3d Tidentity;
        Sophus::SE3d Tkfcur(Rkfc, tkfc);
        // 通过3d点计算基础矩阵
        Eigen::Matrix3d Fkfcur = MultiViewGeometry::computeFundamentalMat12(Tidentity, Tkfcur, pcurframe_->pcalib_leftcam_->K_);

        std::vector<int> vbadkpids;
        vbadkpids.reserve(pcurframe_->nb2dkps_);

        for( const auto &it : pcurframe_->mapkps_ ) 
        {// 遍历所有的2d点
            if( it.second.is3d_ ) {
                continue;
            }

            auto &kp = it.second;

            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);

            // Normalized coord.
            Eigen::Vector3d curpt(kp.unpx_.x, kp.unpx_.y, 1.);
            Eigen::Vector3d kfpt(kfkp.unpx_.x, kfkp.unpx_.y, 1.);
            // 计算2d点的重投影误差
            float epi_err = MultiViewGeometry::computeSampsonDistance(Fkfcur, curpt, kfpt);
            // 如果重投影误差大于阈值，认为是离群点 
            if( epi_err > pslamstate_->fransac_err_ ) {
                vbadkpids.push_back(kp.lmid_);
            }
        }
        // 删除所有的离群点 
        for( const auto & kpid : vbadkpids ) {
            pmap_->removeObsFromCurFrameById(kpid);
        }

        if( pslamstate_->debug_ )
            std::cout << "\n Nb of 2d kps removed : " << vbadkpids.size() << " \n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_EpipolarFiltering");
}


/**
 * @brief  2d-3d相机位姿估计
*/
void VisualFrontEnd::computePose()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_computePose");

    // Get cur nb of 3D kps    
    // 当前帧的3D点数量 
    size_t nb3dkps = pcurframe_->nb3dkps_;
    // 数量太少了，不够计算
    if( nb3dkps < 4 ) {
        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> Not enough kps to compute P3P / PnP\n";
        return;
    }

    // Setup P3P-Ransac computation for OpenGV-based Pose estimation
    // + motion-only BA with Ceres
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vbvs, vwpts;
    std::vector<int> vkpids, voutliersidx, vscales;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vkps;

    vbvs.reserve(nb3dkps);
    vwpts.reserve(nb3dkps);
    vkpids.reserve(nb3dkps);
    voutliersidx.reserve(nb3dkps);

    vkps.reserve(nb3dkps);
    vscales.reserve(nb3dkps);

    bool bdop3p = bp3preq_ || pslamstate_->dop3p_;

    // Store every 3D bvs, MPs and their related ids
    for( const auto &it : pcurframe_->mapkps_ ) 
    {// 对于每一个3d点 
        if( !it.second.is3d_ ) {
            continue;
        }

        auto &kp = it.second;
        // auto plm = pmap_->getMapPoint(kp.lmid_);
        auto plm = pmap_->map_plms_.at(kp.lmid_);
        if( plm == nullptr ) {
            continue;
        }

        if( bdop3p ) {
            vbvs.push_back(kp.bv_);
        }
        // 添加待优化的点 
        vkps.push_back(Eigen::Vector2d(kp.unpx_.x, kp.unpx_.y));
        vwpts.push_back(plm->getPoint());
        vscales.push_back(kp.scale_);
        vkpids.push_back(kp.lmid_);
    }

    Sophus::SE3d Twc = pcurframe_->getTwc();
    bool do_optimize = false;
    bool success = false;

    if( bdop3p ) 
    {// p3p 优化，线性求解，得到初始值
        if( pslamstate_->debug_ ) {
            std::cout << "\n \t>>>P3P Ransac : ";
            std::cout << "\n \t>>> nb 3d pts : " << nb3dkps;
            std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
            std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
            std::cout << "\n\n";
        }

        // Only effective with OpenGV
        bool use_lmeds = true;
        // 优化
        success = 
            MultiViewGeometry::p3pRansac(
                            vbvs, vwpts, 
                            pslamstate_->nransac_iter_, 
                            pslamstate_->fransac_err_, 
                            do_optimize, 
                            pslamstate_->bdo_random, 
                            pcurframe_->pcalib_leftcam_->fx_, 
                            pcurframe_->pcalib_leftcam_->fy_, 
                            Twc,
                            voutliersidx,
                            use_lmeds);

        if( pslamstate_->debug_ )
            std::cout << "\n \t>>> P3P-LMeds nb outliers : " << voutliersidx.size();

        // Check that pose estim. was good enough
        size_t nbinliers = vwpts.size() - voutliersidx.size();
        // 删除错误结果
        if( !success
            || nbinliers < 5
            || Twc.translation().array().isInf().any()
            || Twc.translation().array().isNaN().any() )
        {
            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();

            return;
        } 

        // Pose seems to be OK!

        // Update frame pose
        // 更新帧的位姿
        pcurframe_->setTwc(Twc);

        // Remove outliers before PnP refinement (a bit dirty)
        int k = 0;
        // 删除外点
        for( const auto &idx : voutliersidx ) {
            // MapManager is responsible for all removing operations
            pmap_->removeObsFromCurFrameById(vkpids.at(idx-k));
            vkps.erase(vkps.begin() + idx - k);
            vwpts.erase(vwpts.begin() + idx - k);
            vkpids.erase(vkpids.begin() + idx - k);
            vscales.erase(vscales.begin() + idx - k);
            k++;
        }

        // Clear before robust PnP refinement using Ceres
        voutliersidx.clear();
    }

    // Ceres-based PnP (motion-only BA)
    // pnp位姿求解，使用ceres优化 
    bool buse_robust = true;
    bool bapply_l2_after_robust = pslamstate_->apply_l2_after_robust_;
    
    size_t nbmaxiters = 5;
    // 优化 
    success =
        MultiViewGeometry::ceresPnP(
                        vkps, vwpts, 
                        vscales,
                        Twc, 
                        nbmaxiters, 
                        pslamstate_->robust_mono_th_, 
                        buse_robust, 
                        bapply_l2_after_robust,
                        pcurframe_->pcalib_leftcam_->fx_, pcurframe_->pcalib_leftcam_->fy_,
                        pcurframe_->pcalib_leftcam_->cx_, pcurframe_->pcalib_leftcam_->cy_,
                        voutliersidx);
    
    // Check that pose estim. was good enough
    // 检查位姿估计是否足够好 
    size_t nbinliers = vwpts.size() - voutliersidx.size();

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Ceres PnP nb outliers : " << voutliersidx.size();
    // 删除错误结果 
    if( !success
        || nbinliers < 5
        || voutliersidx.size() > 0.5 * vwpts.size()
        || Twc.translation().array().isInf().any()
        || Twc.translation().array().isNaN().any() )
    {
        if( !bdop3p ) {
            // Weird results, skipping here and applying p3p next
            bp3preq_ = true;
        }
        else if( pslamstate_->mono_ ) {

            if( pslamstate_->debug_ )
                std::cout << "\n \t>>> Not enough inliers for reliable pose est. Resetting KF state\n";

            resetFrame();
        } 
        // else {
            // resetFrame();
            // motion_model_.reset();
        // }

        return;
    } 

    // Pose seems to be OK!

    // Update frame pose
    // 更新帧的位姿 
    pcurframe_->setTwc(Twc);

    // Set p3p req to false as it is triggered either because
    // of bad PnP or by bad klt tracking
    bp3preq_ = false;
    // 删除外点
    // Remove outliers
    for( const auto & idx : voutliersidx ) {
        // MapManager is responsible for all removing operations
        pmap_->removeObsFromCurFrameById(vkpids.at(idx));
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_computePose");
}

/**
 * @brief 判断是否可以初始化
*/
bool VisualFrontEnd::checkReadyForInit()
{
    // 计算当前帧的平均视差 
    double avg_rot_parallax = computeParallax(pcurframe_->kfid_, false);

    std::cout << "\n \t>>> Init current parallax (" << avg_rot_parallax <<" px)\n"; 

    if( avg_rot_parallax > pslamstate_->finit_parallax_ ) {
    // 当具有足够的视差时
        auto cb = std::chrono::high_resolution_clock::now();
        
        // Get prev. KF
        // 获取和当前帧关联的关键帧
        auto pkf = pmap_->map_pkfs_.at(pcurframe_->kfid_);
        if( pkf == nullptr ) {
            return false;
        }

        // Get cur. Frame nb kps
        size_t nbkps = pcurframe_->nbkps_;// 当前帧的特征点数量 
        // 当前帧的特征点数量小于8个时，不进行初始化 
        if( nbkps < 8 ) {
            std::cout << "\nNot enough kps to compute 5-pt Essential Matrix\n";
            return false;
        }

        // Setup Essential Matrix computation for OpenGV-based filtering
        std::vector<int> vkpsids, voutliersidx;
        vkpsids.reserve(nbkps);
        voutliersidx.reserve(nbkps);

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
        vkfbvs.reserve(nbkps);
        vcurbvs.reserve(nbkps);
        // 当前帧和关键帧的旋转差值
        Eigen::Matrix3d Rkfcur = pkf->getTcw().rotationMatrix() * pcurframe_->getTwc().rotationMatrix();
        int nbparallax = 0;
        float avg_rot_parallax = 0.;
        // 这里计算的视差是去除了旋转的视差 
        // Get bvs and compute the rotation compensated parallax for all cur kps
        // for( const auto &kp : pcurframe_->getKeypoints() ) {
        for( const auto &it : pcurframe_->mapkps_ ) {
            // 遍历当前帧的所有特征点 
            auto &kp = it.second;
            // Get the prev. KF related kp if it exists
            auto kfkp = pkf->getKeypointById(kp.lmid_);
            // 查找关键帧中是否存在当前帧的特征点 
            if( kfkp.lmid_ != kp.lmid_ ) {
                continue;// 如果不存在，则跳过 
            }

            // Store the bvs and their ids
            vkfbvs.push_back(kfkp.bv_);
            vcurbvs.push_back(kp.bv_);
            vkpsids.push_back(kp.lmid_);

            // Compute rotation compensated parallax
            // 将关键帧下的方向向量转换到当前帧下 
            Eigen::Vector3d rotbv = Rkfcur * kp.bv_;// 当前帧的特征点在关键帧坐标系下的坐标
            // 当前帧下的特征点坐标 
            Eigen::Vector3d unpx = pcurframe_->pcalib_leftcam_->K_ * rotbv;
            cv::Point2f rotpx(unpx.x() / unpx.z(), unpx.y() / unpx.z());
            // 计算平均视差 
            avg_rot_parallax += cv::norm(rotpx - kfkp.unpx_);
            nbparallax++;
        }
        // 匹配点数目不足
        if( nbparallax < 8 ) {
            std::cout << "\nNot enough prev KF kps to compute 5-pt Essential Matrix\n";
            return false;
        }

        // Average parallax
        // 计算平均视差，去除旋转后的视差
        avg_rot_parallax /= (nbparallax);

        // 确保平均视差足够大 
        if( avg_rot_parallax < pslamstate_->finit_parallax_ ) {
            std::cout << "\n \t>>> Not enough parallax (" << avg_rot_parallax <<" px) to compute 5-pt Essential Matrix\n";
            return false;
        }

        bool do_optimize = true;
        // 设置初始的关键帧位姿为单位矩阵 
        Eigen::Matrix3d Rkfc;
        Eigen::Vector3d tkfc;
        Rkfc.setIdentity();
        tkfc.setZero();

        std::cout << "\n \t>>> 5-pt EssentialMatrix Ransac :";
        std::cout << "\n \t>>> nb pts : " << nbkps;
        std::cout << " / avg. parallax : " << avg_rot_parallax;
        std::cout << " / nransac_iter_ : " << pslamstate_->nransac_iter_;
        std::cout << " / fransac_err_ : " << pslamstate_->fransac_err_;
        std::cout << " / bdo_random : " << pslamstate_->bdo_random;
        std::cout << "\n\n";
        // 5点法计算本质矩阵
        bool success = 
            MultiViewGeometry::compute5ptEssentialMatrix
                    (vkfbvs, vcurbvs, pslamstate_->nransac_iter_, pslamstate_->fransac_err_, 
                    do_optimize, pslamstate_->bdo_random, 
                    pcurframe_->pcalib_leftcam_->fx_, 
                    pcurframe_->pcalib_leftcam_->fy_, 
                    Rkfc, tkfc, 
                    voutliersidx);

        std::cout << "\n \t>>> Epipolar nb outliers : " << voutliersidx.size();
        // 如果计算失败，则返回
        if( !success ) {
            std::cout << "\n \t>>> No pose could be computed from 5-pt EssentialMatrix\n";
            return false;
        }

        // Remove outliers from cur. Frame
        // 删除外点 
        for( const auto & idx : voutliersidx ) {
            // MapManager is responsible for all the removing operations.
            pmap_->removeObsFromCurFrameById(vkpsids.at(idx));
        }

        // Arbitrary scale
        tkfc.normalize();// 归一化平移向量
        tkfc = tkfc.eval() * 0.25;// 乘以一个任意的尺度

        std::cout << "\n \t>>> Essential Mat init : " << tkfc.transpose();
        // 设置当前帧的位姿
        pcurframe_->setTwc(Rkfc, tkfc);
        
        auto ce = std::chrono::high_resolution_clock::now();
        std::cout << "\n \t>>> Essential Mat Intialization run time : " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ce-cb).count()
            << "[ms]" << std::endl;

        return true;
    }

    return false;
}

/**
 * @brief 判断是否需要新建关键帧 
*/
bool VisualFrontEnd::checkNewKfReq()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_checkNewKfReq");

    // Get prev. KF
    // 获取当前帧和其关联的关键帧
    auto pkfit = pmap_->map_pkfs_.find(pcurframe_->kfid_);
    // 如果当前帧没有关联的关键帧，直接返回 false
    if( pkfit == pmap_->map_pkfs_.end() ) {
        return false; // Should not happen
    }
    auto pkf = pkfit->second;

    // Compute median parallax
    double med_rot_parallax = 0.;// 视差值

    // unrot : false / median : true / only_2d : false
    // 1 不去除旋转，2 使用中值，3 不仅仅使用2d点 
    // 旋转也需要考虑进去，中值更稳定 
    med_rot_parallax = computeParallax(pkf->kfid_, true, true, false);

    // Id diff with last KF
    // 当前帧和最近关键帧的id差值
    int nbimfromkf = pcurframe_->id_-pkf->id_;
    // 关键帧条件1 
    // 1. 当前帧的特征点数小于最大值的1/3
    // 2. 当前帧和最近关键帧的id差值大于5
    // 3. 当前没有局部BA，有空闲算力
    if( pcurframe_->noccupcells_ < 0.33 * pslamstate_->nbmaxkps_
        && nbimfromkf >= 5
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }
    // 关键帧条件2 
    // 1. 当前帧的3d点数小于20 
    // 2. 当前帧和最近关键帧的id差值大于2
    if( pcurframe_->nb3dkps_ < 20 &&
        nbimfromkf >= 2 )
    {
        return true;
    }
    // 关键帧否定条件3
    // 1. 当前帧的3d点数大于最大值的1/2
    // 2. 当前正在进行局部BA，或者当前帧和最近关键帧的id差值小于2 
    if( pcurframe_->nb3dkps_ > 0.5 * pslamstate_->nbmaxkps_ 
        && (pslamstate_->blocalba_is_on_ || nbimfromkf < 2) )
    {
        return false;
    }

    // Time diff since last KF in sec.
    double time_diff = pcurframe_->img_time_ - pkf->img_time_;

    // 关键帧条件4
    // 1. 双目模式
    // 2. 当前帧和最近关键帧的时间差大于1s
    // 3. 当前没有局部BA，有空闲算力
    if( pslamstate_->stereo_ && time_diff > 1. 
        && !pslamstate_->blocalba_is_on_ )
    {
        return true;
    }

    // 关键帧条件5
    // cx : 视差值大于最小视差的一半，或者双目模式且当前帧和最近关键帧的id差值大于2
    // c0 : 视差值大于最小视差
    // c1 : 当前帧的3d点数小于最近关键帧的3d点数的0.75
    // c2 : 当前帧的占用的网格数小于最大值的一半，且当前帧的3d点数小于最近关键帧的3d点数的0.85，且当前没有局部BA
    // cx 必须满足，c0,c1,c2满足一个即可 
    bool cx = med_rot_parallax >= pslamstate_->finit_parallax_ / 2.
        || (pslamstate_->stereo_ && !pslamstate_->blocalba_is_on_ && pcurframe_->id_-pkf->id_ > 2);

    bool c0 = med_rot_parallax >= pslamstate_->finit_parallax_;
    bool c1 = pcurframe_->nb3dkps_ < 0.75 * pkf->nb3dkps_;
    bool c2 = pcurframe_->noccupcells_ < 0.5 * pslamstate_->nbmaxkps_
                && pcurframe_->nb3dkps_ < 0.85 * pkf->nb3dkps_
                && !pslamstate_->blocalba_is_on_;
    
    bool bkfreq = (c0 || c1 || c2) && cx;

    if( bkfreq && pslamstate_->debug_ ) {
        
        std::cout << "\n\n----------------------------------------------------------------------";
        std::cout << "\n>>> Check Keyframe conditions :";
        std::cout << "\n> pcurframe_->id_ = " << pcurframe_->id_ << " / prev kf frame_id : " << pkf->id_;
        std::cout << "\n> Prev KF nb 3d kps = " << pkf->nb3dkps_ << " / Cur Frame = " << pcurframe_->nb3dkps_;
        std::cout << " / Cur Frame occup cells = " << pcurframe_->noccupcells_ << " / parallax = " << med_rot_parallax;
        std::cout << "\n-------------------------------------------------------------------\n\n";
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_checkNewKfReq");

    return bkfreq;
}


// This function computes the parallax (in px.) between cur. Frame 
// and the provided KF id.
/**
 * @brief 图像图像之间的视差，用于判断是否需要新建关键帧以及判断是否具备足够视差初始化 
 * @param kfid   当前帧id
 * @param do_unrot 是否去除旋转视差
 * @param bmedian  是否使用中值，否则使用平均值 
 * @param b2donly  是否只计算2d点的视差
*/
float VisualFrontEnd::computeParallax(const int kfid, bool do_unrot, bool bmedian, bool b2donly)
{
    // Get prev. KF
    // 获取上一个关键帧 
    auto pkfit = pmap_->map_pkfs_.find(kfid);
    // 如果没有找到上一个关键帧，返回0 
    if( pkfit == pmap_->map_pkfs_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\n[Visual Front End] Error in computeParallax ! Prev KF #" 
                    << kfid << " does not exist!\n";
        return 0.;
    }

    // Compute relative rotation between cur Frame 
    // and prev. KF if required
    Eigen::Matrix3d Rkfcur(Eigen::Matrix3d::Identity());
    if( do_unrot ) {
        // 关键帧的旋转矩阵 
        Eigen::Matrix3d Rkfw = pkfit->second->getRcw();
        // 当前帧的旋转矩阵 
        Eigen::Matrix3d Rwcur = pcurframe_->getRwc();
        // 旋转矩阵的差值
        Rkfcur = Rkfw * Rwcur;
    }

    // Compute parallax 
    float avg_parallax = 0.; // 平均视差值
    int nbparallax = 0; // 视差值的数量

    std::set<float> set_parallax;

    // Compute parallax for all kps seen in prev. KF{
    for( const auto &it : pcurframe_->mapkps_ ) 
    {// 遍历当前帧的所有特征点
        if( b2donly && it.second.is3d_ ) {// 如果只计算2d特征点的视差，且该特征点是3d点，则跳过 
            continue;
        }
        // 当前帧的特征点 
        auto &kp = it.second;

        // Get prev. KF kp if it exists
        // 关键帧对应的特征点 
        auto kfkp = pkfit->second->getKeypointById(kp.lmid_);
    
        if( kfkp.lmid_ != kp.lmid_ ) {
            continue;
        }

        // Compute parallax with unpx pos.
        cv::Point2f unpx = kp.unpx_;// 当前帧的特征点在归一化平面上的坐标 

        // Rotate bv into KF cam frame and back project into image
        if( do_unrot ) {
            // 去除因为旋转带来的视差变化 
            unpx = pkfit->second->projCamToImage(Rkfcur * kp.bv_);
        }

        // Compute rotation-compensated parallax
        // 计算视差
        float parallax = cv::norm(unpx - kfkp.unpx_);
        avg_parallax += parallax;
        nbparallax++;
        // 视差值的集合 
        if( bmedian ) {
            set_parallax.insert(parallax);
        }
    }

    if( nbparallax == 0 ) {
        return 0.;
    }

    // Average parallax
    // 计算平均视差 
    avg_parallax /= nbparallax;

    // 视差的中值 
    if( bmedian ) 
    {
        auto it = set_parallax.begin();
        std::advance(it, set_parallax.size() / 2);
        avg_parallax = *it;
    }
    // 返回视差值 平均值 / 中位数
    return avg_parallax;
}

// 图像预处理 
void VisualFrontEnd::preprocessImage(cv::Mat &img_raw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_TM_preprocessImage");

    // Set cur raw img
    // left_raw_img_ = img_raw;

    // Update prev img
    // 如果不是关键帧跟踪，那么将当前帧设置为上一帧
    if( !pslamstate_->btrack_keyframetoframe_ ) {
        // cur_img_.copyTo(prev_img_);
        cv::swap(cur_img_, prev_img_);
    }

    // Update cur img
    if( pslamstate_->use_clahe_ ) {
        ptracker_->pclahe_->apply(img_raw, cur_img_);
    } else {
        // 更新当前帧
        cur_img_ = img_raw;
    }

    // Pre-building the pyramid used for KLT speed-up
    // 预先构建金字塔，用于KLT加速  
    if( pslamstate_->do_klt_ ) {

        // If tracking from prev image, swap the pyramid
        // 如果和上一帧跟踪，那么交换金字塔 
        if( !cur_pyr_.empty() && !pslamstate_->btrack_keyframetoframe_ ) {
            prev_pyr_.swap(cur_pyr_);
        }
        // 构建金字塔 
        cv::buildOpticalFlowPyramid(cur_img_, cur_pyr_, pslamstate_->klt_win_size_, pslamstate_->nklt_pyr_lvl_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_TM_preprocessImage");
}


// Reset current Frame state
/**
 * @brief 重置当前帧状态 
*/
void VisualFrontEnd::resetFrame()
{
    auto mapkps = pcurframe_->mapkps_;
    for( const auto &kpit : mapkps ) {
        pmap_->removeObsFromCurFrameById(kpit.first);
    }
    pcurframe_->mapkps_.clear();
    pcurframe_->vgridkps_.clear();
    pcurframe_->vgridkps_.resize( pcurframe_->ngridcells_ );

    // Do not clear those as we keep the same pose
    // and hence keep a chance to retrack the previous map
    //
    // pcurframe_->map_covkfs_.clear();
    // pcurframe_->set_local_mapids_.clear();

    pcurframe_->nbkps_ = 0;
    pcurframe_->nb2dkps_ = 0;
    pcurframe_->nb3dkps_ = 0;
    pcurframe_->nb_stereo_kps_ = 0;

    pcurframe_->noccupcells_ = 0;
}

// Reset VisualFrontEnd
/**
 * @brief 重置视觉前端
*/
void VisualFrontEnd::reset()
{
    cur_img_.release();
    prev_img_.release();

    // left_raw_img_.release();

    cur_pyr_.clear();
    prev_pyr_.clear();
    kf_pyr_.clear();
}
