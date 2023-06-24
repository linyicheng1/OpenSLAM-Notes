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
#include <opencv2/highgui.hpp>

#include "ov2slam.hpp"

/**
 * @brief 构造函数
 * @param pstate SLAM参数 
 * @param pviz 可视化类
*/
SlamManager::SlamManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<RosVisualizer> pviz)
    : pslamstate_(pstate)
    , prosviz_(pviz)
{
    std::cout << "\n SLAM Manager is being created...\n";

    #ifdef OPENCV_CONTRIB
        std::cout << "\n OPENCV CONTRIB FOUND!  BRIEF DESCRIPTOR WILL BE USED!\n";
    #else
        std::cout << "\n OPENCV CONTRIB NOT FOUND!  ORB DESCRIPTOR WILL BE USED!\n";
    #endif

    #ifdef USE_OPENGV
        std::cout << "\n OPENGV FOUND!  OPENGV MVG FUNCTIONS WILL BE USED!\n";
    #else
        std::cout << "\n OPENGV NOT FOUND!  OPENCV MVG FUNCTIONS WILL BE USED!\n";
    #endif

    // We first setup the calibration to init everything related
    // to the configuration of the current run
    std::cout << "\n SetupCalibration()\n";
    // 设置标定参数 
    setupCalibration();

    if( pslamstate_->stereo_ && pslamstate_->bdo_stereo_rect_ ) {
        // 如果是双目的话 
        std::cout << "\n SetupStereoCalibration()\n";
        // 设置双目标定参数 
        setupStereoCalibration();
    }
    else if( pslamstate_->mono_ && pslamstate_->bdo_stereo_rect_ ) {
        pslamstate_->bdo_stereo_rect_ = false;
    }

    // If no stereo rectification required (i.e. mono config or 
    // stereo w/o rectification) and image undistortion required
    // 设置畸变去除映射
    if( !pslamstate_->bdo_stereo_rect_ && pslamstate_->bdo_undist_ ) {
        std::cout << "\n Setup Image Undistortion\n";
        pcalib_model_left_->setUndistMap(pslamstate_->alpha_);
        if( pslamstate_->stereo_ )
            pcalib_model_right_->setUndistMap(pslamstate_->alpha_);
    }
    // 构造当前帧
    if( pslamstate_->mono_ ) {
        pcurframe_.reset( new Frame(pcalib_model_left_, pslamstate_->nmaxdist_) );
    } else if( pslamstate_->stereo_ ) {
        pcurframe_.reset( new Frame(pcalib_model_left_, pcalib_model_right_, pslamstate_->nmaxdist_) );
    } else {
        std::cerr << "\n\n=====================================================\n\n";
        std::cerr << "\t\t YOU MUST CHOOSE BETWEEN MONO / STEREO (and soon RGBD)\n";
        std::cerr << "\n\n=====================================================\n\n";
    }

    // Create all objects to be used within OV²SLAM
    // =============================================
    int tilesize = 50;
    cv::Size clahe_tiles(pcalib_model_left_->img_w_ / tilesize
                        , pcalib_model_left_->img_h_ / tilesize);
                        
    cv::Ptr<cv::CLAHE> pclahe = cv::createCLAHE(pslamstate_->fclahe_val_, clahe_tiles);

    pfeatextract_.reset( new FeatureExtractor(
                                pslamstate_->nbmaxkps_, pslamstate_->nmaxdist_, 
                                pslamstate_->dmaxquality_, pslamstate_->nfast_th_
                            ) 
                        );

    ptracker_.reset( new FeatureTracker(pslamstate_->nmax_iter_, 
                            pslamstate_->fmax_px_precision_, pclahe
                        )
                    );

    // Map Manager will handle Keyframes / MapPoints
    pmap_.reset( new MapManager(pslamstate_, pcurframe_, pfeatextract_, ptracker_) );

    // Visual Front-End processes every incoming frames 
    pvisualfrontend_.reset( new VisualFrontEnd(pslamstate_, pcurframe_, 
                                    pmap_, ptracker_
                                )
                            );

    // Mapper thread handles Keyframes' processing
    // (i.e. triangulation, local map tracking, BA, LC)
    pmapper_.reset( new Mapper(pslamstate_, pmap_, pcurframe_) );
}

/**
 * @brief 主要SLAM线程
*/
void SlamManager::run()
{
    std::cout << "\nOV²SLAM is ready to process incoming images!\n";

    bis_on_ = true;

    cv::Mat img_left, img_right;

    double time = -1.; // Current image timestamp
    double cam_delay = -1.; // Delay between two successive images
    double last_img_time = -1.; // Last received image time

    // Main SLAM loop
    while( !bexit_required_ ) {

        // 0. Get New Images
        // =============================================
        // 从队列中获取一帧图像
        if( getNewImage(img_left, img_right, time) )
        {
            // Update current frame
            frame_id_++;// 帧id +1 
            // 更新当前帧id和时间
            pcurframe_->updateFrame(frame_id_, time);

            // Update cam delay for automatic exit
            if( frame_id_ > 0 ) {
                // 计算两帧图像之间的时间间隔 cam_delay 
                cam_delay = ros::Time::now().toSec() - last_img_time;
                last_img_time += cam_delay;
            } else {
                last_img_time = ros::Time::now().toSec();
            }

            // Display info on current frame state
            // 当前帧的信息输出调试
            if( pslamstate_->debug_ )
                pcurframe_->displayFrameInfo();

            // 1. Send images to the FrontEnd
            // =============================================
            if( pslamstate_->debug_ )
                std::cout << "\n \t >>> [SLAM Node] New image send to Front-End\n";
            // 传入图像，时间戳，返回是否需要关键帧 
            bool is_kf_req = pvisualfrontend_->visualTracking(img_left, time);

            // Save current pose
            // 保存当前帧的位姿 
            Logger::addSE3Pose(time, pcurframe_->getTwc(), is_kf_req);

            if( pslamstate_->breset_req_ ) {
                reset();
                continue;
            }

            // 2. Create new KF if req. / Send new KF to Mapper
            // ================================================
            // 如果需要关键帧，创建关键帧，添加到地图中 
            if( is_kf_req ) 
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n \t >>> [SLAM Node] New Keyframe send to Back-End\n";

                if( pslamstate_->stereo_ ) 
                {// 双目 
                    Keyframe kf(
                        pcurframe_->kfid_,  // id
                        img_left,          // 左图
                        img_right,         // 右图 
                        pvisualfrontend_->cur_pyr_ // 金字塔
                        );

                    pmapper_->addNewKf(kf);
                } 
                else if( pslamstate_->mono_ ) 
                {// 单目 
                    // 创建关键帧 
                    Keyframe kf(pcurframe_->kfid_, img_left);
                    pmapper_->addNewKf(kf);
                }

                if( !bkf_viz_ison_ ) {
                    std::thread kf_viz_thread(&SlamManager::visualizeAtKFsRate, this, time);
                    kf_viz_thread.detach();
                }    
            }

            if( pslamstate_->debug_ || pslamstate_->log_timings_ )
                std::cout << Profiler::getInstance().displayTimeLogs() << std::endl;

            // Frame rate visualization (limit the visualization processing)
            // 可视化线程 
            if( !bframe_viz_ison_ ) {
                std::thread viz_thread(&SlamManager::visualizeAtFrameRate, this, time);
                viz_thread.detach();
            }
        } 
        else {
            // 如果没有获取到图像，就判断是否运行结束，如果结束就保存结果并退出
            // 3. Check if we are done with a sequence!
            // ========================================
            bool c1 = cam_delay > 0;
            bool c2 = ( ros::Time::now().toSec() - last_img_time ) > 100. * cam_delay;
            bool c3 = !bnew_img_available_;

            if( c1 && c2 && c3 )
            {
                bexit_required_ = true;

                // Warn threads to stop and then save the results only in this case of 
                // automatic stop because end of sequence reached 
                // (avoid wasting time when forcing stop by CTRL+C)
                pmapper_->bexit_required_ = true;

                writeResults();

                // Notify exit to ROS
                ros::requestShutdown();
            }
            else {
                std::chrono::milliseconds dura(1);
                std::this_thread::sleep_for(dura);
            }
        }
    }

    std::cout << "\nOV²SLAM is stopping!\n";

    bis_on_ = false;
}

// 添加一帧新的图像 
void SlamManager::addNewMonoImage(const double time, cv::Mat &im0)
{
    // 如果需要在处理之前对图像进行矫正 
    if( pslamstate_->bdo_undist_ ) {
        pcalib_model_left_->rectifyImage(im0, im0);
    }

    // 图像队列 
    std::lock_guard<std::mutex> lock(img_mutex_);
    qimg_left_.push(im0);
    qimg_time_.push(time);
    // 新图像可用标志 
    bnew_img_available_ = true;
}

// 添加一帧新的双目图像 
void SlamManager::addNewStereoImages(const double time, cv::Mat &im0, cv::Mat &im1) 
{
    // 如果需要在处理之前对图像进行矫正 
    if( pslamstate_->bdo_stereo_rect_ || pslamstate_->bdo_undist_ ) {
        pcalib_model_left_->rectifyImage(im0, im0);
        pcalib_model_right_->rectifyImage(im1, im1);
    }
    // 图像队列
    std::lock_guard<std::mutex> lock(img_mutex_);
    qimg_left_.push(im0);
    qimg_right_.push(im1);
    qimg_time_.push(time);
    // 新图像可用标志 
    bnew_img_available_ = true;
}

// 获取一帧新的图像 
bool SlamManager::getNewImage(cv::Mat &iml, cv::Mat &imr, double &time)
{
    std::lock_guard<std::mutex> lock(img_mutex_);
    // 如果没有新的图像可用 
    if( !bnew_img_available_ ) {
        return false;
    }
    
    int k = 0;

    do {
        k++;
        // 获取图像队列中的第一帧图像 
        iml = qimg_left_.front();
        qimg_left_.pop();
        // 获取图像队列中的第一帧时间戳 
        time = qimg_time_.front();
        qimg_time_.pop();
        // 如果是双目图像 
        if( pslamstate_->stereo_ ) {
            // 获取图像队列中的第一帧图像
            imr = qimg_right_.front();
            qimg_right_.pop();
        }
        // 如果不是强制实时
        // 如果是强制实时，那么就取最新的图像，不管图像队列中有多少图像 
        if( !pslamstate_->bforce_realtime_ )
            break;

    } while( !qimg_left_.empty() );
    // 输出强制实时的提示信息 
    if( k > 1 ) {    
        if( pslamstate_->debug_ )
            std::cout << "\n SLAM is late!  Skipped " << k-1 << " frames...\n";
    }
    // 如果图像队列中没有图像了，那么就将新图像可用标志置为false
    if( qimg_left_.empty() ) {
        bnew_img_available_ = false;
    }

    return true;
}

/**
 * @brief 设置标定参数 
*/
void SlamManager::setupCalibration()
{
    // 如果是单目相机
    // 设置单目相机的标定参数 
    pcalib_model_left_.reset( 
                new CameraCalibration(
                        pslamstate_->cam_left_model_, 
                        pslamstate_->fxl_, pslamstate_->fyl_, 
                        pslamstate_->cxl_, pslamstate_->cyl_,
                        pslamstate_->k1l_, pslamstate_->k2l_, 
                        pslamstate_->p1l_, pslamstate_->p2l_,
                        pslamstate_->img_left_w_, 
                        pslamstate_->img_left_h_
                        ) 
                    );

    if( pslamstate_->stereo_ ) 
    {
        // 如果是双目相机 
        // 设置右目相机的标定参数 
        pcalib_model_right_.reset( 
                    new CameraCalibration(
                            pslamstate_->cam_right_model_, 
                            pslamstate_->fxr_, pslamstate_->fyr_, 
                            pslamstate_->cxr_, pslamstate_->cyr_,
                            pslamstate_->k1r_, pslamstate_->k2r_, 
                            pslamstate_->p1r_, pslamstate_->p2r_,
                            pslamstate_->img_right_w_, 
                            pslamstate_->img_right_h_
                            ) 
                        );
        
        // TODO: Change this and directly add the extrinsic parameters within the 
        // constructor (maybe set default parameters on extrinsic with identity / zero)
        // 设置左右相机的外参
        pcalib_model_right_->setupExtrinsic(pslamstate_->T_left_right_);
    }
}

/**
 * @brief 设置双目标定参数, 主要是畸变矫正
*/
void SlamManager::setupStereoCalibration()
{
    // Apply stereorectify and setup the calibration models
    cv::Mat Rl, Rr, Pl, Pr, Q;

    cv::Rect rectleft, rectright;
    // 确保左右相机的相机模型类型保持一致
    if( pcalib_model_left_->model_ != pcalib_model_right_->model_ )
    {
        std::cerr << "\n Left and Right cam have different distortion model.  Cannot use stereo rectifcation!\n";
        return;
    }
    // 如果左右相机都没有畸变，那么就不需要进行矫正 
    if( cv::countNonZero(pcalib_model_left_->Dcv_) == 0 && 
        cv::countNonZero(pcalib_model_right_->Dcv_) == 0 &&
        pcalib_model_right_->Tc0ci_.rotationMatrix().isIdentity(1.e-5) )
    {
        std::cout << "\n No distorsion and R_left_right = I3x3 / NO rectif to apply!";
        return;
    }

    if( pcalib_model_left_->model_ == CameraCalibration::Pinhole )
    {// 如果是针孔相机模型 
        // 计算左右相机的矫正参数
        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
        cv::stereoRectify(
                pcalib_model_left_->Kcv_, pcalib_model_left_->Dcv_,
                pcalib_model_right_->Kcv_, pcalib_model_right_->Dcv_,
                pcalib_model_left_->img_size_, 
                pcalib_model_right_->Rcv_cic0_, 
                pcalib_model_right_->tcv_cic0_,
                Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
                pslamstate_->alpha_,
                pcalib_model_left_->img_size_, 
                &rectleft, &rectright
                );
    }
    else 
    {
        // 如果是鱼眼相机模型
        // 计算左右相机的矫正参数
        cv::fisheye::stereoRectify(
                pcalib_model_left_->Kcv_, pcalib_model_left_->Dcv_,
                pcalib_model_right_->Kcv_, pcalib_model_right_->Dcv_,
                pcalib_model_left_->img_size_, 
                pcalib_model_right_->Rcv_cic0_, 
                pcalib_model_right_->tcv_cic0_,
                Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 
                pcalib_model_left_->img_size_, 
                pslamstate_->alpha_
                );
        
        rectleft = cv::Rect(0, 0, pcalib_model_left_->img_w_, pcalib_model_left_->img_h_);
        rectright = cv::Rect(0, 0, pcalib_model_right_->img_w_, pcalib_model_right_->img_h_);
    }

    std::cout << "\n Alpha : " << pslamstate_->alpha_;

    std::cout << "\n Kl : \n" << pcalib_model_left_->Kcv_;
    std::cout << "\n Kr : \n" << pcalib_model_right_->Kcv_;

    std::cout << "\n Dl : \n" << pcalib_model_left_->Dcv_;
    std::cout << "\n Dr : \n" << pcalib_model_right_->Dcv_;

    std::cout << "\n Rl : \n" << Rl;
    std::cout << "\n Rr : \n" << Rr;
    
    std::cout << "\n Pl : \n" << Pl;
    std::cout << "\n Pr : \n" << Pr;

    // % OpenCV can handle left-right or up-down camera arrangements
    // isVerticalStereo = abs(RCT.P2(2,4)) > abs(RCT.P2(1,4));
    // 设置左右相机之间的相互映射关系 
    pcalib_model_left_->setUndistStereoMap(Rl, Pl, rectleft);
    pcalib_model_right_->setUndistStereoMap(Rr, Pr, rectright);

    // SLAM state keeps track of the initial intrinsic
    // parameters (perhaps to be used for optim...)
    // 设置左右相机的标定参数
    pslamstate_->fxl_ = pcalib_model_left_->fx_;
    pslamstate_->fyl_ = pcalib_model_left_->fy_;
    pslamstate_->cxl_ = pcalib_model_left_->cx_;
    pslamstate_->cyl_ = pcalib_model_left_->cy_;

    pslamstate_->fxr_ = pcalib_model_right_->fx_;
    pslamstate_->fyr_ = pcalib_model_right_->fy_;
    pslamstate_->cxr_ = pcalib_model_right_->cx_;
    pslamstate_->cyr_ = pcalib_model_right_->cy_;
}

/**
 * @brief   SLAM 系统重置函数
*/
void SlamManager::reset()
{
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET REQUIRED!";
    std::cout << "\n=======================================\n";

    pcurframe_->reset();
    pvisualfrontend_->reset();
    pmap_->reset();
    pmapper_->reset();

    pslamstate_->reset();
    Logger::reset();

    frame_id_ = -1;

    std::lock_guard<std::mutex> lock(img_mutex_);
    
    qimg_left_ = std::queue<cv::Mat>(); 
    qimg_right_ = std::queue<cv::Mat>();
    qimg_time_ = std::queue<double>();

    bnew_img_available_ = false;
    
    std::cout << "\n=======================================\n";
    std::cout << "\t RESET APPLIED!";
    std::cout << "\n=======================================\n";
}


// ==========================
//   Visualization functions
// ==========================

/**
 * @brief   以帧率可视化当前帧和轨迹还有点云 
 * @param   time  当前帧的时间戳
*/
void SlamManager::visualizeAtFrameRate(const double time) 
{
    bframe_viz_ison_ = true;
    // 可视化当前帧
    visualizeFrame(pvisualfrontend_->cur_img_, time);
    // 可视化轨迹
    visualizeVOTraj(time);
    // 发布地图点云 
    prosviz_->pubPointCloud(pmap_->pcloud_, time);
    // 正在可视化的标志位
    bframe_viz_ison_ = false;
}

/**
 * @brief   以关键帧率可视化共视图和所有关键帧的轨迹 
*/
void SlamManager::visualizeAtKFsRate(const double time)
{
    bkf_viz_ison_ = true;
    // 可视化共视图
    visualizeCovisibleKFs(time);
    // 可视化所有关键帧的轨迹
    visualizeFullKFsTraj(time);
    // 正在可视化的标志位 
    bkf_viz_ison_ = false;
}

/**
 * @brief   可视化当前帧
 * @param   imleft  左目图像
 * @param   time    当前帧的时间戳
*/
void SlamManager::visualizeFrame(const cv::Mat &imleft, const double time)
{
    // 如果没有订阅可视化话题，则直接返回，节省计算资源 
    if( prosviz_->pub_image_track_.getNumSubscribers() == 0 ) {
        return;
    }

    // Display keypoints
    // 转换为RGB图像，便于可视化特征点 
    cv::Mat img_2_pub;
    cv::cvtColor(imleft, img_2_pub, CV_GRAY2RGB);

    for( const auto &kp : pcurframe_->getKeypoints() ) {
        // 遍历所有特征点 
        cv::Scalar col;

        if(kp.is_retracked_) {
            if(kp.is3d_) {// 正在被跟踪的3D点是绿色的 
                col = cv::Scalar(0,255,0);
            } else {// 正在被跟踪的2D点是黄色的
                col = cv::Scalar(235, 235, 52);
            } 
        } else if(kp.is3d_) {// 没有被跟踪的3D点是蓝色的 
            col = cv::Scalar(255,0,0);
        } else {// 没有被跟踪的2D点是红色的 
            col = cv::Scalar(0,0,255);
        }
        // 画出特征点
        cv::circle(img_2_pub, kp.px_, 4, col, -1);
    }
    // 发布当前帧的图像 
    prosviz_->pubTrackImage(img_2_pub, time);
}

/**
 * @brief   可视化当前帧的轨迹 
*/
void SlamManager::visualizeVOTraj(const double time)
{
    // 只是发布当前帧的位姿，不是轨迹
    prosviz_->pubVO(pcurframe_->getTwc(), time);
}

/**
 * @brief   可视化共视图
 * @param   time  当前帧的时间戳
*/
void SlamManager::visualizeCovisibleKFs(const double time)
{
    // 如果没有订阅可视化话题，则直接返回，节省计算资源 
    if( prosviz_->pub_kfs_pose_.getNumSubscribers() == 0 ) {
        return;
    }
    // 遍历当前帧的共视关键帧 
    for( const auto &covkf : pcurframe_->getCovisibleKfMap() ) {
        // 获取关键帧
        auto pkf = pmap_->getKeyframe(covkf.first);
        if( pkf != nullptr ) {
            // 发布关键帧的位姿
            prosviz_->addVisualKF(pkf->getTwc());
        }
    }
    // 发布共视图
    prosviz_->pubVisualKFs(time);
}

/**
 * @brief   可视化所有关键帧的轨迹
*/
void SlamManager::visualizeFullKFsTraj(const double time)
{
    // 如果没有订阅可视化话题，则直接返回，节省计算资源
    if( prosviz_->pub_kfs_traj_.getNumSubscribers() == 0 ) {
            return;
    }
    // 清空轨迹
    prosviz_->clearKFsTraj();
    // 遍历所有关键帧 
    for( int i = 0 ; i <= pcurframe_->kfid_ ; i++ ) {
        auto pkf = pmap_->getKeyframe(i);
        if( pkf != nullptr ) {
            // 添加关键帧的位姿 
            prosviz_->addKFsTraj(pkf->getTwc());
        }
    }
    // 发布轨迹 
    prosviz_->pubKFsTraj(time);
}

/**
 * @brief   可视化所有关键帧的轨迹
*/
void SlamManager::visualizeFinalKFsTraj()
{
    // 如果没有订阅可视化话题，则直接返回，节省计算资源
    if( prosviz_->pub_final_kfs_traj_.getNumSubscribers() == 0 ) {
        return;
    }
    for( int i = 0 ; i <= pcurframe_->kfid_ ; i++ ) {
        // 遍历所有关键帧
        auto pkf = pmap_->getKeyframe(i);
        if( pkf != nullptr ) {
            // 发布关键帧的位姿
            prosviz_->pubFinalKFsTraj(pkf->getTwc(), pkf->img_time_);
        }
    }
}

// ==========================
// Write Results functions
// ==========================

/**
 * @brief   写入结果
*/
void SlamManager::writeResults()
{
    // Make sure that nothing is running in the background
    // 确保没有后台线程在运行 
    while( pslamstate_->blocalba_is_on_ || pslamstate_->blc_is_on_ ) {
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }
    // 可视化所有关键帧的轨迹 
    visualizeFullKFsTraj(pcurframe_->img_time_);

    // Write Trajectories files
    // 写入轨迹文件，包括KITTI格式和TUM格式 
    Logger::writeTrajectory("ov2slam_traj.txt");
    Logger::writeTrajectoryKITTI("ov2slam_traj_kitti.txt");

    for( const auto & kfid_pkf : pmap_->map_pkfs_ )
    {
        auto pkf = kfid_pkf.second;
        if( pkf != nullptr ) {
            Logger::addKfSE3Pose(pkf->img_time_, pkf->getTwc());
        }
    }
    // 写入关键帧的轨迹文件 
    Logger::writeKfsTrajectory("ov2slam_kfs_traj.txt");

    // Apply full BA on KFs + 3D MPs if required + save
    if( pslamstate_->do_full_ba_ ) 
    {// 结束之后的处理 
        std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
        pmapper_->runFullBA();

        prosviz_->pubPointCloud(pmap_->pcloud_, ros::Time::now().toSec());
        visualizeFinalKFsTraj();

        for( const auto & kfid_pkf : pmap_->map_pkfs_ ) {
            auto pkf = kfid_pkf.second;
            if( pkf != nullptr ) {
                Logger::addKfSE3Pose(pkf->img_time_, pkf->getTwc());
            }
        }

        Logger::writeKfsTrajectory("ov2slam_fullba_kfs_traj.txt");
    }

    // Write full trajectories taking into account LC
    if( pslamstate_->buse_loop_closer_ )
    {
        writeFullTrajectoryLC();
    }
}


void SlamManager::writeFullTrajectoryLC()
{
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTwc;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTpc;
    std::vector<bool> viskf;

    vTwc.reserve(Logger::vframepose_.size());
    vTpc.reserve(Logger::vframepose_.size());
    viskf.reserve(Logger::vframepose_.size());

    size_t kfid = 0;

    Sophus::SE3d Twc, Twkf;

    std::ofstream f;
    std::string filename = "ov2slam_full_traj_wlc.txt";

    std::cout << "\n Going to write the full trajectory w. LC into : " << filename << "\n";

    f.open(filename.c_str());
    f << std::fixed;

    float fid = 0.;

    for( auto & fr : Logger::vframepose_ )
    {
        if( !fr.iskf_ || (fr.iskf_ && !pmap_->map_pkfs_.count(kfid)) ) 
        {
            // Get frame's pose from relative pose w.r.t. prev frame
            Eigen::Map<Eigen::Vector3d> t(fr.tprev_cur_);
            Eigen::Map<Eigen::Quaterniond> q(fr.qprev_cur_);

            vTpc.push_back(Sophus::SE3d(q,t));

            Sophus::SE3d Tprevcur(q,t);

            Twc = Twc * Tprevcur;

            viskf.push_back(false);

        } else {

            // Get keyframe's pose from map manager
            auto pkf = pmap_->map_pkfs_.at(kfid);

            Twc = pkf->getTwc();

            Twkf = Twc;

            kfid++;

            viskf.push_back(true);

            Eigen::Map<Eigen::Vector3d> t(fr.tprev_cur_);
            Eigen::Map<Eigen::Quaterniond> q(fr.qprev_cur_);
            vTpc.push_back(Sophus::SE3d(q,t));
        }

        vTwc.push_back(Twc);

        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.unit_quaternion();

        f << std::setprecision(9) << fid << " " << twc.x() << " " << twc.y() << " " << twc.z()
            << " " << qwc.x() << " " << qwc.y() << " " << qwc.z() << " " << qwc.w() << std::endl;

        f.flush();

        fid += 1.;
    }

    f.close();

    std::cout << "\nFull Trajectory w. LC file written!\n";
    
    // Apply full pose graph for optimal full trajectory w. LC
    pmapper_->pestimator_->poptimizer_->fullPoseGraph(vTwc, vTpc, viskf);
}