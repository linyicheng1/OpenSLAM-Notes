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
#pragma once


#include <vector>
#include <queue>
#include <mutex>

#include "slam_params.hpp"
#include "ros_visualizer.hpp"

#include "logger.hpp"

#include "camera_calibration.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"

#include "frame.hpp"
#include "map_manager.hpp"
#include "visual_front_end.hpp"
#include "mapper.hpp"
#include "estimator.hpp"

// slam 管理类 
class SlamManager {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数 
    SlamManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<RosVisualizer> pviz);
    // 运行线程 
    void run();
    // 获取当前帧 
    bool getNewImage(cv::Mat &iml, cv::Mat &imr, double &time);
    // 添加新的双目图像到SLAM系统中
    void addNewStereoImages(const double time, cv::Mat &im0, cv::Mat &im1);
    // 添加新的单目图像到SLAM系统中
    void addNewMonoImage(const double time, cv::Mat &im0);
    // 设置标定参数 
    void setupCalibration();
    void setupStereoCalibration();
    // 重置SLAM系统 
    void reset();
    // 写入结果 
    void writeResults();
    // 写入完整轨迹 
    void writeFullTrajectoryLC();
    // 可视化 
    void visualizeAtFrameRate(const double time);
    void visualizeFrame(const cv::Mat &imleft, const double time);
    void visualizeVOTraj(const double time);

    void visualizeAtKFsRate(const double time);
    void visualizeCovisibleKFs(const double time);
    void visualizeFullKFsTraj(const double time);
    
    void visualizeFinalKFsTraj();
    // 当前帧id计数
    int frame_id_ = -1;
    // 新图像是否可用 
    bool bnew_img_available_ = false;
    // 是否需要退出
    bool bexit_required_ = false;

    bool bis_on_ = false;
    
    bool bframe_viz_ison_ = false;
    bool bkf_viz_ison_ = false;

    std::shared_ptr<SlamParams> pslamstate_;// slam参数类
    std::shared_ptr<RosVisualizer> prosviz_;// ros可视化类 

    std::shared_ptr<CameraCalibration> pcalib_model_left_;// 左相机标定参数 
    std::shared_ptr<CameraCalibration> pcalib_model_right_;// 右相机标定参数

    std::shared_ptr<Frame> pcurframe_;// 当前帧 

    std::shared_ptr<MapManager> pmap_;// 地图管理类 

    std::unique_ptr<VisualFrontEnd> pvisualfrontend_;// 视觉前端类 
    std::unique_ptr<Mapper> pmapper_;// 地图构建类 

    std::shared_ptr<FeatureExtractor> pfeatextract_;// 特征提取类 
    std::shared_ptr<FeatureTracker> ptracker_;// 特征跟踪类

    std::queue<cv::Mat> qimg_left_, qimg_right_;// 图像队列
    std::queue<double> qimg_time_;// 时间戳队列

    std::mutex img_mutex_;
};