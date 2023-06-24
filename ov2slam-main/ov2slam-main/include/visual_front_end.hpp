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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "slam_params.hpp"
#include "map_manager.hpp"
#include "feature_tracker.hpp"

class MotionModel {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 根据当前的时间戳，预测相机的位姿 
    void applyMotionModel(Sophus::SE3d &Twc, double time) {
        if( prev_time_ > 0 ) 
        {
            // Provided Twc and prevTwc should be equal here
            // as prevTwc is updated right after pose computation
            // 如果和上一帧的位姿不一致，说明出现了问题，直接更新为当前帧的位姿
            // 构造的时候默认就是上一帧的位姿 
            if( !(Twc * prevTwc_.inverse()).log().isZero(1.e-5) )
            {
                // Might happen in case of LC!
                // So update prevPose to stay consistent
                prevTwc_ = Twc;
            }
            // 相对时间 
            double dt = (time - prev_time_);
            // 模型假设相机的运动是匀速的，因此根据上一帧的位姿和相对位姿，预测当前帧的位姿 
            Twc = Twc * Sophus::SE3d::exp(log_relT_ * dt);
        }
    }

    // 根据优化得到的相机位姿，更新运动模型 
    void updateMotionModel(const Sophus::SE3d &Twc, double time) {
        if( prev_time_ < 0. ) {
            // 上一次的时间戳小于0，说明是第一次进入，直接更新时间戳和位姿 
            prev_time_ = time;
            prevTwc_ = Twc;
        } else {
            // 相对时间 
            double dt = time - prev_time_;
            // 更新时间戳 
            prev_time_ = time;
            // 确保时间戳是递增的 
            if( dt < 0. ) {
                std::cerr << "\nGot image older than previous image! LEAVING!\n";
                exit(-1);
            }
            // 计算相对位姿 
            Sophus::SE3d Tprevcur = prevTwc_.inverse() * Twc;
            // 计算相对位姿的对数映射 
            log_relT_ = Tprevcur.log() / dt;
            // 更新上一帧的位姿 
            prevTwc_ = Twc;
        }
    }
    // 重置运动模型 
    void reset() {
        prev_time_ = -1.;
        log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
    }
    

    double prev_time_ = -1.;

    Sophus::SE3d prevTwc_;// 上一帧的位姿
    // 相对位姿的对数映射 
    Eigen::Matrix<double, 6, 1> log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
};

// 视觉前端类
class VisualFrontEnd {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数 
    VisualFrontEnd() {}
    VisualFrontEnd(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframerame, 
        std::shared_ptr<MapManager> pmap, std::shared_ptr<FeatureTracker> ptracker);
    // 前端可视化 
    bool visualTracking(cv::Mat &iml, double time);
    // 单目跟踪函数
    bool trackMono(cv::Mat &im, double time);
    // 双目跟踪函数
    bool trackStereo(cv::Mat &iml, cv::Mat &imr, double time);
    // 图像预处理 
    void preprocessImage(cv::Mat &img_raw);
    // 连续帧之间的跟踪
    void kltTracking();
    // 从关键帧中跟踪
    void kltTrackingFromKF();
    // 基础矩阵筛选离群点 
    void epipolar2d2dFiltering();
    // 计算当前帧的位姿 
    void computePose();
    // 计算视差
    float computeParallax(const int kfid, bool do_unrot=true, bool bmedian=true, bool b2donly=false);
    // 判断当前是否具有足够的视差
    bool checkReadyForInit();
    // 判断是否需要新的关键帧 
    bool checkNewKfReq();
    // 创建新的关键帧 
    void createKeyframe();
    // 运动模型估计
    void applyMotion();
    void updateMotion();
    // 重置当前帧
    void resetFrame();
    // 重置前端跟踪器
    void reset();

    std::shared_ptr<SlamParams> pslamstate_;// SLAM参数
    std::shared_ptr<Frame> pcurframe_;// 当前帧 
    std::shared_ptr<MapManager> pmap_;// 地图管理器 

    std::shared_ptr<FeatureTracker> ptracker_;// 特征跟踪器 

    cv::Mat left_raw_img_;// 原始图像
    cv::Mat cur_img_, prev_img_;// 当前帧和上一帧图像
    std::vector<cv::Mat> cur_pyr_, prev_pyr_;// 当前帧和上一帧图像金字塔
    std::vector<cv::Mat> kf_pyr_;// 关键帧图像金字塔
    
    MotionModel motion_model_;// 运动模型

    bool bp3preq_ = false; 
};
