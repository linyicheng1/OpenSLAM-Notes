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


#include <queue>
#include <vector>
#include <unordered_set>

#include "map_manager.hpp"
#include "multi_view_geometry.hpp"
#include "optimizer.hpp"
#include "estimator.hpp"
#include "loop_closer.hpp"
// 关键帧结构体
struct Keyframe {
    int kfid_;// 关键帧id
    cv::Mat imleft_, imright_;// 左右图像 
    cv::Mat imleftraw_, imrightraw_;// 左右图像原始图像
    // 金字塔图像 
    std::vector<cv::Mat> vpyr_imleft_, vpyr_imright_;
    bool is_stereo_;// 是否为双目图像
    
    Keyframe()
        : kfid_(-1), is_stereo_(false)
    {}
    // 单目构造函数 
    Keyframe(int kfid, const cv::Mat &imleftraw) 
        : kfid_(kfid), imleftraw_(imleftraw.clone()), is_stereo_(false)
    {}

    // 双目构造函数, 传入金字塔图像 
    Keyframe(int kfid, const cv::Mat &imleftraw, const std::vector<cv::Mat> &vpyrleft, 
        const std::vector<cv::Mat> &vpyrright )
        : kfid_(kfid), imleftraw_(imleftraw.clone()) 
        , vpyr_imleft_(vpyrleft), vpyr_imright_(vpyrright)
        , is_stereo_(true)
    {}
    // 双目构造函数
    Keyframe(int kfid, const cv::Mat &imleftraw, const cv::Mat &imrightraw, 
        const std::vector<cv::Mat> &vpyrleft
         )
        : kfid_(kfid)
        , imleftraw_(imleftraw.clone())
        , imrightraw_(imrightraw.clone())
        , vpyr_imleft_(vpyrleft)
    {}
    // 输出关键帧信息
     void displayInfo() {
         std::cout << "\n\n Keyframe struct object !  Info : id #" << kfid_ << " - is stereo : " << is_stereo_;
         std::cout << " - imleft size : " << imleft_.size << " - imright size : " << imright_.size;
         std::cout << " - pyr left size : " << vpyr_imleft_.size() << " - pyr right size : " << vpyr_imright_.size() << "\n\n";
     }
    // 释放图像
     void releaseImages() {
         imleft_.release();
         imright_.release();
         imleftraw_.release();
         imrightraw_.release();
         vpyr_imleft_.clear();
         vpyr_imright_.clear();
     }
};

// 地图构建线程
class Mapper {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数
    Mapper() {}
    Mapper(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap, std::shared_ptr<Frame> pframe);
    // 线程运行函数 
    void run();
    // 当前帧和局部地图匹配 
    bool matchingToLocalMap(Frame &frame);
    // 当前帧和地图匹配
    std::map<int,int> matchToMap(const Frame &frame, const float fmaxprojerr, const float fdistratio, std::unordered_set<int> &set_local_lmids);
    // 合并匹配点
    void mergeMatches(const Frame &frame, const std::map<int,int> &map_kpids_lmids);
    // p3p ransac 计算位姿 
    bool p3pRansac(const Frame &frame, const std::map<int,int>& map_kpids_lmids, Sophus::SE3d &Twc, std::vector<int> &voutlier_ids);
    // pnp 优化位姿 
    bool computePnP(const Frame &frame, const std::map<int,int>& map_kpids_lmids, Sophus::SE3d &Twc, std::vector<int> &voutlier_ids);
    // 三角化 
    void triangulate(Frame &frame);
    void triangulateTemporal(Frame &frame);
    void triangulateStereo(Frame &frame);
    bool triangulate(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr, Eigen::Vector3d &wpt);
    Eigen::Vector3d computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);
    // 整体的ba优化 
    void runFullBA();
    // 位姿图ba优化
    void runFullPoseGraph(std::vector<double*> &vtwc, std::vector<double*> &vqwc, std::vector<double*> &vtprevcur, std::vector<double*> &vqprevcur, std::vector<bool> &viskf);
    // 添加新的关键帧
    bool getNewKf(Keyframe &kf);
    void addNewKf(const Keyframe &kf);

    void reset();

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<MapManager> pmap_;
    std::shared_ptr<Frame> pcurframe_;

    std::shared_ptr<Estimator> pestimator_;
    std::shared_ptr<LoopCloser> ploopcloser_;

    bool bnewkfavailable_ = false;
    bool bwaiting_for_lc_ = false;
    bool bexit_required_ = false; 

    std::queue<Keyframe> qkfs_;

    std::mutex qkf_mutex_;
};