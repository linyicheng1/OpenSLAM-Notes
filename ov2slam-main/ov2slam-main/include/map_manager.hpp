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


#include <mutex>
#include <unordered_map>

#include <pcl_ros/point_cloud.h>

#include "slam_params.hpp"
#include "frame.hpp"
#include "map_point.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"

// 地图管理类 
class MapManager {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数
    MapManager() {}

    MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker);
    // 当前帧成为关键帧前的准备工作 
    void prepareFrame();
    // 添加关键帧 
    void addKeyframe();
    // 添加地图点 
    void addMapPoint(const cv::Scalar &color = cv::Scalar(200));
    void addMapPoint(const cv::Mat &desc, const cv::Scalar &color = cv::Scalar(200));
    // 根据id获取关键帧和地图点 
    std::shared_ptr<Frame> getKeyframe(const int kfid) const;
    std::shared_ptr<MapPoint> getMapPoint(const int lmid) const;
    // 跟新地图点信息
    void updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, const double kfanch_invdepth=-1.);
    // 添加地图点观测 
    void addMapPointKfObs(const int lmid, const int kfid);
    // 设置地图点观测 
    bool setMapPointObs(const int lmid);
    // 更新关键帧之间的共视关系 
    void updateFrameCovisibility(Frame &frame);
    // 合并地图点
    void mergeMapPoints(const int prevlmid, const int newlmid);
    // 删除关键帧 
    void removeKeyframe(const int kfid);
    // 删除地图点
    void removeMapPoint(const int lmid);
    // 删除地图点观测 
    void removeMapPointObs(const int lmid, const int kfid);
    void removeMapPointObs(MapPoint &lm, Frame &frame);
    // 删除当前帧的地图点观测
    void removeObsFromCurFrameById(const int lmid);
    void removeObsFromCurFrameByIdx(const int kpidx);
    // 创建关键帧 
    void createKeyframe(const cv::Mat &im, const cv::Mat &imraw);
    // 往关键帧中添加地图点 
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> &vscales, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<cv::Mat> &vdescs, Frame &frame);
    void addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> &vscales, const std::vector<float> &vangles, 
                const std::vector<cv::Mat> &vdescs, Frame &frame);
    // 从关键帧中提取特征点
    void extractKeypoints(const cv::Mat &im, const cv::Mat &imraw);
    // 提取特征点的描述子 
    void describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, 
                const std::vector<cv::Point2f> &vpts, 
                const std::vector<int> *pvscales = nullptr, 
                std::vector<float> *pvangles = nullptr);
    // klt 双目跟踪 
    void kltStereoTracking(const std::vector<cv::Mat> &vleftpyr, 
                const std::vector<cv::Mat> &vrightpyr);
    // 双目匹配
    void stereoMatching(Frame &frame, const std::vector<cv::Mat> &vleftpyr, 
                const std::vector<cv::Mat> &vrightpyr);

    void guidedMatching(Frame &frame);
    // 三角化
    void triangulate(Frame &frame);
    void triangulateTemporal(Frame &frame);
    void triangulateStereo(Frame &frame);

    Eigen::Vector3d computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr);
    // 计算深度分布
    void getDepthDistribution(const Frame &frame, double &mean_depth, double &std_depth);

    void reset();
    
    int nlmid_, nkfid_;
    int nblms_, nbkfs_;

    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<FeatureExtractor> pfeatextract_;
    std::shared_ptr<FeatureTracker> ptracker_;

    std::shared_ptr<Frame> pcurframe_;

    std::unordered_map<int, std::shared_ptr<Frame>> map_pkfs_;
    std::unordered_map<int, std::shared_ptr<MapPoint>> map_plms_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud_;

    mutable std::mutex kf_mutex_, lm_mutex_;
    mutable std::mutex curframe_mutex_;

    mutable std::mutex map_mutex_, optim_mutex_;
};