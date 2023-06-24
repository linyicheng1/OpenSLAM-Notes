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


#include <unordered_map>
#include <set>
#include <mutex>

#include <Eigen/Core>
#include <opencv2/core.hpp>

// 地图点类 
class MapPoint {

public:
// 内存对齐，解决动态分配Eigen矩阵时的内存对齐问题 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数 
    MapPoint() {}
    // lmid 地图点id，kfid 观测到该地图点的关键帧id，bobs 是否被观测到
    MapPoint(const int lmid, const int kfid, const bool bobs=true);
    // lmid 地图点id，kfid 观测到该地图点的关键帧id，desc 描述子，bobs 是否被观测到
    MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const bool bobs=true);
    // lmid 地图点id，kfid 观测到该地图点的关键帧id，color 颜色，bobs 是否被观测到
    MapPoint(const int lmid, const int kfid, const cv::Scalar &color, const bool bobs=true);
    // lmid 地图点id，kfid 观测到该地图点的关键帧id，desc 描述子，color 颜色，bobs 是否被观测到
    MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const cv::Scalar &color, const bool bobs=true);

    // 操作接口 
    // 设置地图点的3D坐标，kfanch_invdepth 关键帧的逆深度 
    void setPoint(const Eigen::Vector3d &ptxyz, const double kfanch_invdepth=-1.);
    // 获取地图点的3D坐标 
    Eigen::Vector3d getPoint() const;
    // 获取观测到该地图点的关键帧id 
    std::set<int> getKfObsSet() const;
    // 添加观测到该地图点的关键帧id 
    void addKfObs(const int kfid);
    // 移除观测到该地图点的关键帧id 
    void removeKfObs(const int kfid);
    // 添加描述子 
    void addDesc(const int kfid, const cv::Mat &d);
    // 获取地图点状态，被设置为bad的地图点不会被优化
    bool isBad();
    // 计算两个地图点的描述子距离 
    float computeMinDescDist(const MapPoint &lm);

    // For using MapPoint in ordered containers
    // 根据地图点id判定是否相等 
    bool operator< (const MapPoint &mp) const
    {
        return lmid_ < mp.lmid_;
    }

    // MapPoint id
    int lmid_;// 地图点id 

    // True if seen in current frame
    bool isobs_;// 在当前帧是否被观测到 

    // True if MP has been init
    bool is3d_;// 是否初始化

    // Set of observed KF ids
    std::set<int> set_kfids_;// 观测到该地图点的关键帧id 

    // 3D position
    Eigen::Vector3d ptxyz_;// 3D坐标 

    // Anchored position
    int kfid_;// 观测到该地图点的关键帧id 
    double invdepth_;// 关键帧上的逆深度

    // Mean desc and list of descs
    cv::Mat desc_;// 平均描述子
    std::unordered_map<int, cv::Mat> map_kf_desc_;// 描述子, key: 关键帧id, value: 描述子 
    std::unordered_map<int, float> map_desc_dist_;// 描述子距离, key: 关键帧id, value: 描述子距离 

    // For vizu
    cv::Scalar color_ = cv::Scalar(200);// 颜色, 用于可视化 

    mutable std::mutex pt_mutex;// 互斥锁，用于多线程访问
};