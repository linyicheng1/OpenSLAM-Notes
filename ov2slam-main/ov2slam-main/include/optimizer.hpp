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


#include <deque>

#include "map_manager.hpp"

// 优化器类，SLAM的核心 
class Optimizer {

public:
// 构造函数 
    Optimizer(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap)
        : pslamstate_(pslamstate), pmap_(pmap), bstop_localba_(false)
    {
        std::cout << "\n Optimizer Object is created!\n";
    }
// 局部BA
    void localBA(Frame &newframe, const bool buse_robust_cost);
// 全局松弛BA
    void looseBA(const int inikfid, const int nkfid, const bool buse_robust_cost);
// 全局BA
    void fullBA(const bool buse_robust_cost);
// 停止局部BA的信号
    void signalStopLocalBA();
    bool stopLocalBA();
// 局部位姿图优化 
    bool localPoseGraph(Frame &newframe, int kfloop_id, const Sophus::SE3d &newTwc);
// 只优化地图点
    void structureOnlyBA(const std::vector<int> &vlm2optids);
// 全局位姿图优化
    bool fullPoseGraph(std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &vTwc, 
        std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &vTpc, 
        std::vector<bool> &viskf);



    std::shared_ptr<SlamParams> pslamstate_;
    std::shared_ptr<MapManager> pmap_;

    bool bstop_localba_;

    std::mutex localba_mutex_;
};