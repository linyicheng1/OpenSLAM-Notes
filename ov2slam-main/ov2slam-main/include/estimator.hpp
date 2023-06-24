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
#include <deque>

#include "map_manager.hpp"
#include "optimizer.hpp"

// 局部BA线程 
class Estimator {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数 
    Estimator(std::shared_ptr<SlamParams> pslamstate, std::shared_ptr<MapManager> pmap)
        : pslamstate_(pslamstate), pmap_(pmap)
        , poptimizer_( new Optimizer(pslamstate_, pmap_) )
    {
        std::cout << "\n Estimator Object is created!\n";
    }
    // 线程函数
    void run();
    void reset();
    // 处理关键帧
    void processKeyframe();
    // 局部BA
    void applyLocalBA();
    // 地图过滤
    void mapFiltering();
    // 获取新的关键帧 
    bool getNewKf();
    // 添加新的关键帧
    void addNewKf(const std::shared_ptr<Frame> &pkf);

    // slam当前的状态
    std::shared_ptr<SlamParams> pslamstate_;
    // slam 地图 
    std::shared_ptr<MapManager> pmap_;
    // 优化器 
    std::unique_ptr<Optimizer> poptimizer_;
    // 新的关键帧 
    std::shared_ptr<Frame> pnewkf_;
    // 是否存在新的关键帧
    bool bnewkfavailable_ = false;
    bool bexit_required_ = false;
    // 是否需要进行局部BA
    bool blooseba_on_ = false;
    // 关键帧处理队列
    std::queue<std::shared_ptr<Frame>> qpkfs_;

    std::mutex qkf_mutex_;
};