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

#include <iostream>

#include "map_point.hpp"

/**
 * @brief 地图点的构造函数 
 * @param lmid 地图点的id
 * @param kfid 地图点所在的关键帧id
 * @param bobs 地图点是否被观测到 
*/
MapPoint::MapPoint(const int lmid, const int kfid, const bool bobs)
    : lmid_(lmid), isobs_(bobs), kfid_(kfid), invdepth_(-1.)
{
    // 将地图点所在的关键帧id加入到set_kfids_中 
    set_kfids_.insert(kfid);
    is3d_ = false;// 地图点是否是3D点，默认为false 
    ptxyz_.setZero();// 地图点的3D坐标，默认为0 
}

/**
 * @brief 地图点的构造函数
 * @param lmid 地图点的id
 * @param kfid 地图点所在的关键帧id
 * @param desc 地图点的描述子
 * @param bobs 地图点是否被观测到
*/
MapPoint::MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const bool bobs)
    : lmid_(lmid), isobs_(bobs), kfid_(kfid), invdepth_(-1.)
{
    // 将地图点所在的关键帧id加入到set_kfids_中
    set_kfids_.insert(kfid);
    // 将地图点的描述子加入到map_kf_desc_中 
    map_kf_desc_.emplace(kfid, desc);
    // 第一个距离为0 
    map_desc_dist_.emplace(kfid, 0.);
    // 将地图点的描述子赋值给desc_
    desc_ = map_kf_desc_.at(kfid);
    // 地图点是否是3D点，默认为false
    is3d_ = false;
    ptxyz_.setZero();// 地图点的3D坐标，默认为0
}

/**
 * @brief 地图点的构造函数
 * @param lmid 地图点的id 
 * @param kfid 地图点所在的关键帧id
 * @param color 地图点的颜色
 * @param bobs 地图点是否被观测到
*/
MapPoint::MapPoint(const int lmid, const int kfid, const cv::Scalar &color, const bool bobs)
    : lmid_(lmid), isobs_(bobs), kfid_(kfid), invdepth_(-1.), color_(color)
{
    // 将地图点所在的关键帧id加入到set_kfids_中 
    set_kfids_.insert(kfid);
    is3d_ = false;// 地图点是否是3D点，默认为false
    ptxyz_.setZero();// 地图点的3D坐标，默认为0
}

/**
 * @brief 地图点的构造函数
 * @param lmid 地图点的id
 * @param kfid 地图点所在的关键帧id
 * @param desc 地图点的描述子
 * @param color 地图点的颜色
 * @param bobs 地图点是否被观测到
*/
MapPoint::MapPoint(const int lmid, const int kfid, const cv::Mat &desc, 
    const cv::Scalar &color, const bool bobs)
    : lmid_(lmid), isobs_(bobs), kfid_(kfid), invdepth_(-1.), color_(color)
{
    // 将地图点所在的关键帧id加入到set_kfids_中
    set_kfids_.insert(kfid);
    // 将地图点的描述子加入到map_kf_desc_中
    map_kf_desc_.emplace(kfid, desc);
    // 第一个距离为0
    map_desc_dist_.emplace(kfid, 0.);
    // 将地图点的描述子赋值给desc_
    desc_ = map_kf_desc_.at(kfid);
    // 地图点是否是3D点，默认为false
    is3d_ = false;
    ptxyz_.setZero();// 地图点的3D坐标，默认为0
}

/**
 * @brief 设置地图点的3D坐标 
 * @param ptxyz 地图点的3D坐标
 * @param kfanch_invdepth 地图点的逆深度，在当前帧中的逆深度
*/
void MapPoint::setPoint(const Eigen::Vector3d &ptxyz, const double kfanch_invdepth)
{
    std::lock_guard<std::mutex> lock(pt_mutex);
    ptxyz_ = ptxyz;// 设置地图点的3D坐标 
    is3d_ = true;// 地图点是否是3D点，默认为false
    // kfanch_invdepth >= 0. 表示地图点在当前帧中的逆深度有效
    if( kfanch_invdepth >= 0. ) {
        invdepth_ = kfanch_invdepth;
    }
}

/**
 * @brief 设置地图点的3D坐标
*/
Eigen::Vector3d MapPoint::getPoint() const
{
    std::lock_guard<std::mutex> lock(pt_mutex);
    return ptxyz_;// 返回地图点的3D坐标 
}

/**
 * @brief 获取观测到地图点的关键帧id
*/
std::set<int> MapPoint::getKfObsSet() const
{
    std::lock_guard<std::mutex> lock(pt_mutex);
    return set_kfids_;// 返回观测到地图点的关键帧id
}

/**
 * @brief 添加观测到地图点的关键帧id 
*/
void MapPoint::addKfObs(const int kfid)
{
    std::lock_guard<std::mutex> lock(pt_mutex);
    set_kfids_.insert(kfid);// 将观测到地图点的关键帧id加入到set_kfids_中 
}

/**
 * @brief 移除观测到地图点的关键帧id 
*/
void MapPoint::removeKfObs(const int kfid)
{
    std::lock_guard<std::mutex> lock(pt_mutex);
    // 如果kfid不在set_kfids_中，则返回
    if( !set_kfids_.count(kfid) ) {
        return;
    }
    
    // First remove the related id
    set_kfids_.erase(kfid);// 否则将kfid从set_kfids_中移除
    // 如果被删光了，则将desc_、map_kf_desc_、map_desc_dist_清空 
    if( set_kfids_.empty() ) {
        desc_.release();
        map_kf_desc_.clear();
        map_desc_dist_.clear();
        return;
    }

    // Set new KF anchor if removed
    if( kfid == kfid_ ) {
        kfid_ = *set_kfids_.begin();
    }

    // Then figure out the most representative one
    // (we could also use the last one to save time)
    float mindist = desc_.cols * 8.;// 最小距离 
    int minid = -1;// 最小距离对应的关键帧id 

    auto itdesc = map_kf_desc_.find(kfid);// 找到kfid对应的描述子 
    if( itdesc != map_kf_desc_.end() ) {

        for( const auto & kf_d : map_kf_desc_ )
        {// 遍历map_kf_desc_中的所有描述子 
            if( kf_d.first != kfid )
            {
                // 计算描述子之间的距离 
                float dist = cv::norm(itdesc->second, kf_d.second, cv::NORM_HAMMING);
                // 更新描述子之间的距离 
                float & descdist = map_desc_dist_.find(kf_d.first)->second;
                descdist -= dist;

                // Get the lowest one
                if( descdist < mindist ) {
                    mindist = descdist;// 更新最小距离 
                    minid = kf_d.first;
                }
            }
        }

        itdesc->second.release();
        // 删除描述子，和描述子之间的距离 
        map_kf_desc_.erase(kfid);
        map_desc_dist_.erase(kfid);

        // Remove desc / update mean desc
        if( minid > 0 ) {
            desc_ = map_kf_desc_.at(minid);// 选取相对最近的描述子作为地图点的描述子
        }
    }
}

/**
 * @brief 添加地图点的描述子 
 * @param kfid 关键帧id
 * @param d 描述子 
*/
void MapPoint::addDesc(const int kfid, const cv::Mat &d)
{
    std::lock_guard<std::mutex> lock(pt_mutex);
    // 判定是否已经存在该关键帧的描述子 
    auto it = map_kf_desc_.find(kfid);
    if( it != map_kf_desc_.end() ) {
        // 如果存在，则返回 
        return;
    }

    // First add the desc and init its distance score
    // 先将描述子加入到map_kf_desc_中 
    map_kf_desc_.emplace(kfid, d);
    // 再将描述子和距离加入到map_desc_dist_中， 默认距离为0 
    map_desc_dist_.emplace(kfid, 0);
    float & newdescdist = map_desc_dist_.find(kfid)->second;
    // 如果只有一个描述子，则直接将该描述子作为地图点的描述子 
    if( map_kf_desc_.size() == 1 ) {
        desc_ = d;
        return;
    }

    // Then figure out the most representative one
    // (we could also use the last one to save time)
    // 遍历map_kf_desc_中的所有描述子，找到最近的描述子
    float mindist = desc_.cols * 8.;
    int minid = -1;

    // Then update the distance scores for all desc
    for( const auto & kf_d : map_kf_desc_ )
    {// 遍历map_kf_desc_中的所有描述子 
        // 计算描述子之间的距离
        float dist = cv::norm(d, kf_d.second, cv::NORM_HAMMING);

        // Update previous desc
        map_desc_dist_.at(kf_d.first) += dist;// 更新描述子之间的距离 

        // Get the lowest one
        if( dist < mindist ) {
            mindist = dist;// 更新最小距离
            minid = kf_d.first;// 更新最小距离对应的关键帧id 
        }

        // Update new desc
        newdescdist += dist;// 当前描述子和其他描述字距离和
    }

    // Get the lowest one
    // 判定当前描述子是否为最小距离对应的描述子 
    if( newdescdist < mindist ) {
        minid = kfid;
    }
    // 更新最小距离对应的描述子 
    desc_ = map_kf_desc_.at(minid);
}

/**
 * @brief 判定地图点是否为坏点
*/
bool MapPoint::isBad()
{
    std::lock_guard<std::mutex> lock(pt_mutex);

    // Set as bad 3D MPs who are observed by 2 KF
    // or less and not observed by current frame
    if( set_kfids_.size() < 2 ) {
        // 如果观测到该地图点的关键帧数小于2，且是一个3d点，则认为该地图点为坏点
        if( !isobs_ && is3d_ ) {
            is3d_ = false;
            return true;
        }
    } 
    // 如果观测到的关键帧数为0，并且当前帧也观测不到，则认为该地图点为坏点
    if ( set_kfids_.size() == 0 && !isobs_ ) {
        is3d_ = false;
        return true;
    }

    return false;
}

/**
 * @brief 计算地图点的描述子之间的最小距离 
 * @param lm 地图点 
*/
float MapPoint::computeMinDescDist(const MapPoint &lm)
{
    std::lock_guard<std::mutex> lock(pt_mutex);

    float min_dist = 1000.;
    // 两重循环，计算任意两个地图点的描述子之间的最小距离
    for( const auto &kf_desc : map_kf_desc_ ) {
        for( const auto &kf_desc2 : lm.map_kf_desc_ ) {
            float dist = cv::norm(kf_desc.second, kf_desc2.second, cv::NORM_HAMMING);
            if( dist < min_dist ) {
                min_dist = dist;
            }
        }
    }
    // 返回最小距离
    return min_dist;
}