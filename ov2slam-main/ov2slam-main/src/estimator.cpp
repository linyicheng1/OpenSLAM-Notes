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

#include "estimator.hpp"

// 局部BA 优化 优化当前帧的位姿和地图点
void Estimator::run()
{
    std::cout << "\n Estimator is ready to process Keyframes!\n";
    
    while( !bexit_required_ ) {

        if( getNewKf() ) // 判断是否有新的关键帧 
        {
            if( pslamstate_->slam_mode_ ) 
            {
                if( pslamstate_->debug_ )
                    std::cout << "\n [Estimator] Slam-Mode - Processing new KF #" << pnewkf_->kfid_;
                // 计算局部BA 
                applyLocalBA();
                // 进行地图点的筛选 
                mapFiltering();

            } else {
                if( pslamstate_->debug_ )
                    std::cout << "\nNO OPITMIZATION (NEITHER SLAM MODE NOR SW MODE SELECTED) !\n";
            }
        } else {
            std::chrono::microseconds dura(20);
            std::this_thread::sleep_for(dura);
        }
    }

    poptimizer_->signalStopLocalBA();
    
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    std::cout << "\n Estimator thread is exiting.\n";
}


// 进行局部BA优化 
void Estimator::applyLocalBA()
{
    int nmincstkfs = 1;// 最小的共视关键帧数,双目为1,单目为2
    if( pslamstate_->mono_ ) {
        nmincstkfs = 2;
    }
    // 如果当前帧的共视关键帧数小于最小的共视关键帧数, 就不需要进行局部BA优化 
    if( pnewkf_->kfid_ < nmincstkfs ) {
        return;
    }

    if( pnewkf_->nb3dkps_ == 0 ) {
        return;
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.BA_localBA");

    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    // We signal that Estimator is performing BA
    pslamstate_->blocalba_is_on_ = true;// 设置为true, 表示正在进行局部BA优化 

    bool use_robust_cost = true;// 使用鲁棒核函数
    // 进行局部BA优化
    poptimizer_->localBA(*pnewkf_, use_robust_cost);

    // We signal that Estimator is stopping BA
    pslamstate_->blocalba_is_on_ = false;// 设置为false, 表示局部BA优化结束
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.BA_localBA");
}

// 进行地图点的筛选 
void Estimator::mapFiltering()
{
    // pslamstate_->fkf_filtering_ratio_ 强制小于1
    if( pslamstate_->fkf_filtering_ratio_ >= 1. ) {
        return;
    }
    // kf id 大于20 或者闭环检测开启, 就不需要进行地图点的筛选
    if( pnewkf_->kfid_ < 20 || pslamstate_->blc_is_on_ ) {
        return;
    }   

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.BA_map-filtering");
    // 获取关键帧的共视关系    
    auto covkf_map = pnewkf_->getCovisibleKfMap();

    for( auto it = covkf_map.rbegin() ; it != covkf_map.rend() ; it++ ) {
        // 获取关键帧的id
        int kfid = it->first;
        // 第一帧不需要进行筛选 
        // 设置为bnewkfavailable_不需要进行筛选
        if( bnewkfavailable_ || kfid == 0 ) {
            break;
        }
        // 只需要筛选当前帧之前的关键帧
        if( kfid >= pnewkf_->kfid_ ) {
            continue;
        }

        // Only useful if LC enabled
        if( pslamstate_->lckfid_ == kfid ) {
            continue;// 如果当前帧是闭环帧, 就不需要进行筛选
        }
        // 获取关键帧
        auto pkf = pmap_->getKeyframe(kfid);
        if( pkf == nullptr ) {// 如果关键帧为空, 就不需要进行筛选
            // 从共视关系中移除该关键帧  
            pnewkf_->removeCovisibleKf(kfid);
            continue;
        } 
        else if( (int)pkf->nb3dkps_ < pslamstate_->nmin_covscore_ / 2 ) {
            // 如果关键帧的地图点数量小于阈值, 就不需要进行筛选
            // 只有关键点很多，信息冗余的时候才需要进行筛选
            std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
            pmap_->removeKeyframe(kfid);
            continue;
        }

        size_t nbgoodobs = 0;
        size_t nbtot = 0;
        for( const auto &kp : pkf->getKeypoints3d() )
        {// 遍历关键帧的地图点 
            // 获取地图点
            auto plm = pmap_->getMapPoint(kp.lmid_);
            if( plm == nullptr ) {// 如果地图点为空
                // 从关键帧中移除该地图点
                pmap_->removeMapPointObs(kp.lmid_, kfid);
                continue;
            } 
            else if( plm->isBad() ) {
                continue;
            }
            else {
                // 获取地图点的观测帧数量
                size_t nbcokfs = plm->getKfObsSet().size();
                if( nbcokfs > 4 ) {
                    nbgoodobs++;
                }
            } 
            
            nbtot++;
            
            if( bnewkfavailable_ ) {
                break;
            }
        }
        // 计算筛选比例 
        // 如果筛选比例大于阈值, 就移除该关键帧
        float ratio = (float)nbgoodobs / nbtot;
        if( ratio > pslamstate_->fkf_filtering_ratio_ ) {

            // Only useful if LC enabled
            if( pslamstate_->lckfid_ == kfid ) {
                continue;
            }
            std::lock_guard<std::mutex> lock(pmap_->map_mutex_);
            pmap_->removeKeyframe(kfid);
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.BA_map-filtering");
}

// 从队列中获取新的关键帧  
bool Estimator::getNewKf()
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);

    // Check if new KF is available
    if( qpkfs_.empty() ) {// 如果队列为空, 就返回false
        bnewkfavailable_ = false;
        return false;
    }

    // In SLAM-mode, we only processed the last received KF
    // but we trick the covscore if several KFs were waiting
    // to make sure that they are all optimized
    std::vector<int> vkfids;
    vkfids.reserve(qpkfs_.size());
    while( qpkfs_.size() > 1 ) {
        qpkfs_.pop();
        vkfids.push_back(pnewkf_->kfid_);
    }
    pnewkf_ = qpkfs_.front();
    qpkfs_.pop();

    if( !vkfids.empty() ) {
        for( const auto &kfid : vkfids ) {
            pnewkf_->map_covkfs_[kfid] = pnewkf_->nb3dkps_;
        }

        if( pslamstate_->debug_ )
            std::cout << "\n ESTIMATOR is late!  Adding several KFs to BA...\n";
    }
    bnewkfavailable_ = false;

    return true;
}

// 添加新的关键帧到队列中 
void Estimator::addNewKf(const std::shared_ptr<Frame> &pkf)
{
    std::lock_guard<std::mutex> lock(qkf_mutex_);
    qpkfs_.push(pkf);
    bnewkfavailable_ = true;

    // We signal that a new KF is ready
    if( pslamstate_->blocalba_is_on_ 
        && !poptimizer_->stopLocalBA() ) 
    {
        poptimizer_->signalStopLocalBA();
    }
}

void Estimator::reset()
{
    std::lock_guard<std::mutex> lock2(pmap_->optim_mutex_);

    bnewkfavailable_ = false;
    bexit_required_ = false; 

    std::queue<std::shared_ptr<Frame>> empty;
    std::swap(qpkfs_, empty);
}