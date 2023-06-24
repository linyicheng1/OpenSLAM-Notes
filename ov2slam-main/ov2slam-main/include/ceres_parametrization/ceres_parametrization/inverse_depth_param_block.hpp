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


#include <Eigen/Core>


// Mainly for prototyping !
// Current params : [u_anch, v_anch, inv_z]
// 逆深度优化模块 [u_anch, v_anch, inv_z] 在像素坐标系下的锚点坐标和逆深度 
class InvDepthParametersBlock {
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InvDepthParametersBlock() {}
    /**
     * @brief 逆深度优化模块构造函数 
     * @param id 特征点id 
     * @param kfid 关键帧id 
     * @param anch_depth 锚点深度 
    */
    InvDepthParametersBlock(const int id, const int kfid, 
        const double anch_depth)
        : id_(id), kfid_(kfid)
    {
        values_[0] = 1./anch_depth;
    }

    /**
     * @brief 逆深度优化模块复制构造函数
     * @param block 逆深度优化模块 
    */
    InvDepthParametersBlock(const InvDepthParametersBlock &block) {
        id_ = block.id_;// 复制id 
        // 复制数据 
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
    }

    /**
     * @brief 逆深度优化模块赋值运算符重载，用于复制数据 
     * @param block 逆深度优化模块 
     * @return InvDepthParametersBlock& 
    */
    InvDepthParametersBlock& operator = (const InvDepthParametersBlock &block) 
    { 
        id_ = block.id_;
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
        return *this; 
    }
    // 获取逆深度 
    double getInvDepth() {
        return values_[0];
    }
    // 获取数据指针 
    inline double* values() {  
        return values_; 
    }

    static const size_t ndim_ = 1;
    double values_[ndim_] = {0.};
    int id_= -1;
    int kfid_=-1;
};