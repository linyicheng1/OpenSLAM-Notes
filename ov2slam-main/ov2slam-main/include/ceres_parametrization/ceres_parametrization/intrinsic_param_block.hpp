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


#include <cstddef>

// 内参优化模块 
class CalibParametersBlock {
    
public:
    CalibParametersBlock() {}
    /**
     * @brief 内参优化模块构造函数
     * @param id 相机id 
     * @param fx 相机内参fx 
     * @param fy 相机内参fy
     * @param cx 相机内参cx
     * @param cy 相机内参cy
    */
    CalibParametersBlock(const int id, const double fx, const double fy, 
        const double cx, const double cy) 
    {
        id_ = id;
        values_[0] = fx; values_[1] = fy;
        values_[2] = cx; values_[3] = cy;
    }
    /**
     * @brief 内参优化模块复制构造函数 
     * @param block 内参优化模块
    */
    CalibParametersBlock(const CalibParametersBlock &block) {
        id_ = block.id_;// 相机id复制
        // 内参复制 
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
    }

    /**
     * @brief 内参优化模块赋值运算符重载，等于号复制
     * @param block 内参优化模块
    */
    CalibParametersBlock& operator = (const CalibParametersBlock &block) 
    { 
        id_ = block.id_;
        for( size_t i = 0 ; i < ndim_ ; i++ ) {
            values_[i] = block.values_[i];
        }
        return *this; 
    }

    /**
     * @brief 获取所有的内参值 
    */
    void getCalib(double &fx, double &fy, double &cx, double &cy) {
        fx = values_[0]; fy = values_[1];
        cx = values_[2]; cy = values_[3]; 
    }

    /**
     * @brief 获取相机内参指针  
    */
    inline double* values() {  
        return values_; 
    }

    static const size_t ndim_ = 4;
    double values_[ndim_] = {0.,0.,0.,0.};
    int id_ = -1;
};