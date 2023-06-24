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


#include "frame.hpp"

// 特征点提取类 
class FeatureExtractor {

public:
    // 特征提取构造函数 
    FeatureExtractor() {};
    FeatureExtractor(size_t nmaxpts, size_t nmaxdist, double dmaxquality, int nfast_th);

    // 提取 GFTT 特征点 
    std::vector<cv::Point2f> detectGFTT(const cv::Mat &im, const std::vector<cv::Point2f> &vcurkps,
                                        const cv::Mat &roi, int nbmax=-1) const;

    // 提取 FAST 特征点
    std::vector<cv::Point2f> detectGridFAST(const cv::Mat &im, const int ncellsize, 
        const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi);

    // 计算 BRIEF 描述子 
    std::vector<cv::Mat> describeBRIEF(const cv::Mat &im, const std::vector<cv::Point2f> &vpts) const;
    
    // 计算特征点 
    std::vector<cv::Point2f> detectSingleScale(const cv::Mat &im, const int ncellsize, 
            const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi);
    // 设置掩码
    void setMask(const cv::Mat &im, const std::vector<cv::Point2f> &vpts,  const int dist, cv::Mat &mask) const;

    size_t nmaxpts_, nmaxdist_, nmindist_;
    double dmaxquality_, dminquality_;
    int nfast_th_;

    std::vector<int> vumax_;
};