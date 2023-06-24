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

#include "feature_extractor.hpp"

#include <algorithm>
#include <iterator>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> 

#ifdef OPENCV_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#endif
#include <opencv2/video/tracking.hpp>

#ifdef OPENCV_LAMBDA_MISSING

namespace cv {

  class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
  {
    private:
      std::function<void(const Range&)> m_functor;
    public:
      ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor) :
        m_functor(functor)
        { }    
      virtual void operator() (const cv::Range& range) const
        {
          m_functor(range);
        }
  };
  
  inline void parallel_for_(const cv::Range& range, std::function<void(const cv::Range&)> functor, double nstripes=-1.)
    {
      parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
    }
}

#endif


cv::Ptr<cv::GFTTDetector> pgftt_;
cv::Ptr<cv::FastFeatureDetector> pfast_;
#ifdef OPENCV_CONTRIB
cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> pbrief_;
#else
cv::Ptr<cv::DescriptorExtractor> pbrief_;
#endif

bool compare_response(cv::KeyPoint first, cv::KeyPoint second) {
    return first.response > second.response;
}

/**
 * @brief 特征点提取器构造函数 
 * @param nmaxpts 最大特征点数
 * @param nmaxdist 特征点之间最大距离
 * @param dmaxquality 最大质量
 * @param nfast_th FAST角点阈值
*/
FeatureExtractor::FeatureExtractor(size_t nmaxpts, size_t nmaxdist, double dmaxquality, int nfast_th)
    : nmaxpts_(nmaxpts), nmaxdist_(nmaxdist), dmaxquality_(dmaxquality), nfast_th_(nfast_th)
{
    // 最小距离 = 最大距离 / 2 
    nmindist_ = nmaxdist / 2.;
    // 最小质量 = 最大质量 / 2
    dminquality_ = dmaxquality / 2.;

    std::cout << "\n*********************************\n";
    std::cout << "\nFeature Extractor is constructed!\n";
    std::cout << "\n>>> Maximum nb of kps : " << nmaxpts_;
    std::cout << "\n>>> Maximum kps dist : " << nmaxdist_;
    std::cout << "\n>>> Minimum kps dist : " << nmindist_;
    std::cout << "\n>>> Maximum kps qual : " << dmaxquality_;
    std::cout << "\n>>> Minimum kps qual : " << dminquality_;
    std::cout << "\n*********************************\n";
}


/**
 * \brief Detect GFTT features (Harris corners with Shi-Tomasi method) using OpenCV.
 *
 * \param[in] im  Image to process (cv::Mat).
 * \param[in] vcurkps Vector of px positions to filter out detections.
 * \param[in] roi  Initial ROI mask (255 ok / 0 reject) to filter out detections (cv::Mat<CV_8UC1>).
 * \return Vector of deteted kps px pos.
 */
/**
 * @brief GFTT角点检测 
 * @param im 图像
 * @param vcurkps 当前帧已经存在的特征点
 * @param roi 初始ROI
 * @param nbmax 最大特征点数
 * @return 检测到的特征点
*/
std::vector<cv::Point2f> FeatureExtractor::detectGFTT(const cv::Mat &im, const std::vector<cv::Point2f> &vcurkps, const cv::Mat &roi, int nbmax) const
{
    // 1. Check how many kps we need to detect
    size_t nb2detect = nmaxpts_ - vcurkps.size(); // 目标提取的特征点数 
    if( vcurkps.size() >= nmaxpts_ ) {// 如果当前帧特征点数已经超过最大特征点数，直接返回空 
        return std::vector<cv::Point2f>();
    }
    // 重新指定 nb2detect 参数 
    if( nbmax != -1 ) {
        nb2detect = nbmax;
    }

    // 1.1 Init the mask
    // 掩码，用于指定检测区域
    cv::Mat mask;
    if( !roi.empty() ) {
        mask = roi.clone();
    }
    setMask(im, vcurkps, nmaxdist_, mask);

    // 1.2 Extract kps
    std::vector<cv::KeyPoint> vnewkps; // 新提取的特征点
    std::vector<cv::Point2f> vnewpts;// 新提取的特征点坐标 
    vnewkps.reserve(nb2detect);
    vnewpts.reserve(nb2detect);

    // std::cout << "\n*****************************\n";
    // std::cout << "\t\t GFTT \n";
    // std::cout << "\n> Nb max pts : " << nmaxpts_;
    // std::cout << "\n> Nb 2 detect : " << nb2detect;
    // std::cout << "\n> Quality : " << dminquality_;
    // std::cout << "\n> Dist : " << nmaxdist_;

    // Init. detector if not done yet
    if( pgftt_ == nullptr ) {
        // opencv 函数，创建 GFTT 角点检测器 
        pgftt_ = cv::GFTTDetector::create(nmaxpts_, dminquality_, nmaxdist_);
    }
    // 设置检测器参数 
    pgftt_->setQualityLevel(dminquality_);
    pgftt_->setMinDistance(nmaxdist_);
    pgftt_->setMaxFeatures(nb2detect);
    // 检测特征点 
    pgftt_->detect(im, vnewkps, mask);

    // cv::KeyPointsFilter::runByPixelsMask(vnewkps,mask);
    // 转换为 cv::Point2f 格式
    cv::KeyPoint::convert(vnewkps, vnewpts);
    vnewkps.clear();

    // Check number of detections
    size_t nbdetected = vnewpts.size();
    // std::cout << "\n \t>>> Found : " << nbdetected;

    // Compute Corners with Sub-Pixel Accuracy
    if( nbdetected > 0 )
    {// 如果检测到了特征点，对特征点进行亚像素精确化 
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);
        // 亚像素精确化 
        cv::cornerSubPix(im, vnewpts, winSize, zeroZone, criteria);
    }

    // If enough, return kps
    // 如果数量足够了，直接返回 
    if( nbdetected >= 0.66 * nb2detect || nb2detect < 20 ) {
        return vnewpts;
    }

    // Else, detect more
    // 否则，降低阈值，继续检测
    nb2detect -= nbdetected;// 目标数量降低
    std::vector<cv::Point2f> vmorepts;
    vmorepts.reserve(nb2detect);

    // Update mask to force detection around 
    // not observed areas
    mask.release();
    if( !roi.empty() ) {
        mask = roi.clone();
    }
    setMask(im, vcurkps, nmindist_, mask);
    setMask(im, vnewpts, nmindist_, mask);

    // Detect additional kps
    // std::cout << "\n \t>>>  Searching more : " << nb2detect;
    // 降低阈值
    pgftt_->setQualityLevel(dmaxquality_);
    pgftt_->setMinDistance(nmindist_);
    pgftt_->setMaxFeatures(nb2detect);
    pgftt_->detect(im, vnewkps, mask);

    cv::KeyPoint::convert(vnewkps, vmorepts);
    vnewkps.clear();

    nbdetected = vmorepts.size();
    // std::cout << "\n \t>>>  Additionally found : " << nbdetected;

    // Compute Corners with Sub-Pixel Accuracy
    if( nbdetected > 0 )
    {
        // 亚像素精确化 
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vmorepts, winSize, zeroZone, criteria);
    }
    // 插入新的特征点 
    // Insert new detections
    vnewpts.insert(vnewpts.end(), 
                    std::make_move_iterator(vmorepts.begin()),
                    std::make_move_iterator(vmorepts.end())
                );

    // std::cout << "\n \t>>>  Total found : " << vnewpts.size();

    // Return vector of detected kps px pos.
    // 返回新的特征点
    return vnewpts;
}

/**
 * @brief 提取特征点的描述子 
 * @param im 输入图像
 * @param vpts 输入特征点坐标
 * @return 特征点描述子
*/
std::vector<cv::Mat> FeatureExtractor::describeBRIEF(const cv::Mat &im, const std::vector<cv::Point2f> &vpts) const
{
    // 特征点为空，返回空 
    if( vpts.empty() ) {
        // std::cout << "\nNo kps provided to function describeBRIEF() \n";
        return std::vector<cv::Mat>();
    }

    std::vector<cv::KeyPoint> vkps;
    size_t nbkps = vpts.size();
    vkps.reserve(nbkps);
    std::vector<cv::Mat> vdescs;
    vdescs.reserve(nbkps);
    // 转换为 cv::KeyPoint 格式 
    cv::KeyPoint::convert(vpts, vkps);
    // 计算描述子 
    cv::Mat descs;

    if( pbrief_ == nullptr ) {
        // 使用 ORB 描述子 or BRIEF 描述子, 应该都是一样的
        #ifdef OPENCV_CONTRIB
        pbrief_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
        #else
        pbrief_  = cv::ORB::create(500, 1., 0);
        std::cout << "\n\n=======================================================================\n";
        std::cout << " BRIEF CANNOT BE USED ACCORDING TO CMAKELISTS (Opencv Contrib not enabled) \n";
        std::cout << " ORB WILL BE USED INSTEAD!  (BUT NO ROTATION  OR SCALE INVARIANCE ENABLED) \n";
        std::cout << "\n\n=======================================================================\n\n";
        #endif
    }

    // std::cout << "\nCOmputing desc for #" << vkps.size() << " kps\n";
    // 计算描述子 
    pbrief_->compute(im, vkps, descs);

    // std::cout << "\nDesc computed for #" << vkps.size() << " kps\n";
    
    if( vkps.empty() ) {
        return std::vector<cv::Mat>(nbkps, cv::Mat());
    }

    size_t k = 0;
    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {// 遍历所有特征点 
        if( k < vkps.size() ) {
            if( vkps[k].pt == vpts[i] ) {
                // vdescs.push_back(descs.row(k).clone());
                // 保存描述子 
                vdescs.push_back(descs.row(k));
                k++;
            }
            else {
                vdescs.push_back(cv::Mat());
            }
        } else {
            vdescs.push_back(cv::Mat());
        }
    }

    assert( vdescs.size() == vpts.size() );

    // std::cout << "\n \t >>> describeBRIEF : " << vkps.size() << " kps described!\n";
    // 返回描述子 
    return vdescs;
}

/**
 * @brief 提取harris角点 
 * @param im 输入图像
 * @param ncellsize 网格大小
 * @param vcurkps 现有的特征点
 * @param roi 感兴趣区域
*/
std::vector<cv::Point2f> FeatureExtractor::detectSingleScale(const cv::Mat &im, const int ncellsize, 
        const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi) 
{    
    if( im.empty() ) {// 图像为空，返回空 
        // std::cerr << "\n No image provided to detectSingleScale() !\n";
        return std::vector<cv::Point2f>();
    }
    // 图像大小 
    size_t ncols = im.cols;
    size_t nrows = im.rows;
    // 半个网格大小 
    size_t nhalfcell = ncellsize / 4;
    // 网格数量 
    size_t nhcells = nrows / ncellsize;
    size_t nwcells = ncols / ncellsize;
    // 网格总数 
    size_t nbcells = nhcells * nwcells;
    // 提取的特征点 
    std::vector<cv::Point2f> vdetectedpx;
    vdetectedpx.reserve(nbcells);
    // 网格内是否有特征点标志 
    std::vector<std::vector<bool>> voccupcells(
            nhcells+1, 
            std::vector<bool>(nwcells+1, false)
            );
    // 掩膜 
    cv::Mat mask = cv::Mat::ones(im.rows, im.cols, CV_32F);

    // 对于现有的特征点 
    for( const auto &px : vcurkps ) {
        // 设置该网格内有特征点
        voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
        // 绘制掩膜，这部分就不提取了
        cv::circle(mask, px, nhalfcell, cv::Scalar(0.), -1);
    }

    // std::cout << "\n Single Scale detection \n";
    // std::cout << "\n nhcells : " << nhcells << " / nwcells : " << nwcells;
    // std::cout << " / nbcells : " << nhcells * nwcells;
    // std::cout << "\n cellsize : " << ncellsize;

    size_t nboccup = 0;

    std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells);
    std::vector<std::vector<cv::Point2f>> vvsecdetectionspx(nbcells);
    // 每一个网格
    auto cvrange = cv::Range(0, nbcells);

    // 并行处理，每一个网格单独处理 
    parallel_for_(cvrange, [&](const cv::Range& range) {
        for( int i = range.start ; i < range.end ; i++ ) {
        
        size_t r = floor(i / nwcells);
        size_t c = i % nwcells;

        if( voccupcells[r][c] ) {
                nboccup++;
                continue;
        }
        // 计算栅格的起点坐标
        size_t x = c*ncellsize;
        size_t y = r*ncellsize;
        // 取一小块，提取特征点 
        cv::Rect hroi(x,y,ncellsize,ncellsize);

        if( x+ncellsize < ncols-1 && y+ncellsize < nrows-1 ) {
            cv::Mat hmap;
            cv::Mat filtered_im;
            // 高斯滤波 
            cv::GaussianBlur(im(hroi), filtered_im, cv::Size(3,3), 0.);、
            // 计算特征值响应 
            cv::cornerMinEigenVal(filtered_im, hmap, 3, 3);

            double dminval, dmaxval;
            cv::Point minpx, maxpx;
            // 最小最大值 
            cv::minMaxLoc(hmap.mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
            maxpx.x += x;
            maxpx.y += y;

            if( maxpx.x < roi.x || maxpx.y < roi.y 
                || maxpx.x >= roi.x+roi.width 
                || maxpx.y >= roi.y+roi.height )
            {
                continue;
            }
            // 最大值大于阈值，保存特征点
            if( dmaxval >= dmaxquality_ ) {
                vvdetectedpx.at(i).push_back(maxpx);
                cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
            }
            // 添加mask之后看看有没有特征点
            cv::minMaxLoc(hmap.mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
            maxpx.x += x;
            maxpx.y += y;

            if( maxpx.x < roi.x || maxpx.y < roi.y 
                || maxpx.x >= roi.x+roi.width 
                || maxpx.y >= roi.y+roi.height )
            {
                continue;
            }
            // 最大值大于阈值，保存特征点 
            if( dmaxval >= dmaxquality_ ) {
                // 预备特征点 
                vvsecdetectionspx.at(i).push_back(maxpx);
                cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
            }
        }
    }
    });
    // 新提取的特征点，添加到vdetectedpx中 
    for( const auto &vpx : vvdetectedpx ) {
        if( !vpx.empty() ) {
            vdetectedpx.insert(vdetectedpx.end(), vpx.begin(), vpx.end());
        }
    }

    size_t nbkps = vdetectedpx.size();
    // 如果提取的特征点数量没有把网格填满
    if( nbkps+nboccup < nbcells ) {
        size_t nbsec = nbcells - (nbkps+nboccup);
        size_t k = 0;
        for( const auto &vseckp : vvsecdetectionspx ) {
            // 如果有特征点，添加到vdetectedpx中
            if( !vseckp.empty() ) {
                vdetectedpx.push_back(vseckp.back());
                k++;
                if( k == nbsec ) {
                    break;
                }
            }
        }
    }

    nbkps = vdetectedpx.size();
    // 如果提取的特征点数量小于网格数量的1/3，降低阈值
    if( nbkps < 0.33 * (nbcells - nboccup) ) {
        dmaxquality_ /= 2.;
    } 
    else if( nbkps > 0.9 * (nbcells - nboccup) ) {
        // 如果提取的特征数量很多，增加阈值 
        dmaxquality_ *= 1.5;
    }

    // Compute Corners with Sub-Pixel Accuracy
    if( !vdetectedpx.empty() )
    {
        // 亚像素精度细化 
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vdetectedpx, winSize, zeroZone, criteria);
    }

    // std::cout << "\n \t>>> Found : " << nbkps;

    return vdetectedpx;
}


/**
 * @brief 从图像中提取 FAST 特征点
 * @param im 输入图像
*/
std::vector<cv::Point2f> FeatureExtractor::detectGridFAST(const cv::Mat &im, const int ncellsize, 
        const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi)
{    
    if( im.empty() ) {
        // std::cerr << "\n No image provided to detectGridFAST() !\n";
        return std::vector<cv::Point2f>();
    }
    // 图像的大小 
    size_t ncols = im.cols;
    size_t nrows = im.rows;
    // 栅格的大小 
    size_t nhalfcell = ncellsize / 4;

    size_t nhcells = nrows / ncellsize;
    size_t nwcells = ncols / ncellsize;

    size_t nbcells = nhcells * nwcells;
    // 提取的特征点 
    std::vector<cv::Point2f> vdetectedpx;
    vdetectedpx.reserve(nbcells);
    // 栅格是否被占用标志 
    std::vector<std::vector<bool>> voccupcells(
            nhcells+1, 
            std::vector<bool>(nwcells+1, false)
            );
    // mask 
    cv::Mat mask = cv::Mat::ones(im.rows, im.cols, CV_32F);

    // 现有的特征点，设置mask，不再提取
    for( const auto &px : vcurkps ) {
        voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
        cv::circle(mask, px, nhalfcell, cv::Scalar(0), -1);
    }

    size_t nboccup = 0;
    size_t nbempty = 0;

    // Create the FAST detector if not set yet
    if( pfast_ == nullptr ) {
        pfast_ = cv::FastFeatureDetector::create(nfast_th_);
    }

    // std::cout << "\ndetectGridFAST (cellsize: " << ncellsize << ") : \n";
    // std::cout << "\n FAST grid search over #" << nbcells;
    // std::cout << " cells (" << nwcells << ", " << nhcells << ")\n";
    // 备选特征 
    std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells);

    auto cvrange = cv::Range(0, nbcells);
    // 并行提取特征点 
    parallel_for_(cvrange, [&](const cv::Range& range) {
        for( int i = range.start ; i < range.end ; i++ ) {

        size_t r = floor(i / nwcells);
        size_t c = i % nwcells;
        // 如果栅格被占用，跳过
        if( voccupcells[r][c] ) {
                nboccup++;
                continue;
        }

        nbempty++;
        // 栅格的左上角坐标
        size_t x = c*ncellsize;
        size_t y = r*ncellsize;
        // 栅格的区域
        cv::Rect hroi(x,y,ncellsize,ncellsize);

        if( x+ncellsize < ncols-1 && y+ncellsize < nrows-1 ) {
            // 提取FAST特征点 
            std::vector<cv::KeyPoint> vkps;
            pfast_->detect(im(hroi), vkps, mask(hroi));

            if( vkps.empty() ) {
                continue;
            } else {
                // 降序排列
                std::sort(vkps.begin(), vkps.end(), compare_response);
            }
            // 如果响应大于20，认为是有效的特征点 
            if( vkps.at(0).response >= 20 ) {
                cv::Point2f pxpt = vkps.at(0).pt;
                
                pxpt.x += x;
                pxpt.y += y;

                cv::circle(mask, pxpt, nhalfcell, cv::Scalar(0), -1);

                vvdetectedpx.at(i).push_back(pxpt);
            }

        }
    }
    });
    // 合并特征点 
    for( const auto &vpx : vvdetectedpx ) {
        if( !vpx.empty() ) {
            vdetectedpx.insert(vdetectedpx.end(), vpx.begin(), vpx.end());
        }
    }

    size_t nbkps = vdetectedpx.size();

    // Update FAST th.
    // int nfast_th = pfast_->getThreshold();
    // 如果提取的特征点太少，降低阈值 
    if( nbkps < 0.5 * nbempty && nbempty > 10 ) {
        nfast_th_ *= 0.66;
        pfast_->setThreshold(nfast_th_);
    } else if ( nbkps == nbempty ) {
        // 如果提取的特征点太多，增加阈值 
        nfast_th_ *= 1.5;
        pfast_->setThreshold(nfast_th_);
    }


    // Compute Corners with Sub-Pixel Accuracy
    if( !vdetectedpx.empty() )
    {
        // 亚像素细化 
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vdetectedpx, winSize, zeroZone, criteria);
    }

    // std::cout << "\n \t>>> Found : " << vdetectedpx.size();

    return vdetectedpx;
}


/**
 * @brief 设置掩码，将已经检测到的特征点的周围区域设置为0 
 * @param im 输入图像
 * @param vpts 已经检测到的特征点
 * @param dist 特征点周围区域的半径
 * @param mask 输出掩码
*/
void FeatureExtractor::setMask(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const int dist, cv::Mat &mask) const
{
    if( mask.empty() ) {
        mask = cv::Mat(im.rows, im.cols, CV_8UC1, cv::Scalar(255));
    }

    for (auto &pt : vpts) {
        // 半径dist内的区域设置为0, 即不再检测
        cv::circle(mask, pt, dist, 0, -1);
    }
}