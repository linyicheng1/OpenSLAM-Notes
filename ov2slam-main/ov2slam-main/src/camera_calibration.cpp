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

#include "camera_calibration.hpp"

/**
 * @brief 相机标定的构造函数
 * @param model 相机模型
 * @param fx 相机的焦距
 * @param fy 相机的焦距
 * @param cx 相机的光心
 * @param cy 相机的光心
 * @param k1 相机的畸变系数
 * @param k2 相机的畸变系数
 * @param p1 相机的畸变系数
 * @param p2 相机的畸变系数
 * @param img_w 图像的宽度
 * @param img_h 图像的高度
*/
CameraCalibration::CameraCalibration(const std::string &model, double fx, double fy, double cx, double cy,
                        double k1, double k2, double p1, double p2, double img_w, double img_h)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), k1_(k1), k2_(k2), p1_(p1), p2_(p2)
    , img_w_(img_w), img_h_(img_h), img_size_(img_w, img_h)
{
    std::cout << "\n Setting up camera, model selected : " << model << "\n";
    // 针孔相机模型
    if( model == "pinhole") {
        model_ = Pinhole;
        std::cout << "\nPinhole Camera Model created\n";
    }
    else if( model == "fisheye" ) {// 鱼眼相机模型 
        model_ = Fisheye;
        std::cout << "\nFisheye Camera Model selected";
    }
    else {
        std::cout << "\nNo supported camera model provided!"; 
        std::cout << "\nChoosee between: pinhole / fisheye";
        exit(-1);
    }
    // 相机内参矩阵 K
    K_ << fx_, 0., cx_, 0., fy_, cy_, 0., 0., 1.;
    // 相机畸变系数 D
    D_ << k1_, k2_, p1_, p2_;
    // cv 的矩阵
    cv::eigen2cv(K_, Kcv_);
    cv::eigen2cv(D_, Dcv_);
    // 相机内参矩阵的逆矩阵 K^-1
    iK_ = K_.inverse();
    ifx_ = iK_(0,0);
    ify_ = iK_(1,1);
    icx_ = iK_(0,2);
    icy_ = iK_(1,2);

    cv::eigen2cv(Tc0ci_.rotationMatrix(), Rcv_c0ci_);
    cv::eigen2cv(Tc0ci_.translation(), tcv_c0ci_);

    std::cout << "\n Camera Calibration set as : \n\n";
    std::cout << "\n K = \n" << K_;
    std::cout << "\n\n D = " << D_.transpose();

    std::cout << "\n opt K = \n" << Kcv_;
   
    const int nborder = 5;
    roi_rect_ = cv::Rect(cv::Point2i(nborder,nborder), cv::Point2i(img_w_-nborder,img_h_-nborder));
    roi_mask_ = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar(0));
    roi_mask_(roi_rect_) = 255;

    Rrectraw_.setIdentity();
}

/**
 * @brief 设置畸变矫正的映射 
*/
void CameraCalibration::setUndistMap(const double alpha) 
{
    std::cout << "\n\nComputing the undistortion mapping!\n";

    cv::Size img_size(img_w_, img_h_);

    cv::Mat newK;

    // CV_16SC2 = 11 / CV_32FC1 = 5
    // opencv 的畸变矫正映射 
    if( model_ == Pinhole )
    {
        newK = cv::getOptimalNewCameraMatrix(Kcv_, Dcv_, img_size, alpha, img_size, &roi_rect_);
        cv::initUndistortRectifyMap(Kcv_, Dcv_, cv::Mat(), newK, img_size, CV_32FC1, undist_map_x_, undist_map_y_);
    }
    else if ( model_ == Fisheye )
    {
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(Kcv_, Dcv_, img_size, cv::Mat(), newK, alpha);
        cv::fisheye::initUndistortRectifyMap(Kcv_, Dcv_, cv::Mat(), newK, img_size, CV_32FC1, undist_map_x_, undist_map_y_);        
    }
    else {
        std::cout << "\n WRONG CAMERA MODEL CHOSEN : Pinhole / Fisheye supported\n";
        exit(-1);
    }

    newK.copyTo(Kcv_);

    cv::cv2eigen(Kcv_, K_);
    Dcv_.release();
    D_.setZero();

    k1_ = 0.; k2_ = 0.;
    p1_ = 0.; p2_ = 0.;

    fx_ = K_(0,0);
    fy_ = K_(1,1);
    cx_ = K_(0,2);
    cy_ = K_(1,2);

    iK_ = K_.inverse();
    ifx_ = iK_(0,0);
    ify_ = iK_(1,1);
    icx_ = iK_(0,2);
    icy_ = iK_(1,2);

    // Setup roi mask / rect
    setROIMask(roi_rect_);

    std::cout << "\n Undist Camera Calibration set as : \n\n";
    std::cout << "\n K = \n" << K_;
    std::cout << "\n\n D = " << D_.transpose();
    std::cout << "\n\n ROI = " << roi_rect_;
}

/**
 * @brief 设置双目的畸变矫正的映射
*/
void CameraCalibration::setUndistStereoMap(const cv::Mat &R, const cv::Mat &P, const cv::Rect &roi) 
{
    std::cout << "\n\nComputing the stereo rectification mapping!\n";
    
    // CV_16SC2 = 11 / CV_32FC1 = 5
    if( model_ == Pinhole )
    {
        cv::initUndistortRectifyMap(Kcv_, Dcv_, R, P, img_size_, 11, undist_map_x_, undist_map_y_);
    }
    else if ( model_ == Fisheye )
    {
        cv::fisheye::initUndistortRectifyMap(Kcv_, Dcv_, R, P, img_size_, 11, undist_map_x_, undist_map_y_);        
    }

    cv::cv2eigen(R,Rrectraw_);
    
    Eigen::Matrix4d Pe;
    cv::cv2eigen(P,Pe);

    fx_ = Pe(0,0);
    fy_ = Pe(1,1);
    cx_ = Pe(0,2);
    cy_ = Pe(1,2);

    K_ = Pe.block<3,3>(0,0);

    cv::eigen2cv(K_, Kcv_);
    Dcv_.release();
    D_.setZero();

    k1_ = 0.; k2_ = 0.;
    p1_ = 0.; p2_ = 0.;
    
    iK_ = K_.inverse();
    ifx_ = iK_(0,0);
    ify_ = iK_(1,1);
    icx_ = iK_(0,2);
    icy_ = iK_(1,2);

    Tc0ci_ = Sophus::SE3d();
    Tc0ci_.translation() = Eigen::Vector3d(-1. * Pe(0,3) / fx_,
                                           -1. * Pe(1,3) / fx_,
                                           -1. * Pe(2,3) / fx_);

    Tcic0_ = Tc0ci_.inverse();

    cv::eigen2cv(Tc0ci_.rotationMatrix(), Rcv_c0ci_);
    cv::eigen2cv(Tc0ci_.translation(), tcv_c0ci_);

    Rcv_cic0_ = Rcv_c0ci_.clone();
    tcv_cic0_ = -1. * tcv_c0ci_.clone();

    // Setup roi mask / rect
    setROIMask(roi);

    std::cout << "\n Undist+Rect Camera Calibration set as : \n\n";
    std::cout << "\n K = \n" << K_;
    std::cout << "\n\n D = " << D_.transpose();
    std::cout << "\n\n ROI = " << roi_rect_;
}


/**
 * @brief 设置相机外参 
*/
void CameraCalibration::setupExtrinsic(const Sophus::SE3d &Tc0ci)
{
    // 保存外参 
    Tc0ci_ = Tc0ci;
    Tcic0_ = Tc0ci.inverse();

    cv::eigen2cv(Tc0ci_.rotationMatrix(), Rcv_c0ci_);
    cv::eigen2cv(Tc0ci_.translation(), tcv_c0ci_);

    cv::eigen2cv(Tcic0_.rotationMatrix(), Rcv_cic0_);
    cv::eigen2cv(Tcic0_.translation(), tcv_cic0_);

    std::cout << "\n Camera Extrinsic : \n\n";
    std::cout << "\n Rc0ci = \n" << Tc0ci_.rotationMatrix();
    std::cout << "\n\n tc0ci = " << Tc0ci_.translation().transpose();
    // 检查外参是否正确，确保左右相机的位置正确 
    Eigen::Vector3d tc0ci = Tc0ci_.translation();
    if( (fabs(tc0ci.x()) > fabs(tc0ci.y()) && tc0ci.x() < 0)
        || (fabs(tc0ci.x()) < fabs(tc0ci.y()) && tc0ci.y() < 0) )
    {
        std::cerr << "\n\n *************************************** \n";
        std::cerr << "\n \t WARNING! \n";
        std::cerr << "Left and right camera seem to be inverted! \n";
        std::cerr << "\n *************************************** \n\n";
    }
}

/**
 * @brief  设置ROI
 * @param  roi 输入ROI
*/
void CameraCalibration::setROIMask(const cv::Rect &roi)
{
    const int nborder = 5;// roi 边界 
    roi_rect_ = roi; // roi 矩形
    roi_rect_.width -= nborder;// 减去边界 
    roi_rect_.height -= nborder;
    roi_rect_.x += nborder;
    roi_rect_.y += nborder;
    // roi_mask_ 为roi矩形区域的掩码，即roi区域为255，其余为0 
    roi_mask_(cv::Range::all(),cv::Range::all()) = 0;
    roi_mask_(roi_rect_) = 255;
}

/**
 * @brief  矫正图像
 * @param  img 输入图像
 * @param  rect 输出图像
*/ 
void CameraCalibration::rectifyImage(const cv::Mat &img, cv::Mat &rect) const
{
    std::lock_guard<std::mutex> lock(intrinsic_mutex_);
    // 如果矫正图像的map不为空，则进行矫正 
    if( !undist_map_x_.empty() )
        // cv::remap()函数的作用是对图像进行重映射，即把图像中的像素点按照一定的规律重新排布，从而达到缩放、平移、旋转、畸变校正等目的。
        cv::remap(img, rect, undist_map_x_, undist_map_y_, cv::INTER_LINEAR);
    else
        rect = img;
}

/**
 * @brief  相机坐标系到图像坐标系的投影 
 * @param  pt 相机坐标系下的点
 * @return    图像坐标系下的点
*/
cv::Point2f CameraCalibration::projectCamToImage(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(intrinsic_mutex_);
    // 针孔相机模型 
    double invz = 1. / pt.z();
    double x =  pt.x() * invz;
    double y =  pt.y() * invz;

    return cv::Point2f(fx_ * x + cx_, fy_ * y + cy_);
}

/**
 * @brief  相机坐标系到图像坐标系的投影, 带畸变 
*/
cv::Point2f CameraCalibration::projectCamToImageDist(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(intrinsic_mutex_);
    // 针孔相机模型 
    double invz = 1. / pt.z();
    double x =  pt.x() * invz;
    double y =  pt.y() * invz;
    // 如果畸变参数为空，则直接返回 
    if( Dcv_.empty() ) {
        return cv::Point2f(fx_ * x + cx_, fy_ * y + cy_);
    }

    cv::Point3f cvpt(x, y, 1.);
    std::vector<cv::Point3f> vpt;
    vpt.push_back(cvpt);
    std::vector<cv::Point2f> vdistpx;

    cv::Mat R = cv::Mat::zeros(3,1,CV_32F);
    // 添加畸变
    if( model_ == Pinhole ) {
        cv::projectPoints(vpt, R, R, Kcv_, Dcv_, vdistpx);
    } else if( model_ == Fisheye ) {
        vdistpx.push_back(cv::Point2f(x,y));
        cv::fisheye::distortPoints(vdistpx, vdistpx, Kcv_, Dcv_);
    }
    return vdistpx[0];
}

/**
 * @brief  相机坐标系到图像坐标系的投影, 并且去畸变 
*/
cv::Point2f CameraCalibration::projectCamToImageUndist(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(intrinsic_mutex_);
    // 针孔相机模型 
    double invz = 1. / pt.z();
    double x =  pt.x() * invz;
    double y =  pt.y() * invz;
    
    cv::Point2f distpx(fx_ * x + cx_, fy_ * y + cy_);
    // 如果畸变参数为空，则直接返回 
    if( Dcv_.empty() ) {
        return distpx;
    }
    // 畸变去除 
    std::vector<cv::Point2f> vpx, vunpx;

    vpx.push_back(distpx);
    
    if( model_ == Pinhole ) {
        cv::undistortPoints(vpx, vunpx, Kcv_, Dcv_, Kcv_);
    }
    else if ( model_ == Fisheye ) {
        cv::fisheye::undistortPoints(vpx, vunpx, Kcv_, Dcv_, Kcv_);
    }
    
    return vunpx[0];
}

/**
 * @brief  图像点的畸变去除 
*/
cv::Point2f CameraCalibration::undistortImagePoint( const cv::Point2f &pt ) const
{
    std::lock_guard<std::mutex> lock(intrinsic_mutex_);
    // 如果畸变参数为空，则直接返回 
    if( Dcv_.empty() ) {
        return pt;
    }
    
    std::vector<cv::Point2f> vpx, vunpx;

    vpx.push_back(pt);
    
    if( model_ == Pinhole ) {
        // opencv针孔相机模型去畸变 
        cv::undistortPoints(vpx, vunpx, Kcv_, Dcv_, Kcv_);
    } else if ( model_ == Fisheye ) {
        // opencv鱼眼相机模型去畸变
        cv::fisheye::undistortPoints(vpx, vunpx, Kcv_, Dcv_, Kcv_);
    }

    return vunpx[0];
}

// Return Rc0ci
/**
 * @brief  获取相机外参, 旋转
*/
Eigen::Matrix3d CameraCalibration::getRotation() const
{
    std::lock_guard<std::mutex> lock(extrinsic_mutex_);
    return Tc0ci_.rotationMatrix();
}

// Return tc0ci
/**
 * @brief  获取相机外参, 平移 
*/
Eigen::Vector3d CameraCalibration::getTranslation() const
{
    std::lock_guard<std::mutex> lock(extrinsic_mutex_);
    return Tc0ci_.translation();
}

/**
 * @brief  获取相机外参
*/
Sophus::SE3d CameraCalibration::getExtrinsic() const
{
    std::lock_guard<std::mutex> lock(extrinsic_mutex_);
    return Tc0ci_;
}

/**
 * @brief  更新相机外参
 * @param  Tc0ci 相机外参
*/
void CameraCalibration::updateExtrinsic(const Sophus::SE3d &Tc0ci)
{
    std::lock_guard<std::mutex> lock(extrinsic_mutex_);
    Tc0ci_ = Tc0ci;
}

/**
 * @brief  更新相机内参
 * @param  fx 焦距
 * @param  fy 焦距
 * @param  cx 光心
 * @param  cy 光心
*/
void CameraCalibration::updateIntrinsic(const double fx, const double fy, const double cx, const double cy)
{
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    // 内参矩阵
    K_ << fx_, 0., cx_, 0., fy_, cy_, 0., 0., 1.;

    cv::eigen2cv(K_, Kcv_);
    // 逆矩阵 
    iK_ = K_.inverse();
    ifx_ = iK_(0,0);
    ify_ = iK_(1,1);
    icx_ = iK_(0,2);
    icy_ = iK_(1,2);
}

/**
 * @brief  更新畸变系数
 * @param  k1 畸变系数
 * @param  k2 畸变系数
 * @param  p1 畸变系数
 * @param  p2 畸变系数
*/
void CameraCalibration::updateDistCoefs(const double k1, const double k2, 
        const double p1, const double p2)
{
    k1_ = k1; k2_ = k2;
    p1_ = p1; p2_ = p2;
    D_ << k1_, k2_, p1_, p2_;
    cv::eigen2cv(D_, Dcv_);
}