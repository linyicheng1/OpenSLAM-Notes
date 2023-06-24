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


#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <sophus/se3.hpp>

#include "camera_calibration.hpp"
// 结构体，关键点 
struct Keypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int lmid_;// 关键点的id  从0开始

    cv::Point2f px_;// 关键点的像素坐标
    cv::Point2f unpx_;// 关键点的归一化坐标
    Eigen::Vector3d bv_;// 关键点的相机坐标

    int scale_;// 关键点的尺度
    float angle_;// 关键点的方向
    cv::Mat desc_;// 关键点的描述子
    
    bool is3d_;// 关键点是否有3d点

    bool is_stereo_;// 关键点是否有立体点
    cv::Point2f rpx_;// 关键点的立体点的像素坐标
    cv::Point2f runpx_;// 关键点的立体点的归一化坐标
    Eigen::Vector3d rbv_;// 关键点的立体点的相机坐标

    bool is_retracked_;// 关键点是否被重投影 

    Keypoint() : lmid_(-1), scale_(0), angle_(-1.), is3d_(false), is_stereo_(false), is_retracked_(false)
    {}

    // For using kps in ordered containers
    bool operator< (const Keypoint &kp) const
    {
        // 安装id排序 
        return lmid_ < kp.lmid_;
    }
};

// 帧 类
class Frame {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数
    Frame();
    // 单目相机构造函数
    Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize);
    // 双目相机构造函数
    Frame(std::shared_ptr<CameraCalibration> pcalib_left, std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize);
    // 复制构造函数 
    Frame(const Frame &F);
    // 跟新帧的id和时间戳 
    void updateFrame(const int id, const double img_time);
    // 获取所有特征点 
    std::vector<Keypoint> getKeypoints() const;
    // 获取所有的2d特征点
    std::vector<Keypoint> getKeypoints2d() const;
    // 获取所有的3d特征点
    std::vector<Keypoint> getKeypoints3d() const;
    // 获取所有的立体特征点
    std::vector<Keypoint> getKeypointsStereo() const;
    // 获取所有的特征点的像素坐标
    std::vector<cv::Point2f> getKeypointsPx() const;
    // 获取所有的特征点的归一化坐标
    std::vector<cv::Point2f> getKeypointsUnPx() const;
    // 获取所有的特征点的相机坐标
    std::vector<Eigen::Vector3d> getKeypointsBv() const;
    // 获取所有的特征点的id
    std::vector<int> getKeypointsId() const;
    // 获取所有的特征点的描述子
    std::vector<cv::Mat> getKeypointsDesc() const;
    // 获取指定id的特征点
    Keypoint getKeypointById(const int lmid) const;
    // 获取指定id向量的特征点
    std::vector<Keypoint> getKeypointsByIds(const std::vector<int> &vlmids) const;
    // 计算特征点 
    void computeKeypoint(const cv::Point2f &pt, Keypoint &kp);
    Keypoint computeKeypoint(const cv::Point2f &pt, const int lmid);

    // 添加特征点 
    void addKeypoint(const Keypoint &kp);
    void addKeypoint(const cv::Point2f &pt, const int lmid);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const int scale);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale);
    void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale, const float angle);
    
    // 更新特征点
    void updateKeypoint(const cv::Point2f &pt, Keypoint &kp);
    void updateKeypoint(const int lmid, const cv::Point2f &pt);
    void updateKeypointDesc(const int lmid, const cv::Mat &desc);
    void updateKeypointAngle(const int lmid, const float angle);
    bool updateKeypointId(const int prevlmid, const int newlmid, const bool is3d);

    // 计算双目特征点 
    void computeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp);
    void updateKeypointStereo(const int lmid, const cv::Point2f &pt);
    // 删除特征点 
    void removeKeypoint(const Keypoint &kp);
    // 删除指定id的特征点 
    void removeKeypointById(const int lmid);
    // 删除双目特征点 
    void removeStereoKeypoint(const Keypoint &kp);
    // 删除指定id的双目特征点
    void removeStereoKeypointById(const int lmid);

    // 将关键点添加到网格中 
    void addKeypointToGrid(const Keypoint &kp);
    // 将关键点从网格中删除 
    void removeKeypointFromGrid(const Keypoint &kp);
    // 更新网格中的关键点
    void updateKeypointInGrid(const Keypoint &prevkp, const Keypoint &newkp);
    // 获取网格中的关键点
    std::vector<Keypoint> getKeypointsFromGrid(const cv::Point2f &pt) const;
    // 获取关键的网格索引       
    int getKeypointCellIdx(const cv::Point2f &pt) const;
    // 获取周围的关键点 
    std::vector<Keypoint> getSurroundingKeypoints(const Keypoint &kp) const;
    std::vector<Keypoint> getSurroundingKeypoints(const cv::Point2f &pt) const;
    // 指定id的特征点转换为3d特征点
    void turnKeypoint3d(const int lmid);
    // 设置指定id特征点是否被观测到
    bool isObservingKp(const int lmid) const;
    // 获取当前帧的相机姿态 
    Sophus::SE3d getTcw() const;
    Sophus::SE3d getTwc() const;

    Eigen::Matrix3d getRcw() const;
    Eigen::Matrix3d getRwc() const;

    Eigen::Vector3d gettcw() const;
    Eigen::Vector3d gettwc() const;

    void setTwc(const Sophus::SE3d &Twc);
    void setTcw(const Sophus::SE3d &Tcw);

    void setTwc(const Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc);
    void setTcw(const Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);
    // 获取具有共视关系的关键帧 
    std::set<int> getCovisibleKfSet() const;
    // 获取共视图 
    std::map<int,int> getCovisibleKfMap() const;
    // 更新共视图 
    void updateCovisibleKfMap(const std::map<int,int> &cokfs);
    // 添加具有共视关系的关键帧 
    void addCovisibleKf(const int kfid);
    // 删除具有共视关系的关键帧
    void removeCovisibleKf(const int kfid);
    // 减少具有共视关系的关键帧
    void decreaseCovisibleKf(const int kfid);
    // 3d地图点投影到当前帧的图像平面上 
    cv::Point2f projCamToImageDist(const Eigen::Vector3d &pt) const;
    cv::Point2f projCamToImage(const Eigen::Vector3d &pt) const;
    // 3d地图点投影到右目图像平面上
    cv::Point2f projCamToRightImageDist(const Eigen::Vector3d &pt) const;
    cv::Point2f projCamToRightImage(const Eigen::Vector3d &pt) const;

    cv::Point2f projDistCamToImage(const Eigen::Vector3d &pt) const;
    cv::Point2f projDistCamToRightImage(const Eigen::Vector3d &pt) const;
    // 相机坐标系到世界坐标系的转换 
    Eigen::Vector3d projCamToWorld(const Eigen::Vector3d &pt) const;
    // 世界坐标系到相机坐标系的转换
    Eigen::Vector3d projWorldToCam(const Eigen::Vector3d &pt) const;
    // 世界坐标系到图像平面的转换
    cv::Point2f projWorldToImage(const Eigen::Vector3d &pt) const;
    cv::Point2f projWorldToImageDist(const Eigen::Vector3d &pt) const;
    // 世界坐标系到右目图像平面的转换
    cv::Point2f projWorldToRightImage(const Eigen::Vector3d &pt) const;
    cv::Point2f projWorldToRightImageDist(const Eigen::Vector3d &pt) const;
    // 判断是否在图像平面内 
    bool isInImage(const cv::Point2f &pt) const;
    bool isInRightImage(const cv::Point2f &pt) const;

    void displayFrameInfo();

    // For using frame in ordered containers
    // 根据id排序 
    bool operator< (const Frame &f) const {
        return id_ < f.id_;
    }

    void reset();

    // Frame info
    int id_, kfid_;// 帧id，关键帧id 
    double img_time_;// 图像时间戳 

    // Hash Map of observed keypoints
    std::unordered_map<int, Keypoint> mapkps_;// 观测到的特征点 

    // Grid of kps sorted by cell numbers and scale
    // (We use const pointer to reference the keypoints in vkps_
    // HENCE we should only use the grid to read kps)
    std::vector<std::vector<int>> vgridkps_;// 网格中的特征点 
    size_t ngridcells_, noccupcells_, ncellsize_, nbwcells_, nbhcells_;
    // 特征点的数量统计 
    size_t nbkps_, nb2dkps_, nb3dkps_, nb_stereo_kps_;

    // Pose (T cam -> world), (T world -> cam)
    Sophus::SE3d Twc_, Tcw_;// 帧位姿

    /* TODO
    Set a vector of calib ptrs to handle any multicam system.
    Each calib ptr should contain an extrinsic parametrization with a common
    reference frame. If cam0 is the ref., its extrinsic would be the identity.
    Would mean an easy integration of IMU body frame as well.
    */
    // Calibration model
    std::shared_ptr<CameraCalibration> pcalib_leftcam_;// 相机模型
    std::shared_ptr<CameraCalibration> pcalib_rightcam_;// 相机模型

    Eigen::Matrix3d Frl_;// 右目相机到左目相机的基础矩阵 
    cv::Mat Fcv_;// 右目相机到左目相机的基础矩阵

    // Covisible kf ids
    std::map<int,int> map_covkfs_;// 共视关系的关键帧

    // Local MapPoint ids
    std::unordered_set<int> set_local_mapids_;// 局部地图点的id

    // Mutex
    mutable std::mutex kps_mutex_, pose_mutex_;
    mutable std::mutex grid_mutex_, cokfs_mutex_;
};
