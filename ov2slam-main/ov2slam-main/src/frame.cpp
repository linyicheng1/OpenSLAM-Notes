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

#include "frame.hpp"

/**
 * @brief 帧的构造函数,默认构造函数
*/
Frame::Frame()
    : id_(-1), kfid_(0), img_time_(0.), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
      Frl_(Eigen::Matrix3d::Zero()), Fcv_(cv::Mat::zeros(3,3,CV_64F))
{}

/**
 * @brief 单目帧的构造函数 
 * @param[in] pcalib_left 左相机的内参及图像 
 * @param[in] ncellsize 网格的大小
*/
Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize)
    : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0),
      nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
      pcalib_leftcam_(pcalib_left)
{
    // 初始化栅格 
    // Init grid from images size
    nbwcells_ = static_cast<size_t>(ceilf( static_cast<float>(pcalib_leftcam_->img_w_) / ncellsize_ ));
    nbhcells_ = static_cast<size_t>(ceilf( static_cast<float>(pcalib_leftcam_->img_h_) / ncellsize_ ));
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;
    
    vgridkps_.resize( ngridcells_ );
}

/**
 * @brief 双目帧的构造函数 
 * @param[in] pcalib_left 左相机的内参及图像
 * @param[in] pcalib_right 右相机的内参及图像
 * @param[in] ncellsize 网格的大小
*/
Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize)
    : id_(-1), kfid_(0), img_time_(0.), ncellsize_(ncellsize), nbkps_(0), nb2dkps_(0), nb3dkps_(0), nb_stereo_kps_(0),
    pcalib_leftcam_(pcalib_left), pcalib_rightcam_(pcalib_right)
{
    // 左相机到右相机的外参 
    Eigen::Vector3d t = pcalib_rightcam_->Tcic0_.translation();
    Eigen::Matrix3d tskew;
    tskew << 0., -t(2), t(1),
            t(2), 0., -t(0),
            -t(1), t(0), 0.;

    Eigen::Matrix3d R = pcalib_rightcam_->Tcic0_.rotationMatrix();
    // 构造基础矩阵
    Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R * pcalib_leftcam_->iK_;
    cv::eigen2cv(Frl_, Fcv_);

    // 初始化网格 
    // Init grid from images size
    nbwcells_ = ceil( (float)pcalib_leftcam_->img_w_ / ncellsize_ );
    nbhcells_ = ceil( (float)pcalib_leftcam_->img_h_ / ncellsize_ );
    ngridcells_ =  nbwcells_ * nbhcells_ ;
    noccupcells_ = 0;
    
    vgridkps_.resize( ngridcells_ );
}

/**
 * @brief 拷贝构造函数 
 * @param[in] F 帧
*/
Frame::Frame(const Frame &F)
    : id_(F.id_), kfid_(F.kfid_), img_time_(F.img_time_), mapkps_(F.mapkps_), vgridkps_(F.vgridkps_), ngridcells_(F.ngridcells_), noccupcells_(F.noccupcells_),
    ncellsize_(F.ncellsize_), nbwcells_(F.nbwcells_), nbhcells_(F.nbhcells_), nbkps_(F.nbkps_), nb2dkps_(F.nb2dkps_), nb3dkps_(F.nb3dkps_), 
    nb_stereo_kps_(F.nb_stereo_kps_), Twc_(F.Twc_), Tcw_(F.Tcw_), pcalib_leftcam_(F.pcalib_leftcam_),
    pcalib_rightcam_(F.pcalib_rightcam_), Frl_(F.Frl_), Fcv_(F.Fcv_), map_covkfs_(F.map_covkfs_), set_local_mapids_(F.set_local_mapids_)
{}

// Set the image time and id
/**
 * @brief 更新帧的id和时间
 * @param[in] id 帧的id
 * @param[in] time 帧的时间
*/
void Frame::updateFrame(const int id, const double time) 
{
    id_= id;
    img_time_ = time;
}

// Return vector of keypoint objects
/**
 * @brief 返回帧的所有关键点
 * @return 帧的所有关键点
*/
std::vector<Keypoint> Frame::getKeypoints() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second);
    }
    return v;
}


// Return vector of 2D keypoint objects
/**
 * @brief 返回帧的所有2D关键点
 * @return 帧的所有2D关键点
*/
std::vector<Keypoint> Frame::getKeypoints2d() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb2dkps_);
    // 遍历所有关键点 
    for( const auto & kp : mapkps_ ) {
        if( !kp.second.is3d_ ) {// 2D关键点 
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of 3D keypoint objects
/**
 * @brief 返回帧的所有3D关键点
 * @return 帧的所有3D关键点
*/
std::vector<Keypoint> Frame::getKeypoints3d() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb3dkps_);
    // 遍历所有关键点 
    for( const auto &kp : mapkps_ ) {
        if( kp.second.is3d_ ) {// 3D关键点 
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of stereo keypoint objects
/**
 * @brief 返回帧的所有立体关键点
 * @return 帧的所有立体关键点
*/
std::vector<Keypoint> Frame::getKeypointsStereo() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> v;
    v.reserve(nb_stereo_kps_);
    // 遍历所有关键点
    for( const auto &kp : mapkps_ ) {
        if( kp.second.is_stereo_ ) {// 立体关键点
            v.push_back(kp.second);
        }
    }
    return v;
}

// Return vector of keypoints' raw pixel positions
/**
 * @brief 返回帧的所有关键点的像素坐标
 * @return 帧的所有关键点的像素坐标
*/
std::vector<cv::Point2f> Frame::getKeypointsPx() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Point2f> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {// 遍历所有关键点
        // 保存像素坐标 
        v.push_back(kp.second.px_);
    }
    return v;
}

// Return vector of keypoints' undistorted pixel positions
/**
 * @brief 返回帧的所有关键点的去畸变像素坐标
 * @return 帧的所有关键点的去畸变像素坐标
*/
std::vector<cv::Point2f> Frame::getKeypointsUnPx() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Point2f> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {// 遍历所有关键点
        // 保存去畸变像素坐标
        v.push_back(kp.second.unpx_);
    }
    return v;
}

// Return vector of keypoints' bearing vectors、
/**
 * @brief 返回帧的所有关键点的方向向量 
 * @return 帧的所有关键点的方向向量
*/
std::vector<Eigen::Vector3d> Frame::getKeypointsBv() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Eigen::Vector3d> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {// 遍历所有关键点
        v.push_back(kp.second.bv_);// 保存方向向量
    }
    return v;
}

// Return vector of keypoints' related landmarks' id
/**
 * @brief 返回帧的所有关键点对应的地图点的id
 * @return 帧的所有关键点对应的地图点的id
*/
std::vector<int> Frame::getKeypointsId() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<int> v;
    v.reserve(nbkps_);
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.first);
    }
    return v;
}

/**
 * @brief 返回指定地图点id的关键点 
*/
Keypoint Frame::getKeypointById(const int lmid) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return Keypoint();
    }

    return it->second;
}


/**
 * @brief 返回指定地图点id的关键点 
 * @param lmid 地图点id
 * @return 地图点id对应的关键点
*/
std::vector<Keypoint> Frame::getKeypointsByIds(const std::vector<int> &vlmids) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<Keypoint> vkp;
    vkp.reserve(vlmids.size());
    for( const auto &lmid : vlmids ) {
        auto it = mapkps_.find(lmid);
        if( it != mapkps_.end() ) {
            vkp.push_back(it->second);
        }
    }

    return vkp;
}


// Return vector of keypoints' descriptor
/**
 * @brief 返回帧的所有关键点的描述子
 * @return 帧的所有关键点的描述子
*/
std::vector<cv::Mat> Frame::getKeypointsDesc() const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    std::vector<cv::Mat> v;
    v.reserve(nbkps_);
    // 遍历所有关键点
    for( const auto &kp : mapkps_ ) {
        v.push_back(kp.second.desc_);
    }

    return v;
}


// Compute keypoint from raw pixel position
/**
 * @brief 从像素坐标创建关键点
 * @param pt 像素坐标
 * @param kp 关键点
*/
inline void Frame::computeKeypoint(const cv::Point2f &pt, Keypoint &kp)
{
    // 设置像素坐标 
    kp.px_ = pt;
    // 设置去畸变像素坐标 
    kp.unpx_ = pcalib_leftcam_->undistortImagePoint(pt);
    // 设置方向向量
    Eigen::Vector3d hunpx(kp.unpx_.x, kp.unpx_.y, 1.);
    kp.bv_ = pcalib_leftcam_->iK_ * hunpx;
    // 归一化
    kp.bv_.normalize();
}

// Create keypoint from raw pixel position
/**
 * @brief 从像素坐标创建关键点
 * @param pt 像素坐标
 * @param lmid 地图点id
*/
inline Keypoint Frame::computeKeypoint(const cv::Point2f &pt, const int lmid)
{
    Keypoint kp;// 默认构造kp
    kp.lmid_ = lmid;// 设置地图点id 
    computeKeypoint(pt,kp);// 计算关键点
    return kp;
}


// Add keypoint object to vector of kps
/**
 * @brief 添加新的关键点
 * @param kp 关键点
*/
void Frame::addKeypoint(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 判断是否已经存在该关键点 
    if( mapkps_.count(kp.lmid_) ) {
        std::cout << "\nWEIRD!  Trying to add a KP with an already existing lmid... Not gonna do it!\n";
        return;
    }
    // 添加关键点 
    mapkps_.emplace(kp.lmid_, kp);
    // 添加到网格中 
    addKeypointToGrid(kp);
    // 更新关键点数量 
    nbkps_++;
    if( kp.is3d_ ) {
        nb3dkps_++;
    } else {
        nb2dkps_++;
    }
}

// Add new keypoint from raw pixel position
/**
 * @brief 添加新的关键点
 * @param pt 关键点的像素坐标
 * @param lmid 关键点对应的地图点id
*/
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid)
{
    // 计算关键点
    Keypoint kp = computeKeypoint(pt, lmid);
    // 添加关键点
    addKeypoint(kp);
}

// Add new keypoint w. desc
/**
 * @brief 添加新的关键点
 * @param pt 关键点的像素坐标
 * @param lmid 关键点对应的地图点id
 * @param desc 关键点的描述子
*/
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc) 
{
    // 计算关键点
    Keypoint kp = computeKeypoint(pt, lmid);
    // 更新关键点的描述子
    kp.desc_ = desc;
    // 添加关键点
    addKeypoint(kp);
}

// Add new keypoint w. desc & scale
/**
 * @brief 添加新的关键点
 * @param pt 关键点的像素坐标
 * @param lmid 关键点对应的地图点id
 * @param scale 关键点的尺度
*/
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const int scale)
{
    // 计算关键点
    Keypoint kp = computeKeypoint(pt, lmid);
    // 更新关键点的尺度
    kp.scale_ = scale;
    // 添加关键点
    addKeypoint(kp);
}

// Add new keypoint w. desc & scale
/**
 * @brief 添加新的关键点
 * @param pt 关键点的像素坐标
 * @param lmid 关键点对应的地图点id
 * @param desc 关键点的描述子
 * @param scale 关键点的尺度
*/
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale)
{
    // 计算关键点
    Keypoint kp = computeKeypoint(pt, lmid);
    // 更新关键点的描述子、尺度
    kp.desc_ = desc;
    kp.scale_ = scale;
    // 添加关键点
    addKeypoint(kp);
}

// Add new keypoint w. desc & scale & angle
/**
 * @brief 添加新的关键点
 * @param pt 关键点的像素坐标
 * @param lmid 关键点对应的地图点id
 * @param desc 关键点的描述子
 * @param scale 关键点的尺度
 * @param angle 关键点的方向
*/
void Frame::addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc, const int scale, const float angle)
{
    // 计算关键点 
    Keypoint kp = computeKeypoint(pt, lmid);
    // 更新关键点的描述子、尺度、方向
    kp.desc_ = desc;
    kp.scale_ = scale;
    kp.angle_ = angle;
    // 添加关键点
    addKeypoint(kp);
}

/**
 * @brief 更新关键点的位置
*/
void Frame::updateKeypoint(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 找到关键点
    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    } 
    // 获取关键点
    Keypoint upkp = it->second;

    if( upkp.is_stereo_ ) {
        nb_stereo_kps_--;
        upkp.is_stereo_ = false;
    }
    // 计算关键点，去除畸变
    computeKeypoint(pt, upkp);
    // 更新关键点在网格中的位置
    updateKeypointInGrid(it->second, upkp);
    // 更新关键点
    it->second = upkp;
}

/**
 * @brief 更新关键点的描述子
 * @param lmid 关键点id
 * @param desc 描述子
*/
void Frame::updateKeypointDesc(const int lmid, const cv::Mat &desc)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 找到关键点
    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    // 更新描述子
    it->second.desc_ = desc;
}

/**
 * @brief 更新关键点的角度
 * @param lmid 关键点id
 * @param angle 角度
*/
void Frame::updateKeypointAngle(const int lmid, const float angle)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 找到关键点 
    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    // 更新角度
    it->second.angle_ = angle;
}

/**
 * @brief 更新关键点状态
 * @param lmid 关键点id
 * @param newlmid 新的关键点id
 * @param is3d 是否是3d点
*/
bool Frame::updateKeypointId(const int prevlmid, const int newlmid, const bool is3d)
{
    std::unique_lock<std::mutex> lock(kps_mutex_);
    // 如果新的关键帧id已经存在，返回false 
    if( mapkps_.count(newlmid) ) {
        return false;
    }
    // 找到旧的关键点 
    auto it = mapkps_.find(prevlmid);
    if( it == mapkps_.end() ) {// 如果旧的关键点不存在，返回false 
        return false;
    }
    // 更新关键点id 
    Keypoint upkp = it->second;
    lock.unlock();
    upkp.lmid_ = newlmid;
    upkp.is_retracked_ = true;
    upkp.is3d_ = is3d;
    // 先删除旧的关键点，再添加新的关键点
    removeKeypointById(prevlmid);
    addKeypoint(upkp);
    return true;
}

// Compute stereo keypoint from raw pixel position
/**
 * @brief 计算立体匹配的关键点
 * @param pt 关键点右相机像素坐标
 * @param kp 关键点
*/
void Frame::computeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp)
{
    // 保存右相机像素坐标
    kp.rpx_ = pt;
    // 保存右相机归一化坐标
    kp.runpx_ = pcalib_rightcam_->undistortImagePoint(pt);
    // 保存右相机归一化坐标 
    Eigen::Vector3d bv(kp.runpx_.x, kp.runpx_.y, 1.);
    // 计算右相机下的向量 
    bv = pcalib_rightcam_->iK_ * bv.eval();
    bv.normalize();

    kp.rbv_ = bv;

    if( !kp.is_stereo_ ) {// 设置指定关键点为立体匹配点 
        kp.is_stereo_ = true;
        nb_stereo_kps_++;
    }
}

/**
 * @brief 更新关键点的立体匹配
 * @param lmid 关键点id
 * @param pt 关键点像素坐标
*/
void Frame::updateKeypointStereo(const int lmid, const cv::Point2f &pt)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 通过id查找关键点 
    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    // 计算立体匹配 
    computeStereoKeypoint(pt, it->second);
}

/**
 * @brief 移除关键点
 * @param kp 关键点
*/
inline void Frame::removeKeypoint(const Keypoint &kp)
{
    removeKeypointById(kp.lmid_);
}

/**
 * @brief 移除关键点
 * @param lmid 关键点id
*/
void Frame::removeKeypointById(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 通过id查找关键点 
    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    // 从网格中移除关键点
    removeKeypointFromGrid(it->second);
    // 如果是3d点，3d点数减1，否则2d点数减1
    if( it->second.is3d_ ) {
        nb3dkps_--;
    } else {
        nb2dkps_--;
    }
    nbkps_--;
    // 如果是双目点，双目点数减1 
    if( it->second.is_stereo_ ) {
        nb_stereo_kps_--;
    }
    // 所有关键点中移除该关键点 
    mapkps_.erase(lmid);
}

/**
 * @brief 移除双目关键点
 * @param lmid 双目关键点
*/
inline void Frame::removeStereoKeypoint(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    //通过id移除双目关键点 
    removeStereoKeypointById(kp.lmid_);
}

/**
 * @brief 移除指定id的双目关键点 
*/
void Frame::removeStereoKeypointById(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);

    auto it = mapkps_.find(lmid);// 所有关键点中找到指定id的关键点 
    if( it == mapkps_.end() ) {
        return;
    }
    
    if( it->second.is_stereo_ ) {// 确保是双目关键点 
        // 从网格中移除关键点
        it->second.is_stereo_ = false;
        nb_stereo_kps_--;
    }
}

/**
 * @brief 将指定id的关键点设置为3d点 
 * @param lmid 关键点id
*/
void Frame::turnKeypoint3d(const int lmid)
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    // 所有关键点中找到指定id的关键点 
    auto it = mapkps_.find(lmid);
    if( it == mapkps_.end() ) {
        return;
    }
    // 如果不是3d点，将其设置为3d点 
    if( !it->second.is3d_ ) {
        it->second.is3d_ = true;
        nb3dkps_++;// 3d点数量加一
        nb2dkps_--;// 2d点数量减一
    }
}

/**
 * @brief 获取指定id的关键点被观测的次数 
*/
bool Frame::isObservingKp(const int lmid) const
{
    std::lock_guard<std::mutex> lock(kps_mutex_);
    return mapkps_.count(lmid);
}

/**
 * @brief 添加关键点到网格中 
 * @param kp 关键点 
*/
void Frame::addKeypointToGrid(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);
    // 获取关键点所在的网格索引 
    int idx = getKeypointCellIdx(kp.px_);
    // 如果是空的，noccupcells_ 被占用的网格数量加一
    if( vgridkps_.at(idx).empty() ) {
        noccupcells_++;
    }
    // 将关键点的 id 添加到网格中 
    vgridkps_.at(idx).push_back(kp.lmid_);
}

/**
 * @brief 从网格中移除关键点
 * @param kp 关键点
*/
void Frame::removeKeypointFromGrid(const Keypoint &kp)
{
    std::lock_guard<std::mutex> lock(grid_mutex_);
    // 获取关键点所在的网格索引 
    int idx = getKeypointCellIdx(kp.px_);
    // 如果索引越界，直接返回 
    if( idx < 0 || idx >= (int)vgridkps_.size() ) {
        return;
    }

    for( size_t i = 0, iend = vgridkps_.at(idx).size() ; i < iend ; i++ )
    {// 遍历网格中的关键点，如果找到了，就从网格中移除 
        if( vgridkps_.at(idx).at(i) == kp.lmid_ ) {
            // 删除关键点 
            vgridkps_.at(idx).erase(vgridkps_.at(idx).begin() + i);
            // 如果网格中没有关键点了，noccupcells_ 被占用的网格数量减一 
            if( vgridkps_.at(idx).empty() ) {
                noccupcells_--;
            }
            break;
        }
    }
}

/**
 * @brief 更新网格内的特征点 
 * @param prevkp 之前的特征点
 * @param newkp 更新后的特征点
*/
void Frame::updateKeypointInGrid(const Keypoint &prevkp, const Keypoint &newkp)
{
    // First ensure that new kp should move
    // 之前的特征点，从网格中移除 
    int idx = getKeypointCellIdx(prevkp.px_);
    // 更新后的特征点，添加到网格中
    int nidx = getKeypointCellIdx(newkp.px_);

    if( idx == nidx ) {
        // Nothing to do
        return;
    }
    else {
        // First remove kp
        // 之前的特征点，从网格中移除 
        removeKeypointFromGrid(prevkp);
        // Second the new kp is added to the grid
        // 更新后的特征点，添加到网格中
        addKeypointToGrid(newkp);
    }
}

/**
 * @brief 获取特征点对应的网格内的特征点 
 * @param pt 特征点坐标
 * @return 网格内的所有特征点
*/
std::vector<Keypoint> Frame::getKeypointsFromGrid(const cv::Point2f &pt) const
{
    std::lock_guard<std::mutex> lock(grid_mutex_);
    // 网格内的特征点id 
    std::vector<int> voutkpids;
    // 获取特征点所在的网格ID 
    int idx = getKeypointCellIdx(pt);
    // 避免网格ID不合法 
    if( idx < 0 || idx >= (int)vgridkps_.size() ) {
        return std::vector<Keypoint>();
    }
    // 网格内没有特征点
    if( vgridkps_.at(idx).empty() ) {
        return std::vector<Keypoint>();
    }
    // 获取网格内的特征点id 
    for( const auto &id : vgridkps_.at(idx) )
    {
        voutkpids.push_back(id);
    }
    // 返回网格内的特征点 
    return getKeypointsByIds(voutkpids);
}

/**
 * @brief 计算特征点对应的网络 
*/
int Frame::getKeypointCellIdx(const cv::Point2f &pt) const
{
    // 计算特征点所在的网格 
    int r = floor(pt.y / ncellsize_);
    int c = floor(pt.x / ncellsize_);
    // 转换成一维索引 
    return (r * nbwcells_ + c);
}

/**
 * @brief 获取特征点周围的特征点 
 * @brief 目标特征点位置
 * @return 周围的特征点
*/
std::vector<Keypoint> Frame::getSurroundingKeypoints(const Keypoint &kp) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);
    // 计算目标特征点所在的网格  
    int rkp = floor(kp.px_.y / ncellsize_);
    int ckp = floor(kp.px_.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);
    // 在目标特征点周围的网格中查找特征点 
    // [rkp-1, rkp+1] * [ckp-1, ckp+1]
    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
            int idx = r * nbwcells_ + c;
            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
                continue;
            }
            for( const auto &id : vgridkps_.at(idx) ) {
                if( id != kp.lmid_ ) {
                    auto it = mapkps_.find(id);
                    if( it != mapkps_.end() ) {
                        // 找到了特征点，加入到结果中 
                        vkps.push_back(it->second);
                    }
                }
            }
        }
    }
    return vkps;
}

/**
 * @brief 获取特征点周围的特征点 
 * @brief 目标特征点位置
 * @return 周围的特征点
*/
std::vector<Keypoint> Frame::getSurroundingKeypoints(const cv::Point2f &pt) const
{
    std::vector<Keypoint> vkps;
    vkps.reserve(20);
    // 计算目标特征点所在的网格 
    int rkp = floor(pt.y / ncellsize_);
    int ckp = floor(pt.x / ncellsize_);

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> glock(grid_mutex_);
    // 在目标特征点周围的网格中查找特征点 
    // [rkp-1, rkp+1] * [ckp-1, ckp+1]
    for( int r = rkp-1 ; r < rkp+1 ; r++ ) {
        for( int c = ckp-1 ; c < ckp+1 ; c++ ) {
            int idx = r * nbwcells_ + c;
            if( r < 0 || c < 0 || idx > (int)vgridkps_.size() ) {
                continue;
            }
            for( const auto &id : vgridkps_.at(idx) ) {
                auto it = mapkps_.find(id);
                if( it != mapkps_.end() ) {
                    // 找到了特征点，加入到结果中 
                    vkps.push_back(it->second);
                }
            }
        }
    }
    return vkps;
}

/**
 * @brief 获取共视图
 * @return 共视图关系
*/
std::map<int,int> Frame::getCovisibleKfMap() const
{
    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    return map_covkfs_;
}

/**
 * @brief 更新共视图
 * @param cokfs 共视图关系
*/
inline void Frame::updateCovisibleKfMap(const std::map<int,int> &cokfs)
{
    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    map_covkfs_ = cokfs;// 直接覆盖原来的共视关系 
}

/**
 * @brief 添加共视图
 * @param kfid 和当前帧具有共视关系的帧id 
*/
void Frame::addCovisibleKf(const int kfid)
{
    // 如果是自己，直接返回 
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    auto it = map_covkfs_.find(kfid);// 查找是否已经存在 
    if( it != map_covkfs_.end() ) {
        // 如果存在，计数加1 
        it->second += 1;
    } else {
        // 如果不存在，添加到共视图中,计数为1 
        map_covkfs_.emplace(kfid, 1);
    }
}

/**
 * @brief 移除共视图
 * @param kfid 和当前帧具有共视关系的帧id
*/
void Frame::removeCovisibleKf(const int kfid)
{
    // 如果是自己，直接返回 
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    // 从共视图中移除 
    map_covkfs_.erase(kfid);
}

/**
 * @brief 减少共视图 
 * @param kfid 和当前帧具有共视关系的帧id
*/
void Frame::decreaseCovisibleKf(const int kfid)
{
    // 为当前帧 
    if( kfid == kfid_ ) {
        return;
    }

    std::lock_guard<std::mutex> lock(cokfs_mutex_);
    auto it = map_covkfs_.find(kfid);// 获取指定帧的共视关系数量
    if( it != map_covkfs_.end() ) {
        if( it->second != 0 ) {
            it->second -= 1;// 数量减一
            // 减到0，从共视图中移除 
            if( it->second == 0 ) {
                map_covkfs_.erase(it);
            }
        }
    }
}
/**
 * @brief 获取相机位姿
 * @return 相机位姿
*/
Sophus::SE3d Frame::getTcw() const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return Tcw_;
}
/**
 * @brief 获取相机位姿
 * @return 相机位姿
*/
Sophus::SE3d Frame::getTwc() const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return Twc_;
}

/**
 * @brief 获取相机旋转矩阵
 * @return 相机旋转矩阵
*/
Eigen::Matrix3d Frame::getRcw() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Tcw_.rotationMatrix();
}

/**
 * @brief 获取相机旋转矩阵
 * @return 相机旋转矩阵
*/
Eigen::Matrix3d Frame::getRwc() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Twc_.rotationMatrix();
}

/**
 * @brief 获取相机平移向量
 * @return 相机平移向量
*/
Eigen::Vector3d Frame::gettcw() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Tcw_.translation();
}

/**
 * @brief 获取相机平移向量
 * @return 相机平移向量
*/
Eigen::Vector3d Frame::gettwc() const
{
   std::lock_guard<std::mutex> lock(pose_mutex_);
   return Twc_.translation();
}

/**
 * @brief 设置相机位姿
 * @param Twc 相机位姿
*/
void Frame::setTwc(const Sophus::SE3d &Twc)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Twc_ = Twc;
    Tcw_ = Twc.inverse();
}

/**
 * @brief 设置相机位姿
 * @param Tcw 相机位姿
*/
inline void Frame::setTcw(const Sophus::SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    Tcw_ = Tcw;
    Twc_ = Tcw.inverse();
}

/**
 * @brief 设置相机位姿
 * @param Rwc 相机旋转矩阵
 * @param twc 相机平移向量
*/
void Frame::setTwc(const Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    // 旋转矩阵和平移向量
    Twc_.setRotationMatrix(Rwc);
    Twc_.translation() = twc;
    // 逆变换
    Tcw_ = Twc_.inverse();
}

/**
 * @brief 设置相机位姿
 * @param Rcw 相机旋转矩阵
 * @param tcw 相机平移向量
*/
inline void Frame::setTcw(const Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    // 旋转矩阵和平移向量 
    Tcw_.setRotationMatrix(Rcw);
    Tcw_.translation() = tcw;
    // 逆变换
    Twc_ = Tcw_.inverse();
}

/**
 * @brief 相机坐标系到图像坐标系
 * @param pt 相机坐标系下的点
 * @return 图像坐标系下的点
*/
cv::Point2f Frame::projCamToImage(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImage(pt);
}

/**
 * @brief 相机坐标系到右目图像坐标系
 * @param pt 相机坐标系下的点
 * @return 右目图像坐标系下的点
*/
cv::Point2f Frame::projCamToRightImage(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImage(pcalib_rightcam_->Tcic0_ * pt);
}

/**
 * @brief 相机坐标系到图像坐标系
 * @param pt 相机坐标系下的点
 * @return 图像坐标系下的点
*/
cv::Point2f Frame::projCamToImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImageDist(pt);
}

/**
 * @brief 相机坐标系到右目图像坐标系
 * @param pt 相机坐标系下的点
 * @return 右目图像坐标系下的点
*/
cv::Point2f Frame::projCamToRightImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImageDist(pcalib_rightcam_->Tcic0_ * pt);
}

/**
 * @brief 相机坐标系到世界坐标系
 * @param pt 相机坐标系下的点
 * @return 世界坐标系下的点
*/
Eigen::Vector3d Frame::projCamToWorld(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    Eigen::Vector3d wpt = Twc_ * pt;
    return wpt;
}

/**
 * @brief 世界坐标系到相机坐标系
 * @param pt 世界坐标系下的点
 * @return 相机坐标系下的点
*/
Eigen::Vector3d Frame::projWorldToCam(const Eigen::Vector3d &pt) const
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    Eigen::Vector3d campt = Tcw_ * pt;
    return campt;
}

/**
 * @brief 世界坐标系到图像坐标系
 * @param pt 世界坐标系下的点
 * @return 图像坐标系下的点
*/
cv::Point2f Frame::projWorldToImage(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImage(projWorldToCam(pt));
}

/**
 * @brief 世界坐标系到图像坐标系
 * @param pt 世界坐标系下的点
 * @return 图像坐标系下的点
*/
cv::Point2f Frame::projWorldToImageDist(const Eigen::Vector3d &pt) const
{
    return pcalib_leftcam_->projectCamToImageDist(projWorldToCam(pt));
}

/**
 * @brief 世界坐标系到右目图像坐标系
 * @param pt 世界坐标系下的点
 * @return 右目图像坐标系下的点
*/
cv::Point2f Frame::projWorldToRightImage(const Eigen::Vector3d &pt) const
{
    return pcalib_rightcam_->projectCamToImage(pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));
}

/**
 * @brief 世界坐标系到右目图像坐标系
 * @param pt 世界坐标系下的点
 * @return 右目图像坐标系下的点
*/
cv::Point2f Frame::projWorldToRightImageDist(const Eigen::Vector3d &pt) const
{
    // 世界坐标系到右目图像坐标系
    return pcalib_rightcam_->projectCamToImageDist(pcalib_rightcam_->Tcic0_ * projWorldToCam(pt));
}

/**
 * @brief 判断一个点是否在图像中
*/
bool Frame::isInImage(const cv::Point2f &pt) const
{
    // x in [0, img_w], y in [0, img_h]
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_leftcam_->img_w_ && pt.y < pcalib_leftcam_->img_h_);
}

/**
 * @brief 判断一个点是否在右目图像中
*/
bool Frame::isInRightImage(const cv::Point2f &pt) const
{
    // x in [0, img_w], y in [0, img_h]
    return (pt.x >= 0 && pt.y >= 0 && pt.x < pcalib_rightcam_->img_w_ && pt.y < pcalib_rightcam_->img_h_);
}

/**
 * @brief 输出当前帧的信息
*/
void Frame::displayFrameInfo()
{
    std::cout << "\n************************************";
    std::cout << "\nFrame #" << id_ << " (KF #" << kfid_ << ") info:\n";
    std::cout << "\n> Nb kps all (2d / 3d / stereo) : " << nbkps_ << " (" << nb2dkps_ << " / " << nb3dkps_ << " / " << nb_stereo_kps_ << ")";
    std::cout << "\n> Nb covisible kfs : " << map_covkfs_.size();
    std::cout << "\n twc : " << Twc_.translation().transpose();
    std::cout << "\n************************************\n\n";
}

/**
 * @brief 清空当前帧的所有信息 
*/
void Frame::reset()
{
    id_ = -1;
    kfid_ = 0;
    img_time_ = 0.;

    std::lock_guard<std::mutex> lock(kps_mutex_);
    std::lock_guard<std::mutex> lock2(grid_mutex_);
    
    mapkps_.clear();
    vgridkps_.clear();
    vgridkps_.resize( ngridcells_ );

    nbkps_ = 0;
    nb2dkps_ = 0;
    nb3dkps_ = 0;
    nb_stereo_kps_ = 0;

    noccupcells_ = 0;

    Twc_ = Sophus::SE3d();
    Tcw_ = Sophus::SE3d();

    map_covkfs_.clear();
    set_local_mapids_.clear();
}