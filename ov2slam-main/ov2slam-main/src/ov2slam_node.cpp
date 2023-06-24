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
#include <string>
#include <thread>
#include <mutex>
#include <queue>

#include <ros/ros.h>
#include <ros/console.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include "ov2slam.hpp"
#include "slam_params.hpp"

// 传感器读取类 
class SensorsGrabber {

public:
    SensorsGrabber(SlamManager *slam): pslam_(slam) {
        std::cout << "\nSensors Grabber is created...\n";
    }
    // 订阅左相机图像
    void subLeftImage(const sensor_msgs::ImageConstPtr &image) {
        std::lock_guard<std::mutex> lock(img_mutex);
        img0_buf.push(image);// 图像入队列
    }

    // 订阅右相机图像 
    void subRightImage(const sensor_msgs::ImageConstPtr &image) {
        std::lock_guard<std::mutex> lock(img_mutex);
        img1_buf.push(image);// 图像入队列
    }

    // 从Msg中获取灰度图像 
    cv::Mat getGrayImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
    {
        // Get and prepare images
        cv_bridge::CvImageConstPtr ptr;
        try {    
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        } 
        catch(cv_bridge::Exception &e)
        {
            ROS_ERROR("\n\n\ncv_bridge exeception: %s\n\n\n", e.what());
        }

        return ptr->image;
    }

    // extract images with same timestamp from two topics
    // (mostly derived from Vins-Fusion: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
    // 同步处理线程 
    // 从两个话题中提取相同时间戳的图像 
    void sync_process()
    {
        std::cout << "\nStarting the measurements reader thread!\n";
        
        while( !pslam_->bexit_required_ )
        {
            if( pslam_->pslamstate_->stereo_ )
            {// 双目 
                // 图像数据 image0, image1 
                cv::Mat image0, image1;

                std::lock_guard<std::mutex> lock(img_mutex);
                // 两个队列都不为空 
                if (!img0_buf.empty() && !img1_buf.empty())
                {
                    // 获取时间戳
                    double time0 = img0_buf.front()->header.stamp.toSec();
                    double time1 = img1_buf.front()->header.stamp.toSec();

                    // sync tolerance
                    if(time0 < time1 - 0.015)// 图像0的时间戳小于图像1的时间戳，超过0.015s 
                    {
                        img0_buf.pop();// 图像0出队列，这一帧图像丢弃 
                        std::cout << "\n Throw img0 -- Sync error : " << (time0 - time1) << "\n";
                    }
                    else if(time0 > time1 + 0.015)// 图像0的时间戳大于图像1的时间戳，超过0.015s
                    {
                        img1_buf.pop();// 图像1出队列，这一帧图像丢弃
                        std::cout << "\n Throw img1 -- Sync error : " << (time0 - time1) << "\n";
                    }
                    else
                    {
                        // 否则，时间戳相差在0.015s以内，认为是同一帧图像 
                        image0 = getGrayImageFromMsg(img0_buf.front());
                        image1 = getGrayImageFromMsg(img1_buf.front());
                        img0_buf.pop();
                        img1_buf.pop();
                        // 添加图像数据到SLAM系统中 
                        if( !image0.empty() && !image1.empty() ) {
                            pslam_->addNewStereoImages(time0, image0, image1);
                        }
                    }
                }
            } 
            else if( pslam_->pslamstate_->mono_ ) 
            {// 单目 
                cv::Mat image0;// 图像数据 

                std::lock_guard<std::mutex> lock(img_mutex);

                if ( !img0_buf.empty() ) // 有图像数据 
                {
                    // 获取时间戳 
                    double time = img0_buf.front()->header.stamp.toSec();
                    // 获取图像数据 
                    image0 = getGrayImageFromMsg(img0_buf.front());
                    // 图像出队列 
                    img0_buf.pop();
                    // 添加图像数据到SLAM系统中 
                    if( !image0.empty()) {
                        pslam_->addNewMonoImage(time, image0);
                    }
                }
            }

            std::chrono::milliseconds dura(1);
            std::this_thread::sleep_for(dura);
        }

        std::cout << "\n Bag reader SyncProcess thread is terminating!\n";
    }

    std::queue<sensor_msgs::ImageConstPtr> img0_buf;
    std::queue<sensor_msgs::ImageConstPtr> img1_buf;
    std::mutex img_mutex;
    
    SlamManager *pslam_;
};

// 程序入口 
int main(int argc, char** argv)
{
    // Init the node
    ros::init(argc, argv, "ov2slam_node");

    if(argc < 2)
    {
       std::cout << "\nUsage: rosrun ov2slam ov2slam_node parameters_files/params.yaml\n";
       return 1;
    }

    std::cout << "\nLaunching OV²SLAM...\n\n";

    ros::NodeHandle nh("~");

    // Load the parameters
    // 加载配置参数 
    std::string parameters_file = argv[1];

    std::cout << "\nLoading parameters file : " << parameters_file << "...\n";
    // opencv 自带的文件读取类
    const cv::FileStorage fsSettings(parameters_file.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
       std::cout << "Failed to open settings file...";
       return 1;
    } else {
        std::cout << "\nParameters file loaded...\n";
    }
    // 创建参数类,基于读取的配置文件
    std::shared_ptr<SlamParams> pparams;
    pparams.reset( new SlamParams(fsSettings) );

    // Create the ROS Visualizer
    // ROS 下的可视化类 
    std::shared_ptr<RosVisualizer> prosviz;
    prosviz.reset( new RosVisualizer(nh) );

    // Setting up the SLAM Manager
    // SLAM 管理类 
    SlamManager slam(pparams, prosviz);

    // Start the SLAM thread
    // 启动 SLAM 线程 
    std::thread slamthread(&SlamManager::run, &slam);

    // Create the Bag file reader & callback functions
    // 传感器数据读取类 
    SensorsGrabber sb(&slam);

    // Create callbacks according to the topics set in the parameters file
    // 根据配置文件中的话题创建回调函数 左右相机话题 
    ros::Subscriber subleft = nh.subscribe(fsSettings["Camera.topic_left"], 2, &SensorsGrabber::subLeftImage, &sb);
    ros::Subscriber subright = nh.subscribe(fsSettings["Camera.topic_right"], 2, &SensorsGrabber::subRightImage, &sb);

    // Start a thread for providing new measurements to the SLAM
    // 启动传感器数据读取线程
    std::thread sync_thread(&SensorsGrabber::sync_process, &sb);

    // ROS Spin
    ros::spin();

    // Request Slam Manager thread to exit
    slam.bexit_required_ = true;

    // Waiting end of SLAM Manager
    while( slam.bis_on_ ) {
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }

    return 0;
}
