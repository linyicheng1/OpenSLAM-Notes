#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// 使用imu数据预测当前位置 
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
	// 获取当前时间
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {// 第一次进入，记录数据，不进行预测
        latest_time = t;
        init_imu = 0;
        return;
    }
	// 计算相差的时间 
    double dt = t - latest_time;
	// 记录时间 
    latest_time = t;
	// 使用imu数据进行预测
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
	// 局部坐标系下的加速度 + 重力 
    Eigen::Vector3d linear_acceleration{dx, dy, dz};
	// 局部坐标系下的角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};
	// 去除重力的影响，使用上一次的位姿数据 
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;
	// 计算更新后的旋转，使用上一次的角速度和当前角速度的平均值 
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
	// 计算旋转四元数 
	// 四元数旋转 
	// w  1
	// x  theta(x)/2
	// y  theta(y)/2
	// z  theta(z)/2
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
	// 去除重力的影响，使用当前估计的旋转
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
	// 得到平均值作为真实估计值 
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
	// 计算位置更新 
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
	// 计算速度更新 
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

// 获取所有的测量
// 返回值：多帧测量数据组成的向量
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
	// 测量量结构体  
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
    	// 确保含有特征点和imu数据
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
		// 确保IMU时序没问题 
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }
		// 确保特征点时序没问题
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
		// 获取第一帧的特征点
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
		// 删除当前特征点 
        feature_buf.pop();
		// 获取当前帧之前的所有imu数据 
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
		// 最后得到一帧数据 
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// imu 数据回调函数 
// 1、记录当前imu数据 
// 2、预测当前位置并发布出去
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
	// 判断数据获取时间是否正确
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
	// 记录获取当前数据的时间 
    last_imu_t = imu_msg->header.stamp.toSec();
    // 线程锁
    m_buf.lock();
	// 压入imu数据数组 
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();
	// ? 
    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
		// 使用imu数据去预测当前位置
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
		// 发布由imu数据预测得到的位置 tmp_P, tmp_Q, tmp_V 
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// 特征点回调函数 
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {// 第一次进入，不使用该数据
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
	// 保存到结构体中 
    feature_buf.push(feature_msg);
    m_buf.unlock();
	// 标志有数据到来
    con.notify_one();
}

// 重启命令 
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// 重定位信息回调函数 
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);// 记录重定位数据即可 
    m_buf.unlock();
}

// thread: visual-inertial odometry
// vio的主要线程 
// 主要调用函数：
// 1、processIMU    处理imu数据
// 2、setReloFrame 处理重定位数据 
// 3、processImage 处理图像数据 
// @TODO 重定位和图像特征点信息如何得到的  
void process()
{
    while (true)
    {
    	// 所有数据的结构体
    	// 1、一系列的imu测量数据
    	// 2、特征点数据
    	// 将两种数据打包并组成一个向量
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
		// 等待得到特征点数据后调用函数 getMeasurements 
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
		// measurements 包含了待处理的多帧数据 
        m_estimator.lock();
        for (auto &measurement : measurements)
        {// 对每一帧数据单独进行处理
        	// 获取图像数据  
            auto img_msg = measurement.second;
			// 计算预积分数据 
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {// 遍历所有imu数据 
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                {// 在获取图像之前的imu数据 
                    if (current_time < 0)
                        current_time = t;
					// 获取图像和imu数据时间差
                    double dt = t - current_time;
					// 确保时间差大于零
                    ROS_ASSERT(dt >= 0);
					// 保留当前数据
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
					// 调用函数processIMU进行预积分计算 
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {// 工作还是做的细致啊，超过当前图像时间的第一帧IMU数据进行处理
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
					// 线性差值得到imu数据 
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
					// 预积分计算 
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            // 获取重定位信息
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
			// 获取得到所有的重定位数据
			while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {// 遍历所有重定位数据 
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
				// 设置检测得到重定位数据的两帧间的关系 
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
			// 获取图像信息 
            TicToc t_s;
			// 图像的数据结构
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {// 遍历所有特征点 
            	// 得到
                int v = img_msg->channels[0].values[i] + 0.5;
				// 
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
				// x y z p_u p_v velocity_x velocity_y 
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
				// 光流法跟踪得到的速度数据
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
				// 特征点id 相机id 光流得到的速度 
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
			// 处理图像数据 
            estimator.processImage(image, img_msg->header);
			// 输出当前优化调试信息
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
			// 发布数据到ros上 
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            // 发布重定位数据 
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

// vins-mono 入口 
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // 获取配置参数 
	readParameters(n);
	// 设置配置参数 
    estimator.setParameter();// 相机内参矩阵 
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");
    // 发布信息 
    registerPub(n);
    // 接受IMU数据 
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
	// 接收特征点数据 
	ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
	// 接收重启命令
	ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
	// 接收匹配点数据 
	ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);
    // 开启线程执行函数 process
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
