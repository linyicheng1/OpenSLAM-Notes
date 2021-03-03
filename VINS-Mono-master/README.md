# 算法整体流程
算法分为三个部分，分别在3个独立的ros工程中，分别为 feature_tracker vins_estimator pose_graph

运行算法的launch文件，如euroc.launch中则运行该三个节点，并获取配置文件 config/euroc_config.yaml 路径，传递给各程序

## 四个模块之间的数据交互（加上可视化数据的RVIZ）
![384369e6-f3cd-491b-a10c-a69d3461676c.png](https://storage.live.com/items/24342272185BBA7E!4892?authkey=AJzdbBYZIQ_AuAo)

因此，大致的算法思想为feature_tracker模块对图像数据进行处理，得到特征点跟踪信息，vins_estimator为主体程序处理跟踪得到的特征点信息和imu数据，pose_graph则为后端优化以及重定位功能，接下再仔细对每一个模块进行分析 

# feature_tracker 模块
本模块中核心实现为特征点跟踪类，基本流程为获取得到一帧图片数据转换为opencv格式并控制帧率，最后送入FeatureTracker类中跟踪特征点，最后发布特征点数据。核心调用在于95行的
```
95:trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
```

## FeatureTracker类
其中函数readImage的主要流程如下，包含两个较为重要的知识点
* OpenCV函数的光流法如何实现的
* 基础矩阵的鲁棒估计

![26488ab8-8a16-4aea-bcd3-56322243cf36.png](https://storage.live.com/items/24342272185BBA7E!4893?authkey=AJzdbBYZIQ_AuAo)

# vins_estimator 模块 

主要的执行函数如下
* 单独线程 `void process();`
* IMU数据回调函数 `imu_callback`
* 特征点数据回调函数 `feature_callback`
* 重定位匹配得到的数据 `relocalization_callback`

## IMU数据回调函数 

1. 接收IMU数据
2. 通过IMU积分更新当前位置
3. 发布由IMU积分得到的位置（消息：`imu_propagate`）

* 知识点：如何通过当前IMU来积分得到当前位置,函数`predict()`给出了实现，可参考注释学习。

## 特征点数据回调函数 
* 仅实现了保存数据的功能

## 重定位匹配得到的数据 
* 仅实现了保存数据的功能

## 单独处理线程 
本线程是主要处理部分，首先明确如下数据，即输入数据有三个,均使用了队列的数据结构对数据先进先出逐个处理。

* imu数据          `queue<sensor_msgs::ImuConstPtr> imu_buf`
* 跟踪得到的特征点  `queue<sensor_msgs::PointCloudConstPtr> feature_buf`
* 重定位匹配点      `queue<sensor_msgs::PointCloudConstPtr> relo_buf`

主要的流程图如下，核心函数实现类为 Estimator 类 
* 将数据按照采集时间打包成一帧图片特征点+多个IMU数据格式的多个测量
* 调用函数 `processIMU` 进行预积分计算
* 调用函数 `setReloFrame` 添加重定位约束
* 调用函数 `processImage` 处理图像特征点 
* 发布消息处理结果消息
![0a89764c-5a87-46f2-bde6-4378e6c18c8d.png](https://storage.live.com/items/24342272185BBA7E!4894?authkey=AJzdbBYZIQ_AuAo)

因此接下来将逐个介绍相关内容

### 函数 `processIMU` 预积分计算 

由预积分类`IntegrationBase`实现，该类重载了push_back 函数，相当于将imu数据压入该类后就自动计算预积分的更新量。

* 实际上采用中值积分的方法，具体公式推导请参考原始论文中的附录A内容

### 函数 `setReloFrame` 添加重定位约束 

* 保存了重定位所有数据
* 利用时间信息找到在当前滑动窗口下对应的具体那一帧具有重定位信息
* 记录那一帧的信息

在函数中并未对数据进行处理，仅仅是保存得到的数据 

### 函数 `processImage` 处理图像特征点 

对于图片特征点数据的处理是算法中最为核心且复杂的内容之一，主要的流程图如下

![45b4a77b-2701-4d37-ba86-a78be97e1548.png](https://storage.live.com/items/24342272185BBA7E!4895?authkey=AJzdbBYZIQ_AuAo)

因此核心实现函数如下：

1. 选择剔除滑动窗口中的哪一帧数据 `addFeatureCheckParallax`
2. 系统状态初始化 `initialStructure`
3. 求解当前位姿 `solveOdometry`
4. 滑动窗口 `slideWindow`

#### addFeatureCheckParallax 选择剔除帧 

* 遍历得到当前帧与上一关键帧的所有匹配点 
* 遍历所有匹配点计算平均视差
* 平均视差大于阈值则剔除最旧的一帧，否则剔除当前帧

#### initialStructure 系统状态初始化

![9991b8a1-4971-4c34-88b6-d2b39f08fcd1.png](https://storage.live.com/items/24342272185BBA7E!4896?authkey=AJzdbBYZIQ_AuAo)

**几个重要的点**

* 如何通过基础矩阵F来得到两帧之间的旋转和平移
* SfM的具体实现过程
* IMU与视觉数据对齐方式 

对于前面两点，可以参考代码注释和计算机视觉相关的书籍弄懂，这里仅对IMU与视觉数据对齐方式做如下说明

![4f701aec-a76b-48dd-b4d4-df99651f1366.png](https://storage.live.com/items/24342272185BBA7E!4897?authkey=AJzdbBYZIQ_AuAo)

数据对齐的目的在于通过对相机位姿的缩放和平移得到一个与IMU估计一致的估计状态。
主要流程如下
* 陀螺仪偏差校准 
* 速度、重力向量以及尺度因子初始化
* 重力的精细估计
* 完成初始化 
  
具体初始化流程与论文第五节B中进行了详细的描述


#### slideWindow 滑动窗口

**这里并没有做滑动窗口的概率转移相关的计算，单纯的滑动删除一帧**

* 将需要删除的一帧的所有相关数据删除掉
* 所有帧位置挪动一下
* 同时删除掉构造的优化问题中的节点和边的变量


#### solveOdometry 求解当前位姿

* 三角化特征点，得到初始值
* 调用ceres库进行优化整个滑动窗口内的地图点与位姿，是求解vio问题的核心部分

**三角化特征点**

对于一个特征点在大于两个关键帧内被看到，则能够求得其深度，而多于两帧则采用线性最小二乘的方式进行求解，代码中构造矩阵利用svd分解对其进行求解，具体原理可参考多视图几何一书。

**调用ceres库对整体进行优化求解**

![6ab618b0-44d1-4403-9ab8-1406d65cb4b6.png](https://storage.live.com/items/24342272185BBA7E!4898?authkey=AJzdbBYZIQ_AuAo)

**注意：这里仅优化了相机的位姿而没有优化地图点**

* 构造图优化问题
  * 构造所有的节点
    * 所有关键帧内相机的位姿和速度
    * 相机外参
    * 重定位的位姿 
  * 构造所有的边
    * 边缘化约束
    * IMU预积分约束
    * 重投影误差约束
    * 重定位约束

* 后处理部分
  * 离群路标点删除
  * 滑动窗口带来的约束项计算


具体如何求解请参考原论文以及相关资料的理论部分，这里并不打算进行理论部分的说明，后期专门讲理论的部分会补充相关理论。

### 处理结束后发布的消息数据 

在滑动窗口内对窗口内的数据进行处理后将数据发布到整体的位姿图上进行最终的整体优化过程。

* 相机帧率的里程计数据 `odometry`
* IMU帧率的里程计数据 `imu_odometry`
* 关键帧位置 `key_pose`
* 相机外参 `extrinsic`
* 关键帧内所有的路标点 `keyframe_point`
* 重定位帧位置 `relo_relative_pose`

# pose_graph 模块 

位姿图代码主要运行在两个线程中，其中一个进行回环检测而另外一个进行四自由度的位姿图优化
下面的图能够很好的表达该模块的工作流程。
![7313ba7d-600d-4534-8455-cbd64387d2eb.png](https://storage.live.com/items/24342272185BBA7E!4899?authkey=AJzdbBYZIQ_AuAo)

首先我们进行回环检测得到了一个约束项，然后发送给vins_estimator模块作为一个固定约束，因此整个滑动窗口的值都会被矫正到回环之后的位置，然后在另外一个线程进行位姿图的优化，对之前所有关键帧相互之间的约束加上回环约束得到整个图在回环校准后的位姿。

* 线程1：process() 
  * 根据时间信息获取对应的图片数据，然后构造关键帧带入添加到位姿图 ` posegraph.addKeyFrame(keyframe, 1);`
  * 在函数内进行回环检测调用函数 `detectLoop`
  * 具体采用词袋模型进行回环判断并对离群值进行了剔除（原理可参考slam14讲）
* 线程2：optimize4DoF() 在PoseGraph类的构造函数中新建的一个线程
  * 线程内对所有关键帧进行四自由度的优化（由于imu对pitch和yaw角度可观，没有累计误差就不优化了）
  * 构造Ceres优化算法，这个相对简单。误差函数为 重投影误差+回环检测误差


# 理论部分补充（待续）

* 光流法
* 基础矩阵计算以及利用基础矩阵估计相机位姿
* IMU预积分推导
* SfM求解相机位姿
* IMU与视觉信息对齐
* 特征点三角化求解
* 滑动窗口带来的约束求解
* 重投影误差计算
* IMU预积分误差计算
* 使用词袋模型进行回环检测
