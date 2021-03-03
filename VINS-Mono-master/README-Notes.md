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



# vins_estimator 模块 

# pose_graph 模块 


