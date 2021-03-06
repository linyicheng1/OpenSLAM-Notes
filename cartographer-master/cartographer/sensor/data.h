/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CARTOGRAPHER_MAPPING_DATA_H_
#define CARTOGRAPHER_MAPPING_DATA_H_

#include "absl/memory/memory.h"
#include "cartographer/common/time.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {

namespace mapping {
class TrajectoryBuilderInterface;
}

namespace sensor {

// 单纯的数据虚类
class Data {
 public:
  explicit Data(const std::string &sensor_id) : sensor_id_(sensor_id) {}
  virtual ~Data() {}
  // 获取数据采集时间
  virtual common::Time GetTime() const = 0;
  // 获取数据id
  const std::string &GetSensorId() const { return sensor_id_; }
  // 添加到轨迹约束中
  virtual void AddToTrajectoryBuilder(
      mapping::TrajectoryBuilderInterface *trajectory_builder) = 0;

 protected:
  const std::string sensor_id_;// 传感器id
};

}  // namespace sensor
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_DATA_H_
