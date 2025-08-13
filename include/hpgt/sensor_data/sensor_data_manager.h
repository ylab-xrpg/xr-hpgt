// Copyright 2025 Yongjiang Laboratory
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <memory>
#include <string>

#include "hpgt/config/system_config.h"
#include "hpgt/estimator/calib_parameter.h"
#include "hpgt/sensor_data/imu_data.h"
#include "hpgt/sensor_data/pose_data.h"

namespace hpgt {

// Sensor data manager for loading, storing, and accessing data.
class SensorDataManager {
 public:
  using Ptr = std::shared_ptr<SensorDataManager>;

  static SensorDataManager::Ptr Create() {
    return Ptr(new SensorDataManager());
  }

  // Load sensor data and keep them in containers.
  bool LoadSensorData(const SystemConfig::Ptr &system_config);

  // Get data sequence by label.
  const PoseSequence &GetPoseSeqByLabel(const std::string &pose_label) const;

  const ImuSequence &GetImuSeqByLabel(const std::string &imu_label) const;

  // Get sensor config.
  const std::map<std::string, PoseDataConfig> &GetAllPoseConfig() {
    return pose_data_config_;
  }

  const std::map<std::string, ImuDataConfig> &GetAllImuConfig() {
    return imu_data_config_;
  }

  // Get sensor start/end time by label.
  double GetPoseStartTimeByLabel(const std::string &pose_label) const;

  double GetImuStartTimeByLabel(const std::string &imu_label) const;

  double GetPoseEndTimeByLabel(const std::string &pose_label) const;

  double GetImuEndTimeByLabel(const std::string &imu_label) const;

  // Get sensor frequency by label.
  double GetPoseFrequencyByLabel(const std::string &pose_label) const;

  double GetImuFrequencyByLabel(const std::string &imu_label) const;

 private:
  // Sensor data container, indexed by labels.
  std::map<std::string, PoseSequence> pose_data_sequence_;
  std::map<std::string, ImuSequence> imu_data_sequence_;

  // Sensor data config, indexed by labels.
  std::map<std::string, PoseDataConfig> pose_data_config_;
  std::map<std::string, ImuDataConfig> imu_data_config_;
};

}  // namespace hpgt
