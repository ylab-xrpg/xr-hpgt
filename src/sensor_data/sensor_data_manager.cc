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

#include "hpgt/sensor_data/sensor_data_manager.h"

namespace hpgt {

bool SensorDataManager::LoadSensorData(const SystemConfig::Ptr &system_config) {
  spdlog::info("Load sensor data...");

  std::string data_dir = system_config->get_data_dir();
  if (data_dir == "") {
    spdlog::critical("Directory of the sensor data must be specified. ");
    return false;
  }

  // Iterate pose config and load the data.
  for (const auto &node : system_config->get_pose_config()) {
    std::string data_label = node.file_name;
    std::string data_path = data_dir + "/" + node.file_name;

    PoseSequence data;
    if (PoseDataLoader::Load(data_path, data)) {
      spdlog::info("Loaded {} pose frames from {}. ", data.size(), data_label);
    } else {
      return false;
    }

    if (pose_data_sequence_.find(data_label) == pose_data_sequence_.end()) {
      pose_data_sequence_[data_label] = std::move(data);
      pose_data_config_[data_label] = node;
    } else {
      spdlog::critical("Duplicate pose label: {}", data_label);
      return false;
    }
  }

  // Iterate imu config and load the data.
  for (const auto &node : system_config->get_imu_config()) {
    std::string data_label = node.file_name;
    std::string data_path = data_dir + "/" + node.file_name;

    ImuSequence data;
    if (ImuDataLoader::Load(data_path, data)) {
      spdlog::info("Load {} imu frames from {}. ", data.size(), data_label);
    } else {
      return false;
    }

    if (imu_data_sequence_.find(data_label) == imu_data_sequence_.end()) {
      imu_data_sequence_[data_label] = std::move(data);
      imu_data_config_[data_label] = node;
    } else {
      spdlog::critical("Duplicate imu label: {}", data_label);
      return false;
    }
  }

  return true;
}

const PoseSequence &SensorDataManager::GetPoseSeqByLabel(
    const std::string &pose_label) const {
  static const PoseSequence empty_seq;
  if (pose_data_sequence_.find(pose_label) == pose_data_sequence_.end()) {
    spdlog::critical("Failed to get pose sequence, invalid label: {} ",
                     pose_label);
    return empty_seq;
  }

  return pose_data_sequence_.at(pose_label);
}

const ImuSequence &SensorDataManager::GetImuSeqByLabel(
    const std::string &imu_label) const {
  static const ImuSequence empty_seq;
  if (imu_data_sequence_.find(imu_label) == imu_data_sequence_.end()) {
    spdlog::critical("Failed to get IMU sequence, invalid label: {} ",
                     imu_label);
    return empty_seq;
  }

  return imu_data_sequence_.at(imu_label);
}

double SensorDataManager::GetPoseStartTimeByLabel(
    const std::string &pose_label) const {
  if (pose_data_sequence_.find(pose_label) == pose_data_sequence_.end()) {
    spdlog::critical("Failed to get pose start time, invalid label: {} ",
                     pose_label);
    return std::nan("");
  }

  return pose_data_sequence_.at(pose_label).front()->timestamp;
}

double SensorDataManager::GetImuStartTimeByLabel(
    const std::string &imu_label) const {
  if (imu_data_sequence_.find(imu_label) == imu_data_sequence_.end()) {
    spdlog::critical("Failed to get IMU start time, invalid label: {} ",
                     imu_label);
    return std::nan("");
  }

  return imu_data_sequence_.at(imu_label).front()->timestamp;
}

double SensorDataManager::GetPoseEndTimeByLabel(
    const std::string &pose_label) const {
  if (pose_data_sequence_.find(pose_label) == pose_data_sequence_.end()) {
    spdlog::critical("Failed to get pose end time, invalid label: {} ",
                     pose_label);
    return std::nan("");
  }

  return pose_data_sequence_.at(pose_label).back()->timestamp;
}

double SensorDataManager::GetImuEndTimeByLabel(
    const std::string &imu_label) const {
  if (imu_data_sequence_.find(imu_label) == imu_data_sequence_.end()) {
    spdlog::critical("Failed to get IMU end time, invalid label: {} ",
                     imu_label);
    return std::nan("");
  }

  return imu_data_sequence_.at(imu_label).back()->timestamp;
}

double SensorDataManager::GetPoseFrequencyByLabel(
    const std::string &pose_label) const {
  if (pose_data_sequence_.find(pose_label) == pose_data_sequence_.end()) {
    spdlog::critical("Failed to get pose frequency, invalid label: {} ",
                     pose_label);
    return std::nan("");
  }

  if (pose_data_sequence_.at(pose_label).size() < 2) {
    spdlog::critical("Failed to get pose frequency, insufficient pose data. ");
    return std::nan("");
  }

  double freq =
      (pose_data_sequence_.at(pose_label).size() - 1) /
      (GetPoseEndTimeByLabel(pose_label) - GetPoseStartTimeByLabel(pose_label));

  return freq;
}

double SensorDataManager::GetImuFrequencyByLabel(
    const std::string &imu_label) const {
  if (imu_data_sequence_.find(imu_label) == imu_data_sequence_.end()) {
    spdlog::critical("Failed to get IMU frequency, invalid label: {} ",
                     imu_label);
    return std::nan("");
  }

  if (imu_data_sequence_.at(imu_label).size() < 2) {
    spdlog::critical("Failed to get IMU frequency, insufficient IMU data. ");
    return std::nan("");
  }

  double freq =
      (imu_data_sequence_.at(imu_label).size() - 1) /
      (GetPoseEndTimeByLabel(imu_label) - GetPoseStartTimeByLabel(imu_label));

  return freq;
}

}  // namespace hpgt