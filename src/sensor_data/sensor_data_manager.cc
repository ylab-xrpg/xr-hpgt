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

#include <filesystem>

namespace hpgt {

bool SensorDataManager::LoadSensorData(const SystemConfig::Ptr &system_config) {
  spdlog::info("Loading sensor data...");

  if (!system_config) {
    spdlog::critical("System config must be provided before loading data.");
    return false;
  }

  const std::filesystem::path data_dir = system_config->get_data_dir();
  if (data_dir.empty()) {
    spdlog::critical("The sensor data directory must be specified.");
    return false;
  }

  std::map<std::string, PoseSequence> pose_data_sequence;
  std::map<std::string, ImuSequence> imu_data_sequence;
  std::map<std::string, PoseDataConfig> pose_data_config;
  std::map<std::string, ImuDataConfig> imu_data_config;

  // Iterate pose config and load the data.
  for (const auto &node : system_config->get_pose_config()) {
    if (node.file_name.empty()) {
      spdlog::warn("Skipping pose data with no file name provided.");
      continue;
    }

    const std::string &data_label = node.file_name;
    if (pose_data_sequence.find(data_label) != pose_data_sequence.end()) {
      spdlog::critical("Duplicate pose label: {}", data_label);
      return false;
    }
    const std::string data_path = (data_dir / data_label).string();

    PoseSequence data;
    if (!PoseDataLoader::Load(data_path, data)) {
      return false;
    }
    spdlog::info("Loaded {} pose frames from {}.", data.size(), data_label);

    pose_data_sequence.emplace(data_label, std::move(data));
    pose_data_config.emplace(data_label, node);
  }

  // Iterate over the IMU configuration and load the data.
  for (const auto &node : system_config->get_imu_config()) {
    if (node.file_name.empty()) {
      spdlog::warn("Skipping IMU data with no file name provided.");
      continue;
    }

    const std::string &data_label = node.file_name;
    if (imu_data_sequence.find(data_label) != imu_data_sequence.end()) {
      spdlog::critical("Duplicate IMU label: {}", data_label);
      return false;
    }
    const std::string data_path = (data_dir / data_label).string();

    ImuSequence data;
    if (!ImuDataLoader::Load(data_path, data)) {
      return false;
    }
    spdlog::info("Loaded {} IMU frames from {}.", data.size(), data_label);

    imu_data_sequence.emplace(data_label, std::move(data));
    imu_data_config.emplace(data_label, node);
  }

  pose_data_sequence_ = std::move(pose_data_sequence);
  imu_data_sequence_ = std::move(imu_data_sequence);
  pose_data_config_ = std::move(pose_data_config);
  imu_data_config_ = std::move(imu_data_config);

  return true;
}

const PoseSequence &SensorDataManager::GetPoseSeqByLabel(
    const std::string &pose_label) const {
  static const PoseSequence empty_seq;
  const auto iterator = pose_data_sequence_.find(pose_label);
  if (iterator == pose_data_sequence_.end() || iterator->second.empty()) {
    spdlog::critical("Failed to get pose sequence; invalid label: {}",
                     pose_label);
    return empty_seq;
  }

  return iterator->second;
}

const ImuSequence &SensorDataManager::GetImuSeqByLabel(
    const std::string &imu_label) const {
  static const ImuSequence empty_seq;
  const auto iterator = imu_data_sequence_.find(imu_label);
  if (iterator == imu_data_sequence_.end() || iterator->second.empty()) {
    spdlog::critical("Failed to get IMU sequence; invalid label: {}",
                     imu_label);
    return empty_seq;
  }

  return iterator->second;
}

double SensorDataManager::GetPoseStartTimeByLabel(
    const std::string &pose_label) const {
  const auto iterator = pose_data_sequence_.find(pose_label);
  if (iterator == pose_data_sequence_.end() || iterator->second.empty()) {
    spdlog::critical("Failed to get pose start time; invalid label: {}",
                     pose_label);
    return std::nan("");
  }

  return iterator->second.front()->timestamp;
}

double SensorDataManager::GetImuStartTimeByLabel(
    const std::string &imu_label) const {
  const auto iterator = imu_data_sequence_.find(imu_label);
  if (iterator == imu_data_sequence_.end() || iterator->second.empty()) {
    spdlog::critical("Failed to get IMU start time; invalid label: {}",
                     imu_label);
    return std::nan("");
  }

  return iterator->second.front()->timestamp;
}

double SensorDataManager::GetPoseEndTimeByLabel(
    const std::string &pose_label) const {
  const auto iterator = pose_data_sequence_.find(pose_label);
  if (iterator == pose_data_sequence_.end() || iterator->second.empty()) {
    spdlog::critical("Failed to get pose end time; invalid label: {}",
                     pose_label);
    return std::nan("");
  }

  return iterator->second.back()->timestamp;
}

double SensorDataManager::GetImuEndTimeByLabel(
    const std::string &imu_label) const {
  const auto iterator = imu_data_sequence_.find(imu_label);
  if (iterator == imu_data_sequence_.end() || iterator->second.empty()) {
    spdlog::critical("Failed to get IMU end time; invalid label: {}",
                     imu_label);
    return std::nan("");
  }

  return iterator->second.back()->timestamp;
}

double SensorDataManager::GetPoseFrequencyByLabel(
    const std::string &pose_label) const {
  const auto iterator = pose_data_sequence_.find(pose_label);
  if (iterator == pose_data_sequence_.end()) {
    spdlog::critical("Failed to get pose frequency; invalid label: {}",
                     pose_label);
    return std::nan("");
  }

  if (iterator->second.size() < 2) {
    spdlog::critical("Failed to get pose frequency; insufficient pose data.");
    return std::nan("");
  }

  const double freq =
      (iterator->second.size() - 1) / (iterator->second.back()->timestamp -
                                       iterator->second.front()->timestamp);

  return freq;
}

double SensorDataManager::GetImuFrequencyByLabel(
    const std::string &imu_label) const {
  const auto iterator = imu_data_sequence_.find(imu_label);
  if (iterator == imu_data_sequence_.end()) {
    spdlog::critical("Failed to get IMU frequency; invalid label: {}",
                     imu_label);
    return std::nan("");
  }

  if (iterator->second.size() < 2) {
    spdlog::critical("Failed to get IMU frequency; insufficient IMU data.");
    return std::nan("");
  }

  const double freq =
      (iterator->second.size() - 1) / (iterator->second.back()->timestamp -
                                       iterator->second.front()->timestamp);

  return freq;
}

}  // namespace hpgt
