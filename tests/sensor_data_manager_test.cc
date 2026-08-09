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

// clang-format off
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "hpgt/sensor_data/sensor_data_manager.h"
// clang-format on

namespace {

bool WriteFile(const std::filesystem::path& path, const std::string& content) {
  std::ofstream file(path);
  file << content;
  return static_cast<bool>(file);
}

bool Check(bool condition, const std::string& message) {
  if (!condition) {
    spdlog::critical("Test failed: {}", message);
  }
  return condition;
}

}  // namespace

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  if (argc != 3) {
    spdlog::critical("Usage: {} <data_dir> <output_dir>", argv[0]);
    return EXIT_FAILURE;
  }

  const std::filesystem::path data_dir = argv[1];
  const std::filesystem::path output_dir = argv[2];
  std::filesystem::create_directories(output_dir);
  const std::string config_path = (data_dir / "hpgt_config.json").string();

  // Test the supported input formats and loader failure paths.
  const auto imu_without_header_path = output_dir / "imu_without_header.csv";
  const auto imu_with_header_path = output_dir / "imu_with_header.csv";
  const auto pose_whitespace_path = output_dir / "pose_whitespace.txt";
  const auto empty_path = output_dir / "empty.txt";
  const auto wrong_columns_path = output_dir / "wrong_columns.txt";
  const auto duplicate_timestamp_path = output_dir / "duplicate_timestamp.csv";
  const auto duplicate_pose_timestamp_path =
      output_dir / "duplicate_pose_timestamp.txt";
  const auto zero_quaternion_path = output_dir / "zero_quaternion.txt";
  const auto non_finite_path = output_dir / "non_finite.csv";

  if (!Check(WriteFile(imu_without_header_path,
                       "1000000000,0.1,0.2,0.3,1.0,2.0,3.0\n"
                       "2000000000,0.4,0.5,0.6,4.0,5.0,6.0\n") &&
                 WriteFile(imu_with_header_path,
                           "timestamp,wx,wy,wz,ax,ay,az\n"
                           "1000000000, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0\n"
                           "2000000000 ,0.4 ,0.5 ,0.6 ,4.0 ,5.0 ,6.0\n") &&
                 WriteFile(pose_whitespace_path,
                           "1.0   1.0  2.0\t3.0  0.0 0.0 0.0 1.0\n"
                           "2.0 4.0 5.0 6.0 0.0 0.0 0.0 1.0\n") &&
                 WriteFile(empty_path, "\n  \t\n") &&
                 WriteFile(wrong_columns_path, "1,2,3\n") &&
                 WriteFile(duplicate_timestamp_path,
                           "1000000000,0,0,0,0,0,0\n"
                           "1000000000,0,0,0,0,0,0\n") &&
                 WriteFile(duplicate_pose_timestamp_path,
                           "1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
                           "1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n") &&
                 WriteFile(zero_quaternion_path,
                           "1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n") &&
                 WriteFile(non_finite_path, "1000000000,nan,0,0,0,0,0\n"),
             "failed to create loader test fixtures")) {
    return EXIT_FAILURE;
  }

  hpgt::ImuSequence imu_without_header;
  hpgt::ImuSequence imu_with_header;
  hpgt::PoseSequence pose_whitespace;
  if (!Check(hpgt::ImuDataLoader::Load(imu_without_header_path.string(),
                                       imu_without_header) &&
                 imu_without_header.size() == 2 &&
                 imu_without_header.front()->timestamp == 1.,
             "IMU data without a header was not loaded completely") ||
      !Check(hpgt::ImuDataLoader::Load(imu_with_header_path.string(),
                                       imu_with_header) &&
                 imu_with_header.size() == 2 &&
                 imu_with_header.back()->timestamp == 2.,
             "IMU data with a header or comma whitespace was not loaded") ||
      !Check(hpgt::PoseDataLoader::Load(pose_whitespace_path.string(),
                                        pose_whitespace) &&
                 pose_whitespace.size() == 2,
             "pose data with regular whitespace was not loaded")) {
    return EXIT_FAILURE;
  }

  hpgt::ImuSequence failed_imu_load = {hpgt::ImuFrame::Create(
      42., Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())};
  if (!Check(!hpgt::ImuDataLoader::Load(empty_path.string(), failed_imu_load) &&
                 failed_imu_load.size() == 1 &&
                 failed_imu_load.front()->timestamp == 42.,
             "an empty IMU file succeeded or changed the destination") ||
      !Check(!hpgt::ImuDataLoader::Load(wrong_columns_path.string(),
                                        failed_imu_load) &&
                 failed_imu_load.front()->timestamp == 42.,
             "an invalid IMU row succeeded or changed the destination") ||
      !Check(!hpgt::ImuDataLoader::Load(duplicate_timestamp_path.string(),
                                        failed_imu_load) &&
                 failed_imu_load.front()->timestamp == 42.,
             "duplicate IMU timestamps were accepted") ||
      !Check(!hpgt::ImuDataLoader::Load(non_finite_path.string(),
                                        failed_imu_load) &&
                 failed_imu_load.front()->timestamp == 42.,
             "a non-finite IMU value was accepted")) {
    return EXIT_FAILURE;
  }

  hpgt::PoseSequence failed_pose_load = {hpgt::PoseFrame::Create(
      42., Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity())};
  if (!Check(
          !hpgt::PoseDataLoader::Load(empty_path.string(), failed_pose_load) &&
              failed_pose_load.size() == 1 &&
              failed_pose_load.front()->timestamp == 42.,
          "an empty pose file succeeded or changed the destination") ||
      !Check(!hpgt::PoseDataLoader::Load(zero_quaternion_path.string(),
                                         failed_pose_load) &&
                 failed_pose_load.front()->timestamp == 42.,
             "a zero-norm pose quaternion was accepted") ||
      !Check(!hpgt::PoseDataLoader::Load(duplicate_pose_timestamp_path.string(),
                                         failed_pose_load) &&
                 failed_pose_load.front()->timestamp == 42.,
             "duplicate pose timestamps were accepted")) {
    return EXIT_FAILURE;
  }

  spdlog::info("======================================================");
  spdlog::info("=============== TEST: READ SENSOR DATA ===============");
  spdlog::info("======================================================");

  // ===========================================================================

  // Step1: Read system config and set the sensor data path.
  // Set info log to silent.
  spdlog::set_level(spdlog::level::warn);

  auto system_config = hpgt::SystemConfig::Create();

  if (!system_config->FromJson(config_path)) {
    spdlog::critical("Test incomplete. ");
    return EXIT_FAILURE;
  }

  // Enable info log
  spdlog::set_level(spdlog::level::info);

  system_config->set_data_dir(data_dir.string());

  // ===========================================================================

  // Step2: Load sensor data to the manager.
  auto sensor_data_manager = hpgt::SensorDataManager::Create();

  if (!sensor_data_manager->LoadSensorData(system_config)) {
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  // Print the start and end times of the data.
  for (const auto& [label, _] : sensor_data_manager->GetAllPoseConfig()) {
    double start_time = sensor_data_manager->GetPoseStartTimeByLabel(label);
    double end_time = sensor_data_manager->GetPoseEndTimeByLabel(label);

    spdlog::info("The start/end time of {} in seconds: {:.6f} / {:.6f}", label,
                 start_time, end_time);
    if (!Check(
            std::isfinite(sensor_data_manager->GetPoseFrequencyByLabel(label)),
            "pose frequency is not finite")) {
      return EXIT_FAILURE;
    }
  }

  for (const auto& [label, _] : sensor_data_manager->GetAllImuConfig()) {
    double start_time = sensor_data_manager->GetImuStartTimeByLabel(label);
    double end_time = sensor_data_manager->GetImuEndTimeByLabel(label);

    spdlog::info("The start/end time of {} in seconds: {:.6f} / {:.6f}", label,
                 start_time, end_time);
    const auto& sequence = sensor_data_manager->GetImuSeqByLabel(label);
    const double expected_frequency =
        (sequence.size() - 1) /
        (sequence.back()->timestamp - sequence.front()->timestamp);
    if (!Check(std::abs(sensor_data_manager->GetImuFrequencyByLabel(label) -
                        expected_frequency) < 1e-12,
               "IMU frequency does not use the IMU sequence timestamps")) {
      return EXIT_FAILURE;
    }
  }

  const size_t pose_sensor_count =
      sensor_data_manager->GetAllPoseConfig().size();
  const size_t imu_sensor_count = sensor_data_manager->GetAllImuConfig().size();
  system_config->get_pose_config().emplace_back();
  system_config->get_imu_config().emplace_back();
  if (!Check(
          sensor_data_manager->LoadSensorData(system_config) &&
              sensor_data_manager->GetAllPoseConfig().size() ==
                  pose_sensor_count &&
              sensor_data_manager->GetAllImuConfig().size() == imu_sensor_count,
          "empty file-name config nodes were not skipped")) {
    return EXIT_FAILURE;
  }

  hpgt::PoseDataConfig missing_file_config;
  missing_file_config.file_name = "missing_pose_file.txt";
  system_config->get_pose_config().push_back(missing_file_config);
  if (!Check(
          !sensor_data_manager->LoadSensorData(system_config) &&
              sensor_data_manager->GetAllPoseConfig().size() ==
                  pose_sensor_count &&
              sensor_data_manager->GetAllImuConfig().size() == imu_sensor_count,
          "a failed manager load changed previously loaded sensor data")) {
    return EXIT_FAILURE;
  }

  // ===========================================================================

  spdlog::info("Test complete. ");

  return 0;
}
