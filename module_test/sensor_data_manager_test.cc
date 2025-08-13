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
#include <string>

#include "hpgt/sensor_data/sensor_data_manager.h"
// clang-format on

int main() {
  spdlog::set_level(spdlog::level::info);

  std::string work_dir = "../../resource/test_data";
  std::string config_path = work_dir + "/hpgt_config.json";

  spdlog::info("======================================================");
  spdlog::info("=============== TEST: READ SENSOR DATA ===============");
  spdlog::info("======================================================");

  // ===========================================================================

  // Step1: Read system config and set the sensor data path.
  // Set info log to silent.
  spdlog::set_level(spdlog::level::warn);

  auto system_config = hpgt::SystemConfig::Create();

  try {
    if (!system_config->FromJson(config_path)) {
      spdlog::critical("Test incomplete. ");
      std::exit(EXIT_FAILURE);
    }
  } catch (const std::exception& e) {
    spdlog::critical("Failed to parse system config. Error: {}. ", e.what());
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  // Enable info log
  spdlog::set_level(spdlog::level::info);

  system_config->set_data_dir(work_dir);

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
  }

  for (const auto& [label, _] : sensor_data_manager->GetAllImuConfig()) {
    double start_time = sensor_data_manager->GetImuStartTimeByLabel(label);
    double end_time = sensor_data_manager->GetImuEndTimeByLabel(label);

    spdlog::info("The start/end time of {} in seconds: {:.6f} / {:.6f}", label,
                 start_time, end_time);
  }

  // ===========================================================================

  spdlog::info("Test complete. ");

  return 0;
}
