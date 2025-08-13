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

#include <chrono>
#include <string>

#include "hpgt/initializer/spatial_extrinsic_initializer.h"
#include "hpgt/initializer/time_offset_initializer.hpp"

int main() {
  spdlog::set_level(spdlog::level::info);

  std::string work_dir = "../../resource/test_data";
  std::string config_path = work_dir + "/hpgt_config.json";

  spdlog::info("======================================================");
  spdlog::info("====== TEST: Initialize calibration parameters =======");
  spdlog::info("======================================================");

  // ===========================================================================

  // Step 1: Read system config and load sensor data.
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
  system_config->set_data_dir(work_dir);

  auto sensor_data_manager = hpgt::SensorDataManager::Create();
  if (!sensor_data_manager->LoadSensorData(system_config)) {
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  // Enable info log
  spdlog::set_level(spdlog::level::info);

  // ===========================================================================

  // Step 2: Initialize the calibration parameters in our system.
  auto calib_parameter = hpgt::CalibParameter::Create();
  auto toff_initializer = hpgt::TimeOffsetInitializer::Create();
  auto spatial_initializer = hpgt::SpatialExtrinsicInitializer::Create();

  // Time offset.
  auto start_time = std::chrono::high_resolution_clock::now();

  bool toff_init_flag = true;
  if (system_config->get_opt_temporal_param_flag()) {
    toff_init_flag = toff_initializer->Initialize(sensor_data_manager,
                                                  calib_parameter, true);
  }
  if (!toff_init_flag) {
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  spdlog::info("Initialize time offset within {:.6f} seconds. ",
               elapsed.count());

  // Spatial extrinsic parameters.
  start_time = std::chrono::high_resolution_clock::now();

  bool spatial_init_flag = true;
  if (system_config->get_opt_spatial_param_flag()) {
    spatial_init_flag = spatial_initializer->Initialize(
        sensor_data_manager, calib_parameter,
        system_config->get_gravity_magnitude(), true);
  }
  if (!spatial_init_flag) {
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  end_time = std::chrono::high_resolution_clock::now();
  elapsed = end_time - start_time;
  spdlog::info(
      "Initialize spatial extrinsic parameters within {:.6f} seconds. ",
      elapsed.count());

  // ===========================================================================

  spdlog::info("Test complete. ");

  return 0;
}
