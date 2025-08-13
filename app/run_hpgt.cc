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

#include "hpgt/estimator/estimator.h"

int main(int argc, char **argv) {
  spdlog::set_level(spdlog::level::info);

  // ===========================================================================

  // Step 1: Set the system working directory.
  std::string config_path;
  std::string input_data_dir;
  std::string output_calib_path;
  std::string output_traj_path;

  if (argc != 2 && argc != 5) {
    // clang-format off
    spdlog::critical(
        "Invalid input parameters.\n\n"
        "Usage:\n"
        "  {} <work_directory>\n"
        "  {} <config_path> <input_data_dir> <output_calib_path> <output_traj_path>\n\n"
        "Description:\n"
        "  If a single <work_directory> is provided, the program assumes default file organization:\n"
        "    - Config path:             <work_directory>/hpgt_config.json\n"
        "    - Output calibration:      <work_directory>/hpgt_output_calib.json\n"
        "    - Output trajectory:       <work_directory>/hpgt_output_traj.txt\n\n"
        "  Alternatively, you can provide all paths explicitly.\n\n"
        "Examples:\n"
        "  {} ./dataset_folder\n"
        "  {} ./config.json ./data ./result/calib.json ./result/traj.txt",
        argv[0], argv[0], argv[0], argv[0]);
    // clang-format on

    std::exit(EXIT_FAILURE);
  }

  if (argc == 2) {
    // Default file organization.
    std::string config_name = "/hpgt_config.json";
    std::string output_calib_name = "/hpgt_output_calib.json";
    std::string output_traj_name = "/hpgt_output_traj.txt";

    std::string work_dir = argv[1];
    config_path = work_dir + config_name;
    input_data_dir = work_dir;
    output_calib_path = work_dir + output_calib_name;
    output_traj_path = work_dir + output_traj_name;
  } else if (argc == 5) {
    // Set by user.
    config_path = argv[1];
    input_data_dir = argv[2];
    output_calib_path = argv[3];
    output_traj_path = argv[4];
  }

  // ===========================================================================

  // Step 2: Parse the system configuration and load data from multiple sensors.
  spdlog::info("======================================================");
  spdlog::info("================ PARSE SYSTEM CONFIG =================");
  spdlog::info("======================================================");
  auto system_config = hpgt::SystemConfig::Create();
  if (!system_config->FromJson(config_path)) {
    std::exit(EXIT_FAILURE);
  }
  system_config->set_data_dir(input_data_dir);

  auto sensor_data_manager = hpgt::SensorDataManager::Create();
  if (!sensor_data_manager->LoadSensorData(system_config)) {
    std::exit(EXIT_FAILURE);
  }

  // ===========================================================================

  // Step 3: Initialize the system state, including time-invariant calibration
  // parameters and time-varying spline functions.
  auto estimator = hpgt::Estimator::Create(system_config, sensor_data_manager);

  spdlog::info("======================================================");
  spdlog::info("=============== INITIALIZE SYSTEM STATE ==============");
  spdlog::info("======================================================");
  if (!estimator->Initialize()) {
    std::exit(EXIT_FAILURE);
  }

  // ===========================================================================

  // Step 4: Construct nonlinear optimization and perform the refinement based
  // on the initial guesses.
  spdlog::info("======================================================");
  spdlog::info("================= BUILD AND OPTIMIZE =================");
  spdlog::info("======================================================");
  if (!estimator->BuildAndOptimize()) {
    std::exit(EXIT_FAILURE);
  }

  // ===========================================================================

  // Step 5: Print calibration parameter results and save both the parameters
  // and the system trajectory to the specific paths.
  spdlog::info("======================================================");
  spdlog::info("=================== PRINT AND SAVE ===================");
  spdlog::info("======================================================");
  if (!estimator->PrintAndSaveResult(output_calib_path, output_traj_path)) {
    std::exit(EXIT_FAILURE);
  }

  // ===========================================================================

  return 0;
}
