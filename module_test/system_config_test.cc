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

#include "hpgt/config/system_config.h"
// clang-format on

int main() {
  spdlog::set_level(spdlog::level::info);

  std::string work_dir = "../../resource/test_data";
  std::string output_path = work_dir + "/config_template.json";
  std::string input_path = work_dir + "/hpgt_config.json";

  // ===========================================================================

  // Step 1: Generate config template.
  spdlog::info("======================================================");
  spdlog::info("========== TEST 1: GENERATE CONFIG TEMPLATE ==========");
  spdlog::info("======================================================");

  // Initialize system config with the specified number of sensors.
  // Serialize to generate a config file template.
  constexpr int kPoseConfigNUm = 2;
  constexpr int kImuConfigNum = 2;

  auto system_config_o = hpgt::SystemConfig::Create();
  for (size_t i = 0; i < kPoseConfigNUm; ++i) {
    system_config_o->get_pose_config().emplace_back();
  }

  for (size_t i = 0; i < kImuConfigNum; ++i) {
    system_config_o->get_imu_config().emplace_back();
  }

  if (system_config_o->ToJson(output_path)) {
    spdlog::info(
        "Generate a config template with {} pose sensors and {} IMUs. Modify "
        "as needed. ",
        kPoseConfigNUm, kImuConfigNum);
  } else {
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  // ===========================================================================

  // Step 2: Read system config.
  spdlog::info("======================================================");
  spdlog::info("============= TEST 2: READ SYSTEM CONFIG =============");
  spdlog::info("======================================================");

  auto system_config_i = hpgt::SystemConfig::Create();

  try {
    if (!system_config_i->FromJson(input_path)) {
      spdlog::critical("Test incomplete. ");
      std::exit(EXIT_FAILURE);
    }
  } catch (const std::exception& e) {
    spdlog::critical("Failed to parse system config. Error: {}. ", e.what());
    spdlog::critical("Test incomplete. ");
    std::exit(EXIT_FAILURE);
  }

  // ===========================================================================

  spdlog::info("Test complete. ");

  return 0;
}
