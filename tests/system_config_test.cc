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
#include <filesystem>
#include <fstream>
#include <string>

#include "hpgt/config/system_config.h"
// clang-format on

namespace {

bool WriteJson(const std::filesystem::path& path, const nlohmann::json& value) {
  std::ofstream file(path);
  file << value.dump(2);
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

  const std::string output_path =
      (output_dir / "config_template.json").string();
  const std::string input_path = (data_dir / "hpgt_config.json").string();

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

  if (!system_config_i->FromJson(input_path)) {
    spdlog::critical("Test incomplete. ");
    return EXIT_FAILURE;
  }

  // ===========================================================================

  // Step 3: Verify serialization round-trip and validation failure paths.
  const auto round_trip_path = output_dir / "round_trip.json";
  auto round_trip_config = hpgt::SystemConfig::Create();
  if (!Check(system_config_i->ToJson(round_trip_path.string()) &&
                 round_trip_config->FromJson(round_trip_path.string()) &&
                 round_trip_config->get_pose_config().size() ==
                     system_config_i->get_pose_config().size() &&
                 round_trip_config->get_imu_config().size() ==
                     system_config_i->get_imu_config().size() &&
                 round_trip_config->get_spline_knot_interval() ==
                     system_config_i->get_spline_knot_interval(),
             "valid config serialization round-trip failed")) {
    return EXIT_FAILURE;
  }

  nlohmann::json valid_json;
  {
    std::ifstream input_file(input_path);
    if (!Check(static_cast<bool>(input_file),
               "failed to reopen valid config")) {
      return EXIT_FAILURE;
    }
    input_file >> valid_json;
  }

  int invalid_config_index = 0;
  const auto rejects_config = [&](const nlohmann::json& invalid_json) {
    const auto invalid_path =
        output_dir /
        ("invalid_config_" + std::to_string(invalid_config_index++) + ".json");
    if (!WriteJson(invalid_path, invalid_json)) {
      return false;
    }
    auto invalid_config = hpgt::SystemConfig::Create();
    return !invalid_config->FromJson(invalid_path.string());
  };

  nlohmann::json invalid_json = valid_json;
  invalid_json["spline_knot_interval_"] = 0.;
  if (!Check(rejects_config(invalid_json),
             "a non-positive spline interval was accepted")) {
    return EXIT_FAILURE;
  }
  invalid_json = valid_json;
  invalid_json["output_frequency_"] = 0.;
  if (!Check(rejects_config(invalid_json),
             "a non-positive output frequency was accepted")) {
    return EXIT_FAILURE;
  }
  invalid_json = valid_json;
  invalid_json["gravity_magnitude_"] = -9.8;
  if (!Check(rejects_config(invalid_json),
             "a non-positive gravity magnitude was accepted")) {
    return EXIT_FAILURE;
  }
  invalid_json = valid_json;
  invalid_json["imu_config_"][0]["frequency"] = 0.;
  if (!Check(rejects_config(invalid_json),
             "a non-positive IMU frequency was accepted")) {
    return EXIT_FAILURE;
  }
  invalid_json = valid_json;
  invalid_json["imu_config_"][0]["noise"]["gyr_n"] = 0.;
  if (!Check(rejects_config(invalid_json),
             "an invalid IMU noise value was accepted")) {
    return EXIT_FAILURE;
  }
  invalid_json = valid_json;
  invalid_json["pose_config_"][0]["rot_q_BP_init"] = {
      {"x", 0.}, {"y", 0.}, {"z", 0.}, {"w", 0.}};
  if (!Check(rejects_config(invalid_json),
             "a zero-norm config quaternion was accepted")) {
    return EXIT_FAILURE;
  }
  invalid_json = valid_json;
  invalid_json["spline_knot_interval_"] = "invalid";
  if (!Check(rejects_config(invalid_json),
             "a JSON field with the wrong type was accepted")) {
    return EXIT_FAILURE;
  }

  const auto malformed_path = output_dir / "malformed.json";
  {
    std::ofstream malformed_file(malformed_path);
    malformed_file << "{ invalid JSON";
  }
  auto malformed_config = hpgt::SystemConfig::Create();
  if (!Check(!malformed_config->FromJson(malformed_path.string()),
             "malformed JSON was accepted")) {
    return EXIT_FAILURE;
  }

  invalid_json = valid_json;
  invalid_json["pose_config_"].push_back(invalid_json["pose_config_"][0]);
  invalid_json["pose_config_"].back()["file_name"] = "";
  invalid_json["imu_config_"].push_back(invalid_json["imu_config_"][0]);
  invalid_json["imu_config_"].back()["file_name"] = "";
  const auto empty_nodes_path = output_dir / "empty_nodes.json";
  auto empty_nodes_config = hpgt::SystemConfig::Create();
  if (!Check(WriteJson(empty_nodes_path, invalid_json) &&
                 empty_nodes_config->FromJson(empty_nodes_path.string()) &&
                 empty_nodes_config->get_pose_config().size() ==
                     valid_json["pose_config_"].size() &&
                 empty_nodes_config->get_imu_config().size() ==
                     valid_json["imu_config_"].size(),
             "empty file-name config nodes were not removed")) {
    return EXIT_FAILURE;
  }

  const size_t original_pose_count = system_config_i->get_pose_config().size();
  const double original_knot_interval =
      system_config_i->get_spline_knot_interval();
  if (!Check(!system_config_i->FromJson(malformed_path.string()) &&
                 system_config_i->get_pose_config().size() ==
                     original_pose_count &&
                 system_config_i->get_spline_knot_interval() ==
                     original_knot_interval,
             "failed config parsing changed the existing config object")) {
    return EXIT_FAILURE;
  }

  if (!Check(!system_config_i->ToJson(
                 (output_dir / "missing_directory" / "config.json").string()),
             "writing config to an invalid path succeeded")) {
    return EXIT_FAILURE;
  }

  // ===========================================================================

  spdlog::info("Test complete. ");

  return 0;
}
