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

#include <fstream>
#include <iomanip>

#include "hpgt/initializer/time_offset_initializer.hpp"
#include "hpgt/sensor_data/sensor_data_manager.h"

int main(int argc, char **argv) {
  spdlog::set_level(spdlog::level::info);

  // ===========================================================================

  // Step 1: Parse input arguments.
  std::string input_path_1;
  std::string input_path_2;
  std::string output_path;
  std::string exe_mode = "p2i";

  if (argc != 4 && argc != 5) {
    // clang-format off
    spdlog::critical(
        "Invalid input parameters.\n\n"
        "Usage:\n"
        "  {} <input_path_1> <input_path_2> <output_path> [mode: p2i | p2p]\n\n"
        "Description:\n"
        "  - p2i: <input_path_1> is an IMU file and <input_path_2> is a pose file\n"
        "  - p2p: both input paths are pose files\n"
        "  - [mode] defaults to 'p2i' when omitted\n"
        "  - The result is added to timestamps of the second sequence to align\n"
        "    them with the first sequence's clock\n\n"
        "Examples:\n"
        "  {} ./imu.txt ./pose.txt ./time_offset.txt\n"
        "  {} ./pose_1.txt ./pose_2.txt ./time_offset.txt p2p",
        argv[0], argv[0], argv[0]);
    // clang-format on

    std::exit(EXIT_FAILURE);
  }

  input_path_1 = argv[1];
  input_path_2 = argv[2];
  output_path = argv[3];

  if (argc == 5) {
    exe_mode = argv[4];
  }

  // ===========================================================================

  // Step 2: Compute time offset according to input data types.
  double toff_1_2 = 0.;
  auto toff_initializer = hpgt::TimeOffsetInitializer::Create();

  if (exe_mode == "p2p") {
    hpgt::PoseSequence seq_1;
    hpgt::PoseSequence seq_2;

    if (hpgt::PoseDataLoader::Load(input_path_1, seq_1)) {
      spdlog::info("Loaded {} pose frames from data path 1. ", seq_1.size());
    } else {
      std::exit(EXIT_FAILURE);
    }

    if (hpgt::PoseDataLoader::Load(input_path_2, seq_2)) {
      spdlog::info("Loaded {} pose frames from data path 2. ", seq_2.size());
    } else {
      std::exit(EXIT_FAILURE);
    }

    if (!toff_initializer->EstimateToffFromSeq(seq_1, seq_2, toff_1_2)) {
      spdlog::critical(
          "Failed to estimate the time offset between the pose sequences.");
      std::exit(EXIT_FAILURE);
    }

  } else if (exe_mode == "p2i") {
    hpgt::ImuSequence imu_seq;
    hpgt::PoseSequence pose_seq;

    if (hpgt::ImuDataLoader::Load(input_path_1, imu_seq)) {
      spdlog::info("Loaded {} IMU frames from data path 1.", imu_seq.size());
    } else {
      std::exit(EXIT_FAILURE);
    }

    if (hpgt::PoseDataLoader::Load(input_path_2, pose_seq)) {
      spdlog::info("Loaded {} pose frames from data path 2.", pose_seq.size());
    } else {
      std::exit(EXIT_FAILURE);
    }

    if (!toff_initializer->EstimateToffFromSeq(imu_seq, pose_seq, toff_1_2)) {
      spdlog::critical(
          "Failed to estimate the time offset between the IMU and pose "
          "sequences.");
      std::exit(EXIT_FAILURE);
    }

  } else {
    spdlog::critical("Invalid mode '{}'. Only 'p2p' and 'p2i' are supported. ",
                     exe_mode);

    std::exit(EXIT_FAILURE);
  }

  spdlog::info(
      "Estimated time offset (toff_1_2): {}. Add this offset to the timestamps "
      "of the second sequence to align them with the clock of the first "
      "sequence.",
      toff_1_2);

  // ===========================================================================

  // Step 3: Save the result.
  std::ofstream out_file(output_path);
  if (!out_file.is_open()) {
    spdlog::critical("Failed to open output file: {}", output_path);
    std::exit(EXIT_FAILURE);
  }

  out_file << "toff_1_2: " << std::setprecision(12) << toff_1_2 << "\n";
  if (!out_file) {
    spdlog::critical("Failed to write the time-offset result: {}", output_path);
    std::exit(EXIT_FAILURE);
  }
  out_file.close();

  spdlog::info("Time offset saved to '{}'.", output_path);

  // ===========================================================================

  return 0;
}
