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

// clang-format off
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Eigen>
#include <spdlog/spdlog.h>
// clang-format on

namespace hpgt {

// Structure for Pose data.
struct PoseFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<PoseFrame>;

  PoseFrame(const double &t = -1,
            const Eigen::Vector3d &p = Eigen::Vector3d::Zero(),
            const Eigen::Quaterniond &q = Eigen::Quaterniond::Identity());

  static PoseFrame::Ptr Create(
      const double &t = -1, const Eigen::Vector3d &p = Eigen::Vector3d::Zero(),
      const Eigen::Quaterniond &q = Eigen::Quaterniond::Identity());

  // Timestamp in seconds.
  double timestamp;
  // Measurement of translation and rotation (Hamilton quaternion).
  Eigen::Vector3d trans;
  Eigen::Quaterniond rot_q;
};

using PoseSequence = std::vector<PoseFrame::Ptr>;

class PoseDataLoader {
 public:
  /**
   * @brief Load pose data from file.
   *
   * @param[in] data_path pose data path.
   * @param[out] pose_data Container for storing pose data.
   * @return True if the data is loaded.
   */
  static bool Load(const std::string &data_path, PoseSequence &pose_data);
};

}  // namespace hpgt
