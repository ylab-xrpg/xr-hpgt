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

// Structure for IMU data.
struct ImuFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<ImuFrame>;

  ImuFrame(const double &t = -1.,
           const Eigen::Vector3d &a = Eigen::Vector3d::Zero(),
           const Eigen::Vector3d &w = Eigen::Vector3d::Zero());

  static ImuFrame::Ptr Create(
      const double &t = -1., const Eigen::Vector3d &a = Eigen::Vector3d::Zero(),
      const Eigen::Vector3d &w = Eigen::Vector3d::Zero());

  // Timestamp in seconds.
  double timestamp;
  // Measurement of accelerometer and gyroscope.
  Eigen::Vector3d acc;
  Eigen::Vector3d gyr;
};

using ImuSequence = std::vector<ImuFrame::Ptr>;

class ImuDataLoader {
 public:
  /**
   * @brief Load IMU data from file.
   *
   * @param[in] data_path IMU data path.
   * @param[out] imu_data Container for storing IMU data.
   * @return True if the data is loaded.
   */
  static bool Load(const std::string &data_path, ImuSequence &imu_data);
};

}  // namespace hpgt
