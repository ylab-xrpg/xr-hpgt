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

#include <string>

#include "hpgt/config/config_parser.h"

namespace hpgt {

// Class for managing the IMU data config.
struct ImuDataConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Data file name
  std::string file_name = "";

  // Basic sensor properties.
  SensorType sensor_type = SensorType::kImu;
  ImuModelType model_type = ImuModelType::kCalibrated;
  double frequency = 100.;
  bool body_frame_flag = false;

  // Noise standard deviation in continuous time.
  Eigen::Vector4d noise = Eigen::Vector4d(2e-2, 5e-3, 1e-3, 5e-4);

  // Initial values of spatiotemporal calibration parameters.
  double toff_BI_init = 0.;
  Eigen::Vector3d trans_BI_init = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot_q_BI_init = Eigen::Quaterniond::Identity();

  // Initial value of IMU biases.
  Eigen::Vector3d acc_bias_init = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_bias_init = Eigen::Vector3d::Zero();

  // Intrusive json serialization.
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ImuDataConfig, file_name, model_type,
                                 frequency, body_frame_flag, noise,
                                 toff_BI_init, trans_BI_init, rot_q_BI_init,
                                 acc_bias_init, gyr_bias_init);
};

}  // namespace hpgt
