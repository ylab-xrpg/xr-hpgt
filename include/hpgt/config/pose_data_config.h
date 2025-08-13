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

// Class for managing the Pose data config.
struct PoseDataConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Data file name
  std::string file_name = "";

  // Basic sensor properties.
  SensorType sensor_type = SensorType::kPose;
  bool body_frame_flag = false;
  bool world_frame_flag = false;
  bool abs_pose_flag = false;

  // Noise standard deviation in discrete time.
  double trans_noise = 5e-4;
  double rot_noise = 5e-3;

  // Initial values of spatiotemporal calibration parameters.
  double toff_BP_init = 0.;
  Eigen::Vector3d trans_BP_init = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot_q_BP_init = Eigen::Quaterniond::Identity();
  Eigen::Vector3d trans_GW_init = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot_q_GW_init = Eigen::Quaterniond::Identity();

  // Intrusive json serialization.
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(PoseDataConfig, file_name, body_frame_flag,
                                 world_frame_flag, abs_pose_flag, trans_noise,
                                 rot_noise, toff_BP_init, trans_BP_init,
                                 rot_q_BP_init, trans_GW_init, rot_q_GW_init);
};

}  // namespace hpgt
