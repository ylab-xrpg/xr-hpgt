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
#include <map>
#include <memory>
#include <string>

#include <Eigen/Eigen>
#include <sophus/se3.hpp>

#include <hpgt/config/config_parser.h>
// clang-format on

namespace hpgt {

/**
 * @brief Structure for IMU intrinsic parameters.
 */
struct IMUIntrinsic {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<IMUIntrinsic>;

  // Construct a IMU intrinsic object.
  IMUIntrinsic() {
    acc_bias = Sophus::Vector3d::Zero();
    gyr_bias = Sophus::Vector3d::Zero();

    // Initialize mapping coefficients to identity values.
    acc_map_coeff = Sophus::Vector6d::Zero();
    acc_map_coeff(0) = 1.0;
    acc_map_coeff(1) = 1.0;
    acc_map_coeff(2) = 1.0;

    gyr_map_coeff = Sophus::Vector6d::Zero();
    gyr_map_coeff(0) = 1.0;
    gyr_map_coeff(1) = 1.0;
    gyr_map_coeff(2) = 1.0;

    rot_gyr_acc = Sophus::SO3d();
  }

  // Create an instance of IMU intrinsic.
  static IMUIntrinsic::Ptr Create() { return Ptr(new IMUIntrinsic); }

  // Bias for IMU measurements.
  Sophus::Vector3d acc_bias;
  Sophus::Vector3d gyr_bias;

  // Coefficient of the upper triangular mapping matrices (Contains scale and
  // axis non-orthogonality information).
  Sophus::Vector6d acc_map_coeff;
  Sophus::Vector6d gyr_map_coeff;

  // Rotational misalignment from accelerometer to gyroscope.
  Sophus::SO3d rot_gyr_acc;
};

/**
 * @brief Structure for spatiotemporal calibration parameters.
 *
 * The detailed coordinate definition is available in the `system_config.h`
 * file. In our system, we define the body frame as B and the gravity-aligned
 * world frame as G. We use this as a reference for estimating the
 * spatiotemporal calibration parameters of all other sensors.
 *
 * For extrinsic translation, trans_B_A represents A in B's frame.
 * For extrinsic rotation, rot_B_A denotes the transformation from A to B. For
 * time offset, toff_B_A converts the timestamp from A's clock to B's clock.
 *
 * The calibration parameters are indexed by sensor labels.
 */
struct CalibParameter {
 public:
  using Ptr = std::shared_ptr<CalibParameter>;
  static CalibParameter::Ptr Create() { return Ptr(new CalibParameter()); };

  // Basic information.
  std::string body_frame_label = "";
  std::string world_frame_label = "";
  SensorType body_sensor_type = SensorType::kInvalid;
  bool gravity_aligned = false;

  // Time offset.
  std::map<std::string, double> toff_B_Ii;
  std::map<std::string, double> toff_B_Pi;
  // We optimize the increment based on the initial guesses of time offset to
  // avoid instability caused by an excessively large time offset.
  std::map<std::string, double> toff_B_Ii_increment;
  std::map<std::string, double> toff_B_Pi_increment;

  // Extrinsic translation.
  std::map<std::string, Sophus::Vector3d> trans_B_Ii;
  std::map<std::string, Sophus::Vector3d> trans_B_Pi;

  // Extrinsic rotation.
  std::map<std::string, Sophus::SO3d> rot_B_Ii;
  std::map<std::string, Sophus::SO3d> rot_B_Pi;

  // World frame alignment.
  std::map<std::string, Sophus::Vector3d> trans_G_Wi;
  std::map<std::string, Sophus::SO3d> rot_G_Wi;

  // IMU intrinsic parameters.
  std::map<std::string, IMUIntrinsic::Ptr> imu_intri;
};

}  // namespace hpgt
