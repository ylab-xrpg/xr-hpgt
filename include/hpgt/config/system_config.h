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

#include <spdlog/spdlog.h>

#include "hpgt/config/imu_data_config.h"
#include "hpgt/config/pose_data_config.h"
// clang-format on

namespace hpgt {

/**
 * @brief Class for managing system config.
 *
 * The frame we have defined is as follows:
 * - (G) denote the world frame of our system, referenced to the world frame of
 * a pose sensor and gravity-aligned.
 * - (B) denote the body frame of our system, which is aligned to any of the
 * specified input sensors. Our system will output the ground truth trajectory
 * referenced to B, denotes as T_G_B.
 *
 * - (Pi) denote the body frame of the i-th pose sensor.
 * - (Wi) denote the world frame of the i-th pose sensor.
 * - (Ii) denote the body frame of the i-th IMU.
 *
 * The Output trajectory timestamp also references the clock of the body
 * frame. The timestamp conversion between different clock is given by: t_Ii =
 * toff_B_Ii + tau_Ii, where tau_Ii is the timestamp of the i-th sensor's
 * measurements in its own clock, toff_B_Ii is the time offset from the i-th
 * sensor's clock to the body frame's clock, t_Ii is the timestamp of the i-th
 * sensor's measurements in the body frame's clock.
 */
class SystemConfig {
 public:
  using Ptr = std::shared_ptr<SystemConfig>;
  static Ptr Create() { return Ptr(new SystemConfig()); }

  // Deserialize from JSON
  bool FromJson(const std::string &input_path);

  // Serialize to JSON
  bool ToJson(const std::string &output_path);

  // Our system uses cubic B-spline by default, which are 4th-order and ensure
  // 2nd-order kinematic continuity.
  static constexpr int kSplineOrder = 4;

  // Accessors for optimization options.
  const double &get_spline_knot_interval() const {
    return spline_knot_interval_;
  }

  const double &get_max_toff_change() const { return max_toff_change_; }

  const double &get_gravity_magnitude() const { return gravity_magnitude_; }

  const double &get_output_frequency() const { return output_frequency_; }

  const bool &get_opt_temporal_param_flag() const {
    return opt_temporal_param_flag_;
  }

  const bool &get_opt_spatial_param_flag() const {
    return opt_spatial_param_flag_;
  }

  // Accessors for sensor data config.
  const std::vector<PoseDataConfig> &get_pose_config() const {
    return pose_config_;
  }

  std::vector<PoseDataConfig> &get_pose_config() { return pose_config_; }

  const std::vector<ImuDataConfig> &get_imu_config() const {
    return imu_config_;
  }

  std::vector<ImuDataConfig> &get_imu_config() { return imu_config_; }

  // Accessors for sensor data directory.
  void set_data_dir(const std::string &data_dir) { data_dir_ = data_dir; }

  const std::string &get_data_dir() { return data_dir_; }

  // Intrusive json serialization.
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SystemConfig, spline_knot_interval_,
                                 max_toff_change_, gravity_magnitude_,
                                 output_frequency_, opt_temporal_param_flag_,
                                 opt_spatial_param_flag_, pose_config_,
                                 imu_config_);

 private:
  bool CheckAndPrintConfig();

  // Optimization options.
  double spline_knot_interval_ = 0.01;
  double max_toff_change_ = 0.1;
  double gravity_magnitude_ = 9.8;
  double output_frequency_ = 100.;
  bool opt_temporal_param_flag_ = true;
  bool opt_spatial_param_flag_ = true;

  // Sensor data config.
  std::vector<PoseDataConfig> pose_config_;
  std::vector<ImuDataConfig> imu_config_;

  // Sensor data directory.
  std::string data_dir_ = "";
};

}  // namespace hpgt
