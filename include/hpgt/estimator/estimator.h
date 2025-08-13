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
#include <memory>
#include <string>

#include <ceres/ceres.h>

#include "hpgt/config/system_config.h"
#include "hpgt/estimator/calib_parameter.h"
#include "hpgt/sensor_data/sensor_data_manager.h"
#include "hpgt/spline/spline_bundle.hpp"
// clang-format on

namespace hpgt {

// Forward declaration.
class FactorAdder;

/**
 * @brief Class for estimating states for multi-sensor calibration and
 * trajectory optimization. Capable of fusing any number and combination of
 * IMU and pose sequences, with optimization performed in a continuous-time
 * framework.
 */
class Estimator {
 public:
  using Ptr = std::shared_ptr<Estimator>;

  // Type definitions.
  using SplineBundleType = SplineBundle<SystemConfig::kSplineOrder>;
  using SplineMetaType = SplineMeta<SystemConfig::kSplineOrder>;

  // Construct estimator with system config and sensor data manager.
  Estimator(SystemConfig::Ptr system_config,
            SensorDataManager::Ptr sensor_data_manager);

  // Create an instance of estimator.
  static Estimator::Ptr Create(SystemConfig::Ptr system_config,
                               SensorDataManager::Ptr sensor_data_manager) {
    return Ptr(new Estimator(system_config, sensor_data_manager));
  }

  /**
   * @brief Initialize all states to be optimized in our system, including
   * time-invariant calibration parameters and time-varying spline functions.
   *
   * @param[in] time_margin Time margin for constructing spline fuctions.
   * @return True if the system states are initialized.
   */
  bool Initialize(double time_margin = 0.01);

  /**
   * @brief Add optimization factors, construct the problem and perform
   * nonlinear batch optimization.
   *
   * @return True if the build and optimization can be completed.
   */
  bool BuildAndOptimize();

  /**
   * @brief Print calibration parameter results and save the results and the
   * system trajectory (T_GB) to the specific paths.
   *
   * @param output_calib_path File path where the calibration parameter
   * results will be saved.
   * @param output_traj_path File path where the trajectory results will be
   * saved.
   * @return True if we can save the result.
   */
  bool PrintAndSaveResult(const std::string& output_calib_path,
                          const std::string& output_traj_path);

 private:
  // Friend class used to add optimization factors to the current estimator's
  // problem.
  friend class FactorAdder;

  // Ceres problem.
  ceres::Problem problem_;

  // For getting config parameters.
  SystemConfig::Ptr system_config_;
  // For retrieving sensor measurements.
  SensorDataManager::Ptr sensor_data_manager_;

  // For managing calibration parameters to be optimized.
  CalibParameter::Ptr calib_parameter_;
  // For managing system splines to be optimized.
  SplineBundleType::Ptr spline_bundle_;

  // Spline name.
  std::string trans_spline_name_;
  std::string rot_spline_name_;
  std::string spline_sensor_label_;
  // Start / end times for system optimization (i.e., start / end times of
  // splines).
  double opt_start_time_;
  double opt_end_time_;
};

}  // namespace hpgt
