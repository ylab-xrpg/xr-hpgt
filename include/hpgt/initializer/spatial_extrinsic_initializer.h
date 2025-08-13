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

#include <memory>

#include "hpgt/initializer/i2p_extrinsic_initializer.h"
#include "hpgt/initializer/p2p_extrinsic_initializer.h"

namespace hpgt {

/**
 * @brief class for initializing spatial extrinsic parameters, capable of
 * solving for any number and combination of pose and IMU sequences.
 */
class SpatialExtrinsicInitializer {
 public:
  using Ptr = std::shared_ptr<SpatialExtrinsicInitializer>;

  // Create an instance of spatial extrinsic parameters initializer.
  static SpatialExtrinsicInitializer::Ptr Create() {
    return Ptr(new SpatialExtrinsicInitializer);
  }

  /**
   * @brief Initialize spatial extrinsic parameters.
   *
   * We first compute the spatial extrinsic parameters of the sensor as the
   * world frame with respect to all other sensors. Then, we propagate to get
   * the transformation to the body frame and align the world frame to gravity.
   * If the spatial extrinsic parameters needs to be fixed, we only determine
   * the world frame and exit.
   *
   * @param[in] data_manager Sensor data manager.
   * @param[out] calib_param Calibration parameter results.
   * @param[in] g_magnitude Gravity magnitude.
   * @param[in] opt_flag Whether to fix the spatial extrinsic parameters.
   * @return True if the spatial parameters are initialized.
   */
  bool Initialize(const SensorDataManager::Ptr &data_manager,
                  const CalibParameter::Ptr &calib_param,
                  const double &g_magnitude, const bool &opt_flag);
};

}  // namespace hpgt
