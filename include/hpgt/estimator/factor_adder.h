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
#include <string>

#include "hpgt/estimator/estimator.h"

namespace hpgt {

/**
 * @brief Friend class of the estimator, used to add ceres factors to its
 * optimization problem.
 */
class FactorAdder {
 public:
  using Ptr = std::shared_ptr<FactorAdder>;

  // Type definitions.
  using SplineBundleType = Estimator::SplineBundleType;
  using SplineMetaType = Estimator::SplineMetaType;

  // Construct factor adder with the estimator.
  explicit FactorAdder(Estimator* estimator);

  // Create an instance of factor adder.
  static FactorAdder::Ptr Create(Estimator* estimator) {
    return Ptr(new FactorAdder(estimator));
  }

  /**
   * @brief Add a factor to the problem based on an absolute pose measurement.
   *
   * @param[in] label The identifier for the pose sequence.
   * @param[in] pose_frame Pose frame containing measurement information.
   * @param[in] trans_weight Weight for translational measurement.
   * @param[in] rot_weight Weight for rotational measurement.
   * @return True if the factor was successfully added.
   */
  bool AddAbsPoseFactor(const std::string& label,
                        const PoseFrame::Ptr& pose_frame,
                        const double& trans_weight, const double& rot_weight);

  /**
   * @brief Add a factor to the problem based on an accelerometer measurement.
   *
   * @param[in] label The identifier for the IMU sequence.
   * @param[in] imu_frame IMU frame containing measurement information.
   * @param[in] model_type Type of IMU model.
   * @param[in] acc_weight Residual weight for acceleration measurement.
   * @return True if the factor was successfully added.
   */
  bool AddImuAccFactor(const std::string& label, const ImuFrame::Ptr& imu_frame,
                       const ImuModelType& model_type,
                       const double& acc_weight);

  /**
   * @brief Add a factor to the problem based on a gyroscope measurement.
   *
   * @param[in] label The identifier for the IMU sequence.
   * @param[in] imu_frame IMU frame containing measurement information.
   * @param[in] model_type Type of IMU model.
   * @param[in] gyr_weight Residual weight for gyroscope measurement.
   * @return True if the factor was successfully added.
   */
  bool AddImuGyrFactor(const std::string& label, const ImuFrame::Ptr& imu_frame,
                       const ImuModelType& model_type,
                       const double& gyr_weight);

  /**
   * @brief Add prior constraints on relative poses to adjacent knots of the B-
   * spline. We attempt to fix the relative poses of the knots near the start
   * and end points of the B-spline to their initial states to prevent them
   * from diverging due to insufficient constraints.
   *
   * @param index_i Index of the previous pose of adjacent poses
   * @param index_j Index of the subsequent pose of adjacent poses
   * @param[in] trans_weight Weight of the translational component of the prior.
   * @param[in] rot_weight Weight of the rotational component of the prior.
   * @return True if the factor was successfully added.
   */
  bool AddSplineKnotPriorFactor(const int& index_i, const int& index_j,
                                const double& trans_weight,
                                const double& rot_weight);

  /**
   * @brief Given a timestamp of sensor measurement (in the body frame's clock),
   * return the time range of B-spline meta data. Ensure B-spline data
   * corresponding to sensor measurements can be obtained when the time offset
   * changes during the optimization.
   *
   * @param[in] meas_time Timestamp of sensor measurement.
   * @param[in] min_time Minimum timestamp of meta data.
   * @param[in] max_time Maximum timestamp of meta data.
   * @return True if the maximum and minimum timestamps of the metadata can be
   * calculated.
   */
  bool CalMetaMinMaxTime(const double& meas_time, double& min_time,
                         double& max_time);

  /**
   * @brief Add translational spline knot data to the parameter block vector.
   *
   * @param[in] param_block_vector A vector to store pointers to the parameter
   * blocks.
   * @param[in] spline The translational spline object containing knot data.
   * @param[in] spline_meta Spline meta data to be added.
   */
  void AddR3dKnotData(std::vector<double*>& param_block_vector,
                      const SplineBundleType::R3dSplineType& spline,
                      const SplineMetaType& spline_meta);

  /**
   * @brief Add rotational spline knot data to the parameter block vector.
   *
   * @param[in] param_block_vector A vector to store pointers to the parameter
   * blocks.
   * @param[in] spline The rotational spline object containing knot data.
   * @param[in] spline_meta Spline meta data to be added.
   */
  void AddSo3dKnotsData(std::vector<double*>& param_block_vector,
                        const SplineBundleType::So3dSplineType& spline,
                        const SplineMetaType& spline_meta);

 private:
  // Quaterniond manifold for ceres.
  static std::shared_ptr<ceres::EigenQuaternionManifold> kQuatManifold_;

  // The estimator to which we are adding the factor.
  Estimator* estimator_;

  // Copy of the member of the estimator class.
  ceres::Problem& problem_;
  SystemConfig::Ptr system_config_;
  CalibParameter::Ptr calib_parameter_;
  SplineBundleType::Ptr spline_bundle_;

  std::string trans_spline_name_;
  std::string rot_spline_name_;
  double opt_start_time_;
  double opt_end_time_;
};

}  // namespace hpgt
