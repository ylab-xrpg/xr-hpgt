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

#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"
#include "hpgt/sensor_data/imu_data.h"
#include "hpgt/spline/ceres_spline_helper_jet.hpp"
#include "hpgt/spline/spline_meta.hpp"

namespace hpgt {

// Structure for accelerometer measurement factor.
template <int Order>
struct ImuAccFactor {
 public:
  /**
   * @brief Construct a accelerometer factor object.
   *
   * @param[in] trans_meta Meta data for the translation spline.
   * @param[in] rot_meta Meta data for the rotation spline.
   * @param[in] imu_frame IMU frame with measurement information.
   * @param[in] toff_base Initial guesses for time offset for incremental
   * optimization.
   * @param[in] weight Residual weight.
   * @param[in] g_magnitude Gravity magnitude.
   */
  ImuAccFactor(const SplineMeta<Order> &trans_meta,
               const SplineMeta<Order> &rot_meta, const ImuFrame::Ptr imu_frame,
               const double &toff_base, const double &weight,
               const double &g_magnitude)
      : trans_meta_(trans_meta),
        rot_meta_(rot_meta),
        imu_frame_(imu_frame),
        toff_base_(toff_base),
        weight_(weight),
        trans_dt_inv_(1. / trans_meta.segments.front().knot_interval),
        rot_dt_inv_(1. / rot_meta.segments.front().knot_interval),
        gravity_magnitude_(g_magnitude) {}

  // Create an instance of accelerometer factor wrapped in a dynamic auto-diff
  // cost function.
  static auto Create(const SplineMeta<Order> &trans_meta,
                     const SplineMeta<Order> &rot_meta,
                     const ImuFrame::Ptr imu_frame, const double &toff_base,
                     const double &weight, const double &g_magnitude) {
    return new ceres::DynamicAutoDiffCostFunction<ImuAccFactor>(
        new ImuAccFactor(trans_meta, rot_meta, imu_frame, toff_base, weight,
                         g_magnitude));
  }

  /**
   * @brief Overload operator, define the residual calculation method and bind
   * parameter blocks.
   *
   * The parameters blocks include:
   * [ trans_GB_knot | ... | trans_GB_knot | rot_GB_knot | ... | rot_GB_knot |
   *   toff_BI | trans_BI | rot_BI | acc_bias | acc_map_coeff ]
   *
   * @tparam T Type of the parameter. (compatible with ceres::Jet for automatic
   * differentiation)
   * @param params Pointer to the parameter blocks.
   * @param residuals Pointer to the output residuals.
   * @return True if the calculation is completed.
   */
  template <class T>
  bool operator()(T const *const *params, T *residuals) const {
    // Determine the address offsets.
    size_t TRANS_KNOT_OFFSET;
    size_t ROT_KNOT_OFFSET;
    size_t TOFF_BI_OFFSET =
        trans_meta_.NumParameters() + rot_meta_.NumParameters();
    size_t TRANS_BI_OFFSET = TOFF_BI_OFFSET + 1;
    size_t ROT_BI_OFFSET = TRANS_BI_OFFSET + 1;
    size_t ACC_BIAS_OFFSET = ROT_BI_OFFSET + 1;
    size_t ACC_MAP_COEFF_OFFSET = ACC_BIAS_OFFSET + 1;

    // Get the values for time-invariant parameters.
    T toff_BI = T(toff_base_) + params[TOFF_BI_OFFSET][0];
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_BI(params[TRANS_BI_OFFSET]);
    Eigen::Map<const Sophus::SO3<T>> rot_BI(params[ROT_BI_OFFSET]);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> acc_bias(params[ACC_BIAS_OFFSET]);

    auto acc_map_coeff = params[ACC_MAP_COEFF_OFFSET];
    Eigen::Matrix<T, 3, 3> acc_map_mat = Eigen::Matrix<T, 3, 3>::Zero();
    acc_map_mat.diagonal() =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(acc_map_coeff, 3);
    acc_map_mat(0, 1) = *(acc_map_coeff + 3);
    acc_map_mat(0, 2) = *(acc_map_coeff + 4);
    acc_map_mat(1, 2) = *(acc_map_coeff + 5);

    // Determine the address offset for spline knots.
    T t_I = toff_BI + T(imu_frame_->timestamp);
    // For measurements outside the B-spline time range, we simply set their
    // gradients to zero.
    size_t trans_spline_index, rot_spline_index;
    T trans_spline_fraction, rot_spline_fraction;
    if (!trans_meta_.ComputeSplineIndex(t_I, trans_spline_index,
                                        trans_spline_fraction) ||
        !rot_meta_.ComputeSplineIndex(t_I, rot_spline_index,
                                      rot_spline_fraction)) {
      Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
      res.setZero();
      return true;
    }

    TRANS_KNOT_OFFSET = trans_spline_index;
    ROT_KNOT_OFFSET = trans_meta_.NumParameters() + rot_spline_index;

    // Evaluate pose and inertial information from the spline.
    Eigen::Matrix<T, 3, 1> trans_acc_GB;
    CeresSplineHelperJet<T, Order>::template Evaluate<3, 2>(
        params + TRANS_KNOT_OFFSET, trans_spline_fraction, trans_dt_inv_,
        &trans_acc_GB);

    Sophus::SO3<T> rot_GB;
    typename Sophus::SO3<T>::Tangent rot_vel_BB, rot_acc_BB;
    CeresSplineHelperJet<T, Order>::template EvaluateLie<Sophus::SO3>(
        params + ROT_KNOT_OFFSET, rot_spline_fraction, rot_dt_inv_, &rot_GB,
        &rot_vel_BB, &rot_acc_BB);
    // typename Sophus::SO3<T>::Tangent rot_vel_GB = rot_GB * rot_vel_BB;
    // typename Sophus::SO3<T>::Tangent rot_acc_GB = rot_GB * rot_acc_BB;

    // Calculate the intermediate variable.
    Eigen::Matrix<T, 3, 1> gravity(T(0.), T(0.), T(gravity_magnitude_));
    Eigen::Matrix<T, 3, 3> rot_vel_BB_hat = Sophus::SO3<T>::hat(rot_vel_BB);
    Eigen::Matrix<T, 3, 3> rot_acc_BB_hat = Sophus::SO3<T>::hat(rot_acc_BB);
    // Eigen::Matrix<T, 3, 3> rot_vel_BB_hat = Sophus::SO3<T>::hat(rot_vel_GB);
    // Eigen::Matrix<T, 3, 3> rot_acc_BB_hat = Sophus::SO3<T>::hat(rot_acc_GB);
    Eigen::Matrix<T, 3, 1> trans_acc_GI =
        trans_acc_GB + rot_GB.matrix() *
                           (rot_acc_BB_hat + rot_vel_BB_hat * rot_vel_BB_hat) *
                           trans_BI;
    // Eigen::Matrix<T, 3, 1> trans_acc_GI =
    //     trans_acc_GB + (rot_acc_BB_hat + rot_vel_BB_hat * rot_vel_BB_hat) *
    //                        (rot_GB.matrix() * trans_BI);

    // Calculate the predicted values.
    Eigen::Matrix<T, 3, 1> acc_pred =
        acc_map_mat * ((rot_GB * rot_BI).inverse() * (trans_acc_GI + gravity)) +
        acc_bias;

    // Get the measurements.
    Eigen::Matrix<T, 3, 1> acc_meas = imu_frame_->acc.template cast<T>();

    // Calculates the residuals based on the difference between the measurements
    // and predicted values.
    Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
    res = T(weight_) * (acc_pred - acc_meas);

    return true;
  }

 private:
  // Spline meta data.
  SplineMeta<Order> trans_meta_, rot_meta_;
  // IMU frame with measurement information.
  ImuFrame::Ptr imu_frame_;

  // Initial guesses for time offset for incremental optimization.
  double toff_base_;
  // Residual weights.
  double weight_;
  // Inverses of the time intervals for splines.
  double trans_dt_inv_, rot_dt_inv_;
  // Gravity magnitude_.
  double gravity_magnitude_;
};

}  // namespace hpgt
