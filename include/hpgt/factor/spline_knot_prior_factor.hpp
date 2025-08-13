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

#include <Eigen/Eigen>
#include <sophus/se3.hpp>

#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"

namespace hpgt {

// Structure for spline knot prior factor.
struct SplineKnotPriorFactor {
 public:
  /**
   * @brief Construct a spline knot prior factor object.
   *
   * @param[in] delta_trans_meas The translational component of the prior
   * constraint.
   * @param[in] delta_rot_meas The rotational component of the prior constraint.
   * @param[in] trans_weight Residual weight for the translation component.
   * @param[in] rot_weight Residual weight for the rotation component.
   */
  SplineKnotPriorFactor(const Sophus::Vector3d &delta_trans_meas,
                        const Sophus::SO3d &delta_rot_meas,
                        const double &trans_weight, const double &rot_weight)
      : delta_trans_meas_(delta_trans_meas),
        delta_rot_meas_(delta_rot_meas),
        trans_weight_(trans_weight),
        rot_weight_(rot_weight) {}

  // Create an instance of spline knot prior factor wrapped in a dynamic
  // auto-diff cost function.
  static auto Create(const Sophus::Vector3d &delta_trans_meas,
                     const Sophus::SO3d &delta_rot_meas,
                     const double &trans_weight, const double &rot_weight) {
    return new ceres::DynamicAutoDiffCostFunction<SplineKnotPriorFactor>(
        new SplineKnotPriorFactor(delta_trans_meas, delta_rot_meas,
                                  trans_weight, rot_weight));
  }

  /**
   * @brief Overload operator, define the residual calculation method and bind
   * parameter blocks.
   *
   * The parameters blocks include:
   * [ tran_knot_i | rot_knot_i | rot_knot_j | rot_knot_j ]
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
    size_t TRANS_KNOT_I_OFFSET = 0;
    size_t ROT_KNOT_I_OFFSET = TRANS_KNOT_I_OFFSET + 1;
    size_t TRANS_KNOT_J_OFFSET = ROT_KNOT_I_OFFSET + 1;
    size_t ROT_KNOT_J_OFFSET = TRANS_KNOT_J_OFFSET + 1;

    // Obtain the current pose of the knot.
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_knot_i(
        params[TRANS_KNOT_I_OFFSET]);
    Eigen::Map<const Sophus::SO3<T>> rot_knot_i(params[ROT_KNOT_I_OFFSET]);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_knot_j(
        params[TRANS_KNOT_J_OFFSET]);
    Eigen::Map<const Sophus::SO3<T>> rot_knot_j(params[ROT_KNOT_J_OFFSET]);

    // Calculate the predicted values.
    Eigen::Matrix<T, 3, 1> delta_trans_pred =
        rot_knot_i.inverse() * (trans_knot_j - trans_knot_i);
    Sophus::SO3<T> delta_rot_pred = rot_knot_i.inverse() * rot_knot_j;

    // Calculates the residuals based on the difference between the measurements
    // and predicted values.
    Eigen::Map<Eigen::Matrix<T, 6, 1>> res(residuals);
    res.template head<3>() =
        T(trans_weight_) * (delta_trans_pred - delta_trans_meas_);
    res.template tail<3>() =
        T(rot_weight_) * (delta_rot_meas_.inverse() * delta_rot_pred).log();

    return true;
  }

 private:
  // Prior measurements.
  Sophus::Vector3d delta_trans_meas_;
  Sophus::SO3d delta_rot_meas_;

  // Residual weights
  double trans_weight_, rot_weight_;
};

}  // namespace hpgt
