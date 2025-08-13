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
#include "hpgt/sensor_data/pose_data.h"
#include "hpgt/spline/ceres_spline_helper_jet.hpp"
#include "hpgt/spline/spline_meta.hpp"

namespace hpgt {

// Structure for absolute pose measurement factor.
template <int Order>
struct AbsPoseFactor {
 public:
  /**
   * @brief Construct an absolute pose factor object.
   *
   * @param[in] trans_meta Meta data for the translation spline.
   * @param[in] rot_meta Meta data for the rotation spline.
   * @param[in] pose_frame Pose frame with measurement information.
   * @param[in] toff_base Initial guesses for time offset for incremental
   * optimization.
   * @param[in] trans_weight Residual weight for the translation component.
   * @param[in] rot_weight Residual weight for the rotation component.
   */
  AbsPoseFactor(const SplineMeta<Order> &trans_meta,
                const SplineMeta<Order> &rot_meta,
                const PoseFrame::Ptr &pose_frame, const double &toff_base,
                const double &trans_weight, const double &rot_weight)
      : trans_meta_(trans_meta),
        rot_meta_(rot_meta),
        pose_frame_(pose_frame),
        toff_base_(toff_base),
        trans_weight_(trans_weight),
        rot_weight_(rot_weight),
        trans_dt_inv_(1. / trans_meta.segments.front().knot_interval),
        rot_dt_inv_(1. / rot_meta.segments.front().knot_interval) {}

  // Create an instance of absolute pose factor wrapped in a dynamic auto-diff
  // cost function.
  static auto Create(const SplineMeta<Order> &trans_meta,
                     const SplineMeta<Order> &rot_meta,
                     const PoseFrame::Ptr &pose_frame, const double &toff_base,
                     const double &trans_weight, const double &rot_weight) {
    return new ceres::DynamicAutoDiffCostFunction<AbsPoseFactor>(
        new AbsPoseFactor(trans_meta, rot_meta, pose_frame, toff_base,
                          trans_weight, rot_weight));
  }

  /**
   * @brief Overload operator, define the residual calculation method and bind
   * parameter blocks.
   *
   * The parameters blocks include:
   * [ trans_GB_knot | ... | trans_GB_knot | rot_GB_knot | ... | rot_GB_knot |
   *   toff_BP | trans_BP | rot_BP | trans_GW | rot_GW ]
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
    size_t TOFF_BP_OFFSET =
        trans_meta_.NumParameters() + rot_meta_.NumParameters();
    size_t TRANS_BP_OFFSET = TOFF_BP_OFFSET + 1;
    size_t ROT_BP_OFFSET = TRANS_BP_OFFSET + 1;
    size_t TRANS_GW_OFFSET = ROT_BP_OFFSET + 1;
    size_t ROT_GW_OFFSET = TRANS_GW_OFFSET + 1;

    // Get the values for time-invariant parameters.
    T toff_BP = T(toff_base_) + params[TOFF_BP_OFFSET][0];
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_BP(params[TRANS_BP_OFFSET]);
    Eigen::Map<const Sophus::SO3<T>> rot_BP(params[ROT_BP_OFFSET]);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_GW(params[TRANS_GW_OFFSET]);
    Eigen::Map<const Sophus::SO3<T>> rot_GW(params[ROT_GW_OFFSET]);
    Eigen::Matrix<T, 3, 1> trans_WG = -(rot_GW.inverse() * trans_GW);
    Sophus::SO3<T> rot_WG = rot_GW.inverse();

    // Determine the address offset for spline knots.
    T t_P = toff_BP + T(pose_frame_->timestamp);
    // For measurements outside the B-spline time range, we simply set their
    // gradients to zero.
    size_t trans_spline_index, rot_spline_index;
    T trans_spline_fraction, rot_spline_fraction;
    if (!trans_meta_.ComputeSplineIndex(t_P, trans_spline_index,
                                        trans_spline_fraction) ||
        !rot_meta_.ComputeSplineIndex(t_P, rot_spline_index,
                                      rot_spline_fraction)) {
      Eigen::Map<Eigen::Matrix<T, 6, 1>> res(residuals);
      res.setZero();
      return true;
    }

    TRANS_KNOT_OFFSET = trans_spline_index;
    ROT_KNOT_OFFSET = trans_meta_.NumParameters() + rot_spline_index;

    // Evaluate translation and rotation from the splines.
    Eigen::Matrix<T, 3, 1> trans_GB;
    CeresSplineHelperJet<T, Order>::template Evaluate<3, 0>(
        params + TRANS_KNOT_OFFSET, trans_spline_fraction, trans_dt_inv_,
        &trans_GB);

    Sophus::SO3<T> rot_GB;
    CeresSplineHelperJet<T, Order>::template EvaluateLie<Sophus::SO3>(
        params + ROT_KNOT_OFFSET, rot_spline_fraction, rot_dt_inv_, &rot_GB);

    // Calculate the predicted values.
    Eigen::Matrix<T, 3, 1> trans_WP_pred =
        rot_WG * rot_GB * trans_BP + rot_WG * trans_GB + trans_WG;
    Sophus::SO3<T> rot_WP_pred = rot_WG * rot_GB * rot_BP;

    // Get the measurements.
    Eigen::Matrix<T, 3, 1> trans_WP_meas =
        pose_frame_->trans.template cast<T>();
    Sophus::SO3<T> rot_WP_meas(pose_frame_->rot_q.template cast<T>());

    // Calculates the residuals based on the difference between the measurements
    // and predicted values.
    Eigen::Map<Eigen::Matrix<T, 6, 1>> res(residuals);
    res.template head<3>() = T(trans_weight_) * (trans_WP_pred - trans_WP_meas);
    res.template tail<3>() =
        T(rot_weight_) * (rot_WP_meas.inverse() * rot_WP_pred).log();

    return true;
  }

 private:
  // Spline meta data.
  SplineMeta<Order> trans_meta_, rot_meta_;
  // Pose frame with measurement information.
  PoseFrame::Ptr pose_frame_;

  // Initial guesses for time offset for incremental optimization.
  double toff_base_;
  // Residual weights for translation and rotation component.
  double trans_weight_, rot_weight_;
  // Inverses of the time intervals for splines.
  double trans_dt_inv_, rot_dt_inv_;
};

}  // namespace hpgt
