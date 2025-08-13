// Portions of this file are adapted from:
// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Licensed under the BSD-style license found in the original CTraj repository.
//
// ---
//
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

#include <ceres/jet.h>

#include "hpgt/spline/spline_utils.hpp"

namespace hpgt {

/**
 * @brief Helper class for implementing Lie group and Euclidean B-splines in
 * ceres.
 *
 * For more details, please refer to: "Sommer C, et al. Efficient derivative
 * computation for cumulative b-splines on lie groups. CVPR, 2020." or
 * "https://github.com/Unsigned-Long/CTraj"
 *
 * @tparam T Scalar type.
 * @tparam Order B-spline order.
 */
template <class T, int Order>
struct CeresSplineHelperJet {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Order and degree of the spline.
  static constexpr int kN = Order;
  static constexpr int kDeg = Order - 1;

  using MatN = Eigen::Matrix<T, kN, kN>;
  using VecN = Eigen::Matrix<T, kN, 1>;

  /**
   * @brief Calculate the derivatives (kN-dimensional vector) of the time
   * polynomial.
   *
   * @tparam Derivative Order of the derivative to be estimated.
   * @tparam Derived Eigen type to store the results.
   * @param[in] fraction Fractional part to calculate the result.
   * @param[out] result_const Base coefficients result.
   */
  template <int Derivative, class Derived>
  static void BaseCoeffsWithTime(
      const T &fraction, const Eigen::MatrixBase<Derived> &result_const) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, kN)
    auto &result = const_cast<Eigen::MatrixBase<Derived> &>(result_const);

    result.setZero();

    if (Derivative < kN) {
      result[Derivative] = base_coefficients_(Derivative, Derivative);

      T u = fraction;
      for (int j = Derivative + 1; j < kN; j++) {
        result[j] = base_coefficients_(Derivative, j) * u;
        u = u * fraction;
      }
    }
  }

  /**
   * @brief Evaluate Lie group cumulative B-spline and time derivatives.
   *
   * @tparam GroupT Lie group type.
   * @param[in] knots Array of pointers of the spline knots. The size of each
   * knot should be GroupT::num_parameters: 4 for SO(3) and 7 for SE(3).
   * @param[in] fraction Relative position (fractional part) within the current
   * time segment.
   * @param[in] inv_dt Inverse of the time interval in seconds between spline
   * knots.
   * @param[out] value If not nullptr return the value of the spline.
   * @param vel If not nullptr return the velocity (first time derivative) in
   * the body frame.
   * @param[out] acc If not nullptr return the acceleration (second time
   * derivative) in the body frame.
   * @param[out] jerk If not nullptr return the jerk (third time derivative) in
   * the body frame.
   */
  template <template <class> class GroupT>
  static inline void EvaluateLie(T const *const *knots, const T fraction,
                                 const double inv_dt,
                                 GroupT<T> *value = nullptr,
                                 typename GroupT<T>::Tangent *vel = nullptr,
                                 typename GroupT<T>::Tangent *acc = nullptr,
                                 typename GroupT<T>::Tangent *jerk = nullptr) {
    using Group = GroupT<T>;
    using Tangent = typename GroupT<T>::Tangent;
    using Adjoint = typename GroupT<T>::Adjoint;

    VecN base_coeff, coeff, d_coeff, dd_coeff, ddd_coeff;

    BaseCoeffsWithTime<0>(fraction, base_coeff);
    coeff = cumulative_blending_matrix_ * base_coeff;

    if (vel || acc || jerk) {
      BaseCoeffsWithTime<1>(fraction, base_coeff);
      d_coeff = inv_dt * cumulative_blending_matrix_ * base_coeff;

      if (acc || jerk) {
        BaseCoeffsWithTime<2>(fraction, base_coeff);
        dd_coeff = inv_dt * inv_dt * cumulative_blending_matrix_ * base_coeff;

        if (jerk) {
          BaseCoeffsWithTime<3>(fraction, base_coeff);
          ddd_coeff = inv_dt * inv_dt * inv_dt * cumulative_blending_matrix_ *
                      base_coeff;
        }
      }
    }

    if (value) {
      Eigen::Map<Group const> const knot_0(knots[0]);
      *value = knot_0;
    }

    Tangent rot_vel, rot_acc, rot_jerk;
    if (vel || acc || jerk) rot_vel.setZero();
    if (acc || jerk) rot_acc.setZero();
    if (jerk) rot_jerk.setZero();

    for (int i = 0; i < kDeg; ++i) {
      Eigen::Map<Group const> const knot_0(knots[i]);
      Eigen::Map<Group const> const knot_1(knots[i + 1]);

      Group knot_delta = knot_0.inverse() * knot_1;
      Tangent delta = knot_delta.log();

      Group exp_k_delta = Group::exp(delta * coeff[i + 1]);

      if (value) (*value) *= exp_k_delta;

      if (vel || acc || jerk) {
        Adjoint Adj = exp_k_delta.inverse().Adj();

        rot_vel = Adj * rot_vel;
        Tangent rot_vel_current = delta * d_coeff[i + 1];
        rot_vel += rot_vel_current;

        if (acc || jerk) {
          rot_acc = Adj * rot_acc;
          Tangent acc_lie_bracket = Group::lieBracket(rot_vel, rot_vel_current);
          rot_acc += dd_coeff[i + 1] * delta + acc_lie_bracket;

          if (jerk) {
            rot_jerk = Adj * rot_jerk;
            rot_jerk += ddd_coeff[i + 1] * delta +
                        Group::lieBracket(dd_coeff[i + 1] * rot_vel +
                                              T(2) * d_coeff[i + 1] * rot_acc -
                                              d_coeff[i + 1] * acc_lie_bracket,
                                          delta);
          }
        }
      }
    }

    if (vel) *vel = rot_vel;
    if (acc) *acc = rot_acc;
    if (jerk) *jerk = rot_jerk;
  }

  /**
   * @brief Evaluate Euclidean B-spline or time derivatives.
   *
   * @tparam Dim B-spline dimension.
   * @tparam Derivative Order of the derivative to be estimated.
   * @param[in] knots Array of pointers of the spline knots. The size of each
   * knot should be Dim.
   * @param[in] fraction Relative position (fractional part) within the current
   * time segment.
   * @param[in] inv_dt Inverse of the time interval in seconds between spline
   * knots.
   * @param[out] deriv if Derivative = 0 returns value of the spline, otherwise
   * the derivative of the corresponding order.
   */
  template <int Dim, int Derivative>
  static inline void Evaluate(T const *const *knots, const T fraction,
                              const double inv_dt,
                              Eigen::Matrix<T, Dim, 1> *deriv) {
    if (!deriv) return;

    using VecD = Eigen::Matrix<T, Dim, 1>;

    VecN base_coeff, coeff;

    BaseCoeffsWithTime<Derivative>(fraction, base_coeff);
    coeff = ceres::pow(T(inv_dt), Derivative) * blending_matrix_ * base_coeff;

    deriv->setZero();

    for (int i = 0; i < kN; ++i) {
      Eigen::Map<VecD const> const knot_i(knots[i]);

      (*deriv) += coeff[i] * knot_i;
    }
  }

 private:
  // Blending matrix.
  static const MatN blending_matrix_;
  // Cumulative blending matrix.
  static const MatN cumulative_blending_matrix_;
  // Base coefficients matrix.
  static const MatN base_coefficients_;
};

template <class T, int Order>
const typename CeresSplineHelperJet<T, Order>::MatN
    CeresSplineHelperJet<T, Order>::base_coefficients_ =
        SplineUtils::ComputeBaseCoefficients<Order, T>();

template <class T, int Order>
const typename CeresSplineHelperJet<T, Order>::MatN
    CeresSplineHelperJet<T, Order>::blending_matrix_ =
        SplineUtils::ComputeBlendingMatrix<Order, T, false>();

template <class T, int Order>
const typename CeresSplineHelperJet<T, Order>::MatN
    CeresSplineHelperJet<T, Order>::cumulative_blending_matrix_ =
        SplineUtils::ComputeBlendingMatrix<Order, T, true>();

}  // namespace hpgt
