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

// clang-format off
#include <array>
#include <memory>

#include <spdlog/spdlog.h>

#include "hpgt/spline/spline_utils.hpp"
// clang-format on

namespace hpgt {

/**
 * @brief Uniform cumulative B-spline for SO(3) manifold of specified order.
 *
 * For more details, please refer to: "Sommer C, et al. Efficient derivative
 * computation for cumulative b-splines on lie groups. CVPR, 2020." or
 * "https://github.com/Unsigned-Long/CTraj"
 *
 * @tparam Order B-spline order.
 * @tparam Scalar B-spline scalar type.
 */
template <int Order, typename Scalar = double>
class So3Spline {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<So3Spline<Order, Scalar>>;

  // Order and degree of the spline.
  static constexpr int kN = Order;
  static constexpr int kDeg = Order - 1;

  // Matrix and vector definitions.
  using MatN = Eigen::Matrix<Scalar, kN, kN>;
  using VecN = Eigen::Matrix<Scalar, kN, 1>;

  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  using So3 = Sophus::SO3<Scalar>;

  /**
   * @brief Struct to store the Jacobian of the spline.
   * Since B-spline of order N has local support (only N knots influence the
   * value) the Jacobian involves the relevant N knots.
   */
  struct JacobianStruct {
    // Start index of the non-zero elements of the Jacobian.
    size_t start_idx;
    // Value of nonzero Jacobians.
    std::array<Mat3, kN> d_val_d_knot;
  };

  /**
   * @brief Construct a new SO(3) Spline object.
   *
   * @param knot_interval[in] Knot interval of the spline in seconds.
   * @param start_time[in] Start time of the spline in seconds.
   */
  explicit So3Spline(double knot_interval, double start_time = 0)
      : knot_interval_(knot_interval), start_time_(start_time) {
    inv_pow_dt_[0] = 1.;
    inv_pow_dt_[1] = 1. / knot_interval;

    for (size_t i = 2; i < kN; i++) {
      inv_pow_dt_[i] = inv_pow_dt_[i - 1] * inv_pow_dt_[1];
    }
  }

  // Create an instance of Euclidean spline.
  static typename So3Spline<Order, Scalar>::Ptr Create(double knot_interval,
                                                       double start_time = 0) {
    return Ptr(new So3Spline<Order, Scalar>(knot_interval, start_time));
  }

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
      const Scalar &fraction, const Eigen::MatrixBase<Derived> &result_const) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, kN)
    auto &result = const_cast<Eigen::MatrixBase<Derived> &>(result_const);

    result.setZero();

    if (Derivative < kN) {
      result[Derivative] = base_coefficients_(Derivative, Derivative);

      Scalar u = fraction;
      for (int j = Derivative + 1; j < kN; j++) {
        result[j] = base_coefficients_(Derivative, j) * u;
        u = u * fraction;
      }
    }
  }

  /**
   * @brief Compute the index and fractional part of a given timestamp
   * relative to the start time.
   *
   * @param[in] timestamp Input timestamp used to calculate the interval from
   * the start time.
   * @param[out] index Computed time index.
   * @param[out] fraction Relative position (fractional part) within the current
   * time segment.
   * @return True if the timestamp is valid and within range.
   */
  bool ComputeTimeIndex(const double &timestamp, size_t &index,
                        double &fraction) const {
    index = 0;
    fraction = 0.;

    double t = timestamp;
    if (t >= MaxTime()) {
      t = t - 1.e-6;
    } else if (t <= MinTime()) {
      t = t + 1.e-6;
    }

    if (t > MinTime() && t < MaxTime()) {
      double time_since_start = t - start_time_;
      index = std::floor(time_since_start / knot_interval_);
      fraction =
          (time_since_start - static_cast<double>(index) * knot_interval_) /
          knot_interval_;

      return true;
    } else {
      spdlog::critical("Sample time out of range in SO(3) spline.");
      return false;
    }
  }

  /**
   * @brief Evaluate value of the SO(3) spline.
   *
   * @param[in] timestamp Timestamp for evaluating the spline.
   * @param[out] result Result of the SO(3) value.
   * @param[out] jacobian If not nullptr, return the Jacobian of the SO(3) value
   * with respect to knots.
   * @return True if we are able to evaluate the result.
   */
  bool Evaluate(const double &timestamp, So3 &result,
                JacobianStruct *jacobian = nullptr) const {
    size_t index;
    double fraction;
    if (!ComputeTimeIndex(timestamp, index, fraction)) {
      return false;
    }

    VecN base_coeff;
    BaseCoeffsWithTime<0>(fraction, base_coeff);
    VecN coeff = blending_matrix_ * base_coeff;

    result = knots_[index];

    Mat3 jacobian_helper;
    if (jacobian) {
      jacobian->start_idx = index;
      jacobian_helper.setIdentity();
    }

    for (int i = 0; i < kDeg; ++i) {
      const So3 &knot_0 = knots_[index + i];
      const So3 &knot_1 = knots_[index + i + 1];
      const So3 knot_delta = knot_0.inverse() * knot_1;
      Vec3 delta = knot_delta.log();
      Vec3 k_delta = delta * coeff[i + 1];

      if (jacobian) {
        Mat3 Jl_inv_delta, Jl_k_delta;

        SplineUtils::LeftJacobianInvSo3(delta, Jl_inv_delta);
        SplineUtils::LeftJacobianSo3(k_delta, Jl_k_delta);

        jacobian->d_val_d_knot[i] = jacobian_helper;
        jacobian_helper = coeff[i + 1] * result.matrix() * Jl_k_delta *
                          Jl_inv_delta * knot_0.inverse().matrix();
        jacobian->d_val_d_knot[i] -= jacobian_helper;
      }

      result *= So3::exp(k_delta);
    }

    if (jacobian) {
      jacobian->d_val_d_knot[kDeg] = jacobian_helper;
    }

    return true;
  }

  /**
   * @brief Evaluate rotational velocity (first time derivative) of SO(3)
   * B-spline in the body frame
   *
   * @param[in] timestamp Timestamp for evaluating the spline.
   * @param[out] result Result of the velocity in the body frame.
   * @param[out] jacobian If not nullptr, return the Jacobian of the value with
   * respect to knots.
   * @return True if we are able to evaluate the result.
   */
  bool VelocityBody(const double &timestamp, Vec3 &result,
                    JacobianStruct *jacobian = nullptr) const {
    result.setZero();

    size_t index;
    double fraction;
    if (!ComputeTimeIndex(timestamp, index, fraction)) {
      return false;
    }

    VecN base_coeff;
    BaseCoeffsWithTime<0>(fraction, base_coeff);
    VecN coeff = blending_matrix_ * base_coeff;

    BaseCoeffsWithTime<1>(fraction, base_coeff);
    VecN d_coeff = inv_pow_dt_[1] * blending_matrix_ * base_coeff;

    Vec3 delta_vec[kDeg];
    Mat3 R_tmp[kDeg];
    So3 accum;
    So3 exp_k_delta[kDeg];
    Mat3 Jr_delta_inv[kDeg], Jr_k_delta[kDeg];

    for (int i = kDeg - 1; i >= 0; --i) {
      const So3 &knot_0 = knots_[index + i];
      const So3 &knot_1 = knots_[index + i + 1];
      const So3 knot_delta = knot_0.inverse() * knot_1;
      delta_vec[i] = knot_delta.log();

      SplineUtils::RightJacobianInvSo3(delta_vec[i], Jr_delta_inv[i]);
      Jr_delta_inv[i] *= knot_1.inverse().matrix();

      Vec3 k_delta = coeff[i + 1] * delta_vec[i];
      SplineUtils::RightJacobianSo3(-k_delta, Jr_k_delta[i]);

      R_tmp[i] = accum.matrix();
      exp_k_delta[i] = So3::exp(-k_delta);
      accum *= exp_k_delta[i];
    }

    Mat3 d_vel_d_delta[kDeg];
    d_vel_d_delta[0] = d_coeff[1] * R_tmp[0] * Jr_delta_inv[0];
    Vec3 rot_vel = delta_vec[0] * d_coeff[1];
    for (int i = 1; i < kDeg; ++i) {
      d_vel_d_delta[i] =
          R_tmp[i - 1] * So3::hat(rot_vel) * Jr_k_delta[i] * coeff[i + 1] +
          R_tmp[i] * d_coeff[i + 1];
      d_vel_d_delta[i] *= Jr_delta_inv[i];

      rot_vel = exp_k_delta[i] * rot_vel + delta_vec[i] * d_coeff[i + 1];
    }

    if (jacobian) {
      jacobian->start_idx = index;
      for (int i = 0; i < kN; ++i) {
        jacobian->d_val_d_knot[i].setZero();
      }
      for (int i = 0; i < kDeg; ++i) {
        jacobian->d_val_d_knot[i] -= d_vel_d_delta[i];
        jacobian->d_val_d_knot[i + 1] += d_vel_d_delta[i];
      }
    }

    result = rot_vel;

    return true;
  }

  // Add knot to the front/back of the spline.
  void KnotsPushBack(const So3 &knot) { knots_.push_back(knot); }

  void KnotsPushFront(const So3 &knot) { knots_.push_front(knot); }

  // Remove knot from the front/back of the spline.
  void KnotsPopBack() { knots_.pop_back(); }

  void KnotsPopFront() {
    start_time_ += knot_interval_;
    knots_.pop_front();
  }

  // Minimum/Maximum time represented by spline.
  double MinTime() const { return start_time_; }

  double MaxTime() const {
    int knot_size = static_cast<int>(knots_.size());
    return start_time_ + (knot_size - kN + 1) * knot_interval_;
  }

  // Check the timestamp is within the time range of the spline.
  bool TimeStampInRange(const double &timestamp) const {
    return timestamp >= MinTime() && timestamp <= MaxTime();
  }

  // Return reference to the knots.
  So3 &get_knot(int i) { return knots_.at(i); }

  const So3 &get_knot(int i) const { return knots_.at(i); }

  const aligned_deque<So3> &get_knots() const { return knots_; }

  // Return time interval.
  const double get_knot_interval() const { return knot_interval_; }

  // Set start time for spline.
  void set_start_time(const double &start_time) { start_time_ = start_time; }

 private:
  // Blending matrix.
  static const MatN blending_matrix_;

  // Base coefficients matrix.
  static const MatN base_coefficients_;

  // Array with inverse powers of delta time.
  std::array<Scalar, kN> inv_pow_dt_;

  // The knots of the B-spline.
  aligned_deque<So3> knots_;
  // Knot interval in seconds.
  double knot_interval_;
  // Start time in seconds.
  double start_time_;
};

template <int Order, typename Scalar>
const typename So3Spline<Order, Scalar>::MatN
    So3Spline<Order, Scalar>::blending_matrix_ =
        SplineUtils::ComputeBlendingMatrix<Order, Scalar, true>();

template <int Order, typename Scalar>
const typename So3Spline<Order, Scalar>::MatN
    So3Spline<Order, Scalar>::base_coefficients_ =
        SplineUtils::ComputeBaseCoefficients<Order, Scalar>();

}  // namespace hpgt
