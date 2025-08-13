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
 * @brief Uniform B-spline of specific dimension and order in Euclidean space.
 *
 * For more details, please refer to: "Sommer C, et al. Efficient derivative
 * computation for cumulative b-splines on lie groups. CVPR, 2020." or
 * "https://github.com/Unsigned-Long/CTraj"
 *
 * @tparam Dim B-spline dimension.
 * @tparam Order B-spline order.
 * @tparam Scalar B-spline scalar type.
 */
template <int Dim, int Order, typename Scalar = double>
class EuclideanSpline {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<EuclideanSpline<Dim, Order, Scalar>>;

  // Dimension of the spline.
  static constexpr int kDim = Dim;

  // Order and degree of the spline.
  static constexpr int kN = Order;
  static constexpr int kDeg = Order - 1;

  // Matrix and vector definitions.
  using MatN = Eigen::Matrix<Scalar, kN, kN>;
  using VecN = Eigen::Matrix<Scalar, kN, 1>;

  using MatD = Eigen::Matrix<Scalar, kDim, kDim>;
  using VecD = Eigen::Matrix<Scalar, kDim, 1>;

  /**
   * @brief Struct to store the Jacobian of the spline.
   * Since B-spline of order N has local support (only N knots influence the
   * value) the Jacobian involves the relevant N knots.
   */
  struct JacobianStruct {
    // Start index of the non-zero elements of the Jacobian.
    size_t start_idx;
    // Value of nonzero Jacobians.
    std::array<Scalar, kN> d_val_d_knot;
  };

  /**
   * @brief Construct a new Euclidean Spline object.
   *
   * @param knot_interval[in] Knot interval of the spline in seconds.
   * @param start_time[in] Start time of the spline in seconds.
   */
  explicit EuclideanSpline(double knot_interval, double start_time = 0)
      : knot_interval_(knot_interval), start_time_(start_time) {
    inv_pow_dt_[0] = 1.;
    inv_pow_dt_[1] = 1. / knot_interval_;

    for (size_t i = 2; i < kN; i++) {
      inv_pow_dt_[i] = inv_pow_dt_[i - 1] * inv_pow_dt_[1];
    }
  }

  // Create an instance of Euclidean spline.
  static typename EuclideanSpline<Dim, Order, Scalar>::Ptr Create(
      double knot_interval, double start_time = 0) {
    return Ptr(
        new EuclideanSpline<Dim, Order, Scalar>(knot_interval, start_time));
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
   * @return True if the timestamp is valid and within range
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
      spdlog::critical("Sample time out of range in Euclidean spline.");
      return false;
    }
  }

  /**
   * @brief Evaluate value (Derivative = 0) or time derivative (Derivative != 0)
   * of the spline.
   *
   * @tparam Derivative Order of the derivative to be estimated.
   * @param[in] timestamp Timestamp for evaluating the spline.
   * @param[out] result Result of the spline value or derivative. Euclidean
   * vector of dimension kDim.
   * @param[out] jacobian If not nullptr, return the Jacobian of the value with
   * respect to knots.
   * @return True if we are able to evaluate the result.
   */
  template <int Derivative = 0>
  bool Evaluate(const double &timestamp, VecD &result,
                JacobianStruct *jacobian = nullptr) const {
    result.setZero();

    size_t index;
    double fraction;
    if (!ComputeTimeIndex(timestamp, index, fraction)) {
      return false;
    }

    VecN base_coeff;
    BaseCoeffsWithTime<Derivative>(fraction, base_coeff);
    VecN coeff = inv_pow_dt_[Derivative] * (blending_matrix_ * base_coeff);

    for (int i = 0; i < kN; i++) {
      result += coeff[i] * knots_[index + i];

      if (jacobian) {
        jacobian->d_val_d_knot[i] = coeff[i];
      }
    }

    if (jacobian) {
      jacobian->start_idx = index;
    }

    return true;
  }

  // Alias for first derivative of spline.
  bool Velocity(const double &timestamp, VecD &result,
                JacobianStruct *jacobian = nullptr) const {
    return Evaluate<1>(timestamp, result, jacobian);
  }

  // Alias for second derivative of spline.
  bool Acceleration(const double &timestamp, VecD &result,
                    JacobianStruct *jacobian = nullptr) const {
    return Evaluate<2>(timestamp, result, jacobian);
  }

  // Add knot to the front/back of the spline.
  void KnotsPushBack(const VecD &knot) { knots_.push_back(knot); }

  void KnotsPushFront(const VecD &knot) { knots_.push_front(knot); }

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
  VecD &get_knot(int i) { return knots_.at(i); }

  const VecD &get_knot(int i) const { return knots_.at(i); }

  const aligned_deque<VecD> &get_knots() const { return knots_; }

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
  aligned_deque<VecD> knots_;
  // Knot interval in seconds.
  double knot_interval_;
  // Start time in seconds.
  double start_time_;
};

template <int Dim, int Order, typename Scalar>
const typename EuclideanSpline<Dim, Order, Scalar>::MatN
    EuclideanSpline<Dim, Order, Scalar>::blending_matrix_ =
        SplineUtils::ComputeBlendingMatrix<Order, Scalar, false>();

template <int Dim, int Order, typename Scalar>
const typename EuclideanSpline<Dim, Order, Scalar>::MatN
    EuclideanSpline<Dim, Order, Scalar>::base_coefficients_ =
        SplineUtils::ComputeBaseCoefficients<Order, Scalar>();

}  // namespace hpgt
