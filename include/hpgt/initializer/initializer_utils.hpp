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
#include <cmath>

#include <Eigen/Eigen>
// clang-format on

namespace hpgt {

/**
 * @brief A utility class for algebraic computations required in initializers.
 */
class InitializerUtils {
 public:
  /**
   * @brief Skew-symmetric matrix from a given 3x1 vector.
   *
   * @param[in] v 3*1 vector to be made a Skew-symmetric.
   * @return 3*3 Skew-symmetric matrix.
   */
  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> Skew(
      const Eigen::MatrixBase<Derived> &r) {
    typedef typename Derived::Scalar scalar;

    Eigen::Matrix<scalar, 3, 3> r_skew = Eigen::Matrix<scalar, 3, 3>::Zero();
    r_skew << static_cast<scalar>(0.), -r[2], r[1], r[2],
        static_cast<scalar>(0.), -r[0], -r[1], r[0], static_cast<scalar>(0.);

    return r_skew;
  }

  /**
   * @brief Rotation angle in degrees for quaternion.
   *
   * @param[in] q Input quaternion.
   * @return Rotation angle in degrees.
   */
  template <typename Derived>
  static double QuatAngleDegree(const Eigen::QuaternionBase<Derived> &q) {
    typedef typename Derived::Scalar scalar;

    Eigen::Quaternion<scalar> q_normalize = q.normalized();
    if (q_normalize.w() < 0) {
      q_normalize.coeffs() = -q_normalize.coeffs();
    }

    scalar angle = 2 * std::acos(q_normalize.w()) * 180. / M_PI;

    return angle;
  }

  /**
   * @brief Left product matrix of quaternion.
   * Note that the default order of quaternion here is (qw, qx, qy, qz).
   *
   * @param[in] q Input quaternion.
   * @return 4*4 left product matrix.
   */
  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> QuatLeftProd(
      const Eigen::QuaternionBase<Derived> &q) {
    typedef typename Derived::Scalar scalar;

    Eigen::Matrix<scalar, 4, 4> left_P;
    left_P(0, 0) = q.w();
    left_P.template block<1, 3>(0, 1) = -q.vec().transpose();
    left_P.template block<3, 1>(1, 0) = q.vec();
    left_P.template block<3, 3>(1, 1) =
        q.w() * Eigen::Matrix<scalar, 3, 3>::Identity() + Skew(q.vec());

    return left_P;
  }

  /**
   * @brief Right product matrix of quaternion.
   * Note that the default order of quaternion here is (qw, qx, qy, qz).
   *
   * @param[in] q Input quaternion.
   * @return 4*4 right product matrix.
   */
  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> QuatRightProd(
      const Eigen::QuaternionBase<Derived> &q) {
    typedef typename Derived::Scalar scalar;

    Eigen::Matrix<scalar, 4, 4> right_P;
    right_P(0, 0) = q.w();
    right_P.template block<1, 3>(0, 1) = -q.vec().transpose();
    right_P.template block<3, 1>(1, 0) = q.vec();
    right_P.template block<3, 3>(1, 1) =
        q.w() * Eigen::Matrix<scalar, 3, 3>::Identity() - Skew(q.vec());

    return right_P;
  }
};

}  // namespace hpgt
