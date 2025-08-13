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
#include <cstdint>
#include <deque>

#include <Eigen/Eigen>
#include <sophus/so3.hpp>
// clang-format on

namespace hpgt {

// Eigen memory-aligned container type definition.
template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

/**
 * @brief A utility class for mathematical operations related to splines and
 * polynomials.
 *
 * For more details, please refer to: "Sommer C, et al. Efficient derivative
 * computation for cumulative b-splines on lie groups. CVPR, 2020." or
 * "https://github.com/Unsigned-Long/CTraj"
 */
class SplineUtils {
 public:
  /**
   * @brief Compute blending matrix for uniform B-spline evaluation.
   *
   * @tparam N Order of the spline.
   * @tparam Scalar Scalar type.
   * @tparam Cumulative If the spline should be cumulative.
   * @return Blending matrix.
   */
  template <int N, typename Scalar = double, bool Cumulative = false>
  static Eigen::Matrix<Scalar, N, N> ComputeBlendingMatrix() {
    Eigen::Matrix<double, N, N> blending_mat;
    blending_mat.setZero();

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double sum = 0;

        for (int s = j; s < N; ++s) {
          sum += std::pow(-1., s - j) * Combinations(N, s - j) *
                 std::pow(N - s - 1.0, N - 1.0 - i);
        }
        blending_mat(j, i) = Combinations(N - 1, N - 1 - i) * sum;
      }
    }

    if (Cumulative) {
      for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
          blending_mat.row(i) += blending_mat.row(j);
        }
      }
    }

    uint64_t factorial = 1;
    for (int i = 2; i < N; ++i) {
      factorial *= i;
    }

    return (blending_mat / factorial).template cast<Scalar>();
  }

  /**
   * @brief Compute base coefficient matrix for polynomials of size N.
   *
   * This matrix is used to compute derivatives of the time polynomial for SO(3)
   * spline. In each row starting from 0 contains the derivative coefficients of
   * the polynomial. For Order=5 we get the following matrix:
   * \f[
   * \begin{bmatrix}
   *   1 & 1 & 1 & 1 & 1
   * \\0 & 1 & 2 & 3 & 4
   * \\0 & 0 & 2 & 6 & 12
   * \\0 & 0 & 0 & 6 & 24
   * \\0 & 0 & 0 & 0 & 24
   * \\ \end{bmatrix}
   * \f]
   *
   * @tparam N Order of the polynomial.
   * @tparam Scalar Scalar type.
   * @return Base coefficients.
   */
  template <int N, typename Scalar = double>
  static Eigen::Matrix<Scalar, N, N> ComputeBaseCoefficients() {
    Eigen::Matrix<double, N, N> base_coeff;

    base_coeff.setZero();
    base_coeff.row(0).setOnes();

    const int kDeg = N - 1;
    int order = kDeg;
    for (int n = 1; n < N; n++) {
      for (int i = kDeg - order; i < N; i++) {
        base_coeff(n, i) = (order - kDeg + i) * base_coeff(n - 1, i);
      }
      order--;
    }

    return base_coeff.template cast<Scalar>();
  }

  /**
   * @brief Right Jacobian for SO(3).
   *
   * @param[in] phi Input 3x1 vector.
   * @param[out] J_phi Result 3x3 matrix.
   */
  template <typename Derived1, typename Derived2>
  static void RightJacobianSo3(const Eigen::MatrixBase<Derived1> &phi,
                               const Eigen::MatrixBase<Derived2> &J_phi) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

    using Scalar = typename Derived1::Scalar;

    auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J.setIdentity();

    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
      Scalar phi_norm = std::sqrt(phi_norm2);
      Scalar phi_norm3 = phi_norm2 * phi_norm;

      J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
      J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    } else {
      // Taylor's expansion around 0
      J -= phi_hat / 2;
      J += phi_hat2 / 6;
    }
  }

  /**
   * @brief Right inverse Jacobian for SO(3).
   *
   * @param[in] phi Input 3x1 vector.
   * @param[out] J_phi Result 3x3 matrix.
   */
  template <typename Derived1, typename Derived2>
  static void RightJacobianInvSo3(const Eigen::MatrixBase<Derived1> &phi,
                                  const Eigen::MatrixBase<Derived2> &J_phi) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

    using Scalar = typename Derived1::Scalar;

    auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J.setIdentity();
    J += phi_hat / 2;

    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
      Scalar phi_norm = std::sqrt(phi_norm2);

      J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                           (2 * phi_norm * std::sin(phi_norm)));
    } else {
      // Taylor's expansion around 0
      J += phi_hat2 / 12;
    }
  }

  /**
   * @brief Left Jacobian for SO(3).
   *
   * @param[in] phi Input 3x1 vector.
   * @param[out] J_phi Result 3x3 matrix.
   */
  template <typename Derived1, typename Derived2>
  static void LeftJacobianSo3(const Eigen::MatrixBase<Derived1> &phi,
                              const Eigen::MatrixBase<Derived2> &J_phi) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

    using Scalar = typename Derived1::Scalar;

    auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J.setIdentity();

    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
      Scalar phi_norm = std::sqrt(phi_norm2);
      Scalar phi_norm3 = phi_norm2 * phi_norm;

      J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
      J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    } else {
      // Taylor's expansion around 0
      J += phi_hat / 2;
      J += phi_hat2 / 6;
    }
  }

  /**
   * @brief Left inverse Jacobian for SO(3).
   *
   * @param[in] phi Input 3x1 vector.
   * @param[out] J_phi Result 3x3 matrix.
   */
  template <typename Derived1, typename Derived2>
  static void LeftJacobianInvSo3(const Eigen::MatrixBase<Derived1> &phi,
                                 const Eigen::MatrixBase<Derived2> &J_phi) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

    using Scalar = typename Derived1::Scalar;

    auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J.setIdentity();
    J -= phi_hat / 2;

    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
      Scalar phi_norm = std::sqrt(phi_norm2);

      J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                           (2 * phi_norm * std::sin(phi_norm)));
    } else {
      // Taylor's expansion around 0
      J += phi_hat2 / 12;
    }
  }

 private:
  /**
   * @brief Compute the binomial coefficient.
   *
   * Computes number of combinations that include k objects out of n.
   *
   * @param[in] n The total number of objects.
   * @param[in] k The number of objects to choose.
   * @return Binomial coefficient.
   */
  static uint64_t Combinations(uint64_t n, uint64_t k) {
    if (k > n) {
      return 0;
    }
    uint64_t r = 1;
    for (uint64_t d = 1; d <= k; ++d) {
      r *= n--;
      r /= d;
    }
    return r;
  }
};

}  // namespace hpgt
