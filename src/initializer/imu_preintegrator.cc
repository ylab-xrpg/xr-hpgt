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

#include "hpgt/initializer/imu_preintegrator.h"

namespace hpgt {

ImuPreintegrator::ImuPreintegrator(
    const Eigen::Vector3d &acc_start, const Eigen::Vector3d &gyr_start,
    const double &g_magnitude, const Eigen::Vector3d &ba,
    const Eigen::Vector3d &bg, const double &acc_noise,
    const double &acc_b_noise, const double &gyr_noise,
    const double &gyr_b_noise) {
  acc_m_start_ = acc_start;
  gyr_m_start_ = gyr_start;
  acc_m_0_ = acc_start;
  gyr_m_0_ = gyr_start;

  gravity_in_G_ = Eigen::Vector3d(0., 0., g_magnitude);

  linearized_ba_ = ba;
  linearized_bg_ = bg;

  // Constructing noise covariance matrix based on standard deviation.
  noise_covariance_ = Eigen::Matrix<double, 18, 18>::Zero();
  noise_covariance_.block<3, 3>(0, 0) =
      acc_noise * acc_noise * Eigen::Matrix3d::Identity();
  noise_covariance_.block<3, 3>(3, 3) =
      gyr_noise * gyr_noise * Eigen::Matrix3d::Identity();
  noise_covariance_.block<3, 3>(6, 6) =
      acc_noise * acc_noise * Eigen::Matrix3d::Identity();
  noise_covariance_.block<3, 3>(9, 9) =
      gyr_noise * gyr_noise * Eigen::Matrix3d::Identity();
  noise_covariance_.block<3, 3>(12, 12) =
      acc_b_noise * acc_b_noise * Eigen::Matrix3d::Identity();
  noise_covariance_.block<3, 3>(15, 15) =
      gyr_b_noise * gyr_b_noise * Eigen::Matrix3d::Identity();
}

ImuPreintegrator::Ptr ImuPreintegrator::Create(
    const Eigen::Vector3d &acc_start, const Eigen::Vector3d &gyr_start,
    const double &g_magnitude, const Eigen::Vector3d &ba,
    const Eigen::Vector3d &bg, const double &acc_noise,
    const double &acc_b_noise, const double &gyr_noise,
    const double &gyr_b_noise) {
  return Ptr(new ImuPreintegrator(acc_start, gyr_start, g_magnitude, ba, bg,
                                  acc_noise, acc_b_noise, gyr_noise,
                                  gyr_b_noise));
}

void ImuPreintegrator::Propagate(const double &dt, const Eigen::Vector3d &acc,
                                 const Eigen::Vector3d &gyr) {
  // Sensor measurements propagation.
  acc_m_1_ = acc;
  gyr_m_1_ = gyr;
  dt_curr_ = dt;
  dt_sum_ += dt;

  dt_buf_.push_back(dt);
  acc_data_buf_.push_back(acc);
  gyr_data_buf_.push_back(gyr);

  // Call the midpoint integration function.
  MidPointIntegrate();

  // Measurements propagation.
  acc_m_0_ = acc_m_1_;
  gyr_m_0_ = gyr_m_1_;
  alpha_last_ = alpha_curr_;
  q_curr_.normalize();
  q_last_ = q_curr_;
  beta_last_ = beta_curr_;
}

void ImuPreintegrator::MidPointIntegrate() {
  // Preintegrator propagation.
  Eigen::Vector3d gyro_rec_mid = 0.5 * (gyr_m_0_ + gyr_m_1_) - linearized_bg_;
  q_curr_ = q_last_ * Eigen::Quaterniond(1, 0.5 * gyro_rec_mid[0] * dt_curr_,
                                         0.5 * gyro_rec_mid[1] * dt_curr_,
                                         0.5 * gyro_rec_mid[2] * dt_curr_)
                          .normalized();
  Eigen::Vector3d acc_rec_mid = 0.5 * (q_last_ * (acc_m_0_ - linearized_ba_)) +
                                0.5 * (q_curr_ * (acc_m_1_ - linearized_ba_));
  beta_curr_ = beta_last_ + acc_rec_mid * dt_curr_;
  alpha_curr_ = alpha_last_ + beta_last_ * dt_curr_ +
                0.5 * acc_rec_mid * dt_curr_ * dt_curr_;
  // ba and bg remain unchanged during propagation.

  // ===========================================================================

  // intermediate value
  Eigen::Vector3d acc_rec_0 = acc_m_0_ - linearized_ba_;
  Eigen::Vector3d acc_rec_1 = acc_m_1_ - linearized_ba_;

  Eigen::Matrix3d gyro_rec_mid_skew = InitializerUtils::Skew(gyro_rec_mid);
  Eigen::Matrix3d acc_rec_0_skew = InitializerUtils::Skew(acc_rec_0);
  Eigen::Matrix3d acc_rec_1_skew = InitializerUtils::Skew(acc_rec_1);

  // ===========================================================================

  // Jacobian of single-step state propagation, [alpha, q, beta, ba, bg]_15*15.
  Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
  F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(0, 3) =
      -0.25 * (q_last_ * acc_rec_0_skew * dt_curr_ * dt_curr_) +
      -0.25 * (q_curr_ * acc_rec_1_skew) *
          (Eigen::Matrix3d::Identity() - gyro_rec_mid_skew * dt_curr_) *
          dt_curr_ * dt_curr_;
  F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt_curr_;
  F.block<3, 3>(0, 9) =
      -0.25 * (q_last_.toRotationMatrix() + q_curr_.toRotationMatrix()) *
      dt_curr_ * dt_curr_;
  F.block<3, 3>(0, 12) =
      -0.25 * (q_curr_ * acc_rec_1_skew) * dt_curr_ * dt_curr_ * -dt_curr_;
  F.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() - gyro_rec_mid_skew * dt_curr_;
  F.block<3, 3>(3, 12) = -1. * Eigen::Matrix3d::Identity() * dt_curr_;
  F.block<3, 3>(6, 3) =
      -0.5 * (q_last_ * acc_rec_0_skew) * dt_curr_ +
      -0.5 * (q_curr_ * acc_rec_1_skew) *
          (Eigen::Matrix3d::Identity() - gyro_rec_mid_skew * dt_curr_) *
          dt_curr_;
  F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(6, 9) =
      -0.5 * (q_last_.toRotationMatrix() + q_curr_.toRotationMatrix()) *
      dt_curr_;
  F.block<3, 3>(6, 12) =
      -0.5 * (q_curr_ * acc_rec_1_skew) * dt_curr_ * -dt_curr_;
  F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

  // ===========================================================================

  // Jacobian of single-step error propagation, [alpha, q, beta, ba,
  // bg]_15*[acc_0_n, gyr_0_n, acc_1_n, gyr_1_n, acc_0_b_n, gyr_0_b_n]_18.
  Eigen::Matrix<double, 15, 18> G = Eigen::Matrix<double, 15, 18>::Zero();
  G.block<3, 3>(0, 0) = 0.25 * q_last_.toRotationMatrix() * dt_curr_ * dt_curr_;
  G.block<3, 3>(0, 3) =
      -0.25 * (q_curr_ * acc_rec_1_skew) * dt_curr_ * dt_curr_ * 0.5 * dt_curr_;
  G.block<3, 3>(0, 6) = 0.25 * q_curr_.toRotationMatrix() * dt_curr_ * dt_curr_;
  G.block<3, 3>(0, 9) = G.block<3, 3>(0, 3);
  G.block<3, 3>(3, 3) = 0.5 * Eigen::Matrix3d::Identity() * dt_curr_;
  G.block<3, 3>(3, 9) = 0.5 * Eigen::Matrix3d::Identity() * dt_curr_;
  G.block<3, 3>(6, 0) = 0.5 * q_last_.toRotationMatrix() * dt_curr_;
  G.block<3, 3>(6, 3) =
      -0.5 * (q_curr_ * acc_rec_1_skew) * dt_curr_ * 0.5 * dt_curr_;
  G.block<3, 3>(6, 6) = 0.5 * q_curr_.toRotationMatrix() * dt_curr_;
  G.block<3, 3>(6, 9) = G.block<3, 3>(6, 3);
  G.block<3, 3>(9, 12) = Eigen::Matrix3d::Identity();
  G.block<3, 3>(12, 15) = Eigen::Matrix3d::Identity();

  // ===========================================================================

  // Jacobian and covariance propagation.
  step_state_jacobian_ = F;
  step_noise_jacobian_ = G;
  state_jacobian_ = F * state_jacobian_;
  // Covariance propagation.
  meas_covariance_ = F * meas_covariance_ * F.transpose() +
                     G * noise_covariance_ * G.transpose();

  // ===========================================================================
}

}  // namespace hpgt
