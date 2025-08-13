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
#include <vector>

#include "hpgt/initializer/imu_preintegrator.h"
#include "hpgt/initializer/initializer_helper.h"
#include "hpgt/initializer/initializer_utils.hpp"
#include "hpgt/sensor_data/sensor_data_manager.h"

namespace hpgt {

/**
 * @brief Elements for solving spatial extrinsic parameters from IMU to pose.
 *
 * Each element contains the pose and IMU preintegration results for a certain
 * time interval.
 */
struct I2PSolverElement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<I2PSolverElement>;

  /**
   * @brief Construct a I2P extrinsic parameter solver element.
   *
   * @param[in] integrator IMU preintegration result over a time interval.
   * @param[in] p_start Translational part of the pose measurement at the start
   * time.
   * @param[in] q_start Rotational part of the pose measurement at the start
   * time.
   * @param[in] p_end Translational part of the pose measurement at the end
   * time.
   * @param[in] q_end Rotational part of the pose measurement at the end time.
   * @param[in] interval Time interval.
   * @param[in] high_quality Flag indicating whether the element is of high
   * quality.
   */
  I2PSolverElement(const std::shared_ptr<ImuPreintegrator> &integrator,
                   const Eigen::Vector3d &p_start,
                   const Eigen::Quaterniond &q_start,
                   const Eigen::Vector3d &p_end,
                   const Eigen::Quaterniond &q_end, const double &interval,
                   const bool &high_quality = false)
      : preintegrator(integrator),
        trans_WP_start(p_start),
        rot_q_WP_start(q_start),
        trans_WP_end(p_end),
        rot_q_WP_end(q_end),
        time_interval(interval),
        high_quality_flag(high_quality) {}

  // Create an instance of I2P solver element.
  static I2PSolverElement::Ptr Create(
      const std::shared_ptr<ImuPreintegrator> &integrator,
      const Eigen::Vector3d &p_0, const Eigen::Quaterniond &q_0,
      const Eigen::Vector3d &p_1, const Eigen::Quaterniond &q_1,
      const double &interval, const bool &high_quality = false) {
    return Ptr(new I2PSolverElement(integrator, p_0, q_0, p_1, q_1, interval,
                                    high_quality));
  }

  // IMU preintegration result.
  std::shared_ptr<ImuPreintegrator> preintegrator;
  // Pose measurement.
  Eigen::Vector3d trans_WP_start;
  Eigen::Quaterniond rot_q_WP_start;
  Eigen::Vector3d trans_WP_end;
  Eigen::Quaterniond rot_q_WP_end;
  // Length of the time interval.
  double time_interval;
  // Flag for high quality element.
  bool high_quality_flag;
};

/**
 * @brief Class for initializing spatial extrinsic parameters between pose and
 * IMU sequences.
 *
 * For more details, please refer to: "Qin T, et al. VINS-Mono: A Robust and
 * Versatile Monocular Visual-Inertial State Estimator. IEEE T-RO, 2018."
 */
class I2PExtrinsicInitializer {
 public:
  using Ptr = std::shared_ptr<I2PExtrinsicInitializer>;

  using I2PSolverElements = std::vector<I2PSolverElement::Ptr>;

  // Create an instance of I2P extrinsic parameters initializer.
  static I2PExtrinsicInitializer::Ptr Create() {
    return Ptr(new I2PExtrinsicInitializer);
  }

  /**
   * @brief Estimate initial guesses of spatial extrinsic parameters between
   * pose and IMU data sequence.
   *
   * We perform linear least squares optimization based on the motion
   * constraints between IMU preintegration and relative pose.
   *
   * @param[in] pose_seq Pose sequence.
   * @param[in] imu_seq IMU sequence.
   * @param[in] toff Time offset between pose and IMU (toff_PI).
   * @param[in] g_magnitude Gravity magnitude.
   * @param[out] trans_PI Initial guess of the extrinsic translation (I in P).
   * @param[out] rot_q_PI Initial guess of the extrinsic rotation (I to P).
   * @param[out] rot_q_GW Initial guess of the gravity-aligned rotation (W to
   * G).
   * @return True if we are able to initialize the extrinsic parameters.
   */
  bool EstimateFromSeq(const PoseSequence &pose_seq, const ImuSequence &imu_seq,
                       const double &toff, const double &g_magnitude,
                       Eigen::Vector3d &trans_PI, Eigen::Quaterniond &rot_q_PI,
                       Eigen::Quaterniond &rot_q_GW);

  // Set the parameters.
  void set_time_margin(const double &v) { time_margin_ = v; }

  void set_element_interval_thresh(const double &v) {
    element_interval_thresh_ = v;
  }

  void set_element_trans_thresh(const double &v) { element_trans_thresh_ = v; }

  void set_element_rot_thresh(const double &v) { element_rot_thresh_ = v; }

  void set_min_element_num(const int &v) { min_element_num_ = v; }

  void set_max_element_num(const int &v) { max_element_num_ = v; }

  void set_robust_kernel_coeff(const int &v) { robust_kernel_coeff_ = v; }

  void set_refine_iter_num(const int &v) { refine_iter_num_ = v; }

 private:
  /**
   * @brief Retrieve IMU data from an IMU sequence for a given time period and
   * calculate the preintegration result.
   *
   * @param[in] imu_seq IMU sequence.
   * @param[in] time_0 Start time of preintegration.
   * @param[in] time_1 End time of preintegration.
   * @param[in] g_magnitude Gravity magnitude.
   * @param[out] integrator Preintegration result.
   * @return True if preintegration is completed.
   */
  bool Preintegrate(const ImuSequence &imu_seq, const double &time_0,
                    const double &time_1, const double &g_magnitude,
                    ImuPreintegrator::Ptr &integrator);

  /**
   * @brief Estimating the initial guess of the extrinsic rotation.
   *
   * @param[in] solver_elements Solver elements for constructing the linear
   * solving system.
   * @param[out] rot_q_PI Initial guess of the extrinsic rotation (I to P).
   */
  void EstimateRot(const I2PSolverElements &solver_elements,
                   Eigen::Quaterniond &rot_q_PI);

  /**
   * @brief Simultaneously estimate the initial guesses of the extrinsic
   * translation (I in P) and the gravity vector represented in W.
   *
   * @param[in] solver_elements Solver elements for constructing the linear
   * solving system.
   * @param[in] rot_q_PI The solved extrinsic rotation (I to P).
   * @param[out] trans_PI Initial guess of the extrinsic translation (I in P).
   * @param[out] gravity_W Initial guess of the gravity vector represented in W.
   */
  void EstimateTransGravityAlign(const I2PSolverElements &solver_elements,
                                 const Eigen::Quaterniond &rot_q_PI,
                                 Eigen::Vector3d &trans_PI,
                                 Eigen::Vector3d &gravity_W);

  /**
   * @brief Iteratively refine the initial guesses of the extrinsic translation
   * (I in P) and the gravity vector represented in W using the constraint of
   * gravity magnitude.
   *
   * @param[in] solver_elements Solver elements for constructing the linear
   * solving system.
   * @param[in] rot_q_PI The solved extrinsic rotation (I to P).
   * @param[in] gravity_magnitude Gravity magnitude
   * @param[out] trans_PI Initial guess of the extrinsic translation (I in P).
   * @param[out] gravity_W Initial guess of the gravity vector represented in W.
   */
  void RefineGravityAlign(const I2PSolverElements &solver_elements,
                          const Eigen::Quaterniond &rot_q_PI,
                          const double &g_magnitude, Eigen::Vector3d &trans_PI,
                          Eigen::Vector3d &gravity_W);

  /**
   * @brief Obtain the tangent space basis of the gravity vector.
   *
   * @param[in] gravity Current gravity vector.
   * @return 3*2 Tangent space basis.
   */
  Eigen::Matrix<double, 3, 2> GravityTangentBasis(
      const Eigen::Vector3d &gravity);

  // Initializer settings
  double time_margin_ = 0.05;
  double element_interval_thresh_ = 0.1;
  double element_trans_thresh_ = 0.2;
  double element_rot_thresh_ = 5;
  int min_element_num_ = 100;
  int max_element_num_ = 500;
  int robust_kernel_coeff_ = 5;
  int refine_iter_num_ = 4;
};

}  // namespace hpgt
