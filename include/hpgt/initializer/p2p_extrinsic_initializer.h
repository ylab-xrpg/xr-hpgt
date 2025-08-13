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

#include "hpgt/initializer/initializer_helper.h"
#include "hpgt/initializer/initializer_utils.hpp"
#include "hpgt/sensor_data/sensor_data_manager.h"

namespace hpgt {

/**
 * @brief Elements for solving spatial extrinsic parameters between two pose
 * sequences.
 *
 * Each element contains two poses measured at the same timestamp from two
 * sequences.
 */
struct P2PSolverElement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<P2PSolverElement>;

  /**
   * @brief Construct a P2P extrinsic parameter solver element.
   *
   * @param[in] p_i Translational part of the pose measurement for sequence i.
   * @param[in] q_i Rotational part of the pose measurement for sequence i.
   * @param[in] p_j Translational part of the pose measurement for sequence j.
   * @param[in] q_j Rotational part of the pose measurement for sequence j.
   * @param[in] high_quality Flag indicating whether the element is of high
   * quality.
   */
  P2PSolverElement(const Eigen::Vector3d &p_i, const Eigen::Quaterniond &q_i,
                   const Eigen::Vector3d &p_j, const Eigen::Quaterniond &q_j,
                   const bool &high_quality = false)
      : trans_Wi_Pi(p_i),
        rot_q_Wi_Pi(q_i),
        trans_Wj_Pj(p_j),
        rot_q_Wj_Pj(q_j),
        high_quality_flag(high_quality) {}

  // Create an instance of P2P solver element.
  static P2PSolverElement::Ptr Create(const Eigen::Vector3d &p_i,
                                      const Eigen::Quaterniond &q_i,
                                      const Eigen::Vector3d &p_j,
                                      const Eigen::Quaterniond &q_j,
                                      const bool &high_quality = false) {
    return Ptr(new P2PSolverElement(p_i, q_i, p_j, q_j, high_quality));
  }

  // Pose for sequence 1.
  Eigen::Vector3d trans_Wi_Pi;
  Eigen::Quaterniond rot_q_Wi_Pi;
  // Pose for sequence 2.
  Eigen::Vector3d trans_Wj_Pj;
  Eigen::Quaterniond rot_q_Wj_Pj;
  // Flag for high quality element.
  bool high_quality_flag;
};

/**
 * @brief Class for initializing spatial extrinsic parameters between two pose
 * sequences.
 */
class P2PExtrinsicInitializer {
 public:
  using Ptr = std::shared_ptr<P2PExtrinsicInitializer>;

  using P2PSolverElements = std::vector<P2PSolverElement::Ptr>;

  // Create an instance of P2P extrinsic parameters initializer.
  static P2PExtrinsicInitializer::Ptr Create() {
    return Ptr(new P2PExtrinsicInitializer);
  }

  /**
   * @brief Estimate initial guesses of spatial extrinsic parameters between
   * two pose data sequence.
   *
   * We compute the transformation between body frames by solving the AXXB
   * problem and then realize the alignment of the world frames by the Umeyama
   * algorithm.
   *
   * @param[in] pose_seq_i Pose sequence 1.
   * @param[in] pose_seq_j Pose sequence 2.
   * @param[in] toff Time offset between pose sequences (toff_P1_P2).
   * @param[out] trans_Pi_Pj Initial guess of the translation between body
   * frames (P2 in P1).
   * @param[out] rot_q_Pi_Pj Initial guess of the rotation between body frames
   * (P2 to P1).
   * @param[out] trans_Wi_Wj Initial guess of the translation between world
   * frames (W2 in W1).
   * @param[out] rot_q_Wi_Wj Initial guess of the rotation between world frames
   * (W2 to W1).
   * @return True if we are able to initialize the extrinsic parameters.
   */
  bool EstimateFromSeq(const PoseSequence &pose_seq_i,
                       const PoseSequence &pose_seq_j, const double &toff,
                       Eigen::Vector3d &trans_Pi_Pj,
                       Eigen::Quaterniond &rot_q_Pi_Pj,
                       Eigen::Vector3d &trans_Wi_Wj,
                       Eigen::Quaterniond &rot_q_Wi_Wj);

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

 private:
  /**
   * @brief Estimating the initial guess of the transformation between body
   * frames (P2 to P1) by solving the AXXB problem.
   *
   * For more details, please refer to: "Andreff N, et al. Robot hand-eye
   * calibration using structure-from-motion. IJRR, 2001."
   *
   * @param[in] solver_elements Solver elements for constructing the AXXB
   * problem.
   * @param[out] trans_Pi_Pj Initial guess of the translation between body
   * frames (P2 in P1).
   * @param[out] rot_q_Pi_Pj Initial guess of the rotation between body frames
   * (P2 to P1).
   */
  bool SolveAXXB(const P2PSolverElements &solver_elements,
                 Eigen::Vector3d &trans_Pi_Pj, Eigen::Quaterniond &rot_q_Pi_Pj);

  /**
   * @brief Estimating the initial guess of the transformation between world
   * frames (W2 to W1) based on Umeyama alignment. Please ensure that the body
   * frame alignment is completed in advance.
   *
   * For more details, please refer to: "Umeyama S. Least-squares estimation of
   * transformation parameters between two point patterns. IEEE T-PAMI, 1991."
   *
   * @param[in] solver_elements Solver elements for constructing the linear
   * solving system.
   * @param[out] trans_Wi_Wj Initial guess of the translation between world
   * frames (P2 in P1).
   * @param[out] rot_q_Wi_Wj Initial guess of the rotation between world frames
   * (P2 to P1).
   */
  bool SolveUmeyama(const P2PSolverElements &solver_elements,
                    Eigen::Vector3d &trans_Wi_Wj,
                    Eigen::Quaterniond &rot_q_Wi_Wj);

  // Initializer settings
  double time_margin_ = 0.05;
  double element_interval_thresh_ = 0.1;
  double element_trans_thresh_ = 0.2;
  double element_rot_thresh_ = 5;
  int min_element_num_ = 100;
  int max_element_num_ = 500;
  int robust_kernel_coeff_ = 5;
};

}  // namespace hpgt
