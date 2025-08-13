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

#include "hpgt/initializer/p2p_extrinsic_initializer.h"

#include <unsupported/Eigen/KroneckerProduct>

namespace hpgt {

bool P2PExtrinsicInitializer::EstimateFromSeq(const PoseSequence &pose_seq_i,
                                              const PoseSequence &pose_seq_j,
                                              const double &toff,
                                              Eigen::Vector3d &trans_Pi_Pj,
                                              Eigen::Quaterniond &rot_q_Pi_Pj,
                                              Eigen::Vector3d &trans_Wi_Wj,
                                              Eigen::Quaterniond &rot_q_Wi_Wj) {
  // Set default values.
  trans_Pi_Pj = Eigen::Vector3d::Zero();
  rot_q_Pi_Pj = Eigen::Quaterniond::Identity();
  trans_Wi_Wj = Eigen::Vector3d::Zero();
  rot_q_Wi_Wj = Eigen::Quaterniond::Identity();

  // ===========================================================================

  // Step 1: Set the joint start and end times (under the clock of the pose
  // sequence 1).
  double start_time_i = pose_seq_i.front()->timestamp + time_margin_;
  double end_time_i = pose_seq_i.back()->timestamp - time_margin_;
  double start_time_j = pose_seq_j.front()->timestamp + toff + time_margin_;
  double end_time_j = pose_seq_j.back()->timestamp + toff - time_margin_;

  double joint_start_time =
      start_time_i > start_time_j ? start_time_i : start_time_j;
  double joint_end_time = end_time_i < end_time_j ? end_time_i : end_time_j;
  if (joint_start_time < 0) {
    spdlog::warn("Start time for P2P initialization is negative, set to 0. ");
    joint_start_time = 0.;
  }
  if (joint_start_time > joint_end_time) {
    spdlog::critical(
        "Timestamp exception in P2P initialization: {:.9f} / {:.9f}.",
        joint_start_time, joint_end_time);
    spdlog::critical("Start time should be less than end time. ");

    return false;
  }

  // ===========================================================================

  // Step 2: Construct the solver elements from pose sequences.
  P2PSolverElements solver_elements;

  // We use pose sequence 1 as the baseline.
  auto reference_pose_ptr = pose_seq_i.begin();
  double reference_time, current_time, time_interval;
  Eigen::Quaterniond reference_q, current_q, delta_q;
  Eigen::Vector3d reference_p, current_p, delta_p;
  double delta_p_norm, delta_q_angle;
  int high_quality_num = 0, low_quality_num = 0;
  // Iterate through pose measurements and construct the solver elements.
  for (auto pose_iter = pose_seq_i.begin(); pose_iter != pose_seq_i.end();
       ++pose_iter) {
    current_time = (*pose_iter)->timestamp;
    if (current_time < joint_start_time) {
      reference_pose_ptr = pose_iter;
      continue;
    } else if (current_time > joint_end_time) {
      break;
    }

    // Reference and current poses.
    reference_time = (*reference_pose_ptr)->timestamp;
    reference_p = (*reference_pose_ptr)->trans;
    reference_q = (*reference_pose_ptr)->rot_q;
    current_p = (*pose_iter)->trans;
    current_q = (*pose_iter)->rot_q;

    // Calculate differences.
    time_interval = current_time - reference_time;
    delta_p = reference_q.inverse() * (current_p - reference_p);
    delta_q = reference_q.inverse() * current_q;
    delta_p_norm = delta_p.norm();
    delta_q_angle = InitializerUtils::QuatAngleDegree(delta_q);

    // Construct solver elements when the forward search meets the conditions.
    if (time_interval < element_interval_thresh_ / 3.) {
      continue;
    } else if (delta_p_norm >= element_trans_thresh_ ||
               delta_q_angle >= element_rot_thresh_ ||
               time_interval >= element_interval_thresh_) {
      // Get the pose of the same timestamp in sequence 2.
      bool get_pose_flag = false;
      Eigen::Vector3d p_j;
      Eigen::Quaterniond q_j;

      get_pose_flag = InitializerHelper::GetTargetPose(
          pose_seq_j, current_time - toff, p_j, q_j);

      if (!get_pose_flag) {
        reference_pose_ptr = pose_iter;
        continue;
      }

      // Construct the solver element.
      auto solver_element =
          P2PSolverElement::Create(current_p, current_q, p_j, q_j);

      // Determine if it is a high-quality element, i.e. fully rotated.
      if (delta_q_angle < element_rot_thresh_) {
        solver_element->high_quality_flag = false;
        low_quality_num++;
      } else {
        solver_element->high_quality_flag = true;
        high_quality_num++;
      }
      solver_elements.push_back(solver_element);
      reference_pose_ptr = pose_iter;
    }
  }

  // ===========================================================================

  // Step 3: Select the solver elements, prioritizing high quality.
  if ((high_quality_num + low_quality_num) < min_element_num_) {
    spdlog::critical(
        "Insufficient solver elements for initializing the P2P extrinsic "
        "parameters. Please increase the motion duration and motion stimuli. ");
    return false;
  }
  if (low_quality_num > high_quality_num) {
    spdlog::warn(
        "Insufficient motion stimuli in P2P extrinsic parameters "
        "initialization, may lead to inaccurate calibration results, please "
        "perform more rapid rotation. ");
  }

  // We need to control the scale of the linear solver and prioritize
  // high quality solver elements.
  int sample_num = 0;
  P2PSolverElements target_elements;
  target_elements.reserve(max_element_num_);
  for (auto &element : solver_elements) {
    if (sample_num >= max_element_num_) {
      break;
    }
    if (element->high_quality_flag) {
      target_elements.push_back(element);
      sample_num++;
    }
  }
  for (auto &element : solver_elements) {
    if (sample_num >= max_element_num_) {
      break;
    }
    if (!element->high_quality_flag) {
      target_elements.push_back(element);
      sample_num++;
    }
  }

  // ===========================================================================

  // Step 4: Initialize the spatial extrinsic parameters between two pose
  // sequences based on selected solver elements.

  // Solve the AXXB problem to initialize the transformation between body frames
  // (P2 to P1).
  if (!SolveAXXB(target_elements, trans_Pi_Pj, rot_q_Pi_Pj)) {
    spdlog::critical(
        "Fail to solve AXXB problem in P2P extrinsic parameters "
        "initialization");
    return false;
  }

  // Transform the poses to align the body frames.
  for (auto &element : target_elements) {
    element->trans_Wi_Pi =
        element->rot_q_Wi_Pi * trans_Pi_Pj + element->trans_Wi_Pi;
    element->rot_q_Wi_Pi = element->rot_q_Wi_Pi * rot_q_Pi_Pj;
  }

  // Solve the Umeyama problem to initialize the transformation between world
  // frames (W2 to W1).
  if (!SolveUmeyama(target_elements, trans_Wi_Wj, rot_q_Wi_Wj)) {
    spdlog::critical(
        "Fail to solve Umeyama problem in P2P extrinsic parameters "
        "initialization");
    return false;
  }

  // ===========================================================================

  rot_q_Pi_Pj.normalize();
  rot_q_Wi_Wj.normalize();

  return true;
}

bool P2PExtrinsicInitializer::SolveAXXB(
    const P2PSolverElements &solver_elements, Eigen::Vector3d &trans_Pi_Pj,
    Eigen::Quaterniond &rot_q_Pi_Pj) {
  // Set default values.
  trans_Pi_Pj = Eigen::Vector3d::Zero();
  rot_q_Pi_Pj = Eigen::Quaterniond::Identity();

  // ===========================================================================

  // Step 1: Calculate relative poses from solver elements and fill the linear
  // solving matrix.
  size_t rel_pose_num = solver_elements.size() - 1;
  Eigen::MatrixXd rot_solver_M = Eigen::MatrixXd::Zero(9 * rel_pose_num, 9);
  Eigen::MatrixXd trans_solver_N = Eigen::MatrixXd::Zero(3 * rel_pose_num, 4);
  for (size_t i = 0; i < rel_pose_num; ++i) {
    Eigen::Vector3d rel_trans_Pi =
        solver_elements[i]->rot_q_Wi_Pi.inverse() *
        (solver_elements[i + 1]->trans_Wi_Pi - solver_elements[i]->trans_Wi_Pi);

    Eigen::Quaterniond rel_rot_q_Pi =
        solver_elements[i]->rot_q_Wi_Pi.inverse() *
        solver_elements[i + 1]->rot_q_Wi_Pi;
    Eigen::Matrix3d rel_rot_R_Pi = rel_rot_q_Pi.toRotationMatrix();

    Eigen::Quaterniond rel_rot_q_Pj =
        solver_elements[i]->rot_q_Wj_Pj.inverse() *
        solver_elements[i + 1]->rot_q_Wj_Pj;
    Eigen::Matrix3d rel_rot_R_Pj = rel_rot_q_Pj.toRotationMatrix();

    rot_solver_M.block<9, 9>(9 * i, 0) =
        Eigen::MatrixXd::Identity(9, 9) -
        Eigen::kroneckerProduct(rel_rot_R_Pi, rel_rot_R_Pj);
    trans_solver_N.block<3, 3>(3 * i, 0) =
        rel_rot_R_Pi - Eigen::MatrixXd::Identity(3, 3);
    trans_solver_N.block<3, 1>(3 * i, 3) = rel_trans_Pi;
  }

  // ===========================================================================

  // Step 2: Solve the extrinsic rotation between body frames.
  Eigen::JacobiSVD<Eigen::MatrixXd> rot_svd(
      rot_solver_M, Eigen::ComputeFullV | Eigen::ComputeFullU);

  if (!rot_svd.computeV()) {
    spdlog::critical(
        "SVD decomposition failed when solving for the rotation part of AXXB.");
    return false;
  }

  Eigen::Matrix3d rot_R_alpha;
  rot_R_alpha.row(0) = rot_svd.matrixV().block<3, 1>(0, 8).transpose();
  rot_R_alpha.row(1) = rot_svd.matrixV().block<3, 1>(3, 8).transpose();
  rot_R_alpha.row(2) = rot_svd.matrixV().block<3, 1>(6, 8).transpose();

  double det = rot_R_alpha.determinant();
  double alpha = std::pow(std::abs(det), 4. / 3.) / det;

  Eigen::HouseholderQR<Eigen::Matrix3d> qr_decomp(rot_R_alpha / alpha);
  Eigen::Matrix3d orthogonal = qr_decomp.householderQ();
  Eigen::Matrix3d adjusted = orthogonal.transpose() * rot_R_alpha / alpha;
  Eigen::Vector3d adjusted_diagonal = adjusted.diagonal();

  Eigen::Matrix3d rot_R_Pi_Pj;
  for (int i = 0; i < 3; i++) {
    rot_R_Pi_Pj.block<3, 1>(0, i) =
        int(adjusted_diagonal(i) >= 0 ? 1 : -1) * orthogonal.col(i);
  }
  rot_q_Pi_Pj = Eigen::Quaterniond(rot_R_Pi_Pj);

  // ===========================================================================

  // Step 3: Solve the extrinsic translation between body frames.
  Eigen::MatrixXd trans_solver_b = Eigen::MatrixXd::Zero(3 * rel_pose_num, 1);
  for (size_t i = 0; i < rel_pose_num; ++i) {
    Eigen::Vector3d rel_trans_Pj =
        solver_elements[i]->rot_q_Wj_Pj.inverse() *
        (solver_elements[i + 1]->trans_Wj_Pj - solver_elements[i]->trans_Wj_Pj);
    trans_solver_b.block<3, 1>(3 * i, 0) = rot_R_Pi_Pj * rel_trans_Pj;
  }

  Eigen::Vector4d trans_solver_result =
      trans_solver_N.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
          .solve(trans_solver_b);
  trans_solver_result /= trans_solver_result(3);

  trans_Pi_Pj = trans_solver_result.head<3>();

  // ===========================================================================

  return true;
}

bool P2PExtrinsicInitializer::SolveUmeyama(
    const P2PSolverElements &solver_elements, Eigen::Vector3d &trans_Wi_Wj,
    Eigen::Quaterniond &rot_q_Wi_Wj) {
  // Set default values.
  trans_Wi_Wj = Eigen::Vector3d::Zero();
  rot_q_Wi_Wj = Eigen::Quaterniond::Identity();

  // Convert the points for the Umeyama alignment into matrix form.
  const int kPointDim = 3;
  const int kPointNum = solver_elements.size();
  Eigen::MatrixXd point_matrix_x(kPointDim, kPointNum);
  Eigen::MatrixXd point_matrix_y(kPointDim, kPointNum);
  for (int i = 0; i < kPointNum; ++i) {
    point_matrix_x.col(i) = solver_elements[i]->trans_Wi_Pi;
    point_matrix_y.col(i) = solver_elements[i]->trans_Wj_Pj;
  }

  // Mean.
  Eigen::VectorXd point_mean_x = point_matrix_x.rowwise().mean();
  Eigen::VectorXd point_mean_y = point_matrix_y.rowwise().mean();

  // Variance.
  double point_sigma_y =
      1. / kPointNum *
      ((point_matrix_y.colwise() - point_mean_y).colwise().squaredNorm().sum());

  // Covariance matrix.
  Eigen::MatrixXd point_cov_xy = Eigen::MatrixXd::Zero(kPointDim, kPointDim);
  for (int i = 0; i < kPointNum; ++i) {
    Eigen::VectorXd point_centered_x = point_matrix_x.col(i) - point_mean_x;
    Eigen::VectorXd point_centered_y = point_matrix_y.col(i) - point_mean_y;
    point_cov_xy += point_centered_x * point_centered_y.transpose();
  }
  point_cov_xy /= kPointNum;

  // Perform SVD on the covariance matrix.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      point_cov_xy, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singular_values = svd.singularValues();
  // Check the decomposition result.
  if (singular_values.head(kPointDim - 1).array().abs().maxCoeff() <
      std::numeric_limits<double>::epsilon()) {
    spdlog::critical(
        "Degenerate covariance rank, unable to achieve Umeyama alignment. ");
    return false;
  }

  // Ensure a RHS coordinate system (Kabsch algorithm).
  Eigen::MatrixXd eye_3d = Eigen::MatrixXd::Identity(kPointDim, kPointDim);
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
    eye_3d(kPointDim - 1, kPointDim - 1) = -1;
  }

  // Extrinsic rotation between world frames.
  Eigen::Matrix3d rot_R_Wi_Wj =
      svd.matrixU() * eye_3d * svd.matrixV().transpose();

  // Scale result.
  double scale = 1.0 / point_sigma_y * singular_values.head(kPointDim).sum();

  // Extrinsic translation between world frames.
  trans_Wi_Wj = point_mean_x - scale * rot_R_Wi_Wj * point_mean_y;
  // Convert to quaternion form.
  rot_q_Wi_Wj = Eigen::Quaterniond(rot_R_Wi_Wj);

  return true;
}

}  // namespace hpgt
