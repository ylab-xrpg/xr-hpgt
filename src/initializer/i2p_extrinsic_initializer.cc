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

#include "hpgt/initializer/i2p_extrinsic_initializer.h"

namespace hpgt {

bool I2PExtrinsicInitializer::EstimateFromSeq(
    const PoseSequence &pose_seq, const ImuSequence &imu_seq,
    const double &toff, const double &g_magnitude, Eigen::Vector3d &trans_PI,
    Eigen::Quaterniond &rot_q_PI, Eigen::Quaterniond &rot_q_GW) {
  // Set default values.
  trans_PI = Eigen::Vector3d::Zero();
  rot_q_PI = Eigen::Quaterniond::Identity();
  rot_q_GW = Eigen::Quaterniond::Identity();

  // ===========================================================================

  // Step 1: Set the joint start and end times (under the clock of the pose
  // sequence).
  double pose_start_time = pose_seq.front()->timestamp + time_margin_;
  double pose_end_time = pose_seq.back()->timestamp - time_margin_;
  double imu_start_time = imu_seq.front()->timestamp + toff + time_margin_;
  double imu_end_time = imu_seq.back()->timestamp + toff - time_margin_;

  double joint_start_time =
      pose_start_time > imu_start_time ? pose_start_time : imu_start_time;
  double joint_end_time =
      pose_end_time < imu_end_time ? pose_end_time : imu_end_time;
  if (joint_start_time < 0) {
    spdlog::warn("Start time for I2P initialization is negative, set to 0. ");
    joint_start_time = 0.;
  }
  if (joint_start_time > joint_end_time) {
    spdlog::critical(
        "Timestamp exception in I2P initialization: {:.9f} / {:.9f}.",
        joint_start_time, joint_end_time);
    spdlog::critical("Start time should be less than end time. ");

    return false;
  }

  // ===========================================================================

  // Step 2: Construct the solver elements from pose and IMU sequences.
  I2PSolverElements solver_elements;

  auto reference_pose_ptr = pose_seq.begin();
  double reference_time, current_time, time_interval;
  Eigen::Quaterniond reference_q, current_q, delta_q;
  Eigen::Vector3d reference_p, current_p, delta_p;
  double delta_p_norm, delta_q_angle;
  int high_quality_num = 0, low_quality_num = 0;
  // Iterate through pose measurements and construct the solver elements.
  for (auto pose_iter = pose_seq.begin(); pose_iter != pose_seq.end();
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
    if (time_interval < element_interval_thresh_ / 3) {
      continue;
    } else if (delta_p_norm >= element_trans_thresh_ ||
               delta_q_angle >= element_rot_thresh_ ||
               time_interval >= element_interval_thresh_) {
      bool preinteg_flag = false;
      std::shared_ptr<ImuPreintegrator> integrator;
      // IMU preintegration.
      preinteg_flag =
          Preintegrate(imu_seq, reference_time - toff, current_time - toff,
                       g_magnitude, integrator);

      if (!preinteg_flag) {
        reference_pose_ptr = pose_iter;
        continue;
      }

      // Construct the solver element.
      auto solver_element =
          I2PSolverElement::Create(integrator, reference_p, reference_q,
                                   current_p, current_q, time_interval);

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
        "Insufficient solver elements for initializing the i2p extrinsic "
        "parameters. Please increase the motion duration and motion stimuli. ");
    return false;
  }
  if (low_quality_num > high_quality_num) {
    spdlog::warn(
        "Insufficient motion stimuli in I2P extrinsic parameters "
        "initialization, may lead to inaccurate calibration results, please "
        "perform more rapid rotation. ");
  }

  // We need to control the scale of the linear solver and prioritize
  // high quality solver elements.
  int sample_num = 0;
  I2PSolverElements target_elements;
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

  // Step 4: Build the linear solving system based on selected solver elements
  // to calculate the initial guesses of extrinsic parameters.
  Eigen::Vector3d trans_PI_result = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot_q_PI_result = Eigen::Quaterniond::Identity();
  Eigen::Vector3d gravity_in_W = Eigen::Vector3d::Zero();

  // Step 4.1: Initialize the extrinsic rotation.
  EstimateRot(target_elements, rot_q_PI_result);

  // Step 4.2: Simultaneously initialize the extrinsic translation and the
  // gravity vector represented in W.
  EstimateTransGravityAlign(target_elements, rot_q_PI_result, trans_PI_result,
                            gravity_in_W);

  // Step 4.3: Refine the initial guesses using the constraint of gravity
  // magnitude.
  RefineGravityAlign(target_elements, rot_q_PI_result, g_magnitude,
                     trans_PI_result, gravity_in_W);

  trans_PI = trans_PI_result;
  rot_q_PI = rot_q_PI_result.normalized();
  rot_q_GW = Eigen::Quaterniond::FromTwoVectors(gravity_in_W.normalized(),
                                                Eigen::Vector3d::UnitZ());

  // ===========================================================================

  return true;
}

bool I2PExtrinsicInitializer::Preintegrate(const ImuSequence &imu_seq,
                                           const double &time_0,
                                           const double &time_1,
                                           const double &g_magnitude,
                                           ImuPreintegrator::Ptr &integrator) {
  // Step 1: Retrieve IMU data for the given time period.
  ImuSequence integ_data;

  if (imu_seq.size() < 2) {
    spdlog::error("Insufficient IMU measurements to perform preintegration. ");
    integrator = nullptr;
    return false;
  }

  if (time_0 > time_1) {
    spdlog::error(
        "Unexpected increase in preintegration time, t_0: {:.9f}, t_1: "
        "{:.9f}. ",
        time_0, time_1);
    integrator = nullptr;
    return false;
  }

  // Assert the preintegration time is within the time range of the IMU.
  // Upper bound of the start IMU data.
  auto start_ub =
      std::upper_bound(imu_seq.begin(), imu_seq.end(), time_0,
                       [](double timestamp, const ImuFrame::Ptr &frame) {
                         return timestamp < frame->timestamp;
                       });

  // Upper bound of the end IMU data.
  auto end_ub =
      std::upper_bound(imu_seq.begin(), imu_seq.end(), time_1,
                       [](double timestamp, const ImuFrame::Ptr &frame) {
                         return timestamp < frame->timestamp;
                       });

  if (start_ub == imu_seq.begin() || end_ub == imu_seq.begin() ||
      start_ub == imu_seq.end() || end_ub == imu_seq.end()) {
    spdlog::error(
        "Specified preintegration time: ({:.9f}, {:.9f}) is out of the time "
        "range of the IMU measurements: ({:.9f}, {:.9f}). ",
        time_0, time_1, imu_seq.front()->timestamp, imu_seq.back()->timestamp);
    integrator = nullptr;
    return false;
  }

  // Start data.
  auto start_data = ImuFrame::Create();
  InitializerHelper::LinInterpImuFrame(*(start_ub - 1), *start_ub, time_0,
                                       start_data);
  integ_data.push_back(start_data);

  // Middle data.
  for (auto iter = start_ub; iter < end_ub; iter++) {
    integ_data.push_back(*iter);
  }

  // End data.
  auto end_data = ImuFrame::Create();
  InitializerHelper::LinInterpImuFrame(*(end_ub - 1), *end_ub, time_1,
                                       end_data);
  integ_data.push_back(end_data);

  // ===========================================================================

  // Step 2: Check the IMU data.
  // Loop through and ensure we do not have an zero dt values
  for (size_t i = 0; i < integ_data.size() - 1; ++i) {
    // This shouldn not happen.
    if (integ_data.at(i + 1)->timestamp < integ_data.at(i)->timestamp) {
      spdlog::error("Preintegration data error, timestamp decrease. ");
      return false;
    }

    if (std::abs(integ_data.at(i + 1)->timestamp -
                 integ_data.at(i)->timestamp) < 1e-6) {
      spdlog::debug(
          "Zero dt between IMU measurements {:.9f} and {:.9f}, remove the "
          "latter. ",
          integ_data.at(i)->timestamp, integ_data.at(i + 1)->timestamp);
      integ_data.erase(integ_data.begin() + i);
      --i;
    }
  }

  if (integ_data.size() < 2) {
    spdlog::error(
        "Insufficient IMU measurements within the specified preintegration "
        "time: ({:.9f}, {:.9f}). ",
        time_0, time_1);
    integrator = nullptr;
    return false;
  }

  // Preintegrate at least three data to prevent degeneration.
  if (integ_data.size() == 2) {
    auto extra_data = ImuFrame::Create();
    double extra_data_time =
        (integ_data.front()->timestamp + integ_data.back()->timestamp) / 2;
    InitializerHelper::LinInterpImuFrame(integ_data.front(), integ_data.back(),
                                         extra_data_time, extra_data);
    integ_data.insert(integ_data.begin() + 1, extra_data);

    spdlog::debug(
        "There are only two IMU data for preintegration within ({:.9f}, "
        "{:.9f}). "
        "Generate an extra IMU measurement at {:.9f}. ",
        integ_data.front()->timestamp, integ_data.back()->timestamp,
        extra_data_time);
  }

  // ===========================================================================

  // Step 3: Preintegration.
  // Loop through and compute the preintegration.
  for (size_t i = 0; i < integ_data.size(); ++i) {
    double timestamp = integ_data.at(i)->timestamp;
    Eigen::Vector3d acc = integ_data.at(i)->acc;
    Eigen::Vector3d gyr = integ_data.at(i)->gyr;

    spdlog::trace(
        "Preintegration data: t({:.9f}), acc_m({:.6f}, {:.6f}, {:.6f}), "
        "gyr_m({:.6f}, {:.6f}, {:.6f}). ",
        timestamp, acc.x(), acc.y(), acc.z(), gyr.x(), gyr.y(), gyr.z());

    if (i == 0) {
      integrator = ImuPreintegrator::Create(acc, gyr, g_magnitude);
    } else {
      double dt = timestamp - integ_data.at(i - 1)->timestamp;
      integrator->Propagate(dt, integ_data.at(i)->acc, integ_data.at(i)->gyr);
    }
  }

  // ===========================================================================

  return true;
}

void I2PExtrinsicInitializer::EstimateRot(
    const I2PSolverElements &solver_elements, Eigen::Quaterniond &rot_q_PI) {
  // Construct the over-determined equations for solving the extrinsic rotation.
  Eigen::MatrixXd solver_A(solver_elements.size() * 4, 4);
  solver_A.setZero();
  int matrix_index = 0;
  for (auto &element : solver_elements) {
    Eigen::Quaterniond rel_rot_q_I = element->preintegrator->get_q_curr();
    Eigen::Quaterniond rel_rot_q_P =
        element->rot_q_WP_start.inverse() * element->rot_q_WP_end;
    double screw_congruence_factor =
        InitializerUtils::QuatAngleDegree(rel_rot_q_I) /
        InitializerUtils::QuatAngleDegree(rel_rot_q_P);
    if (screw_congruence_factor < 1) {
      screw_congruence_factor = 1. / screw_congruence_factor;
    }
    // Robust kernal based on screw congruence theorem.
    double robust_kernel =
        1. / std::exp(robust_kernel_coeff_ * (screw_congruence_factor - 1));

    // Sub-matrix corresponding to the current solver element.
    solver_A.block<4, 4>(matrix_index * 4, 0) =
        robust_kernel * (InitializerUtils::QuatLeftProd(rel_rot_q_P) -
                         InitializerUtils::QuatRightProd(rel_rot_q_I));

    matrix_index++;
  }

  // Solve the homogeneous linear least squares problem using the SVD algorithm.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      solver_A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d rot_v4d_PI = svd.matrixV().col(3);
  rot_q_PI = Eigen::Quaterniond(rot_v4d_PI(0), rot_v4d_PI(1), rot_v4d_PI(2),
                                rot_v4d_PI(3));
}

void I2PExtrinsicInitializer::EstimateTransGravityAlign(
    const I2PSolverElements &solver_elements,
    const Eigen::Quaterniond &rot_q_PI, Eigen::Vector3d &trans_PI,
    Eigen::Vector3d &gravity_W) {
  // The total dimension of the states to be solved.
  const int kStateDim = 3 * (solver_elements.size() + 1) + 3 * 2;

  // Construct the linear solving system for initializing the extrinsic
  // translation and the gravity in W.
  Eigen::MatrixXd solver_A(kStateDim, kStateDim);
  solver_A.setZero();
  Eigen::VectorXd solver_b(kStateDim);
  solver_b.setZero();
  int matrix_index = 0;
  for (auto &element : solver_elements) {
    double dt = element->time_interval;
    Eigen::Matrix3d rot_R_PI = rot_q_PI.toRotationMatrix();
    Eigen::Matrix3d rot_R_WP_start = element->rot_q_WP_start.toRotationMatrix();
    Eigen::Matrix3d rot_R_WP_end = element->rot_q_WP_end.toRotationMatrix();
    Eigen::Matrix3d rot_R_WI_start = rot_R_WP_start * rot_R_PI;

    // Sub-matrix corresponding to the current solver element.
    Eigen::Matrix<double, 6, 12> single_A;
    single_A.setZero();
    Eigen::Matrix<double, 6, 1> single_b;
    single_b.setZero();

    single_A.block<3, 3>(0, 0) = -rot_R_WI_start.inverse() * dt;
    single_A.block<3, 3>(0, 6) = 1. / 2 * rot_R_WI_start.inverse() * dt * dt;
    single_A.block<3, 3>(0, 9) =
        rot_R_WI_start.inverse() * rot_R_WP_end - rot_R_PI.inverse();
    single_A.block<3, 3>(3, 0) = -rot_R_WI_start.inverse();
    single_A.block<3, 3>(3, 3) = rot_R_WI_start.inverse();
    single_A.block<3, 3>(3, 6) = rot_R_WI_start.inverse() * dt;

    single_b.block<3, 1>(0, 0) =
        element->preintegrator->get_alpha_curr() -
        rot_R_WI_start.inverse() * element->trans_WP_end +
        rot_R_WI_start.inverse() * element->trans_WP_start;
    single_b.block<3, 1>(3, 0) = element->preintegrator->get_beta_curr();

    Eigen::Matrix<double, 12, 12> At_A = single_A.transpose() * single_A;
    Eigen::Matrix<double, 12, 1> At_b = single_A.transpose() * single_b;

    // Fill the sub-matrix into the corresponding block.
    solver_A.block<6, 6>(matrix_index * 3, matrix_index * 3) +=
        At_A.topLeftCorner<6, 6>();
    solver_A.bottomRightCorner<6, 6>() += At_A.bottomRightCorner<6, 6>();
    solver_A.block<6, 6>(matrix_index * 3, kStateDim - 6) +=
        At_A.topRightCorner<6, 6>();
    solver_A.block<6, 6>(kStateDim - 6, matrix_index * 3) +=
        At_A.bottomLeftCorner<6, 6>();

    solver_b.segment<6>(matrix_index * 3) += At_b.topLeftCorner<6, 1>();
    solver_b.tail<6>() += At_b.bottomRightCorner<6, 1>();

    ++matrix_index;
  }

  // Solve the non-homogeneous linear least squares problem.
  Eigen::VectorXd solver_x(kStateDim);
  solver_x = solver_A.ldlt().solve(solver_b);

  trans_PI = solver_x.tail<3>();
  gravity_W = solver_x.segment<3>(kStateDim - 6);
}

void I2PExtrinsicInitializer::RefineGravityAlign(
    const I2PSolverElements &solver_elements,
    const Eigen::Quaterniond &rot_q_PI, const double &g_magnitude,
    Eigen::Vector3d &trans_PI, Eigen::Vector3d &gravity_W) {
  // The total dimension of the states to be solved.
  const int kStateDim = 3 * (solver_elements.size() + 1) + 2 + 3;
  // Gravity vector constrained by magnitude.
  Eigen::Vector3d standard_gravity = gravity_W.normalized() * g_magnitude;

  // Iterate and refine the extrinsic translation and the gravity in W.
  Eigen::MatrixXd solver_A(kStateDim, kStateDim);
  solver_A.setZero();
  Eigen::VectorXd solver_b(kStateDim);
  solver_b.setZero();
  Eigen::VectorXd solver_x(kStateDim);
  solver_x.setZero();
  int matrix_index = 0;
  for (int i = 0; i < refine_iter_num_; ++i) {
    // Calculate the tangent space basis of the current gravity vector.
    Eigen::Matrix<double, 3, 2> tangent_basis =
        GravityTangentBasis(standard_gravity);
    solver_A.setZero();
    solver_b.setZero();
    matrix_index = 0;
    for (auto &element : solver_elements) {
      double dt = element->time_interval;
      Eigen::Matrix3d rot_R_PI = rot_q_PI.toRotationMatrix();
      Eigen::Matrix3d rot_R_WP_start =
          element->rot_q_WP_start.toRotationMatrix();
      Eigen::Matrix3d rot_R_WP_end = element->rot_q_WP_end.toRotationMatrix();
      Eigen::Matrix3d rot_R_WI_start = rot_R_WP_start * rot_R_PI;

      // Sub-matrix corresponding to the current solver element.
      Eigen::Matrix<double, 6, 11> single_A;
      single_A.setZero();
      Eigen::Matrix<double, 6, 1> single_b;
      single_b.setZero();

      single_A.block<3, 3>(0, 0) = -rot_R_WI_start.inverse() * dt;
      single_A.block<3, 2>(0, 6) =
          1. / 2 * rot_R_WI_start.inverse() * tangent_basis * dt * dt;
      single_A.block<3, 3>(0, 8) =
          rot_R_WI_start.inverse() * rot_R_WP_end - rot_R_PI.inverse();
      single_A.block<3, 3>(3, 0) = -rot_R_WI_start.inverse();
      single_A.block<3, 3>(3, 3) = rot_R_WI_start.inverse();
      single_A.block<3, 2>(3, 6) =
          rot_R_WI_start.inverse() * tangent_basis * dt;

      single_b.block<3, 1>(0, 0) =
          element->preintegrator->get_alpha_curr() -
          rot_R_WI_start.inverse() * element->trans_WP_end +
          rot_R_WI_start.inverse() * element->trans_WP_start -
          1. / 2 * rot_R_WI_start.inverse() * standard_gravity * dt * dt;
      single_b.block<3, 1>(3, 0) =
          element->preintegrator->get_beta_curr() -
          rot_R_WI_start.inverse() * standard_gravity * dt;

      Eigen::Matrix<double, 11, 11> At_A = single_A.transpose() * single_A;
      Eigen::Matrix<double, 11, 1> At_b = single_A.transpose() * single_b;

      // Fill the sub-matrix into the corresponding block.
      solver_A.block<6, 6>(matrix_index * 3, matrix_index * 3) +=
          At_A.topLeftCorner<6, 6>();
      solver_A.bottomRightCorner<5, 5>() += At_A.bottomRightCorner<5, 5>();
      solver_A.block<6, 5>(matrix_index * 3, kStateDim - 5) +=
          At_A.topRightCorner<6, 5>();
      solver_A.block<5, 6>(kStateDim - 5, matrix_index * 3) +=
          At_A.bottomLeftCorner<5, 6>();

      solver_b.segment<6>(matrix_index * 3) += At_b.topLeftCorner<6, 1>();
      solver_b.tail<5>() += At_b.bottomRightCorner<5, 1>();

      ++matrix_index;
    }

    // Solve the non-homogeneous linear least squares problem.
    solver_x = solver_A.ldlt().solve(solver_b);

    // Coefficients of the tangent space basis.
    Eigen::Vector2d basis_delta = solver_x.segment<2>(kStateDim - 5);
    // Update the gravity.
    standard_gravity =
        (standard_gravity + tangent_basis * basis_delta).normalized() *
        g_magnitude;
  }

  trans_PI = solver_x.tail<3>();
  gravity_W = standard_gravity;
}

Eigen::Matrix<double, 3, 2> I2PExtrinsicInitializer::GravityTangentBasis(
    const Eigen::Vector3d &gravity) {
  Eigen::Vector3d normalized_gravity = gravity.normalized();
  Eigen::Vector3d basis_0, basis_1;
  Eigen::Vector3d basis_temp = Eigen::Vector3d::UnitZ();
  if (normalized_gravity == basis_temp) {
    basis_temp = Eigen::Vector3d::UnitX();
  }

  // Calculate basis 0 perpendicular to normalized gravity.
  basis_0 = (basis_temp -
             normalized_gravity * (normalized_gravity.transpose() * basis_temp))
                .normalized();
  // basis 1 = normalized gravity × basis 0
  basis_1 = normalized_gravity.cross(basis_0);

  Eigen::Matrix<double, 3, 2> basis;
  basis.block<3, 1>(0, 0) = basis_0;
  basis.block<3, 1>(0, 1) = basis_1;

  return basis;
}

}  // namespace hpgt
