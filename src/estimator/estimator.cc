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

#include "hpgt/estimator/estimator.h"

#include "hpgt/estimator/factor_adder.h"
#include "hpgt/initializer/spatial_extrinsic_initializer.h"
#include "hpgt/initializer/time_offset_initializer.hpp"

namespace hpgt {

Estimator::Estimator(SystemConfig::Ptr system_config,
                     SensorDataManager::Ptr sensor_data_manager) {
  system_config_ = system_config;
  sensor_data_manager_ = sensor_data_manager;

  trans_spline_name_ = "trans_spline";
  rot_spline_name_ = "rot_spline";
  spline_sensor_label_ = "";
  opt_start_time_ = -1.;
  opt_end_time_ = -1.;
}

bool Estimator::Initialize(double time_margin) {
  auto toff_initializer = TimeOffsetInitializer::Create();
  auto spatial_initializer = SpatialExtrinsicInitializer::Create();

  calib_parameter_ = CalibParameter::Create();

  // ===========================================================================

  // Step 1: Initialize the time-invariant calibration parameters.
  // Step 1.1: Initialize the time offset.
  spdlog::info("--------------- Initialize time offset ---------------");

  // If the time offsets need to be optimized, perform initialization.
  bool opt_temporal_param_flag = system_config_->get_opt_temporal_param_flag();
  bool toff_init_flag = toff_initializer->Initialize(
      sensor_data_manager_, calib_parameter_, opt_temporal_param_flag);
  if (!toff_init_flag) {
    spdlog::critical("Fail to initialize system time offset. ");
    return false;
  }

  // If not, set to the initial values.
  if (!opt_temporal_param_flag) {
    for (const auto &[label, config] :
         sensor_data_manager_->GetAllPoseConfig()) {
      calib_parameter_->toff_B_Pi[label] = config.toff_BP_init;
      calib_parameter_->toff_B_Pi_increment[label] = 0.;
    }
    for (const auto &[label, config] :
         sensor_data_manager_->GetAllImuConfig()) {
      calib_parameter_->toff_B_Ii[label] = config.toff_BI_init;
      calib_parameter_->toff_B_Ii_increment[label] = 0.;
    }
  }

  // Step 1.2: Initialize spatial extrinsic parameters.
  spdlog::info("------------ Initialize spatial parameters -----------");

  // If the spatial extrinsic parameters need to be optimized, perform
  // initialization.
  bool opt_spatial_param_flag = system_config_->get_opt_spatial_param_flag();
  bool spatial_init_flag = spatial_initializer->Initialize(
      sensor_data_manager_, calib_parameter_,
      system_config_->get_gravity_magnitude(), opt_spatial_param_flag);
  if (!spatial_init_flag) {
    spdlog::critical(
        "Fail to initialize system spatial extrinsic parameters. ");
    return false;
  }

  // If not, set to the initial values.
  if (!opt_spatial_param_flag) {
    for (const auto &[label, config] :
         sensor_data_manager_->GetAllPoseConfig()) {
      calib_parameter_->trans_B_Pi[label] = config.trans_BP_init;
      calib_parameter_->rot_B_Pi[label] = Sophus::SO3d(config.rot_q_BP_init);
      calib_parameter_->trans_G_Wi[label] = config.trans_GW_init;
      calib_parameter_->rot_G_Wi[label] = Sophus::SO3d(config.rot_q_GW_init);
    }
    for (const auto &[label, config] :
         sensor_data_manager_->GetAllImuConfig()) {
      calib_parameter_->trans_B_Ii[label] = config.trans_BI_init;
      calib_parameter_->rot_B_Ii[label] = Sophus::SO3d(config.rot_q_BI_init);
    }
  }

  // Step 1.3: Initialize other time-invariant states (intrinsic parameters).
  for (const auto &[label, config] : sensor_data_manager_->GetAllImuConfig()) {
    calib_parameter_->imu_intri[label] = IMUIntrinsic::Create();
  }

  // ===========================================================================

  // Step 2: Determine the start / stop times of the continuous-time spline
  // function
  // Step 2.1: Determine the joint start / stop times for all sensor sequences.
  spdlog::info("-------------- Initialize system spline --------------");

  for (const auto &[label, config] : sensor_data_manager_->GetAllPoseConfig()) {
    double tau_start = sensor_data_manager_->GetPoseStartTimeByLabel(label);
    double tau_end = sensor_data_manager_->GetPoseEndTimeByLabel(label);

    double t_start = calib_parameter_->toff_B_Pi.at(label) + tau_start;
    double t_end = calib_parameter_->toff_B_Pi.at(label) + tau_end;

    if (t_start > opt_start_time_) {
      opt_start_time_ = t_start;
    }
    if (t_end < opt_end_time_ || opt_end_time_ < 0) {
      opt_end_time_ = t_end;
    }
  }

  for (const auto &[label, config] : sensor_data_manager_->GetAllImuConfig()) {
    double tau_start = sensor_data_manager_->GetImuStartTimeByLabel(label);
    double tau_end = sensor_data_manager_->GetImuEndTimeByLabel(label);

    double t_start = calib_parameter_->toff_B_Ii.at(label) + tau_start;
    double t_end = calib_parameter_->toff_B_Ii.at(label) + tau_end;

    if (t_start > opt_start_time_) {
      opt_start_time_ = t_start;
    }
    if (t_end < opt_end_time_ || opt_end_time_ < 0) {
      opt_end_time_ = t_end;
    }
  }

  // Step 2.2: Consider the time margin to determine the start / end times of
  // the spline.
  const double kKnotInterval = system_config_->get_spline_knot_interval();
  const double kSplineKnotDeltaTime =
      static_cast<double>(SystemConfig::kSplineOrder - 2) / 2. * kKnotInterval;
  // Ensure the time range for constructing and optimizing the B-spline is
  // within the time range of sensor's measurement.
  time_margin += kSplineKnotDeltaTime + kKnotInterval +
                 system_config_->get_max_toff_change();

  opt_start_time_ = opt_start_time_ + time_margin;
  opt_end_time_ = opt_end_time_ - time_margin;
  if (opt_end_time_ < 0 || opt_end_time_ < 0 ||
      opt_end_time_ < opt_start_time_) {
    spdlog::critical(
        "System spline start/stop time in body frame exception: {:.9f}/{:.9f}");
  }

  // ===========================================================================

  // Step 3: Initialize the continuous-time pose spline function (time-varying)
  // in our system.
  // Step 3.1: Construct spline bundle with the spline information.
  auto trans_spline_info =
      SplineInfo(trans_spline_name_, SplineType::EuclideanSpline,
                 opt_start_time_, opt_end_time_, kKnotInterval);
  auto rot_spline_info =
      SplineInfo(rot_spline_name_, SplineType::So3Spline, opt_start_time_,
                 opt_end_time_, kKnotInterval);
  spline_bundle_ =
      SplineBundleType::Create({trans_spline_info, rot_spline_info});

  auto &trans_spline = spline_bundle_->GetR3dSpline(trans_spline_name_);
  auto &rot_spline = spline_bundle_->GetSo3dSpline(rot_spline_name_);
  opt_start_time_ = trans_spline.MinTime();
  opt_end_time_ = trans_spline.MaxTime();
  size_t knot_size = trans_spline.get_knots().size();

  // Time check.
  if (!(std::abs(trans_spline.MinTime() - rot_spline.MinTime()) < 1.e-9) ||
      !(std::abs(trans_spline.MaxTime() - rot_spline.MaxTime()) < 1.e-9) ||
      trans_spline.get_knots().size() != rot_spline.get_knots().size()) {
    spdlog::critical("Inconsistent parameters for system B-splines. ");
    return false;
  } else if ((opt_end_time_ - opt_start_time_) <
             2 * SystemConfig::kSplineOrder * time_margin) {
    spdlog::critical(
        "The time range for optimization is too small. Please input "
        "measurement data covering a longer time period. ");
    return false;
  }
  spdlog::info(
      "Construct system spline with start / end time: {:.6f} / {:.6f}, and "
      "{} control points. ",
      opt_start_time_, opt_end_time_, knot_size);

  // Step 3.2: Determine the reference pose sequence to initialize the knots.
  for (const auto &[label, config] : sensor_data_manager_->GetAllPoseConfig()) {
    if (!config.abs_pose_flag) {
      continue;
    }

    if (spline_sensor_label_ == "") {
      spline_sensor_label_ = label;
    }

    if (sensor_data_manager_->GetPoseFrequencyByLabel(label) >
        sensor_data_manager_->GetPoseFrequencyByLabel(spline_sensor_label_)) {
      spline_sensor_label_ = label;
    }
  }

  if (spline_sensor_label_ == "") {
    spdlog::critical(
        "No sensor measurement applicable to initialize spline knots. ");
    return false;
  } else {
    spdlog::info("Initialize spline knots with measurement of {}. ",
                 spline_sensor_label_);
  }

  // Step 3.3: Initialize control points based on interpolation of the reference
  // pose sequence.
  Eigen::Quaterniond rot_q_P_B =
      calib_parameter_->rot_B_Pi.at(spline_sensor_label_)
          .inverse()
          .unit_quaternion();
  Eigen::Vector3d trans_P_B =
      -(rot_q_P_B * calib_parameter_->trans_B_Pi.at(spline_sensor_label_));
  Eigen::Vector3d trans_G_W =
      calib_parameter_->trans_G_Wi.at(spline_sensor_label_);
  Eigen::Quaterniond rot_q_G_W =
      calib_parameter_->rot_G_Wi.at(spline_sensor_label_).unit_quaternion();
  double toff_P_B = -calib_parameter_->toff_B_Pi.at(spline_sensor_label_);

  auto &spline_pose_seq =
      sensor_data_manager_->GetPoseSeqByLabel(spline_sensor_label_);
  double current_knot_time = opt_start_time_ - kSplineKnotDeltaTime + toff_P_B;
  for (size_t i = 0; i < knot_size; ++i) {
    Eigen::Vector3d trans_W_P;
    Eigen::Quaterniond rot_q_W_P;

    if (!InitializerHelper::GetTargetPose(spline_pose_seq, current_knot_time,
                                          trans_W_P, rot_q_W_P)) {
      spdlog::critical("Fail to initialize system spline knots. ");
      return false;
    }

    Eigen::Vector3d knot_trans;
    Eigen::Quaterniond knot_rot_q;
    // Convert to the form T_G_B.
    knot_trans =
        rot_q_G_W * rot_q_W_P * trans_P_B + rot_q_G_W * trans_W_P + trans_G_W;
    knot_rot_q = rot_q_G_W * rot_q_W_P * rot_q_P_B;
    trans_spline.get_knot(i) = knot_trans;
    rot_spline.get_knot(i) = Sophus::SO3d(knot_rot_q);

    current_knot_time += kKnotInterval;
  }

  spdlog::info("Complete the initialization of B-spline knots. ");

  // ===========================================================================

  return true;
}

bool Estimator::BuildAndOptimize() {
  // Extend time range of sensor measurements to cover entire B-spline after
  // time offsets are optimized. For measurements outside the B-spline time
  // range, we simply set their gradients to zero.
  const double kMeasStartTime =
      opt_start_time_ - system_config_->get_max_toff_change();
  const double kMeasEndTime =
      opt_end_time_ + system_config_->get_max_toff_change();

  auto factor_adder = FactorAdder::Create(this);

  // ===========================================================================

  // Step 1: Add the factors constructed from pose measurements.
  for (const auto &[label, config] : sensor_data_manager_->GetAllPoseConfig()) {
    double toff_B_P = calib_parameter_->toff_B_Pi.at(label);
    if (config.abs_pose_flag) {
      // Step 1.1: Add absolute pose factor.
      int abs_factor_count = 0;
      double kPoseTransWeight = 1. / config.trans_noise;
      double kPoseRotWeight = 1. / config.rot_noise;
      for (auto const &pose_frame :
           sensor_data_manager_->GetPoseSeqByLabel(label)) {
        if (pose_frame->timestamp + toff_B_P <= kMeasStartTime ||
            pose_frame->timestamp + toff_B_P >= kMeasEndTime)
          continue;

        factor_adder->AddAbsPoseFactor(label, pose_frame, kPoseTransWeight,
                                       kPoseRotWeight);
        ++abs_factor_count;
      }

      spdlog::info("Add a total of {} absolute pose factors. ",
                   abs_factor_count);
    } else {
      // Step 1.2: Add relative pose factor, to handle trajectories that have
      // cumulative error.
      // TODO: Advanced features will be released after the paper is accepted.
      spdlog::info(
          "Advanced features will be released after the paper is accepted.");
    }
  }

// ===========================================================================

// Step 2: Add the factors constructed from IMU measurements, including
// acceleration and angular velocity.
for (const auto &[label, config] : sensor_data_manager_->GetAllImuConfig()) {
  int imu_factor_count = 0;
  double toff_B_I = calib_parameter_->toff_B_Ii.at(label);
  ImuModelType imu_model_type = config.model_type;
  // Convert the continuous-time noise into discrete-time noise.
  double kAccWeight = 1. / (config.noise[0] * std::sqrt(config.frequency));
  double kGyrWeight = 1. / (config.noise[2] * std::sqrt(config.frequency));

  for (auto const &imu_frame : sensor_data_manager_->GetImuSeqByLabel(label)) {
    if (imu_frame->timestamp + toff_B_I <= kMeasStartTime ||
        imu_frame->timestamp + toff_B_I >= kMeasEndTime)
      continue;

    factor_adder->AddImuAccFactor(label, imu_frame, imu_model_type, kAccWeight);
    factor_adder->AddImuGyrFactor(label, imu_frame, imu_model_type, kGyrWeight);
    ++imu_factor_count;
  }

  spdlog::info("Add a total of {} imu factors. ", imu_factor_count);
}

// ===========================================================================

// // Step 3: Add prior constraints on relative poses to adjacent knots of the
// // B-spline.
// int knots_num =
//     spline_bundle_->GetR3dSpline(trans_spline_name_).get_knots().size();
// int knot_prior_factor_count = 0;
// // Set the weights to match those of the sensors used in constructing the
// // B-spline.
// double kKnotPriorTransWeight = 4. /
// sensor_data_manager_->GetAllPoseConfig()
//                                         .at(spline_sensor_label_)
//                                         .trans_noise;
// double kKnotPriorRotWeight = 4. / sensor_data_manager_->GetAllPoseConfig()
//                                       .at(spline_sensor_label_)
//                                       .rot_noise;
// for (int i = 0; i < system_config_->kSplineOrder - 1; ++i) {
//   // We attempt to fix the relative poses of the knots near the start and
//   // end points of the B-spline to their initial states to prevent them
//   // from diverging due to insufficient constraints.
//   int front_i = i;
//   int front_j = i + 1;
//   int rear_j = knots_num - 1 - i;
//   int rear_i = rear_j - 1;

// factor_adder->AddSplineKnotPriorFactor(
//     front_i, front_j, kKnotPriorTransWeight, kKnotPriorRotWeight);
// factor_adder->AddSplineKnotPriorFactor(rear_i, rear_j,
// kKnotPriorTransWeight,
//                                        kKnotPriorRotWeight);

//   ++knot_prior_factor_count;
// }

// spdlog::info("Add a total of {} spline knot prior factors. ",
//              knot_prior_factor_count);

// ===========================================================================

// Step 4: Set options for ceres solver and solve the problem.
ceres::Solver::Options options;
ceres::Solver::Summary summary;
options.trust_region_strategy_type = ceres::DOGLEG;
// options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
options.max_num_iterations = 15;
options.num_threads = 10;
options.minimizer_progress_to_stdout = true;

// Solve the problem.
spdlog::info("Perform gradient descent, it may take some time... ");
ceres::Solve(options, &problem_, &summary);

// ===========================================================================

return summary.IsSolutionUsable();
}

bool Estimator::PrintAndSaveResult(const std::string &output_calib_path,
                                   const std::string &output_traj_path) {
  // Copy the current system configuration as the basis for the output.
  SystemConfig::Ptr output_config = system_config_;

  // ===========================================================================

  // Step 1: Print calibration parameters and update the results in the
  // output configuration.

  spdlog::info("--------- Print calibration parameters result --------");

  // Step 1.1: Pose sensors.
  for (const auto &[label, config] : sensor_data_manager_->GetAllPoseConfig()) {
    spdlog::info("- Calibration parameters for [{}]. ", label);

    double toff_B_Pi = calib_parameter_->toff_B_Pi.at(label) +
                       calib_parameter_->toff_B_Pi_increment.at(label);
    Eigen::Vector3d trans_B_Pi = calib_parameter_->trans_B_Pi.at(label);
    Eigen::Quaterniond rot_q_B_Pi =
        calib_parameter_->rot_B_Pi.at(label).unit_quaternion();
    Eigen::Vector3d trans_G_Wi = calib_parameter_->trans_G_Wi.at(label);
    Eigen::Quaterniond rot_q_G_Wi =
        calib_parameter_->rot_G_Wi.at(label).unit_quaternion();

    // Print the results.
    spdlog::info("Time offset (toff_BP) for [{}] is optimized to: {:.6f}",
                 label, toff_B_Pi);

    spdlog::info(
        "Translation (tran_BP) for body frame is optimized to:  [{:.6f}, "
        "{:.6f}, {:.6f}]",
        trans_B_Pi.x(), trans_B_Pi.y(), trans_B_Pi.z());
    spdlog::info(
        "Rotation (rot_q_BP) for body frame is optimized to:    [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_q_B_Pi.x(), rot_q_B_Pi.y(), rot_q_B_Pi.z(), rot_q_B_Pi.w());

    spdlog::info(
        "Translation (tran_GW) for world frame is optimized to: [{:.6f}, "
        "{:.6f}, {:.6f}]",
        trans_G_Wi.x(), trans_G_Wi.y(), trans_G_Wi.z());
    spdlog::info(
        "Rotation (rot_q_GW) for world frame is optimized to:   [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_q_G_Wi.x(), rot_q_G_Wi.y(), rot_q_G_Wi.z(), rot_q_G_Wi.w());

    // Updata the results in the output configuration.
    for (auto &pose_config : output_config->get_pose_config()) {
      if (pose_config.file_name == label) {
        pose_config.toff_BP_init = toff_B_Pi;
        pose_config.trans_BP_init = trans_B_Pi;
        pose_config.rot_q_BP_init = rot_q_B_Pi;
        pose_config.trans_GW_init = trans_G_Wi;
        pose_config.rot_q_GW_init = rot_q_G_Wi;
      }
    }
  }

  // ===========================================================================

  // Step 1.2: IMUs.
  for (const auto &[label, config] : sensor_data_manager_->GetAllImuConfig()) {
    spdlog::info("- Calibration parameters for [{}]. ", label);

    double toff_B_Ii = calib_parameter_->toff_B_Ii.at(label) +
                       calib_parameter_->toff_B_Ii_increment.at(label);
    Eigen::Vector3d trans_B_Ii = calib_parameter_->trans_B_Ii.at(label);
    Eigen::Quaterniond rot_q_B_Ii =
        calib_parameter_->rot_B_Ii.at(label).unit_quaternion();
    Eigen::Vector3d acc_bias_i =
        calib_parameter_->imu_intri.at(label)->acc_bias;
    Eigen::Vector3d gyr_bias_i =
        calib_parameter_->imu_intri.at(label)->gyr_bias;
    Eigen::Matrix<double, 6, 1> acc_map_coeff_i =
        calib_parameter_->imu_intri.at(label)->acc_map_coeff;
    Eigen::Matrix<double, 6, 1> gyr_map_coeff_i =
        calib_parameter_->imu_intri.at(label)->gyr_map_coeff;
    Eigen::Quaterniond rot_gyr_acc_i =
        calib_parameter_->imu_intri.at(label)->rot_gyr_acc.unit_quaternion();

    // Print the results.
    spdlog::info("Time offset (toff_BI) for [{}] is optimized to: {:.6f}",
                 label, toff_B_Ii);
    spdlog::info(
        "Translation (tran_BI) for body frame is optimized to:  [{:.6f}, "
        "{:.6f}, {:.6f}]",
        trans_B_Ii.x(), trans_B_Ii.y(), trans_B_Ii.z());
    spdlog::info(
        "Rotation (rot_q_BI) for body frame is optimized to:    [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_q_B_Ii.x(), rot_q_B_Ii.y(), rot_q_B_Ii.z(), rot_q_B_Ii.w());

    spdlog::info("Accelerometer bias is optimized to: [{:.6f}, {:.6f}, {:.6f}]",
                 acc_bias_i.x(), acc_bias_i.y(), acc_bias_i.z());
    spdlog::info("Gyroscope bias is optimized to:     [{:.6f}, {:.6f}, {:.6f}]",
                 gyr_bias_i.x(), gyr_bias_i.y(), gyr_bias_i.z());

    spdlog::info(
        "Accelerometer mapping coefficient is optimized to: [{:.6f}, {:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
        acc_map_coeff_i(0), acc_map_coeff_i(1), acc_map_coeff_i(2),
        acc_map_coeff_i(3), acc_map_coeff_i(4), acc_map_coeff_i(5));
    spdlog::info(
        "Gyroscope mapping coefficient is optimized to:     [{:.6f}, {:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
        gyr_map_coeff_i(0), gyr_map_coeff_i(1), gyr_map_coeff_i(2),
        gyr_map_coeff_i(3), gyr_map_coeff_i(4), gyr_map_coeff_i(5));

    spdlog::info(
        "Rotational misalignment (rot_gyr_acc_i) is optimized to: [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_gyr_acc_i.x(), rot_gyr_acc_i.y(), rot_gyr_acc_i.z(),
        rot_gyr_acc_i.w());

    // Updata the results in the output configuration.
    for (auto &imu_config : output_config->get_imu_config()) {
      if (imu_config.file_name == label) {
        imu_config.toff_BI_init = toff_B_Ii;
        imu_config.trans_BI_init = trans_B_Ii;
        imu_config.rot_q_BI_init = rot_q_B_Ii;
        imu_config.acc_bias_init = acc_bias_i;
        imu_config.gyr_bias_init = gyr_bias_i;
      }
    }
  }

  // ===========================================================================

  // Step 2: Save the calibration parameter and the trajectory result.
  spdlog::info("------------------- Save the result ------------------");

  // Save the updated system configuration (calibration parameter).
  output_config->ToJson(output_calib_path);

  // Save the system B-spline trajectory result (denoted as T_GB).
  std::ofstream file(output_traj_path);
  if (!file.is_open()) {
    spdlog::critical("Failed to open the output trajectory file: ",
                     output_traj_path);
    return false;
  }
  file << std::fixed << std::setprecision(9);

  const double kOutputInterval = 1. / system_config_->get_output_frequency();
  // We ignore parts of the trajectory with large errors at the start and end
  // points.
  const double kOutputMargin =
      SystemConfig::kSplineOrder * system_config_->get_spline_knot_interval();
  const double kOutputEndTime =
      spline_bundle_->GetR3dSpline(trans_spline_name_).MaxTime() -
      kOutputMargin;
  double current_time =
      spline_bundle_->GetR3dSpline(trans_spline_name_).MinTime() +
      kOutputMargin;
  while (current_time < kOutputEndTime) {
    Sophus::Vector3d trans;
    Sophus::SO3d rot;
    spline_bundle_->GetR3dSpline(trans_spline_name_)
        .Evaluate(current_time, trans);
    spline_bundle_->GetSo3dSpline(rot_spline_name_).Evaluate(current_time, rot);
    Eigen::Quaterniond quat = rot.unit_quaternion();

    file << current_time << " " << trans.x() << " " << trans.y() << " "
         << trans.z() << " " << quat.x() << " " << quat.y() << " " << quat.z()
         << " " << quat.w() << std::endl;

    current_time += kOutputInterval;
  }

  file.close();
  spdlog::info("Write system B-spline trajectory to file: {}",
               output_traj_path);

  // ===========================================================================

  return true;
}

}  // namespace hpgt
