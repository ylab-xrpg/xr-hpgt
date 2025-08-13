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

#include "hpgt/estimator/factor_adder.h"

#include "hpgt/factor/abs_pose_factor.hpp"
#include "hpgt/factor/imu_acc_factor.hpp"
#include "hpgt/factor/imu_gyr_factor.hpp"
#include "hpgt/factor/spline_knot_prior_factor.hpp"

namespace hpgt {

// Quaterniond manifold for ceres.
std::shared_ptr<ceres::EigenQuaternionManifold> FactorAdder::kQuatManifold_(
    new ceres::EigenQuaternionManifold());

FactorAdder::FactorAdder(Estimator *estimator)
    : estimator_(estimator),
      problem_(estimator->problem_),
      system_config_(estimator->system_config_),
      calib_parameter_(estimator->calib_parameter_),
      spline_bundle_(estimator->spline_bundle_),
      trans_spline_name_(estimator->trans_spline_name_),
      rot_spline_name_(estimator->rot_spline_name_),
      opt_start_time_(estimator->opt_start_time_),
      opt_end_time_(estimator->opt_end_time_) {}

bool FactorAdder::AddAbsPoseFactor(const std::string &label,
                                   const PoseFrame::Ptr &pose_frame,
                                   const double &trans_weight,
                                   const double &rot_weight) {
  // Step 1: Calculate the meta data for the translational and rotational
  // B-splines corresponding to the measurement.
  SplineMeta<SystemConfig::kSplineOrder> trans_meta, rot_meta;

  // Calculate the start and end times of the meta data.
  double meta_min_time = -1.;
  double meta_max_time = -1.;
  const double kMeasTimestamp =
      calib_parameter_->toff_B_Pi.at(label) + pose_frame->timestamp;

  if (!CalMetaMinMaxTime(kMeasTimestamp, meta_min_time, meta_max_time)) {
    return false;
  }

  if (!spline_bundle_->TimeInRangeForR3dSpline(meta_min_time,
                                               trans_spline_name_) ||
      !spline_bundle_->TimeInRangeForR3dSpline(meta_max_time,
                                               trans_spline_name_) ||
      !spline_bundle_->TimeInRangeForSo3dSpline(meta_min_time,
                                                rot_spline_name_) ||
      !spline_bundle_->TimeInRangeForSo3dSpline(meta_max_time,
                                                rot_spline_name_)) {
    spdlog::critical(
        "The min/max times of the spline meta exception for [{}] at {:.9f} ",
        label, pose_frame->timestamp);
    return false;
  }

  spline_bundle_->CalculateR3dSplineMeta(
      trans_spline_name_, {{meta_min_time, meta_max_time}}, trans_meta);
  spline_bundle_->CalculateSo3dSplineMeta(
      rot_spline_name_, {{meta_min_time, meta_max_time}}, rot_meta);

  // ===========================================================================

  // Step 2: Create an absolute pose factor and add parameter blocks.
  auto abs_pose_factor = AbsPoseFactor<SystemConfig::kSplineOrder>::Create(
      trans_meta, rot_meta, pose_frame, calib_parameter_->toff_B_Pi.at(label),
      trans_weight, rot_weight);

  // Parameter blocks for spline knots.
  for (size_t i = 0; i < trans_meta.NumParameters(); ++i) {
    abs_pose_factor->AddParameterBlock(3);
  }
  for (size_t i = 0; i < rot_meta.NumParameters(); ++i) {
    abs_pose_factor->AddParameterBlock(4);
  }

  // toff_BP
  abs_pose_factor->AddParameterBlock(1);
  // trans_BP
  abs_pose_factor->AddParameterBlock(3);
  // rot_BP
  abs_pose_factor->AddParameterBlock(4);
  // trans_GW
  abs_pose_factor->AddParameterBlock(3);
  // rot_GW
  abs_pose_factor->AddParameterBlock(4);
  // residual
  abs_pose_factor->SetNumResiduals(6);

  // ===========================================================================

  // Step 3: Organize the parameter block pointers in a vector and add the
  // residual block to the problem.
  std::vector<double *> param_block_vector;

  // Add the spline knot data to the parameter block vector.
  AddR3dKnotData(param_block_vector,
                 spline_bundle_->GetR3dSpline(trans_spline_name_), trans_meta);
  AddSo3dKnotsData(param_block_vector,
                   spline_bundle_->GetSo3dSpline(rot_spline_name_), rot_meta);

  // Add the calibration data to the vector.
  auto toff_BP_data = &calib_parameter_->toff_B_Pi_increment.at(label);
  auto trans_BP_data = calib_parameter_->trans_B_Pi.at(label).data();
  auto rot_BP_data = calib_parameter_->rot_B_Pi.at(label).data();
  auto trans_GW_data = calib_parameter_->trans_G_Wi.at(label).data();
  auto rot_GW_data = calib_parameter_->rot_G_Wi.at(label).data();

  param_block_vector.push_back(toff_BP_data);
  param_block_vector.push_back(trans_BP_data);
  param_block_vector.push_back(rot_BP_data);
  param_block_vector.push_back(trans_GW_data);
  param_block_vector.push_back(rot_GW_data);

  // Add the residual block.
  problem_.AddResidualBlock(abs_pose_factor, nullptr, param_block_vector);
  // Set manifold for quaternion blocks.
  problem_.SetManifold(rot_BP_data, kQuatManifold_.get());
  problem_.SetManifold(rot_GW_data, kQuatManifold_.get());

  // ===========================================================================

  // Step 4: Handle cases that require fixed parameter blocks.
  if (label == calib_parameter_->body_frame_label) {
    problem_.SetParameterBlockConstant(toff_BP_data);
    problem_.SetParameterBlockConstant(trans_BP_data);
    problem_.SetParameterBlockConstant(rot_BP_data);
  }

  if (label == calib_parameter_->world_frame_label) {
    problem_.SetParameterBlockConstant(trans_GW_data);
    if (!calib_parameter_->gravity_aligned) {
      problem_.SetParameterBlockConstant(rot_GW_data);
    }
  }

  if (!system_config_->get_opt_temporal_param_flag()) {
    problem_.SetParameterBlockConstant(toff_BP_data);
  }

  if (!system_config_->get_opt_spatial_param_flag()) {
    problem_.SetParameterBlockConstant(trans_BP_data);
    problem_.SetParameterBlockConstant(rot_BP_data);
    problem_.SetParameterBlockConstant(trans_GW_data);
    problem_.SetParameterBlockConstant(rot_GW_data);
  }

  // ===========================================================================

  return true;
}

bool FactorAdder::AddImuAccFactor(const std::string &label,
                                  const ImuFrame::Ptr &imu_frame,
                                  const ImuModelType &model_type,
                                  const double &acc_weight) {
  // Step 1: Calculate the meta data for the translational and rotational
  // B-splines corresponding to the measurement.
  SplineMeta<SystemConfig::kSplineOrder> trans_meta, rot_meta;

  // Calculate the start and end times of the meta data.
  double meta_min_time = -1.;
  double meta_max_time = -1.;
  const double kMeasTimestamp =
      calib_parameter_->toff_B_Ii.at(label) + imu_frame->timestamp;

  if (!CalMetaMinMaxTime(kMeasTimestamp, meta_min_time, meta_max_time)) {
    return false;
  }

  if (!spline_bundle_->TimeInRangeForR3dSpline(meta_min_time,
                                               trans_spline_name_) ||
      !spline_bundle_->TimeInRangeForR3dSpline(meta_max_time,
                                               trans_spline_name_) ||
      !spline_bundle_->TimeInRangeForSo3dSpline(meta_min_time,
                                                rot_spline_name_) ||
      !spline_bundle_->TimeInRangeForSo3dSpline(meta_max_time,
                                                rot_spline_name_)) {
    spdlog::critical(
        "The min/max times of the spline meta exception for [{}] at {:.9f} ",
        label, imu_frame->timestamp);
    return false;
  }

  spline_bundle_->CalculateR3dSplineMeta(
      trans_spline_name_, {{meta_min_time, meta_max_time}}, trans_meta);
  spline_bundle_->CalculateSo3dSplineMeta(
      rot_spline_name_, {{meta_min_time, meta_max_time}}, rot_meta);

  // ===========================================================================

  // Step 2: Create an accelerometer factor and add parameter blocks.
  auto imu_acc_factor = ImuAccFactor<SystemConfig::kSplineOrder>::Create(
      trans_meta, rot_meta, imu_frame, calib_parameter_->toff_B_Ii.at(label),
      acc_weight, system_config_->get_gravity_magnitude());

  // Parameter blocks for spline knots.
  for (size_t i = 0; i < trans_meta.NumParameters(); ++i) {
    imu_acc_factor->AddParameterBlock(3);
  }
  for (size_t i = 0; i < rot_meta.NumParameters(); ++i) {
    imu_acc_factor->AddParameterBlock(4);
  }

  // toff_BI
  imu_acc_factor->AddParameterBlock(1);
  // trans_BI
  imu_acc_factor->AddParameterBlock(3);
  // rot_BI
  imu_acc_factor->AddParameterBlock(4);
  // acc_bias
  imu_acc_factor->AddParameterBlock(3);
  // acc_map_coeff
  imu_acc_factor->AddParameterBlock(6);
  // residual
  imu_acc_factor->SetNumResiduals(3);

  // ===========================================================================

  // Step 3: Organize the parameter block pointers in a vector and add the
  // residual block to the problem.
  std::vector<double *> param_block_vector;

  // Add the spline knot data to the parameter block vector.
  AddR3dKnotData(param_block_vector,
                 spline_bundle_->GetR3dSpline(trans_spline_name_), trans_meta);
  AddSo3dKnotsData(param_block_vector,
                   spline_bundle_->GetSo3dSpline(rot_spline_name_), rot_meta);

  // Add the calibration data to the vector.
  auto toff_BI_data = &calib_parameter_->toff_B_Ii_increment.at(label);
  auto trans_BI_data = calib_parameter_->trans_B_Ii.at(label).data();
  auto rot_BI_data = calib_parameter_->rot_B_Ii.at(label).data();
  auto acc_bias_data = calib_parameter_->imu_intri.at(label)->acc_bias.data();
  auto acc_map_coeff_data =
      calib_parameter_->imu_intri.at(label)->acc_map_coeff.data();

  param_block_vector.push_back(toff_BI_data);
  param_block_vector.push_back(trans_BI_data);
  param_block_vector.push_back(rot_BI_data);
  param_block_vector.push_back(acc_bias_data);
  param_block_vector.push_back(acc_map_coeff_data);

  // Add the residual block.
  problem_.AddResidualBlock(imu_acc_factor, nullptr, param_block_vector);
  // Set manifold for quaternion blocks.
  problem_.SetManifold(rot_BI_data, kQuatManifold_.get());

  // ===========================================================================

  // Step 4: Handle cases that require fixed parameter blocks.
  if (label == calib_parameter_->body_frame_label) {
    problem_.SetParameterBlockConstant(toff_BI_data);
    problem_.SetParameterBlockConstant(trans_BI_data);
    problem_.SetParameterBlockConstant(rot_BI_data);
  }

  if (!system_config_->get_opt_temporal_param_flag()) {
    problem_.SetParameterBlockConstant(toff_BI_data);
  }

  if (!system_config_->get_opt_spatial_param_flag()) {
    problem_.SetParameterBlockConstant(trans_BI_data);
    problem_.SetParameterBlockConstant(rot_BI_data);
  }

  if (model_type == ImuModelType::kCalibrated) {
    problem_.SetParameterBlockConstant(acc_map_coeff_data);
  }

  // ===========================================================================

  return true;
}

bool FactorAdder::AddImuGyrFactor(const std::string &label,
                                  const ImuFrame::Ptr &imu_frame,
                                  const ImuModelType &model_type,
                                  const double &gyr_weight) {
  // Step 1: Calculate the meta data for the rotational B-spline corresponding
  // to the measurement.
  SplineMeta<SystemConfig::kSplineOrder> rot_meta;

  // Calculate the start and end times of the meta data.
  double meta_min_time = -1.;
  double meta_max_time = -1.;
  const double kMeasTimestamp =
      calib_parameter_->toff_B_Ii.at(label) + imu_frame->timestamp;

  if (!CalMetaMinMaxTime(kMeasTimestamp, meta_min_time, meta_max_time)) {
    return false;
  }

  if (!spline_bundle_->TimeInRangeForSo3dSpline(meta_min_time,
                                                rot_spline_name_) ||
      !spline_bundle_->TimeInRangeForSo3dSpline(meta_max_time,
                                                rot_spline_name_)) {
    spdlog::critical(
        "The min/max times of the spline meta exception for [{}] at {:.9f} ",
        label, imu_frame->timestamp);
    return false;
  }

  spline_bundle_->CalculateSo3dSplineMeta(
      rot_spline_name_, {{meta_min_time, meta_max_time}}, rot_meta);

  // ===========================================================================

  // Step 2: Create an gyroscope factor and add parameter blocks.
  auto imu_gyr_factor = ImuGyrFactor<SystemConfig::kSplineOrder>::Create(
      rot_meta, imu_frame, calib_parameter_->toff_B_Ii.at(label), gyr_weight);

  // Parameter blocks for spline knots.
  for (size_t i = 0; i < rot_meta.NumParameters(); ++i) {
    imu_gyr_factor->AddParameterBlock(4);
  }

  // toff_BI
  imu_gyr_factor->AddParameterBlock(1);
  // rot_BI
  imu_gyr_factor->AddParameterBlock(4);
  // gyr_bias
  imu_gyr_factor->AddParameterBlock(3);
  // gyr_map_coeff
  imu_gyr_factor->AddParameterBlock(6);
  // rot_gyr_acc
  imu_gyr_factor->AddParameterBlock(4);
  // residual
  imu_gyr_factor->SetNumResiduals(3);

  // ===========================================================================

  // Step 3: Organize the parameter block pointers in a vector and add the
  // residual block to the problem.
  std::vector<double *> param_block_vector;

  // Add the spline knot data to the parameter block vector.
  AddSo3dKnotsData(param_block_vector,
                   spline_bundle_->GetSo3dSpline(rot_spline_name_), rot_meta);

  // Add the calibration data to the vector.
  auto toff_BI_data = &calib_parameter_->toff_B_Ii_increment.at(label);
  auto rot_BI_data = calib_parameter_->rot_B_Ii.at(label).data();
  auto gyr_bias_data = calib_parameter_->imu_intri.at(label)->gyr_bias.data();
  auto gyr_map_coeff_data =
      calib_parameter_->imu_intri.at(label)->gyr_map_coeff.data();
  auto rot_gyr_acc_data =
      calib_parameter_->imu_intri.at(label)->rot_gyr_acc.data();

  param_block_vector.push_back(toff_BI_data);
  param_block_vector.push_back(rot_BI_data);
  param_block_vector.push_back(gyr_bias_data);
  param_block_vector.push_back(gyr_map_coeff_data);
  param_block_vector.push_back(rot_gyr_acc_data);

  // Add the residual block.
  problem_.AddResidualBlock(imu_gyr_factor, nullptr, param_block_vector);
  // Set manifold for quaternion blocks.
  problem_.SetManifold(rot_BI_data, kQuatManifold_.get());
  problem_.SetManifold(rot_gyr_acc_data, kQuatManifold_.get());

  // ===========================================================================

  // Step 4: Handle cases that require fixed parameter blocks.
  if (label == calib_parameter_->body_frame_label) {
    problem_.SetParameterBlockConstant(toff_BI_data);
    problem_.SetParameterBlockConstant(rot_BI_data);
  }

  if (!system_config_->get_opt_temporal_param_flag()) {
    problem_.SetParameterBlockConstant(toff_BI_data);
  }

  if (!system_config_->get_opt_spatial_param_flag()) {
    problem_.SetParameterBlockConstant(rot_BI_data);
  }

  if (model_type == ImuModelType::kCalibrated) {
    problem_.SetParameterBlockConstant(gyr_map_coeff_data);
    problem_.SetParameterBlockConstant(rot_gyr_acc_data);
  } else if (model_type == ImuModelType::kScale) {
    problem_.SetParameterBlockConstant(rot_gyr_acc_data);
  }

  // ===========================================================================

  return true;
}

bool FactorAdder::AddSplineKnotPriorFactor(const int &index_i,
                                           const int &index_j,
                                           const double &trans_weight,
                                           const double &rot_weight) {
  // Step 1: Calculate the relative poses of the B-spline knots in the initial
  // state.
  Eigen::Vector3d trans_i =
      spline_bundle_->GetR3dSpline(trans_spline_name_).get_knot(index_i);
  Sophus::SO3d rot_i =
      spline_bundle_->GetSo3dSpline(rot_spline_name_).get_knot(index_i);
  Eigen::Vector3d trans_j =
      spline_bundle_->GetR3dSpline(trans_spline_name_).get_knot(index_j);
  Sophus::SO3d rot_j =
      spline_bundle_->GetSo3dSpline(rot_spline_name_).get_knot(index_j);

  Eigen::Vector3d delta_trans_prior = rot_i.inverse() * (trans_j - trans_i);
  Sophus::SO3d delta_rot_prior = rot_i.inverse() * rot_j;

  // ===========================================================================

  // Step 2: Create an knot prior factor and add parameter blocks.
  auto knot_prior_factor = SplineKnotPriorFactor::Create(
      delta_trans_prior, delta_rot_prior, trans_weight, rot_weight);

  // The translational component of the previous knot.
  knot_prior_factor->AddParameterBlock(3);
  // The rotational component of the previous knot.
  knot_prior_factor->AddParameterBlock(4);
  // The translational component of the subsequent knot.
  knot_prior_factor->AddParameterBlock(3);
  // The rotational component of the subsequent knot.
  knot_prior_factor->AddParameterBlock(4);
  // Residual。
  knot_prior_factor->SetNumResiduals(6);

  // ===========================================================================

  // Step 3: Organize the parameter block pointers in a vector and add the
  // residual block to the problem.
  std::vector<double *> param_block_vector;

  auto &trans_spline = spline_bundle_->GetR3dSpline(trans_spline_name_);
  auto &rot_spline = spline_bundle_->GetSo3dSpline(rot_spline_name_);

  auto *trans_data_i =
      const_cast<double *>(trans_spline.get_knot(index_i).data());
  auto *rot_data_i = const_cast<double *>(rot_spline.get_knot(index_i).data());
  auto *trans_data_j =
      const_cast<double *>(trans_spline.get_knot(index_j).data());
  auto *rot_data_j = const_cast<double *>(rot_spline.get_knot(index_j).data());

  param_block_vector.push_back(trans_data_i);
  param_block_vector.push_back(rot_data_i);
  param_block_vector.push_back(trans_data_j);
  param_block_vector.push_back(rot_data_j);

  // Add the residual block.
  problem_.AddResidualBlock(knot_prior_factor, nullptr, param_block_vector);
  problem_.SetManifold(rot_data_i, kQuatManifold_.get());
  // Set manifold for quaternion blocks.
  problem_.SetManifold(rot_data_j, kQuatManifold_.get());

  return true;
}

bool FactorAdder::CalMetaMinMaxTime(const double &meas_time, double &min_time,
                                    double &max_time) {
  // Obtain the maximum and minimum timestamps of metadata based on the
  // maximum change in time offset, and ensure their validity.
  const double kMaxToffChange = system_config_->get_max_toff_change();

  if ((meas_time < opt_start_time_ - 2 * kMaxToffChange) ||
      (meas_time > opt_end_time_ + 2 * kMaxToffChange)) {
    spdlog::critical(
        "The specified measurement time is not within the valid time range of "
        "the system. ");
    return false;
  }

  if (meas_time < opt_start_time_) {
    min_time = opt_start_time_;
    max_time = opt_start_time_ + kMaxToffChange;
  } else if (meas_time - kMaxToffChange < opt_start_time_ &&
             meas_time >= opt_start_time_) {
    min_time = opt_start_time_;
    max_time = meas_time + kMaxToffChange;
  } else if (meas_time > opt_end_time_) {
    min_time = opt_end_time_ - kMaxToffChange;
    max_time = opt_end_time_;
  } else if (meas_time + kMaxToffChange > opt_end_time_ &&
             meas_time <= opt_end_time_) {
    min_time = meas_time - kMaxToffChange;
    max_time = opt_end_time_;
  } else {
    min_time = meas_time - kMaxToffChange;
    max_time = meas_time + kMaxToffChange;
  }

  return true;
}

void FactorAdder::AddR3dKnotData(std::vector<double *> &param_block_vector,
                                 const SplineBundleType::R3dSplineType &spline,
                                 const SplineMetaType &spline_meta) {
  for (const auto &segment : spline_meta.segments) {
    // Compute time index, 'knot_interval * 0.5' is the treatment for numerical
    // accuracy
    size_t index;
    double fraction;
    spline.ComputeTimeIndex(segment.start_time + segment.knot_interval * 0.5,
                            index, fraction);

    // Iterate over control points in the segment.
    for (std::size_t i = index; i < index + segment.NumParameters(); ++i) {
      auto *data =
          const_cast<double *>(spline.get_knot(static_cast<int>(i)).data());

      // Add the control point as a parameter block.
      problem_.AddParameterBlock(data, 3);

      // Store the control point data in the parameter block vector.
      param_block_vector.push_back(data);
    }
  }
}

void FactorAdder::AddSo3dKnotsData(
    std::vector<double *> &param_block_vector,
    const SplineBundleType::So3dSplineType &spline,
    const SplineMetaType &spline_meta) {
  for (const auto &segment : spline_meta.segments) {
    // Compute time index, 'knot_interval * 0.5' is the treatment for numerical
    // accuracy
    size_t index;
    double fraction;
    spline.ComputeTimeIndex(segment.start_time + segment.knot_interval * 0.5,
                            index, fraction);

    // Iterate over control points in the segment.
    for (std::size_t i = index; i < index + segment.NumParameters(); ++i) {
      auto *data =
          const_cast<double *>(spline.get_knot(static_cast<int>(i)).data());

      // Add the control point as a parameter block.
      problem_.AddParameterBlock(data, 4, kQuatManifold_.get());

      // Store the control point data in the parameter block vector.
      param_block_vector.push_back(data);
    }
  }
}

}  // namespace hpgt
