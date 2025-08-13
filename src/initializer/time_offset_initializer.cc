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

#include "hpgt/initializer/time_offset_initializer.hpp"

namespace hpgt {

bool TimeOffsetInitializer::Initialize(
    const SensorDataManager::Ptr& data_manager,
    const CalibParameter::Ptr& calib_param, const bool& opt_flag) {
  // Step 1: Find the sensor serving as the body frame.
  for (const auto& [label, config] : data_manager->GetAllPoseConfig()) {
    if (!config.body_frame_flag) {
      continue;
    }

    if (calib_param->body_frame_label != "") {
      spdlog::critical("Only one body frame is allowed.");
      return false;
    }
    calib_param->body_frame_label = label;
    calib_param->body_sensor_type = config.sensor_type;
    calib_param->toff_B_Pi[label] = 0.;
    calib_param->toff_B_Pi_increment[label] = 0.;

    spdlog::info("Body frame is set to: [{}]", label);
    spdlog::info("Time offset (toff_BP) for [{}] is initialized to: {:.6f}",
                 label, calib_param->toff_B_Pi.at(label));
  }

  for (const auto& [label, config] : data_manager->GetAllImuConfig()) {
    if (!config.body_frame_flag) {
      continue;
    }

    if (calib_param->body_frame_label != "") {
      spdlog::critical("Only one body frame is allowed.");
      return false;
    }
    calib_param->body_frame_label = label;
    calib_param->body_sensor_type = config.sensor_type;
    calib_param->toff_B_Ii[label] = 0.;
    calib_param->toff_B_Ii_increment[label] = 0.;

    spdlog::info("Body frame is set to: [{}]", label);
    spdlog::info("Time offset (toff_BP) for [{}] is initialized to: {:.6f}",
                 label, calib_param->toff_B_Ii.at(label));
  }

  if (calib_param->body_frame_label == "" ||
      calib_param->body_sensor_type == SensorType::kInvalid) {
    spdlog::critical("Fail to set body frame. ");
    return false;
  }

  // Skip initialization if time offset parameters are to be fixed.
  if (!opt_flag) {
    spdlog::info(
        "Skip initialization as the time offset parameters need to be fixed. ");
    return true;
  }

  // ===========================================================================

  // Step 2: Iterate all other sensors and initialize the time offset
  double toff_B_Xi;
  // Pose sensors.
  for (const auto& [label, config] : data_manager->GetAllPoseConfig()) {
    if (label == calib_param->body_frame_label) {
      continue;
    }

    bool success_flag = false;
    if (calib_param->body_sensor_type == SensorType::kPose) {
      success_flag = EstimateToffFromSeq(
          data_manager->GetPoseSeqByLabel(calib_param->body_frame_label),
          data_manager->GetPoseSeqByLabel(label), toff_B_Xi);
    } else if (calib_param->body_sensor_type == SensorType::kImu) {
      success_flag = EstimateToffFromSeq(
          data_manager->GetImuSeqByLabel(calib_param->body_frame_label),
          data_manager->GetPoseSeqByLabel(label), toff_B_Xi);
    }

    if (!success_flag) {
      return false;
    }

    calib_param->toff_B_Pi[label] = toff_B_Xi;
    calib_param->toff_B_Pi_increment[label] = 0.;
    spdlog::info("Time offset (toff_BP) for [{}] is initialized to: {:.6f}",
                 label, toff_B_Xi);
  }

  // IMU sensors.
  for (const auto& [label, config] : data_manager->GetAllImuConfig()) {
    if (label == calib_param->body_frame_label) {
      continue;
    }

    bool success_flag = false;
    if (calib_param->body_sensor_type == SensorType::kPose) {
      success_flag = EstimateToffFromSeq(
          data_manager->GetPoseSeqByLabel(calib_param->body_frame_label),
          data_manager->GetImuSeqByLabel(label), toff_B_Xi);
    } else if (calib_param->body_sensor_type == SensorType::kImu) {
      success_flag = EstimateToffFromSeq(
          data_manager->GetImuSeqByLabel(calib_param->body_frame_label),
          data_manager->GetImuSeqByLabel(label), toff_B_Xi);
    }

    if (!success_flag) {
      return false;
    }

    calib_param->toff_B_Ii[label] = toff_B_Xi;
    calib_param->toff_B_Ii_increment[label] = 0.;
    spdlog::info("Time offset (toff_BI) for [{}] is initialized to: {:.6f}",
                 label, toff_B_Xi);
  }

  // ===========================================================================

  return true;
}

bool TimeOffsetInitializer::SignalCorrelation(
    const Eigen::MatrixXd& signal_i, const Eigen::RowVectorXd& signal_j,
    Eigen::MatrixXd& correlation, const bool& same_size) {
  // Step 1: Determine the dimensions of the correlation result and expansion
  // matrices.
  int rows_i = signal_i.rows(), correlation_rows = rows_i,
      expansion_rows = rows_i;

  constexpr int kSignalMinCols = 2;
  int cols_i = signal_i.cols(), cols_j = signal_j.cols();
  if (cols_i <= kSignalMinCols || cols_j <= kSignalMinCols) {
    spdlog::critical(
        "The signal sampling point for calculating correlation must "
        "be greater than {}. ",
        kSignalMinCols);
    return false;
  }

  int correlation_cols, expansion_cols;
  if (same_size) {
    correlation_cols = cols_i;
    expansion_cols = cols_i + cols_j - 1;
  } else {
    correlation_cols = cols_i + cols_j - 1;
    expansion_cols = cols_i + 2 * (cols_j - 1);
  }

  // ===========================================================================

  // Step 2: Expand the signal 0 matrix.
  Eigen::MatrixXd signal_i_expansion(expansion_rows, expansion_cols);
  if (same_size) {
    int left_duplicate_num = (cols_j - 1) / 2;
    int right_duplicate_num = cols_j - left_duplicate_num - 1;
    Eigen::MatrixXd expansion_left(rows_i, left_duplicate_num);
    Eigen::MatrixXd expansion_right(rows_i, right_duplicate_num);
    for (int i = 0; i < left_duplicate_num; ++i) {
      expansion_left.col(i) = signal_i.col(0);
    }
    for (int i = 0; i < right_duplicate_num; ++i) {
      expansion_right.col(i) = signal_i.col(cols_i - 1);
    }
    signal_i_expansion << expansion_left, signal_i, expansion_right;
  } else {
    Eigen::MatrixXd expansion_zeros(rows_i, cols_j - 1);
    expansion_zeros.setZero();
    signal_i_expansion << expansion_zeros, signal_i, expansion_zeros;
  }

  // ===========================================================================

  // Step 3: Calculate the correlation result.
  correlation.resize(correlation_rows, correlation_cols);
  for (int i = 0; i < correlation_rows; ++i) {
    for (int j = 0; j < correlation_cols; ++j) {
      correlation(i, j) =
          (signal_i_expansion.row(i).segment(j, cols_j).array() *
           signal_j.array())
              .sum();
    }
  }

  // ===========================================================================

  return true;
}

bool TimeOffsetInitializer::CorrelationMaxIndex(
    const Eigen::RowVectorXd& correlation_function, double& max_index) {
  correlation_function.row(0).maxCoeff(&max_index);

  // Determine the start and end index.
  int fitting_start_index, fitting_end_index;
  fitting_start_index = max_index - refine_fitting_range_;
  fitting_end_index = max_index + refine_fitting_range_;

  if (fitting_start_index < 0 ||
      fitting_end_index > correlation_function.cols()) {
    spdlog::critical(
        "The range of the correlation function is insufficient for calculating "
        "the index of the maximum value. ");
    return false;
  }

  // Get the correlation values in range.
  std::vector<double> fitting_values;
  std::vector<double> fitting_index(2 * refine_fitting_range_ + 1);
  for (int i = fitting_start_index; i <= fitting_end_index; ++i) {
    fitting_values.push_back(correlation_function(i));
  }
  std::iota(fitting_index.begin(), fitting_index.end(), 0);

  // Calculate the coefficient matrix for quadratic polynomial fitting.
  constexpr int kFittingDegree = 2;
  Eigen::MatrixXd design_matrix(fitting_index.size(), kFittingDegree + 1);
  Eigen::VectorXd target_vector(fitting_values.size());
  for (size_t i = 0; i < fitting_index.size(); ++i) {
    target_vector(i) = fitting_values[i];
    for (int j = 0; j <= kFittingDegree; ++j) {
      design_matrix(i, j) = std::pow(fitting_index[i], j);
    }
  }

  // Maximum index = -2*a/b
  Eigen::VectorXd fitting_coeffs =
      design_matrix.householderQr().solve(target_vector);
  max_index -= fitting_coeffs[1] / (2 * fitting_coeffs[2]) +
               fitting_index[refine_fitting_range_];

  return true;
}

}  // namespace hpgt
