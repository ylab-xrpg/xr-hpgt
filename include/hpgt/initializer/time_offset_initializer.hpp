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
#include "hpgt/sensor_data/sensor_data_manager.h"

namespace hpgt {

/**
 * @brief class for initializing time offsets, capable of aligning any
 * number and combination of pose and IMU sequences.
 *
 * For more details, please refer to: "Shu Z, et al. A Spatiotemporal Hand-Eye
 * Calibration for Trajectory Alignment in Visual (-Inertial) Odometry
 * Evaluation[J]. IEEE RA-L, 2024."
 */
class TimeOffsetInitializer {
 public:
  using Ptr = std::shared_ptr<TimeOffsetInitializer>;

  using DynamicV3D = Eigen::Matrix<double, 3, Eigen::Dynamic>;

  // Create an instance of time offset initializer.
  static TimeOffsetInitializer::Ptr Create() {
    return Ptr(new TimeOffsetInitializer);
  }

  /**
   * @brief Initialize the time offsets.
   *
   * We initial the time offset between each sensor and the body sensor, i.e.
   * toff_B_Xi, and save the results to the CalibParameter object. If the time
   * offset needs to be fixed, we only determine the body frame and exit.
   *
   * @param[in] data_manager Sensor data manager.
   * @param[out] calib_param Calibration parameter results.
   * @param[in] opt_flag Whether to fix the time offset parameters.
   * @return True if the time offsets are initialized.
   */
  bool Initialize(const SensorDataManager::Ptr &data_manager,
                  const CalibParameter::Ptr &calib_param,
                  const bool &opt_flag = true);

  /**
   *
   * @brief Estimate time offset between two data sequence. Templated design for
   * compatibility with both IMU and pose sequences.
   *
   * We initialize the time offset based on screw theory, i.e. consistency of
   * angular velocity. For the input sequence_i (X) and sequence_j (Y), we
   * compute toff_i_j such that t_j = toff_i_j + tau_j.
   *
   * @tparam SeqT_i Data type of sequence i.
   * @tparam SeqT_j Data type of sequence j.
   * @param[in] seq_i Data sequence i.
   * @param[in] seq_j Data sequence j.
   * @param[out] toff Time offset result.
   * @return True if we are able to initialize the time offset.
   */
  template <typename SeqT_i, typename SeqT_j>
  bool EstimateToffFromSeq(const SeqT_i &seq_i, const SeqT_j &seq_j,
                           double &toff) {
    // Step 1: Generate 3D angular velocity from sequences.
    std::vector<double> sample_timestamps_i, sample_timestamps_j;
    DynamicV3D angular_vel_i, angular_vel_j;

    if (!GenerateAngularVel(seq_i, sample_timestamps_i, angular_vel_i) ||
        !GenerateAngularVel(seq_j, sample_timestamps_j, angular_vel_j)) {
      return false;
    }

    size_t sample_size_i = sample_timestamps_i.size();
    size_t sample_size_j = sample_timestamps_j.size();
    if (sample_size_i < 2 || sample_size_j < 2) {
      spdlog::critical(
          "Insufficient data used to initialize the time offset. ");
      return false;
    }

    // ===========================================================================

    // Step 2: Smooth the 3D angular velocity and calculate its magnitude.
    Eigen::RowVectorXd smooth_kernel = Eigen::RowVectorXd::Constant(
        smooth_kernal_size_, static_cast<double>(1. / smooth_kernal_size_));
    Eigen::MatrixXd angular_vel_filtered_i;
    Eigen::MatrixXd angular_vel_filtered_j;
    if (!SignalCorrelation(angular_vel_i, smooth_kernel, angular_vel_filtered_i,
                           true) ||
        !SignalCorrelation(angular_vel_j, smooth_kernel, angular_vel_filtered_j,
                           true)) {
      return false;
    }

    Eigen::RowVectorXd angular_vel_norm_i(sample_size_i);
    Eigen::RowVectorXd angular_vel_norm_j(sample_size_j);
    for (size_t i = 0; i < sample_size_i; ++i) {
      angular_vel_norm_i(i) = angular_vel_filtered_i.col(i).norm();
    }
    for (size_t i = 0; i < sample_size_j; ++i) {
      angular_vel_norm_j(i) = angular_vel_filtered_j.col(i).norm();
    }

    // ===========================================================================

    // Step 3: Calculate the correlation function and map the index of the
    // maximum value to initialize the time offset.
    Eigen::MatrixXd correlation_function(angular_vel_norm_i.rows(),
                                         sample_size_i + sample_size_j - 1);
    if (!SignalCorrelation(angular_vel_norm_i, angular_vel_norm_j,
                           correlation_function, false)) {
      return false;
    }

    double max_index;
    if (!CorrelationMaxIndex(correlation_function, max_index)) {
      return false;
    }
    toff = sample_timestamps_i.front() - sample_timestamps_j.back() +
           (max_index)*velocity_sample_interval_;

    // ===========================================================================

    return true;
  }

  // Set the parameters.
  void set_velocity_calc_range(const double &v) { velocity_calc_range_ = v; }

  void set_velocity_sample_interval(const double &v) {
    velocity_sample_interval_ = v;
  }

  void set_smooth_kernal_size(const double &v) { smooth_kernal_size_ = v; }

  void set_refine_fitting_range(const double &v) { refine_fitting_range_ = v; }

 private:
  /**
   * @brief Generate angular velocities for a given data sequence at specified
   * timestamps. Templated design for compatibility with both IMU and pose
   * sequences.
   *
   * @tparam SeqT Data sequence type.
   * @param[in] seq Data sequence.
   * @param[out] timestamps Timestamps used for calculating angular velocities.
   * @param[out] angular_vel Result angular velocities in matrix form.
   * @return True if we are able to generate angular velocity.
   */
  template <typename SeqT>
  bool GenerateAngularVel(const SeqT &seq, std::vector<double> &timestamps,
                          DynamicV3D &angular_vel) {
    // Step 1: Calculate the number of samples between start and stop times.
    double start_time = seq.front()->timestamp + velocity_calc_range_;
    double end_time = seq.back()->timestamp - velocity_calc_range_;
    if (start_time < 0) {
      spdlog::warn(
          "Start time for time offset initialization is negative, set to 0. ");
      start_time = 0.;
    }
    if (start_time > end_time) {
      spdlog::critical(
          "Timestamp exception in time offset initialization: {:.9f} / {:.9f}.",
          start_time, end_time);
      spdlog::critical("Start time should be less than end time. ");
      return false;
    }
    double duration = end_time - start_time;

    size_t sample_num =
        static_cast<size_t>(duration / velocity_sample_interval_);
    timestamps.reserve(sample_num);
    angular_vel.resize(3, sample_num);
    angular_vel.setZero();

    // ===========================================================================

    // Step 2: Calculate the angular velocity at each sampling time.
    double time_curr = start_time;
    Eigen::Vector3d single_angular_vel;

    bool get_vel_flag;
    for (size_t i = 0; i < sample_num; ++i) {
      get_vel_flag = false;
      // Specialize the data sequence and compute the target angular velocity.
      if constexpr (std::is_same<SeqT, PoseSequence>::value) {
        get_vel_flag = InitializerHelper::GetTargetAngularVel(
            seq, time_curr, velocity_calc_range_, single_angular_vel);
      } else if constexpr (std::is_same<SeqT, ImuSequence>::value) {
        get_vel_flag = InitializerHelper::GetTargetAngularVel(
            seq, time_curr, single_angular_vel);
      } else {
        spdlog::critical(
            "Invalid sequence type in generating angular velocity. ");
        return false;
      }

      // Save the results.
      if (get_vel_flag) {
        angular_vel.col(i) = single_angular_vel;
        timestamps.push_back(time_curr);
      }

      time_curr += velocity_sample_interval_;
    }

    // ===========================================================================

    return true;
  }

  /**
   * @brief Calculate the correlation of two discrete signals.
   *
   * @param[in] signal_i Signal i.
   * @param[in] signal_j Signal j, used as kernel.
   * @param[out] output Correlation result.
   * @param[in] same_size Whether to keep the same size as input signal 0.
   * @return True if we are able to calculate the correlation function.
   */
  bool SignalCorrelation(const Eigen::MatrixXd &signal_i,
                         const Eigen::RowVectorXd &signal_j,
                         Eigen::MatrixXd &correlation, const bool &same_size);

  /**
   * @brief Get the index of the correlation maximum. To improve accuracy, we
   * use a polynomial fitting for the approximation.
   *
   * @param[in] correlation_function Correlation function.
   * @param[out] max_index Index of the maximum value.
   * @return True if the index is found.
   */
  bool CorrelationMaxIndex(const Eigen::RowVectorXd &correlation_function,
                           double &max_index);

  // Initializer settings
  double velocity_calc_range_ = 0.05;
  double velocity_sample_interval_ = 0.01;
  double smooth_kernal_size_ = 5;
  double refine_fitting_range_ = 3;
};

}  // namespace hpgt
