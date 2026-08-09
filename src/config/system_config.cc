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

#include "hpgt/config/system_config.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool IsFinite(const Eigen::Vector3d& vector) { return vector.allFinite(); }

bool IsFinite(const Eigen::Vector4d& vector) { return vector.allFinite(); }

bool IsValidQuaternion(const Eigen::Quaterniond& quaternion) {
  return quaternion.coeffs().allFinite() &&
         quaternion.squaredNorm() > std::numeric_limits<double>::epsilon();
}

}  // namespace

namespace hpgt {

bool SystemConfig::FromJson(const std::string& input_path) {
  spdlog::info("Reading system config from: {}", input_path);

  std::ifstream input_file(input_path);
  if (!input_file.is_open()) {
    spdlog::critical("Failed to open the input system config file: {}",
                     input_path);
    return false;
  }

  try {
    nlohmann::json nlm_json;
    input_file >> nlm_json;

    SystemConfig parsed_config;
    nlm_json.get_to(parsed_config);
    if (!parsed_config.CheckAndPrintConfig()) {
      spdlog::critical("System config validation failed: {}", input_path);
      return false;
    }

    parsed_config.data_dir_ = data_dir_;
    *this = std::move(parsed_config);
  } catch (const nlohmann::json::exception& exception) {
    spdlog::critical("Failed to parse system config '{}': {}", input_path,
                     exception.what());
    return false;
  } catch (const std::exception& exception) {
    spdlog::critical("Failed to read system config '{}': {}", input_path,
                     exception.what());
    return false;
  }

  return true;
}

bool SystemConfig::ToJson(const std::string& output_path) {
  spdlog::info("Writing system config to: {}", output_path);

  std::ofstream output_file(output_path);
  if (!output_file.is_open()) {
    spdlog::critical("Failed to open the output system config file: {}",
                     output_path);
    return false;
  }

  try {
    const nlohmann::json nlm_json = *this;
    output_file << nlm_json.dump(2);
  } catch (const std::exception& exception) {
    spdlog::critical("Failed to serialize system config '{}': {}", output_path,
                     exception.what());
    return false;
  }

  if (!output_file) {
    spdlog::critical("Failed to write the system config file: {}", output_path);
    return false;
  }

  return true;
}

bool SystemConfig::CheckAndPrintConfig() {
  // Step 1: Check and print the optimization options.
  spdlog::info("---------------- Optimization options ----------------");
  if (!std::isfinite(spline_knot_interval_) || spline_knot_interval_ <= 0.) {
    spdlog::critical("B-spline knot interval must be finite and positive.");
    return false;
  }
  spdlog::info("B-spline knot interval:         {:.3f}", spline_knot_interval_);
  if (!std::isfinite(max_toff_change_)) {
    spdlog::critical("Maximum time-offset change must be finite.");
    return false;
  }
  if (max_toff_change_ < 0.01) {
    spdlog::warn(
        "Maximum time-offset change is too small; clamping it to 0.01 s.");
    max_toff_change_ = 0.01;
  } else if (max_toff_change_ > 0.5) {
    spdlog::warn(
        "Maximum time-offset change is too large; clamping it to 0.5 s.");
    max_toff_change_ = 0.5;
  }
  spdlog::info("Maximum time-offset change:     {:.3f}", max_toff_change_);
  if (!std::isfinite(gravity_magnitude_) || gravity_magnitude_ <= 0.) {
    spdlog::critical("Gravity magnitude must be finite and positive.");
    return false;
  }
  spdlog::info("Gravity magnitude:              {:.3f}", gravity_magnitude_);
  if (!std::isfinite(output_frequency_) || output_frequency_ <= 0.) {
    spdlog::critical(
        "Output trajectory frequency must be finite and positive.");
    return false;
  }
  spdlog::info("Output trajectory frequency:    {:.1f}", output_frequency_);
  spdlog::info("Whether to optimize the temporal calibration parameters: {}",
               opt_temporal_param_flag_);
  spdlog::info("Whether to optimize the spatial calibration parameters:  {}",
               opt_spatial_param_flag_);

  // ===========================================================================

  // Step 2: Check and print the sensor data config.
  spdlog::info("----------------- Sensor data config -----------------");
  int valid_pose_count = 0;
  int valid_imu_count = 0;
  int body_frame_count = 0;
  int world_frame_count = 0;
  constexpr double kMinNoiseStd = 1e-6;

  pose_config_.erase(
      std::remove_if(pose_config_.begin(), pose_config_.end(),
                     [](const PoseDataConfig& node) {
                       if (!node.file_name.empty()) {
                         return false;
                       }
                       spdlog::warn(
                           "Skipping pose data with no file name provided.");
                       return true;
                     }),
      pose_config_.end());
  imu_config_.erase(
      std::remove_if(imu_config_.begin(), imu_config_.end(),
                     [](const ImuDataConfig& node) {
                       if (!node.file_name.empty()) {
                         return false;
                       }
                       spdlog::warn(
                           "Skipping IMU data with no file name provided.");
                       return true;
                     }),
      imu_config_.end());

  // Step 2.1: Check the pose data config.
  for (auto& node : pose_config_) {
    spdlog::info("- Pose node [{}].", node.file_name);

    // Basic sensor properties.
    spdlog::info("Used as the body frame:  {}", node.body_frame_flag);
    spdlog::info("Used as the world frame: {}", node.world_frame_flag);
    spdlog::info("Provides absolute pose measurements: {}", node.abs_pose_flag);

    if (node.world_frame_flag && !node.abs_pose_flag) {
      spdlog::critical(
          "Absolute measurements must be used when the pose sensor is used as "
          "the world frame.");
      return false;
    }

    // Noise setting.
    if (!std::isfinite(node.trans_noise) || !std::isfinite(node.rot_noise) ||
        node.trans_noise < kMinNoiseStd || node.rot_noise < kMinNoiseStd) {
      spdlog::critical(
          "Noise std [{:.6f}, {:.6f}] too small, may cause "
          "optimization instability.",
          node.trans_noise, node.rot_noise);
      return false;
    }
    spdlog::info("Noise std of the translational part in discrete time: {:.6f}",
                 node.trans_noise);
    spdlog::info("Noise std of the rotational part in discrete time:    {:.6f}",
                 node.rot_noise);

    // Initial values.
    if (!std::isfinite(node.toff_BP_init) || !IsFinite(node.trans_BP_init) ||
        !IsFinite(node.trans_GW_init) ||
        !IsValidQuaternion(node.rot_q_BP_init) ||
        !IsValidQuaternion(node.rot_q_GW_init)) {
      spdlog::critical(
          "Invalid initial calibration value for pose sensor [{}].",
          node.file_name);
      return false;
    }
    node.rot_q_BP_init.normalize();
    node.rot_q_GW_init.normalize();

    if (!opt_temporal_param_flag_) {
      spdlog::info("Time offset (toff_BP) is fixed as: {:.6f}",
                   node.toff_BP_init);
    }

    if (!opt_spatial_param_flag_) {
      spdlog::info(
          "Body frame translation (trans_BP) is fixed as:  [{:.6f}, {:.6f}, "
          "{:.6f}]",
          node.trans_BP_init.x(), node.trans_BP_init.y(),
          node.trans_BP_init.z());
      spdlog::info(
          "Body frame rotation (rot_q_BP) is fixed as:     [{:.6f}, {:.6f}, "
          "{:.6f}, {:.6f}]",
          node.rot_q_BP_init.x(), node.rot_q_BP_init.y(),
          node.rot_q_BP_init.z(), node.rot_q_BP_init.w());

      spdlog::info(
          "World frame translation (trans_GW) is fixed as: [{:.6f}, "
          "{:.6f}, {:.6f}]",
          node.trans_GW_init.x(), node.trans_GW_init.y(),
          node.trans_GW_init.z());
      spdlog::info(
          "World frame rotation (rot_q_GW) is fixed as:    [{:.6f}, {:.6f}, "
          "{:.6f}, {:.6f}]",
          node.rot_q_GW_init.x(), node.rot_q_GW_init.y(),
          node.rot_q_GW_init.z(), node.rot_q_GW_init.w());
    }

    if (!node.abs_pose_flag) {
      spdlog::info(
          "For relative pose measurements, the noise standard deviation and "
          "world-frame transformation are ignored.");
    }

    // Number count.
    ++valid_pose_count;
    if (node.body_frame_flag) {
      ++body_frame_count;
    }
    if (node.world_frame_flag) {
      ++world_frame_count;
    }
  }

  // ===========================================================================

  // Step 2.2: Check the IMU data config.
  for (auto& node : imu_config_) {
    spdlog::info("- IMU node [{}].", node.file_name);

    // Basic sensor properties.
    spdlog::info("Used as the body frame: {}", node.body_frame_flag);

    if (node.model_type == ImuModelType::kInvalid) {
      spdlog::warn("Invalid IMU model; using the 'calibrated' model.");
      node.model_type = ImuModelType::kCalibrated;
    }
    spdlog::info("IMU model type: {}", ToString(node.model_type));

    if (!std::isfinite(node.frequency) || node.frequency <= 0.) {
      spdlog::critical("IMU frequency for [{}] must be finite and positive.",
                       node.file_name);
      return false;
    }
    spdlog::info("IMU frequency:  {}", node.frequency);

    // Noise setting.
    if (!IsFinite(node.noise) || node.noise[0] < kMinNoiseStd ||
        node.noise[1] < kMinNoiseStd || node.noise[2] < kMinNoiseStd ||
        node.noise[3] < kMinNoiseStd) {
      spdlog::critical(
          "Noise std [{:.6f}, {:.6f}, {:.6f}, {:.6f}] too "
          "small, may cause optimization instability.",
          node.noise[0], node.noise[1], node.noise[2], node.noise[3]);
      return false;
    }
    spdlog::info(
        "Noise std of [na, ba, ng, bg] in continuous time: [{:.6f}, {:.6f}, "
        "{:.6f}, {:.6f}]",
        node.noise[0], node.noise[1], node.noise[2], node.noise[3]);

    // Initial values.
    if (!std::isfinite(node.toff_BI_init) || !IsFinite(node.trans_BI_init) ||
        !IsFinite(node.acc_bias_init) || !IsFinite(node.gyr_bias_init) ||
        !IsValidQuaternion(node.rot_q_BI_init)) {
      spdlog::critical("Invalid initial calibration value for IMU [{}].",
                       node.file_name);
      return false;
    }
    node.rot_q_BI_init.normalize();

    if (!opt_temporal_param_flag_) {
      spdlog::info("Time offset (toff_BI) is fixed as: {:.6f}",
                   node.toff_BI_init);
    }

    if (!opt_spatial_param_flag_) {
      spdlog::info(
          "Body frame translation (trans_BI) is fixed as:  [{:.6f}, {:.6f}, "
          "{:.6f}]",
          node.trans_BI_init.x(), node.trans_BI_init.y(),
          node.trans_BI_init.z());
      spdlog::info(
          "Body frame rotation (rot_q_BI) is fixed as:     [{:.6f}, {:.6f}, "
          "{:.6f}, {:.6f}]",
          node.rot_q_BI_init.x(), node.rot_q_BI_init.y(),
          node.rot_q_BI_init.z(), node.rot_q_BI_init.w());
    }

    // Number count.
    ++valid_imu_count;
    if (node.body_frame_flag) {
      ++body_frame_count;
    }
  }

  // ===========================================================================

  // Step 3: Check the number of input sensors and body frames.
  if (valid_pose_count < 1 || (valid_pose_count + valid_imu_count) < 2) {
    spdlog::critical(
        "Insufficient number of sensors. At least two sensors are required, "
        "and at least one must provide pose information.");
    return false;
  } else {
    spdlog::info("- Fusing {} pose sensors and {} IMUs.", valid_pose_count,
                 valid_imu_count);
  }

  if (body_frame_count != 1) {
    spdlog::critical("Exactly one sensor must define the body frame; found {}.",
                     body_frame_count);
    return false;
  }

  if (world_frame_count != 1) {
    spdlog::critical(
        "Exactly one sensor must define the world frame; found {}.",
        world_frame_count);
    return false;
  }

  // ===========================================================================

  return true;
}

}  // namespace hpgt
