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

#include "hpgt/initializer/imu_preintegrator.h"
#include "hpgt/sensor_data/sensor_data_manager.h"

namespace hpgt {

/**
 * @brief A helper class that provides static methods for initializing our
 * system.
 */
class InitializerHelper {
 public:
  /**
   * @brief Get the 3D angular velocity at a specified timestamp from a
   * pose sequence.
   *
   * @param[in] pose_seq Pose sequence.
   * @param[in] timestamp Target timestamp.
   * @param[in] calc_range Time range for calculating velocity from pose
   * differentials.
   * @param[out] angular_vel Angular velocity at target timestamp.
   * @return true if we are able to get the angular velocity.
   */
  static bool GetTargetAngularVel(const PoseSequence &pose_seq,
                                  const double &timestamp,
                                  const double &calc_range,
                                  Eigen::Vector3d &angular_vel);

  /**
   * @brief Get the 3D angular velocity at a specified timestamp from a
   * IMU sequence.
   *
   * @param[in] pose_seq IMU sequence.
   * @param[in] timestamp Target timestamp.
   * @param[out] angular_vel Angular velocity at target timestamp.
   * @return True if we are able to get the angular velocity.
   */
  static bool GetTargetAngularVel(const ImuSequence &imu_seq,
                                  const double &timestamp,
                                  Eigen::Vector3d &angular_vel);

  /**
   * @brief Get the target pose at a specified timestamp from a pose sequence.
   *
   * @param[in] pose_seq Pose sequence.
   * @param[in] timestamp Target timestamp.
   * @param[out] trans Translation part of the result pose.
   * @param[out] rot_q Rotation part of the result pose.
   * @return True if we are able to get the pose.
   */
  static bool GetTargetPose(const PoseSequence &pose_seq,
                            const double &timestamp, Eigen::Vector3d &trans,
                            Eigen::Quaterniond &rot_q);

  /**
   * @brief Linear interpolation between two pose frames to get the result with
   * target timestamp.
   *
   * @param[in] data_0 First pose frame.
   * @param[in] data_1 Second pose frame.
   * @param[in] timestamp Target timestamp.
   * @param[out] data_result Result pose with target timestamp.
   */
  static void LinInterpPoseFrame(const PoseFrame::Ptr &data_0,
                                 const PoseFrame::Ptr &data_1,
                                 const double &timestamp,
                                 PoseFrame::Ptr &data_result);

  /**
   * @brief Linear interpolation between two IMU frames to get the result with
   * target timestamp.
   *
   * @param[in] data_0 First IMU frame.
   * @param[in] data_1 Second IMU frame.
   * @param[in] timestamp Target timestamp.
   * @param[out] data_result Result IMU with target timestamp.
   */
  static void LinInterpImuFrame(const ImuFrame::Ptr &data_0,
                                const ImuFrame::Ptr &data_1,
                                const double &timestamp,
                                ImuFrame::Ptr &data_result);
};

}  // namespace hpgt
