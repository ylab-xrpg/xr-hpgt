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

#include "hpgt/initializer/initializer_helper.h"

namespace hpgt {

bool InitializerHelper::GetTargetAngularVel(const PoseSequence &pose_seq,
                                            const double &timestamp,
                                            const double &calc_range,
                                            Eigen::Vector3d &angular_vel) {
  angular_vel = Eigen::Vector3d::Zero();

  const double kCalVelHalfRange = calc_range / 2.;
  double time_0 = timestamp - kCalVelHalfRange;
  double time_1 = timestamp + kCalVelHalfRange;

  // Get the two poses for differentiation.
  Eigen::Vector3d trans_0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d trans_1 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot_q_0 = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond rot_q_1 = Eigen::Quaterniond::Identity();
  if (!GetTargetPose(pose_seq, time_0, trans_0, rot_q_0) ||
      !GetTargetPose(pose_seq, time_1, trans_1, rot_q_1)) {
    return false;
  }

  // Calculate rotation differences to derive angular velocity.
  Eigen::Quaterniond delta_rot_q = rot_q_0.inverse() * rot_q_1;
  Eigen::AngleAxisd delta_aa(delta_rot_q);

  angular_vel = delta_aa.angle() * delta_aa.axis() / kCalVelHalfRange / 2.;

  return true;
}

bool InitializerHelper::GetTargetAngularVel(const ImuSequence &imu_seq,
                                            const double &timestamp,
                                            Eigen::Vector3d &angular_vel) {
  angular_vel = Eigen::Vector3d::Zero();

  if (imu_seq.size() < 2) {
    spdlog::error(
        "Insufficient IMU measurements to get target angle velocity. ");
    return false;
  }

  // Find the bounding IMU frames.
  auto target_ub =
      std::upper_bound(imu_seq.begin(), imu_seq.end(), timestamp,
                       [](double timestamp, const ImuFrame::Ptr &frame) {
                         return timestamp < frame->timestamp;
                       });

  if (target_ub == imu_seq.begin() || target_ub == imu_seq.end()) {
    spdlog::error(
        "Specified angular velocity sampling time: {:.9f} is out of the "
        "time range of the IMU measurements: {:.9f} {:.9f}. ",
        timestamp, imu_seq.front()->timestamp, imu_seq.back()->timestamp);
    return false;
  }

  auto imu_0 = *(target_ub--);
  auto imu_1 = *(target_ub);

  if (std::abs(timestamp - imu_0->timestamp) > 0.1 ||
      std::abs(timestamp - imu_1->timestamp) > 0.1) {
    spdlog::error(
        "Missing IMU data for more than 0.1s around {:.9f}, unable to "
        "get an accurate angular velocity. ",
        timestamp);
    return false;
  }

  // Interpolation.
  auto target_imu = ImuFrame::Create();
  LinInterpImuFrame(imu_0, imu_1, timestamp, target_imu);

  angular_vel = target_imu->gyr;

  return true;
}

bool InitializerHelper::GetTargetPose(const PoseSequence &pose_seq,
                                      const double &timestamp,
                                      Eigen::Vector3d &trans,
                                      Eigen::Quaterniond &rot_q) {
  trans = Eigen::Vector3d::Zero();
  rot_q = Eigen::Quaterniond::Identity();

  if (pose_seq.size() < 2) {
    spdlog::error("Insufficient pose frames to get target pose. ");
    return false;
  }

  // Find the bounding poses.
  auto target_ub =
      std::upper_bound(pose_seq.begin(), pose_seq.end(), timestamp,
                       [](double timestamp, const PoseFrame::Ptr &frame) {
                         return timestamp < frame->timestamp;
                       });

  if (target_ub == pose_seq.begin() || target_ub == pose_seq.end()) {
    spdlog::error(
        "Specified pose interpolation time: {:.9f} is out of the range "
        "of the pose data: ({:.9f}, {:.9f}). ",
        timestamp, pose_seq.front()->timestamp, pose_seq.back()->timestamp);
    return false;
  }

  PoseFrame::Ptr pose_0 = *(target_ub--);
  PoseFrame::Ptr pose_1 = *(target_ub);

  if (std::abs(timestamp - pose_0->timestamp) > 1.0 ||
      std::abs(timestamp - pose_1->timestamp) > 1.0) {
    spdlog::warn(
        "Missing pose data for more than 1.0s around {}, unable to "
        "perform an accurate interpolation. ",
        timestamp);
  }

  // Interpolation.
  auto target_pose = PoseFrame::Create();
  LinInterpPoseFrame(pose_0, pose_1, timestamp, target_pose);

  trans = target_pose->trans;
  rot_q = target_pose->rot_q;

  return true;
}

void InitializerHelper::LinInterpPoseFrame(const PoseFrame::Ptr &data_0,
                                           const PoseFrame::Ptr &data_1,
                                           const double &timestamp,
                                           PoseFrame::Ptr &data_result) {
  double lambda =
      (timestamp - data_0->timestamp) / (data_1->timestamp - data_0->timestamp);

  data_result->timestamp = timestamp;
  // The translation part is interpolated using LERP.
  data_result->trans = (1 - lambda) * data_0->trans + lambda * data_1->trans;
  // The rotation component is interpolated using SLERP.
  data_result->rot_q = data_0->rot_q.slerp(lambda, data_1->rot_q).normalized();
};

void InitializerHelper::LinInterpImuFrame(const ImuFrame::Ptr &data_0,
                                          const ImuFrame::Ptr &data_1,
                                          const double &timestamp,
                                          ImuFrame::Ptr &data_result) {
  // Time-distance lambda
  double lambda =
      (timestamp - data_0->timestamp) / (data_1->timestamp - data_0->timestamp);

  // LERP between the two frames.
  data_result->timestamp = timestamp;
  data_result->acc = (1 - lambda) * data_0->acc + lambda * data_1->acc;
  data_result->gyr = (1 - lambda) * data_0->gyr + lambda * data_1->gyr;
}

}  // namespace hpgt
