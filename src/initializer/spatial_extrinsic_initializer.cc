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

#include "hpgt/initializer/spatial_extrinsic_initializer.h"

namespace hpgt {

bool SpatialExtrinsicInitializer::Initialize(
    const SensorDataManager::Ptr &data_manager,
    const CalibParameter::Ptr &calib_param, const double &g_magnitude,
    const bool &opt_flag) {
  // Step 1: Determine the world and local frames of our system.
  for (const auto &[label, config] : data_manager->GetAllPoseConfig()) {
    if (!config.world_frame_flag) {
      continue;
    }

    if (calib_param->world_frame_label != "") {
      spdlog::critical("Only one world frame is allowed.");
      return false;
    }
    calib_param->world_frame_label = label;
    calib_param->trans_B_Pi[label] = Sophus::Vector3d::Zero();
    calib_param->rot_B_Pi[label] = Sophus::SO3d::exp(Sophus::Vector3d::Zero());
    calib_param->trans_G_Wi[label] = Sophus::Vector3d::Zero();
    calib_param->rot_G_Wi[label] = Sophus::SO3d::exp(Sophus::Vector3d::Zero());
  }

  if (calib_param->world_frame_label == "") {
    spdlog::critical("Fail to set world frame. ");
    return false;
  } else {
    spdlog::info("World frame is set to: [{}]", calib_param->world_frame_label);
  }

  if (calib_param->body_frame_label == "") {
    spdlog::critical(
        "The time offset initialization needs to completed before. ");
    return false;
  } else {
    spdlog::info("Body frame is set to:  [{}]", calib_param->body_frame_label);
  }

  // Skip initialization if time offset parameters are to be fixed.
  if (!opt_flag) {
    if (data_manager->GetAllImuConfig().size() != 0) {
      calib_param->gravity_aligned = true;
    }
    spdlog::info(
        "Skip initialization as the spatial extrinsic parameters need to be "
        "fixed. ");
    return true;
  }

  // ===========================================================================

  // Step 2: We first solve the spatial extrinsic parameters of the world
  // frame sensor (denoted as G, M) with respect to each other sensor.
  Eigen::Vector3d trans_M_Xi;
  Eigen::Quaterniond rot_M_Xi;

  Eigen::Vector3d trans_G_Wi;
  Eigen::Quaterniond rot_q_G_Wi;
  Eigen::Quaterniond rot_q_gravity_align;
  bool gravity_aligned = false;

  auto P2P_initializer = P2PExtrinsicInitializer::Create();
  auto I2P_initializer = I2PExtrinsicInitializer::Create();

  // Iterate all other sensors and initialize the spatial extrinsic parameters.
  // Step 2.1: Pose sensors.
  double toff_MB = -calib_param->toff_B_Pi.at(calib_param->world_frame_label);
  for (const auto &[label, config] : data_manager->GetAllPoseConfig()) {
    if (label == calib_param->world_frame_label) {
      continue;
    }

    bool success_flag = false;
    success_flag = P2P_initializer->EstimateFromSeq(
        data_manager->GetPoseSeqByLabel(calib_param->world_frame_label),
        data_manager->GetPoseSeqByLabel(label),
        toff_MB + calib_param->toff_B_Pi.at(label), trans_M_Xi, rot_M_Xi,
        trans_G_Wi, rot_q_G_Wi);

    if (!success_flag) {
      return false;
    }

    // Body frame.
    calib_param->trans_B_Pi[label] = trans_M_Xi;
    calib_param->rot_B_Pi[label] = Sophus::SO3d(rot_M_Xi);

    // World frame.
    calib_param->trans_G_Wi[label] = trans_G_Wi;
    calib_param->rot_G_Wi[label] = Sophus::SO3d(rot_q_G_Wi);
  }

  // Step 2.2: IMU sensors.
  for (const auto &[label, config] : data_manager->GetAllImuConfig()) {
    bool success_flag = false;
    success_flag = I2P_initializer->EstimateFromSeq(
        data_manager->GetPoseSeqByLabel(calib_param->world_frame_label),
        data_manager->GetImuSeqByLabel(label),
        toff_MB + calib_param->toff_B_Ii.at(label), g_magnitude, trans_M_Xi,
        rot_M_Xi, rot_q_gravity_align);

    if (!success_flag) {
      return false;
    }

    calib_param->trans_B_Ii[label] = trans_M_Xi;
    calib_param->rot_B_Ii[label] = Sophus::SO3d(rot_M_Xi);

    gravity_aligned = true;
  }

  // ===========================================================================

  // Step 3: Propagate the previous results.
  // Get the transformation between the world frame sensor and the body frame.
  Sophus::Vector3d trans_MB, trans_BM;
  Sophus::SO3d rot_q_MB, rot_q_BM;
  if (calib_param->body_sensor_type == SensorType::kPose) {
    trans_MB = calib_param->trans_B_Pi.at(calib_param->body_frame_label);
    rot_q_MB = calib_param->rot_B_Pi.at(calib_param->body_frame_label);
  } else if (calib_param->body_sensor_type == SensorType::kImu) {
    trans_MB = calib_param->trans_B_Ii.at(calib_param->body_frame_label);
    rot_q_MB = calib_param->rot_B_Ii.at(calib_param->body_frame_label);
  }

  trans_BM = -(rot_q_MB.inverse() * trans_MB);
  rot_q_BM = rot_q_MB.inverse();

  // Propagate with the body frame.
  for (auto &[label, trans_B_Pi] : calib_param->trans_B_Pi) {
    trans_B_Pi = rot_q_BM * trans_B_Pi + trans_BM;
  }

  for (auto &[label, rot_B_Pi] : calib_param->rot_B_Pi) {
    rot_B_Pi = rot_q_BM * rot_B_Pi;
  }

  for (auto &[label, trans_B_Ii] : calib_param->trans_B_Ii) {
    trans_B_Ii = rot_q_BM * trans_B_Ii + trans_BM;
  }

  for (auto &[label, rot_B_Ii] : calib_param->rot_B_Ii) {
    rot_B_Ii = rot_q_BM * rot_B_Ii;
  }

  // Align the world frame with gravity.
  if (gravity_aligned) {
    Sophus::SO3d rot_gravity_align = Sophus::SO3d(rot_q_gravity_align);
    for (auto &[label, trans_G_Wi] : calib_param->trans_G_Wi) {
      trans_G_Wi = rot_gravity_align * trans_G_Wi;
    }

    for (auto &[label, rot_G_Wi] : calib_param->rot_G_Wi) {
      rot_G_Wi = rot_gravity_align * rot_G_Wi;
    }

    calib_param->gravity_aligned = true;
  }

  // ===========================================================================

  // Step 4: Print the final results.
  // Spatial extrinsic parameters for pose sensors.
  for (const auto &[label, config] : data_manager->GetAllPoseConfig()) {
    spdlog::info("- Spatial extrinsic parameters for [{}]. ", label);

    Eigen::Vector3d trans_B_Pi = calib_param->trans_B_Pi.at(label);
    Eigen::Quaterniond rot_q_B_Pi =
        calib_param->rot_B_Pi.at(label).unit_quaternion();
    Eigen::Vector3d trans_G_Wi = calib_param->trans_G_Wi.at(label);
    Eigen::Quaterniond rot_q_G_Wi =
        calib_param->rot_G_Wi.at(label).unit_quaternion();

    // Body frame.
    spdlog::info(
        "Translation (tran_BP) for body frame is initialized to:  [{:.6f}, "
        "{:.6f}, {:.6f}]",
        trans_B_Pi.x(), trans_B_Pi.y(), trans_B_Pi.z());
    spdlog::info(
        "Rotation (rot_q_BP) for body frame is initialized to:    [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_q_B_Pi.x(), rot_q_B_Pi.y(), rot_q_B_Pi.z(), rot_q_B_Pi.w());

    // World frame.
    spdlog::info(
        "Translation (tran_GW) for world frame is initialized to: [{:.6f}, "
        "{:.6f}, {:.6f}]",
        trans_G_Wi.x(), trans_G_Wi.y(), trans_G_Wi.z());
    spdlog::info(
        "Rotation (rot_q_GW) for world frame is initialized to:   [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_q_G_Wi.x(), rot_q_G_Wi.y(), rot_q_G_Wi.z(), rot_q_G_Wi.w());
  }

  // Spatial extrinsic parameters for IMUs.
  for (const auto &[label, config] : data_manager->GetAllImuConfig()) {
    spdlog::info("- Spatial extrinsic parameters for [{}]. ", label);

    Eigen::Vector3d trans_B_Ii = calib_param->trans_B_Ii.at(label);
    Eigen::Quaterniond rot_q_B_Ii =
        calib_param->rot_B_Ii.at(label).unit_quaternion();
    spdlog::info(
        "Translation (tran_BI) for body frame is initialized to:  [{:.6f}, "
        "{:.6f}, {:.6f}]",
        trans_B_Ii.x(), trans_B_Ii.y(), trans_B_Ii.z());
    spdlog::info(
        "Rotation (rot_q_BI) for body frame is initialized to:    [{:.6f}, "
        "{:.6f}, {:.6f}, {:.6f}]",
        rot_q_B_Ii.x(), rot_q_B_Ii.y(), rot_q_B_Ii.z(), rot_q_B_Ii.w());
  }

  // ===========================================================================

  return true;
}

}  // namespace hpgt
