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

#include <filesystem>
#include <fstream>
#include <iomanip>

#include "hpgt/initializer/initializer_helper.h"
#include "hpgt/sensor_data/pose_data.h"
#include "hpgt/spline/euclidean_spline.hpp"
#include "hpgt/spline/so3_spline.hpp"
#include "hpgt/spline/spline_bundle.hpp"

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  if (argc != 3) {
    spdlog::critical("Usage: {} <data_dir> <output_dir>", argv[0]);
    return EXIT_FAILURE;
  }

  // Working directory and data path.
  const std::filesystem::path data_dir = argv[1];
  const std::filesystem::path output_dir = argv[2];
  std::filesystem::create_directories(output_dir);

  const std::string data_path = (data_dir / "mocap.txt").string();
  const std::string output_traj_path =
      (output_dir / "spline_traj.txt").string();
  const std::string output_imu_path = (output_dir / "spline_imu.txt").string();

  // ===========================================================================

  // Step 1: Load discrete pose sequence.
  hpgt::PoseSequence pose_seq;
  if (hpgt::PoseDataLoader::Load(data_path, pose_seq)) {
    spdlog::info("Loaded {} pose frames.", pose_seq.size());
  } else {
    spdlog::critical("Spline test failed while loading pose data.");
    return EXIT_FAILURE;
  }

  // ===========================================================================

  // Step 2: Initialize the splines.
  // Define spline parameters.
  constexpr int kOrder = 4;
  constexpr double kKnotInterval = 0.01;
  constexpr double kOutputInterval = 0.01;
  double time_margin = 0.02;
  const std::string trans_spline_name = "trans_spline";
  const std::string rot_spline_name = "rot_spline";

  // Calculate the start and end time of the spline. Time margin and knot
  // delta is considered.
  const double kSeqStartTime = pose_seq.front()->timestamp;
  const double kSeqEndTime = pose_seq.back()->timestamp;
  const double kSplineKnotDeltaTime =
      static_cast<double>(kOrder - 2) / 2. * kKnotInterval;
  time_margin += kSplineKnotDeltaTime + kKnotInterval;

  const double kSplineStartTime = kSeqStartTime + time_margin;
  const double kSplineEndTime = kSeqEndTime - time_margin;

  // Specify the spline information.
  auto trans_spline_info =
      hpgt::SplineInfo(trans_spline_name, hpgt::SplineType::EuclideanSpline,
                       kSplineStartTime, kSplineEndTime, kKnotInterval);
  auto rot_spline_info =
      hpgt::SplineInfo(rot_spline_name, hpgt::SplineType::So3Spline,
                       kSplineStartTime, kSplineEndTime, kKnotInterval);

  // Create a B-spline bundle with two splines, one for translation and one for
  // rotation.
  auto spline_bundle =
      hpgt::SplineBundle<kOrder>::Create({trans_spline_info, rot_spline_info});
  auto& trans_spline = spline_bundle->GetR3dSpline(trans_spline_name);
  auto& rot_spline = spline_bundle->GetSo3dSpline(rot_spline_name);

  spdlog::info(
      "Constructed R(3) B-spline from {:.6f} to {:.6f} seconds with {} "
      "control points.",
      trans_spline.MinTime(), trans_spline.MaxTime(),
      trans_spline.get_knots().size());
  spdlog::info(
      "Constructed SO(3) B-spline from {:.6f} to {:.6f} seconds with {} "
      "control points.",
      rot_spline.MinTime(), rot_spline.MaxTime(),
      rot_spline.get_knots().size());

  // ===========================================================================

  // Step 3: Initialize the knots of the spline. Values are obtained from the
  // discrete pose sequence.
  // Note the time delta of the knot.
  size_t knot_size = trans_spline.get_knots().size();
  double current_knot_time = kSplineStartTime - kSplineKnotDeltaTime;
  for (size_t i = 0; i < knot_size; ++i) {
    Eigen::Vector3d knot_trans;
    Eigen::Quaterniond knot_rot_q;

    if (hpgt::InitializerHelper::GetTargetPose(pose_seq, current_knot_time,
                                               knot_trans, knot_rot_q)) {
      trans_spline.get_knot(i) = knot_trans;
      rot_spline.get_knot(i) = Sophus::SO3d(knot_rot_q);
    } else {
      spdlog::critical("Spline test failed while initializing knots.");
      return EXIT_FAILURE;
    }

    current_knot_time += kKnotInterval;
  }

  // ===========================================================================

  // Step 4: Generate pose and inertial data from the spline.
  spdlog::info("======================================================");
  spdlog::info("========== TEST:GENERATE DATA FROM B-SPLINE ==========");
  spdlog::info("======================================================");

  // Open output files.
  std::ofstream output_traj_file(output_traj_path);
  if (!output_traj_file.is_open()) {
    spdlog::critical("Failed to open the output spline trajectory file: {}",
                     output_traj_path);
    return EXIT_FAILURE;
  }
  output_traj_file << std::fixed << std::setprecision(9);

  std::ofstream output_imu_file(output_imu_path);
  if (!output_imu_file.is_open()) {
    spdlog::critical("Failed to open the output spline IMU file: {}",
                     output_imu_path);
    return EXIT_FAILURE;
  }
  output_imu_file << std::fixed << std::setprecision(9);

  // Loop and sample.
  constexpr double kGravityMagnitude = 9.8;
  double current_sample_time = trans_spline.MinTime();
  int output_data_count = 0;
  const Eigen::Vector3d gravity_in_W(0., 0., kGravityMagnitude);
  while (current_sample_time < trans_spline.MaxTime()) {
    // Pose data.
    Eigen::Vector3d sample_trans;
    Sophus::SO3d sample_rot;
    if (!trans_spline.Evaluate(current_sample_time, sample_trans) ||
        !rot_spline.Evaluate(current_sample_time, sample_rot)) {
      spdlog::critical("Spline test failed while evaluating pose data.");
      return EXIT_FAILURE;
    }

    // Inertial data.
    Eigen::Vector3d sample_acc;
    Eigen::Vector3d sample_gyr;
    if (!trans_spline.Acceleration(current_sample_time, sample_acc) ||
        !rot_spline.VelocityBody(current_sample_time, sample_gyr)) {
      spdlog::critical("Spline test failed while evaluating inertial data.");
      return EXIT_FAILURE;
    }

    // Transform the acceleration to body frame.
    sample_acc = sample_rot.inverse() * (sample_acc + gravity_in_W);

    // Output to file stream.
    const Eigen::Quaterniond sample_quat = sample_rot.unit_quaternion();
    output_traj_file << current_sample_time << " " << sample_trans.x() << " "
                     << sample_trans.y() << " " << sample_trans.z() << " "
                     << sample_quat.x() << " " << sample_quat.y() << " "
                     << sample_quat.z() << " " << sample_quat.w() << '\n';

    output_imu_file << static_cast<long>(current_sample_time * 1.e9) << ", "
                    << sample_gyr.x() << ", " << sample_gyr.y() << ", "
                    << sample_gyr.z() << ", " << sample_acc.x() << ", "
                    << sample_acc.y() << ", " << sample_acc.z() << '\n';

    current_sample_time += kOutputInterval;
    ++output_data_count;
  }

  output_traj_file.close();
  output_imu_file.close();

  spdlog::info("Generated {} pose and inertial samples from the B-spline.",
               output_data_count);
  spdlog::info("Output data written to: {}", output_dir.string());

  // ===========================================================================

  // Step 5: Validation of spline segment calculation.
  spdlog::info("======================================================");
  spdlog::info("========= TEST: CALCULATE SEGMENT OF SPLINE  =========");
  spdlog::info("======================================================");

  // Calculate a segment from the spline.
  const double kMetaMinTime = 1034777.8383324;
  const double kMetaMaxTime = 1034778.0233321;
  hpgt::TimeSpan kTimeSpan({kMetaMinTime, kMetaMaxTime});
  hpgt::TimeSpanList kTimeSpanList = {kTimeSpan};

  hpgt::SplineMeta<kOrder> trans_meta, rot_meta;
  if (!spline_bundle->CalculateR3dSplineMeta(trans_spline_name, kTimeSpanList,
                                             trans_meta) ||
      !spline_bundle->CalculateSo3dSplineMeta(rot_spline_name, kTimeSpanList,
                                              rot_meta)) {
    spdlog::critical("Spline test failed while calculating spline metadata.");
    return EXIT_FAILURE;
  }

  // Print the result.
  hpgt::SplineSegment<kOrder> trans_segment = trans_meta.segments[0];
  hpgt::SplineSegment<kOrder> rot_segment = rot_meta.segments[0];
  spdlog::info("Calculated spline segment from {:.6f} to {:.6f} seconds.",
               kMetaMinTime, kMetaMaxTime);
  spdlog::info("R(3) spline segment information:");
  spdlog::info("Min time: {:.6f}, max time: {:.6f}, number of knots: {}",
               trans_segment.MinTime(), trans_segment.MaxTime(),
               trans_segment.NumParameters());
  spdlog::info("SO(3) spline segment information:");
  spdlog::info("Min time: {:.6f}, max time: {:.6f}, number of knots: {}",
               rot_segment.MinTime(), rot_segment.MaxTime(),
               rot_segment.NumParameters());

  // Compute time index within the segment.
  const double kSampleTime = 1034777.9;
  size_t index;
  double fraction;
  if (!rot_segment.ComputeTimeIndex(kSampleTime, index, fraction)) {
    spdlog::critical("Spline test failed while computing the time index.");
    return EXIT_FAILURE;
  }
  spdlog::info("Sampled the R(3) segment at {}:", kSampleTime);
  spdlog::info("The resulting index and fraction are {}, {:.6f}, respectively.",
               index, fraction);

  // ===========================================================================

  spdlog::info("Spline test completed.");

  return 0;
}
