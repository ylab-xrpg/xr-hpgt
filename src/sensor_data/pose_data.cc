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

#include "hpgt/sensor_data/pose_data.h"

namespace hpgt {

PoseFrame::PoseFrame(const double &t, const Eigen::Vector3d &p,
                     const Eigen::Quaterniond &q)
    : timestamp(t), trans(p), rot_q(q) {}

PoseFrame::Ptr PoseFrame::Create(const double &t, const Eigen::Vector3d &p,
                                 const Eigen::Quaterniond &q) {
  return Ptr(new PoseFrame(t, p, q));
}

bool PoseDataLoader::Load(const std::string &data_path,
                          PoseSequence &pose_data) {
  std::ifstream file(data_path);
  if (!file.is_open()) {
    spdlog::critical("Failed to open pose data file: {}", data_path);
    return false;
  }

  std::string line;
  double last_timestamp = std::numeric_limits<double>::lowest();
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    std::vector<double> values;

    try {
      while (std::getline(ss, value, ' ')) {
        values.push_back(std::stod(value));
      }
    } catch (const std::exception &e) {
      spdlog::critical(
          "Timestamped poses in the pose file must be in TUM format, i.e. "
          "\"timestamp(s) tx(m) ty(m) tz(m) qx qy qz qw\". ");
      return false;
    }

    if (values.size() != 8) {
      spdlog::critical(
          "Timestamped poses in the pose file must be in TUM format, i.e. "
          "\"timestamp(s) tx(m) ty(m) tz(m) qx qy qz qw\". ");
      return false;
    }

    double timestamp(values[0]);
    Eigen::Vector3d trans(values[1], values[2], values[3]);
    Eigen::Quaterniond rot(values[7], values[4], values[5], values[6]);

    if (timestamp < last_timestamp) {
      spdlog::critical(
          "Timestamps in the pose file must be in ascending order. ");
      return false;
    }

    pose_data.push_back(PoseFrame::Create(timestamp, trans, rot));
    last_timestamp = timestamp;
  }

  return true;
}

}  // namespace hpgt
