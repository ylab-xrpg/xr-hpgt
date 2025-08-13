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

#include "hpgt/sensor_data/imu_data.h"

namespace hpgt {

ImuFrame::ImuFrame(const double &t, const Eigen::Vector3d &a,
                   const Eigen::Vector3d &w)
    : timestamp(t), acc(a), gyr(w) {}

ImuFrame::Ptr ImuFrame::Create(const double &t, const Eigen::Vector3d &a,
                               const Eigen::Vector3d &w) {
  return Ptr(new ImuFrame(t, a, w));
}

bool ImuDataLoader::Load(const std::string &data_path, ImuSequence &imu_data) {
  std::ifstream file(data_path);
  if (!file.is_open()) {
    spdlog::critical("Failed to open imu data file: {}", data_path);
    return false;
  }

  std::string line;
  double last_timestamp = std::numeric_limits<double>::lowest();

  // Skip the first line (assumed to be header)
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    std::vector<double> values;

    try {
      while (std::getline(ss, value, ',')) {
        values.push_back(std::stod(value));
      }
    } catch (const std::exception &e) {
      spdlog::critical(
          "Each line in the IMU file must be in format "
          "\"timestamp (ns), wx (rad/s), wy (rad/s), wz (rad/s), "
          "ax (m/s^2), ay (m/s^2), az (m/s^2)\". ");
      return false;
    }

    if (values.size() == 7) {
      values[0] /= 1e9;
    } else {
      spdlog::critical(
          "Each line in the IMU file must be in format "
          "\"timestamp (ns), wx (rad/s), wy (rad/s), wz (rad/s), "
          "ax (m/s^2), ay (m/s^2), az (m/s^2)\". ");
      return false;
    }

    double timestamp(values[0]);
    Eigen::Vector3d acc(values[4], values[5], values[6]);
    Eigen::Vector3d gyr(values[1], values[2], values[3]);

    if (timestamp < last_timestamp) {
      spdlog::critical(
          "Timestamps in the IMU file must be in ascending order. ");
      return false;
    }

    imu_data.push_back(ImuFrame::Create(timestamp, acc, gyr));
    last_timestamp = timestamp;
  }

  return true;
}

}  // namespace hpgt
