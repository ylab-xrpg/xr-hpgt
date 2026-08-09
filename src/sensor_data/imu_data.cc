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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>

namespace {

bool IsHeaderLine(const std::string &line) {
  std::string lower_line = line;
  std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return lower_line.find("timestamp") != std::string::npos;
}

bool ParseImuLine(const std::string &line, std::vector<double> &values) {
  const size_t last_character = line.find_last_not_of(" \t\r\n");
  if (last_character == std::string::npos || line[last_character] == ',') {
    return false;
  }

  std::stringstream stream(line);
  std::string value;
  values.reserve(7);
  while (std::getline(stream, value, ',')) {
    try {
      size_t parsed_size = 0;
      const double parsed_value = std::stod(value, &parsed_size);
      if (value.find_first_not_of(" \t\r\n", parsed_size) !=
          std::string::npos) {
        return false;
      }
      values.push_back(parsed_value);
    } catch (const std::exception &) {
      return false;
    }
  }
  return values.size() == 7;
}

}  // namespace

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
    spdlog::critical("Failed to open IMU data file: {}", data_path);
    return false;
  }

  ImuSequence parsed_data;
  std::string line;
  size_t line_number = 0;
  bool first_content_line = true;
  double last_timestamp = std::numeric_limits<double>::lowest();
  while (std::getline(file, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    std::vector<double> values;
    if (!ParseImuLine(line, values)) {
      if (first_content_line && IsHeaderLine(line)) {
        first_content_line = false;
        continue;
      }
      spdlog::critical(
          "Invalid IMU data in '{}' at line {}. Expected "
          "\"timestamp (ns), wx (rad/s), wy (rad/s), wz (rad/s), "
          "ax (m/s^2), ay (m/s^2), az (m/s^2)\".",
          data_path, line_number);
      return false;
    }
    first_content_line = false;

    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
      spdlog::critical("Non-finite IMU value in '{}' at line {}.", data_path,
                       line_number);
      return false;
    }
    values[0] /= 1e9;

    const double timestamp = values[0];
    const Eigen::Vector3d acc(values[4], values[5], values[6]);
    const Eigen::Vector3d gyr(values[1], values[2], values[3]);

    if (timestamp <= last_timestamp) {
      spdlog::critical(
          "IMU timestamps in '{}' must be strictly increasing; invalid value "
          "at line {}.",
          data_path, line_number);
      return false;
    }

    parsed_data.push_back(ImuFrame::Create(timestamp, acc, gyr));
    last_timestamp = timestamp;
  }

  if (parsed_data.empty()) {
    spdlog::critical("IMU data file contains no measurements: {}", data_path);
    return false;
  }

  imu_data = std::move(parsed_data);
  return true;
}

}  // namespace hpgt
