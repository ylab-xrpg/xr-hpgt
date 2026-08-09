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

}  // namespace

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

  PoseSequence parsed_data;
  std::string line;
  size_t line_number = 0;
  bool first_content_line = true;
  double last_timestamp = std::numeric_limits<double>::lowest();
  while (std::getline(file, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    std::stringstream ss(line);
    std::vector<double> values;
    values.reserve(8);
    double value = 0.;
    while (ss >> value) {
      values.push_back(value);
    }

    if (!ss.eof() || values.size() != 8) {
      if (first_content_line && IsHeaderLine(line)) {
        first_content_line = false;
        continue;
      }
      spdlog::critical(
          "Invalid pose data in '{}' at line {}. Expected TUM format "
          "\"timestamp(s) tx(m) ty(m) tz(m) qx qy qz qw\".",
          data_path, line_number);
      return false;
    }
    first_content_line = false;

    if (!std::all_of(values.begin(), values.end(),
                     [](double element) { return std::isfinite(element); })) {
      spdlog::critical("Non-finite pose value in '{}' at line {}.", data_path,
                       line_number);
      return false;
    }

    const double timestamp = values[0];
    const Eigen::Vector3d trans(values[1], values[2], values[3]);
    const Eigen::Quaterniond rot(values[7], values[4], values[5], values[6]);

    if (timestamp <= last_timestamp) {
      spdlog::critical(
          "Pose timestamps in '{}' must be strictly increasing; invalid value "
          "at line {}.",
          data_path, line_number);
      return false;
    }

    if (rot.squaredNorm() <= std::numeric_limits<double>::epsilon()) {
      spdlog::critical("Invalid zero-norm quaternion in '{}' at line {}.",
                       data_path, line_number);
      return false;
    }

    parsed_data.push_back(PoseFrame::Create(timestamp, trans, rot));
    last_timestamp = timestamp;
  }

  if (parsed_data.empty()) {
    spdlog::critical("Pose data file contains no measurements: {}", data_path);
    return false;
  }

  pose_data = std::move(parsed_data);
  return true;
}

}  // namespace hpgt
