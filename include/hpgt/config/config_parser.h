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

// clang-format off
#include <string>

#include <Eigen/Eigen>
#include <nlohmann/json.hpp>
// clang-format on

namespace hpgt {

// Enumeration for IMU model types.
enum class ImuModelType { kCalibrated, kScale, kScaleMisalignment, kInvalid };

// Enumeration for sensor types.
enum class SensorType { kPose, kImu, kInvalid };

inline std::string ToString(const ImuModelType& model_type) {
  switch (model_type) {
    case ImuModelType::kCalibrated:
      return "calibrated";
    case ImuModelType::kScale:
      return "scale";
    case ImuModelType::kScaleMisalignment:
      return "scale_misalignment";
    default:
      return "invalid";
  }
}

}  // namespace hpgt

namespace nlohmann {

// Template specialization for JSON serialization
template <>
struct adl_serializer<Eigen::Vector3d> {
  static void to_json(json& j, const Eigen::Vector3d& v) {
    j = json{{"x", v.x()}, {"y", v.y()}, {"z", v.z()}};
  }

  static void from_json(const json& j, Eigen::Vector3d& v) {
    v.x() = j.at("x").get<double>();
    v.y() = j.at("y").get<double>();
    v.z() = j.at("z").get<double>();
  }
};

template <>
struct adl_serializer<Eigen::Quaterniond> {
  static void to_json(json& j, const Eigen::Quaterniond& q) {
    j = json{{"x", q.x()}, {"y", q.y()}, {"z", q.z()}, {"w", q.w()}};
  }

  static void from_json(const json& j, Eigen::Quaterniond& q) {
    q.x() = j.at("x").get<double>();
    q.y() = j.at("y").get<double>();
    q.z() = j.at("z").get<double>();
    q.w() = j.at("w").get<double>();
  }
};

template <>
struct adl_serializer<Eigen::Vector4d> {
  static void to_json(json& j, const Eigen::Vector4d& v) {
    j = json{
        {"acc_n", v[0]}, {"acc_b", v[1]}, {"gyr_n", v[2]}, {"gyr_b", v[3]}};
  }

  static void from_json(const json& j, Eigen::Vector4d& v) {
    v[0] = j.at("acc_n").get<double>();
    v[1] = j.at("acc_b").get<double>();
    v[2] = j.at("gyr_n").get<double>();
    v[3] = j.at("gyr_b").get<double>();
  }
};

template <>
struct adl_serializer<hpgt::ImuModelType> {
  static void to_json(json& j, const hpgt::ImuModelType& m) {
    switch (m) {
      case hpgt::ImuModelType::kCalibrated:
        j = "calibrated";
        break;
      case hpgt::ImuModelType::kScale:
        j = "scale";
        break;
      case hpgt::ImuModelType::kScaleMisalignment:
        j = "scale_misalignment";
        break;
      default:
        j = "";
        break;
    }
  }

  static void from_json(const json& j, hpgt::ImuModelType& m) {
    std::string str = j.get<std::string>();
    if (str == "calibrated")
      m = hpgt::ImuModelType::kCalibrated;
    else if (str == "scale")
      m = hpgt::ImuModelType::kScale;
    else if (str == "scale_misalignment")
      m = hpgt::ImuModelType::kScaleMisalignment;
    else
      m = hpgt::ImuModelType::kInvalid;
  }
};

}  // namespace nlohmann
