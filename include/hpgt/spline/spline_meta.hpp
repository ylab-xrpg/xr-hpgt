// Portions of this file are adapted from:
// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Licensed under the BSD-style license found in the original CTraj repository.
//
// ---
//
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
#include <memory>
#include <vector>

#include <ceres/jet.h>
#include <spdlog/spdlog.h>
// clang-format on

namespace hpgt {

/**
 * @brief Spline segment of specified order. Segment data is represented by time
 * information without storing the knot's data.
 *
 * @tparam Order B-spline order.
 */
template <int Order>
struct SplineSegment {
  // Order and degree of the spline.
  static constexpr int kN = Order;
  static constexpr int kDeg = Order - 1;

  SplineSegment(double t0, double dt, size_t n = 0)
      : start_time(t0), knot_interval(dt), knot_num(n) {}

  /**
   * @brief Compute the index and fractional part of a given timestamp
   * relative to the segment start time.
   *
   * @tparam T Scalar type.
   * @param[in] timestamp Input timestamp used to calculate the interval from
   * the start time.
   * @param[out] index Computed time index.
   * @param[out] fraction Relative position (fractional part) within the current
   * time segment.
   * @return True if the timestamp is valid and within range.
   */
  template <typename T>
  bool ComputeTimeIndex(const T &timestamp, size_t &index, T &fraction) const {
    T t = timestamp;
    if (t >= T(MaxTime()))
      t = t - T(1.e-6);
    else if (t <= T(MinTime()))
      t = t + T(1.e-6);

    if (t > T(MinTime()) && t < T(MaxTime())) {
      T time_since_start = (t - T(start_time)) / T(knot_interval);
      index = PotentiallyUnsafeFloor(time_since_start);
      fraction = time_since_start - T(index);
      return true;
    } else {
      return false;
    }
  }

  // Number of knot parameters.
  size_t NumParameters() const { return knot_num; }

  // Minimum/Maximum time represented by spline.
  double MinTime() const { return start_time; }

  double MaxTime() const {
    return start_time + (knot_num - kDeg) * knot_interval;
  }

  template <typename T>
  size_t PotentiallyUnsafeFloor(T x) const {
    return static_cast<size_t>(std::floor(x));
  }

  // This way of treating Jets are potentially unsafe, hence the function name.
  template <typename Scalar, int N>
  size_t PotentiallyUnsafeFloor(const ceres::Jet<Scalar, N> &x) const {
    return static_cast<size_t>(ceres::floor(x.a));
  };

  // First valid time.
  double start_time;
  // Time interval of knots.
  double knot_interval;
  // Number of knots.
  size_t knot_num;
};

/**
 * @brief Spline meta class with multiple spline segments.
 *
 * @tparam Order B-spline order.
 */
template <int Order>
struct SplineMeta {
  // Total number of knot parameters.
  size_t NumParameters() const {
    size_t n = 0;
    for (auto &segment_meta : segments) {
      n += segment_meta.NumParameters();
    }
    return n;
  }

  /**
   * @brief Compute the relative index and fractional part of a given timestamp
   * for the spline meta.
   *
   * @tparam T Scalar type.
   * @param[in] timestamp Input timestamp used to calculate the interval from
   * the start time.
   * @param[out] index Computed time index (compared to meta start point).
   * @param[out] fraction Relative position (fractional part) within the current
   * spline meta.
   * @return True if the timestamp is valid and within range.
   */
  template <typename T>
  bool ComputeSplineIndex(const T &timestamp, size_t &index,
                          T &fraction) const {
    index = 0;
    for (auto const &seg : segments) {
      size_t idx = 0;
      if (seg.ComputeTimeIndex(timestamp, idx, fraction)) {
        index += idx;
        return true;
      } else {
        index += seg.NumParameters();
      }
    }

    return false;
  }

  // Container for multiple spline segments.
  std::vector<SplineSegment<Order>> segments;
};

}  // namespace hpgt
