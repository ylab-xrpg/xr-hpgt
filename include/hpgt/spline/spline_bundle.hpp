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

#include <map>
#include <memory>
#include <string>

#include "hpgt/spline/euclidean_spline.hpp"
#include "hpgt/spline/so3_spline.hpp"
#include "hpgt/spline/spline_meta.hpp"

namespace hpgt {

// Define time types
using TimeSpan = std::pair<double, double>;
using TimeSpanList = std::initializer_list<TimeSpan>;

enum class SplineType { EuclideanSpline, So3Spline };

/**
 * @brief Structure to hold information about a spline.
 */
struct SplineInfo {
  SplineInfo(std::string name, SplineType type, double st, double et, double dt)
      : name(std::move(name)),
        type(type),
        start_time(st),
        end_time(et),
        knot_interval(dt) {}

  // Name of spline, as a search key for access in spline bundle.
  std::string name;
  // Type of spline, Euclidean or SO(3) is supported.
  SplineType type;
  // Start/end time and knot interval of the uniform spline.
  double start_time, end_time, knot_interval;
};

/**
 * @brief Class for managing a bundle of B-splines of different types and of the
 * same order. Provide interfaces for calculating the spline meta data for
 * specified time span list.
 *
 * @tparam Order B-spline order.
 */
template <int Order>
class SplineBundle {
 public:
  using Ptr = std::shared_ptr<SplineBundle<Order>>;

  static constexpr int kN = Order;

  // Type definitions.
  using R3dSplineType = EuclideanSpline<3, Order, double>;
  using So3dSplineType = So3Spline<Order, double>;
  // Eigen::Vector3d
  using R3dSplineKnotType = typename R3dSplineType::VecD;
  // Sophus::SO3d
  using So3dSplineKnotType = typename So3dSplineType::So3;
  using SplineMetaType = SplineMeta<Order>;

  // Construct spline bundle with spline information.
  explicit SplineBundle(const std::vector<SplineInfo> &splines) {
    for (const auto &spline : splines) {
      AddSpline(spline);
    }
  }

  // Create an instance of spline bundle.
  static Ptr Create(const std::vector<SplineInfo> &splines) {
    return Ptr(new SplineBundle<Order>(splines));
  }

  /**
   * @brief Calculate R(3) spline meta data for a givin time span list.
   *
   * @param[in] name Name of the R(3) spline.
   * @param[in] times List of time spans for which the spline meta data is
   * computed.
   * @param[out] spline_meta Spline meta data result.
   */
  bool CalculateR3dSplineMeta(const std::string &name,
                              const TimeSpanList &times,
                              SplineMetaType &spline_meta) const {
    return CalculateSplineMeta(r3d_splines_.at(name), times, spline_meta);
  }

  // Calculate SO(3) spline meta data for a givin time span list.
  bool CalculateSo3dSplineMeta(const std::string &name,
                               const TimeSpanList &times,
                               SplineMetaType &spline_meta) const {
    return CalculateSplineMeta(so3d_splines_.at(name), times, spline_meta);
  }

  // Check if a time is within the valid range.
  bool TimeInRangeForR3dSpline(double time, const std::string &name) const {
    // left closed right open interval
    auto &spline = r3d_splines_.at(name);
    return time >= spline.MinTime() - 1e-6 && time < spline.MaxTime() + 1e-6;
  }

  bool TimeInRangeForSo3dSpline(double time, const std::string &name) const {
    // left closed right open interval
    auto &spline = so3d_splines_.at(name);
    return time >= spline.MinTime() - 1e-6 && time < spline.MaxTime() + 1e-6;
  }

  template <class SplineType>
  bool TimeInRange(double time, const SplineType &spline) const {
    // left closed right open interval
    return time >= spline.MinTime() - 1e-6 && time < spline.MaxTime() + 1e-6;
  }

  // Retrieve the 3D spline for a given name.
  R3dSplineType &GetR3dSpline(const std::string &name) {
    return r3d_splines_.at(name);
  }

  So3dSplineType &GetSo3dSpline(const std::string &name) {
    return so3d_splines_.at(name);
  }

 private:
  // Add a new spline based on the spline infomation.
  void AddSpline(const SplineInfo &spline) {
    switch (spline.type) {
      // Insert a new R(3) spline.
      case SplineType::EuclideanSpline: {
        r3d_splines_.insert({spline.name, R3dSplineType(spline.knot_interval,
                                                        spline.start_time)});
        ExtendKnots(r3d_splines_.at(spline.name), spline.end_time,
                    R3dSplineKnotType::Zero());
      } break;
      // Insert a new SO(3) spline.
      case SplineType::So3Spline: {
        so3d_splines_.insert({spline.name, So3dSplineType(spline.knot_interval,
                                                          spline.start_time)});
        ExtendKnots(so3d_splines_.at(spline.name), spline.end_time,
                    So3dSplineKnotType());
      } break;
    }
  }

  /**
   * @brief Extent the spline knots with default value.
   *
   * @tparam SplineType Type of the spline.
   * @tparam KnotType Type of the spline knot.
   * @param[in] spline Extended spline.
   * @param[in] end_time End time of spline.
   * @param[out] init Default value of the knots.
   */
  template <class SplineType, class KnotType>
  void ExtendKnots(SplineType &spline, double end_time, const KnotType &init) {
    while ((spline.get_knots().size() < kN) || (spline.MaxTime() < end_time)) {
      spline.KnotsPushBack(init);
    }
  }

  // Calculate R(3) spline meta data for a givin time span list.
  template <class SplineType>
  bool CalculateSplineMeta(const SplineType &spline, const TimeSpanList &times,
                           SplineMetaType &spline_meta) const {
    double master_dt = spline.get_knot_interval();
    double master_t0 = spline.MinTime();
    size_t current_segment_start = 0;
    size_t current_segment_end = 0;

    for (auto tt : times) {
      size_t index_1, index_2;
      double fraction_1, fraction_2;
      if (!spline.ComputeTimeIndex(tt.first, index_1, fraction_1) ||
          !spline.ComputeTimeIndex(tt.second, index_2, fraction_2)) {
        return false;
      }

      // Create new segment, or extend the current one
      if (spline_meta.segments.empty() || index_1 > current_segment_end) {
        double segment_t0 = master_t0 + master_dt * double(index_1);
        spline_meta.segments.emplace_back(segment_t0, master_dt);
        current_segment_start = index_1;
      } else {
        index_1 = current_segment_end + 1;
      }

      auto &current_segment_meta = spline_meta.segments.back();

      for (size_t i = index_1; i < (index_2 + kN); ++i) {
        current_segment_meta.knot_num += 1;
      }

      current_segment_end =
          current_segment_start + current_segment_meta.knot_num - 1;
    }  // for times

    return true;
  }

  // Container for R(3) splines.
  std::map<std::string, R3dSplineType> r3d_splines_;
  // Container for SO(3)d splines.
  std::map<std::string, So3dSplineType> so3d_splines_;
};

}  // namespace hpgt
