# HPGT configuration reference

HPGT reads a JSON configuration file before loading sensor data. The complete
schema is illustrated by
[`data/simulated/config_template.json`](../data/simulated/config_template.json).
The current JSON parser expects every field shown in the template; C++ default
values document the intended defaults but do not make JSON keys optional.

Each entry in `pose_config_` or `imu_config_` represents a generic measurement
stream. Application-specific roles such as motion capture, auxiliary IMU, or
device under test are not hard-coded; every input is handled as a pose or IMU
stream according to its configuration.

All configured values must be finite. Blank `file_name` entries are skipped,
which makes it possible to keep unused template nodes, but the remaining sensor
set must still satisfy all frame and sensor-count constraints.

## Contents

- [Coordinate frames and parameter notation](#coordinate-frames-and-parameter-notation)
- [System-level fields](#system-level-fields)
- [Pose-sensor fields](#pose-sensor-fields)
- [IMU fields](#imu-fields)
- [Vector and quaternion representation](#vector-and-quaternion-representation)
- [Frame-selection constraints](#frame-selection-constraints)
- [Initial values and reusable output](#initial-values-and-reusable-output)
- [Current scope](#current-scope)

## Coordinate frames and parameter notation

HPGT uses the following frames:

- `G`: the system world frame. It is based on the selected pose-sensor world
  frame and becomes gravity-aligned when IMU information is available.
- `B`: the system body frame and the frame of the output trajectory.
- `Pi`: the body frame of pose sensor `i`.
- `Wi`: the world frame in which pose sensor `i` reports `T_Wi_Pi`.
- `Ii`: the body frame of IMU `i`.

The output trajectory is `T_G_B`. For a parameter named `trans_B_A`, the vector
is the origin of frame `A` expressed in frame `B`. `rot_B_A` maps coordinates
from `A` to `B`.

Time-offset names follow the same convention. If `tau_Pi` and `tau_Ii` are
timestamps in the sensors' own clocks, their timestamps in the body clock are:

```text
t_B = tau_Pi + toff_B_Pi
t_B = tau_Ii + toff_B_Ii
```

Consequently, a positive `toff_B_Pi` or `toff_B_Ii` is added to the sensor
timestamp to express it in the body-frame clock.

## System-level fields

| JSON field | Type | Unit | Default | Requirements and effect |
| --- | --- | --- | --- | --- |
| `spline_knot_interval_` | number | s | `0.01` | Must be positive. Sets the cubic B-spline knot spacing. Smaller values increase temporal resolution and computation. |
| `max_toff_change_` | number | s | `0.1` | Finite values are clamped to `[0.01, 0.5]`. Defines the allowed time-offset increment around initialization and contributes to spline margins. |
| `gravity_magnitude_` | number | m/s² | `9.8` | Must be positive. Sets the gravity magnitude used by IMU factors and initialization. |
| `output_frequency_` | number | Hz | `100.0` | Must be positive. Sets the sampling rate of the output trajectory. |
| `opt_temporal_param_flag_` | boolean | — | `true` | When `true`, time offsets are initialized from signal correlation and refined; the body-frame sensor offset remains the fixed clock reference. When `false`, configured offset initial values are used as fixed values. |
| `opt_spatial_param_flag_` | boolean | — | `true` | When `true`, extrinsics are initialized from the measurements and refined subject to frame gauges. When `false`, configured spatial initial values are fixed. |
| `pose_config_` | array | — | empty | Pose-sensor nodes described below. At least one valid node is required. |
| `imu_config_` | array | — | empty | IMU nodes described below. |

The system requires at least two valid sensors in total and at least one pose
sensor.

## Pose-sensor fields

Each entry in `pose_config_` contains:

| JSON field | Type | Unit | Default | Requirements and effect |
| --- | --- | --- | --- | --- |
| `file_name` | string | — | `""` | File name relative to `<input_data_dir>`. An empty name skips the node. Labels must be unique among pose sensors. |
| `body_frame_flag` | boolean | — | `false` | Selects `B`. Exactly one pose or IMU node across the entire configuration must set this to `true`. |
| `world_frame_flag` | boolean | — | `false` | Selects the pose-sensor world frame used to define `G`. Exactly one pose node must set this to `true`. |
| `abs_pose_flag` | boolean | — | `false` | Must be `true` for the current absolute-pose optimization and for the world-frame sensor. Relative-pose factors are not implemented in this release. |
| `trans_noise` | number | m | `5e-4` | Discrete-time translation standard deviation; must be at least `1e-6`. Its reciprocal is used as the translation residual weight. |
| `rot_noise` | number | rad | `5e-3` | Discrete-time rotation standard deviation; must be at least `1e-6`. Its reciprocal is used as the rotation residual weight. |
| `toff_BP_init` | number | s | `0.0` | Pose-clock to body-clock offset. Used as a fixed value when temporal optimization is disabled. |
| `trans_BP_init` | vector3 | m | zero | Initial/fixed translation from `Pi` to `B`. |
| `rot_q_BP_init` | quaternion | — | identity | Initial/fixed rotation from `Pi` to `B`. |
| `trans_GW_init` | vector3 | m | zero | Initial/fixed translation from `Wi` to `G`. |
| `rot_q_GW_init` | quaternion | — | identity | Initial/fixed rotation from `Wi` to `G`. |

With spatial optimization enabled, HPGT computes its own spatial initial guess
from the sensor sequences before nonlinear refinement. With optimization
disabled, the configured `trans_*_init` and `rot_q_*_init` values are used
directly.

## IMU fields

Each entry in `imu_config_` contains:

| JSON field | Type | Unit | Default | Requirements and effect |
| --- | --- | --- | --- | --- |
| `file_name` | string | — | `""` | File name relative to `<input_data_dir>`. An empty name skips the node. Labels must be unique among IMUs. |
| `model_type` | string | — | `"calibrated"` | One of `calibrated`, `scale`, or `scale_misalignment`; an unknown value currently falls back to `calibrated` with a warning. |
| `frequency` | number | Hz | `100.0` | Must be positive. Used with continuous-time measurement noise to form discrete residual weights. |
| `body_frame_flag` | boolean | — | `false` | Selects `B`. Exactly one pose or IMU node must set this flag. |
| `noise.acc_n` | number | m/s²/√Hz | `0.02` | Accelerometer white-noise density; must be at least `1e-6`. Used by the current accelerometer factor. |
| `noise.acc_b` | number | m/s³/√Hz | `0.005` | Accelerometer bias random-walk density; validated and serialized, but not used by the current static-bias factor. |
| `noise.gyr_n` | number | rad/s/√Hz | `0.001` | Gyroscope white-noise density; must be at least `1e-6`. Used by the current gyroscope factor. |
| `noise.gyr_b` | number | rad/s²/√Hz | `0.0005` | Gyroscope bias random-walk density; validated and serialized, but not used by the current static-bias factor. |
| `toff_BI_init` | number | s | `0.0` | IMU-clock to body-clock offset. Used as a fixed value when temporal optimization is disabled. |
| `trans_BI_init` | vector3 | m | zero | Initial/fixed translation from `Ii` to `B`. |
| `rot_q_BI_init` | quaternion | — | identity | Initial/fixed rotation from `Ii` to `B`. |
| `acc_bias_init` | vector3 | m/s² | zero | Serialized accelerometer-bias field. All three IMU models optimize a bias, but the current runtime state starts at zero rather than reading this value. |
| `gyr_bias_init` | vector3 | rad/s | zero | Serialized gyroscope-bias field. All three IMU models optimize a bias, but the current runtime state starts at zero rather than reading this value. |

The current factor weights are:

```text
accelerometer_weight = 1 / (acc_n * sqrt(frequency))
gyroscope_weight     = 1 / (gyr_n * sqrt(frequency))
```

### IMU intrinsic models

All models optimize accelerometer and gyroscope biases. Their additional
degrees of freedom are:

| `model_type` | Mapping matrices | Gyroscope-to-accelerometer rotation |
| --- | --- | --- |
| `calibrated` | Fixed to identity | Fixed to identity |
| `scale` | Optimize both six-parameter upper-triangular mapping matrices | Fixed to identity |
| `scale_misalignment` | Optimize both six-parameter upper-triangular mapping matrices | Optimized |

Each mapping matrix contains three diagonal scale coefficients and three
upper-triangular cross-axis coefficients. The gyroscope-to-accelerometer
rotation is named `rot_gyr_acc` in the implementation.

Mapping coefficients and `rot_gyr_acc` are currently reported in the runtime
log but are not fields in the JSON schema. The output calibration JSON retains
the selected `model_type` and updates the serialized offsets, extrinsics, and
biases; it does not persist those additional scale/misalignment estimates.

## Vector and quaternion representation

JSON vectors use named components:

```json
{"x": 0.0, "y": 0.0, "z": 0.0}
```

JSON quaternions always use `x`, `y`, `z`, and `w` keys:

```json
{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
```

Quaternion values must be finite and have non-zero norm. HPGT normalizes
configuration quaternions during validation.

## Frame-selection constraints

Before loading data, HPGT enforces the following rules:

1. Exactly one sensor across `pose_config_` and `imu_config_` has
   `body_frame_flag: true`.
2. Exactly one pose sensor has `world_frame_flag: true`.
3. A world-frame pose sensor also has `abs_pose_flag: true`.
4. At least one valid pose sensor and at least two valid sensors in total are
   configured.
5. File names are unique within their sensor type.

The selected body sensor defines both `B` and the clock of the output
trajectory. The selected world pose sensor provides the world-frame gauge.

## Initial values and reusable output

When temporal or spatial optimization is disabled, the corresponding `*_init`
values are fixed calibration parameters. When optimization is enabled, HPGT
initializes those parameters from the measurements and then refines them; the
configured temporal/spatial guesses are not used to replace that initializer.

`RunHPGT` writes an output JSON with the estimated serialized parameters. That
file has the same schema as the input and can be supplied as the configuration
for another run. In particular, the estimated temporal and spatial parameters
can be reused in a later fixed-calibration run:

```json
{
  "opt_temporal_param_flag_": false,
  "opt_spatial_param_flag_": false
}
```

The snippet only highlights the two flags; retain every other required field
from the generated JSON. Although estimated IMU biases are written to
`acc_bias_init` and `gyr_bias_init`, the current implementation initializes its
runtime bias states to zero on every run. Those two output fields therefore
record the result but do not yet seed a subsequent optimization.

## Current scope

The optimizer currently adds factors only for pose nodes with
`abs_pose_flag: true`. Relative-pose support is reserved for future work, so a
node with `abs_pose_flag: false` does not contribute pose constraints in this
release. At least one absolute pose sequence is also needed to initialize the
continuous-time spline.
