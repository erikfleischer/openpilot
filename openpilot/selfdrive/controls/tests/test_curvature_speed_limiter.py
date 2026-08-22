from types import SimpleNamespace

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  N, T_IDXS, T_DIFFS, CRUISE_MIN_ACCEL, ISO_LATERAL_ACCEL, MODEL_T_IDXS,
  CURVATURE_LIMIT_LEAD_TIME, get_A_LAT_max_from_personality, get_cruise_min_accel_factor,
  curvature_from_model, speed_limit_from_curvature, apply_curvature_speed_limit,
)


def make_model(velocity_x, yaw_rate):
  return SimpleNamespace(
    velocity=SimpleNamespace(x=list(velocity_x)),
    orientationRate=SimpleNamespace(z=list(yaw_rate)),
  )


class TestCurvatureSpeedLimiter(OpenpilotTestCase):
  def test_straight_path_does_not_reduce_cruise(self):
    v = 20.0
    model = make_model(np.full(ModelConstants.IDX_N, v), np.zeros(ModelConstants.IDX_N))
    v_cruise = np.full(N + 1, 30.0)
    limited = apply_curvature_speed_limit(v_cruise, model, log.LongitudinalPersonality.standard)
    np.testing.assert_array_equal(limited, v_cruise)

  def test_constant_radius_speed_limit(self):
    kappa = 0.01  # 1/m
    v = 20.0
    a_lat = get_A_LAT_max_from_personality(log.LongitudinalPersonality.standard)
    yaw_rate = kappa * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    yaw_rates = np.full(ModelConstants.IDX_N, yaw_rate)

    kappa_t = curvature_from_model(velocity_x, yaw_rates)
    np.testing.assert_allclose(kappa_t, kappa, rtol=1e-6)

    v_lim = speed_limit_from_curvature(kappa_t, a_lat)
    np.testing.assert_allclose(v_lim, np.sqrt(a_lat / kappa), rtol=1e-6)

  def test_personality_ordering(self):
    relaxed = get_A_LAT_max_from_personality(log.LongitudinalPersonality.relaxed)
    standard = get_A_LAT_max_from_personality(log.LongitudinalPersonality.standard)
    aggressive = get_A_LAT_max_from_personality(log.LongitudinalPersonality.aggressive)

    assert relaxed <= standard <= aggressive <= ISO_LATERAL_ACCEL

    kappa = 0.02
    v_relaxed = speed_limit_from_curvature(kappa, relaxed)
    v_standard = speed_limit_from_curvature(kappa, standard)
    v_aggressive = speed_limit_from_curvature(kappa, aggressive)
    assert v_relaxed[()] <= v_standard[()] <= v_aggressive[()]

  def test_backward_propagation_braking_lead_in(self):
    # Curvature grows with time so late horizon limits speed; back-prop must keep profile achievable.
    personality = log.LongitudinalPersonality.standard
    cruise_min_accel = CRUISE_MIN_ACCEL * get_cruise_min_accel_factor(personality)
    v = 25.0
    # κ(t) rises from ~0 to 0.04 over the model horizon
    kappa_model = 0.04 * (MODEL_T_IDXS / MODEL_T_IDXS[-1])
    yaw_rates = kappa_model * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    model = make_model(velocity_x, yaw_rates)

    v_cruise = np.full(N + 1, 40.0)
    limited = apply_curvature_speed_limit(v_cruise, model, personality)

    assert np.any(limited < v_cruise)
    for i in range(N):
      max_from_next = limited[i + 1] - cruise_min_accel * T_DIFFS[i + 1]
      assert limited[i] <= max_from_next + 1e-6

    a_lat = get_A_LAT_max_from_personality(personality)
    kappa = curvature_from_model(velocity_x, yaw_rates)
    v_lim_raw = speed_limit_from_curvature(kappa, a_lat)
    v_needed_at_0 = v_lim_raw[-1]
    for i in range(N - 1, -1, -1):
      v_needed_at_0 = min(v_lim_raw[i], v_needed_at_0 - cruise_min_accel * T_DIFFS[i + 1])
    if v_needed_at_0 < v_cruise[0]:
      assert limited[0] < v_cruise[0]

  def test_lead_time_applies_late_horizon_limits_earlier(self):
    # Late-horizon curve: accel back-prop is feasible, then 1s look-ahead pulls it earlier.
    personality = log.LongitudinalPersonality.standard
    cruise_min_accel = CRUISE_MIN_ACCEL * get_cruise_min_accel_factor(personality)
    v = 25.0
    kappa_model = np.where(MODEL_T_IDXS >= 5.0, 0.04, 1e-6)
    yaw_rates = kappa_model * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    model = make_model(velocity_x, yaw_rates)

    v_cruise = np.full(N + 1, 40.0)
    limited = apply_curvature_speed_limit(v_cruise, model, personality)

    a_lat = get_A_LAT_max_from_personality(personality)
    kappa = curvature_from_model(velocity_x, yaw_rates)
    v_lim_accel_only = speed_limit_from_curvature(kappa, a_lat)
    for i in range(N - 1, -1, -1):
      v_lim_accel_only[i] = min(v_lim_accel_only[i], v_lim_accel_only[i + 1] - cruise_min_accel * T_DIFFS[i + 1])
    v_lim_accel_only = np.minimum(v_cruise, v_lim_accel_only)

    v_lim_lead = np.interp(T_IDXS + CURVATURE_LIMIT_LEAD_TIME, T_IDXS, v_lim_accel_only)
    expected = np.minimum(v_lim_accel_only, v_lim_lead)
    np.testing.assert_allclose(limited, expected, atol=1e-6)
    assert np.all(limited <= v_lim_lead + 1e-6)
    assert np.any(limited < v_lim_accel_only - 1e-6)

    for i in range(N):
      max_from_next = limited[i + 1] - cruise_min_accel * T_DIFFS[i + 1]
      assert limited[i] <= max_from_next + 1e-5

  def test_cruise_min_accel_factor_ordering(self):
    relaxed = get_cruise_min_accel_factor(log.LongitudinalPersonality.relaxed)
    standard = get_cruise_min_accel_factor(log.LongitudinalPersonality.standard)
    aggressive = get_cruise_min_accel_factor(log.LongitudinalPersonality.aggressive)
    assert relaxed < standard < aggressive

  def test_low_speed_floor_no_nans(self):
    # Near-zero velocity floored by MIN_SPEED; still finite curvature
    yaw_rates = np.full(ModelConstants.IDX_N, 0.1)
    velocity_x = np.zeros(ModelConstants.IDX_N)
    kappa = curvature_from_model(velocity_x, yaw_rates)
    assert np.all(np.isfinite(kappa))
    np.testing.assert_allclose(kappa, 0.1 / MIN_SPEED, rtol=1e-6)

    model = make_model(velocity_x, np.zeros(ModelConstants.IDX_N))
    v_cruise = np.full(N + 1, 25.0)
    limited = apply_curvature_speed_limit(v_cruise, model, log.LongitudinalPersonality.standard)
    assert np.all(np.isfinite(limited))
    np.testing.assert_array_equal(limited, v_cruise)

  def test_invalid_model_unchanged(self):
    v_cruise = np.full(N + 1, 25.0)

    empty = make_model([], [])
    np.testing.assert_array_equal(
      apply_curvature_speed_limit(v_cruise, empty, log.LongitudinalPersonality.standard),
      v_cruise,
    )

    short = make_model([1.0], [0.0])
    np.testing.assert_array_equal(
      apply_curvature_speed_limit(v_cruise, short, log.LongitudinalPersonality.standard),
      v_cruise,
    )
