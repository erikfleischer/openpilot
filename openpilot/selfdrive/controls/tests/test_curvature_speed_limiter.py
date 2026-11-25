from types import SimpleNamespace

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  N, T_IDXS, T_DIFFS, CRUISE_MIN_ACCEL, ISO_LATERAL_ACCEL, MODEL_T_IDXS,
  CURVATURE_LIMIT_LEAD_TIME, get_A_TOTAL_max_from_personality, get_cruise_min_accel_factor,
  curvature_from_model, speed_limit_from_curvature,
  apply_curvature_speed_limit,
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
    a_total = get_A_TOTAL_max_from_personality(log.LongitudinalPersonality.standard)
    yaw_rate = kappa * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    yaw_rates = np.full(ModelConstants.IDX_N, yaw_rate)

    kappa_t = curvature_from_model(velocity_x, yaw_rates)
    np.testing.assert_allclose(kappa_t, kappa, rtol=1e-6)

    v_lim = speed_limit_from_curvature(kappa_t, a_total)
    np.testing.assert_allclose(v_lim, np.sqrt(a_total / kappa), rtol=1e-6)

  def test_personality_ordering(self):
    relaxed = get_A_TOTAL_max_from_personality(log.LongitudinalPersonality.relaxed)
    standard = get_A_TOTAL_max_from_personality(log.LongitudinalPersonality.standard)
    aggressive = get_A_TOTAL_max_from_personality(log.LongitudinalPersonality.aggressive)

    assert relaxed <= standard <= aggressive <= ISO_LATERAL_ACCEL

    kappa = 0.02
    v_relaxed = speed_limit_from_curvature(kappa, relaxed)
    v_standard = speed_limit_from_curvature(kappa, standard)
    v_aggressive = speed_limit_from_curvature(kappa, aggressive)
    assert v_relaxed[()] <= v_standard[()] <= v_aggressive[()]

  def test_future_curvature_does_not_lower_current_speed(self):
    # A curve several seconds out must not pull the current speed limit down.
    personality = log.LongitudinalPersonality.standard
    v = 25.0
    kappa_model = np.where(MODEL_T_IDXS >= 5.0, 0.04, 1e-6)
    yaw_rates = kappa_model * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    model = make_model(velocity_x, yaw_rates)

    v_cruise = np.full(N + 1, 40.0)
    limited = apply_curvature_speed_limit(v_cruise, model, personality)

    assert limited[0] == v_cruise[0]
    assert np.any(limited < v_cruise)

  def test_lead_time_applies_late_horizon_limits_earlier(self):
    # Rising curvature: the 1s look-ahead pulls later limits forward.
    personality = log.LongitudinalPersonality.standard
    cruise_min_accel = CRUISE_MIN_ACCEL * get_cruise_min_accel_factor(personality)
    v = 25.0
    kappa_model = 0.04 * (MODEL_T_IDXS / MODEL_T_IDXS[-1])
    yaw_rates = kappa_model * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    model = make_model(velocity_x, yaw_rates)

    v_cruise = np.full(N + 1, 40.0)
    limited = apply_curvature_speed_limit(v_cruise, model, personality)

    a_total_max = get_A_TOTAL_max_from_personality(personality)
    kappa = curvature_from_model(velocity_x, yaw_rates)
    v_lim = speed_limit_from_curvature(kappa, a_total_max)
    a_long = np.zeros_like(v_lim)
    a_long[:-1] = (v_lim[1:] - v_lim[:-1]) / T_DIFFS[1:]
    a_brake = np.clip(a_long, cruise_min_accel, 0.0)
    a_lat_allowed = np.sqrt(np.maximum(a_total_max**2 - a_brake**2, 0.0))
    v_lim = np.minimum(v_lim, speed_limit_from_curvature(kappa, a_lat_allowed))
    before_lead = np.minimum(v_cruise, v_lim)
    v_lim_lead = np.interp(T_IDXS + CURVATURE_LIMIT_LEAD_TIME, T_IDXS, v_lim)
    v_lim = np.minimum(v_lim, v_lim_lead)
    expected = np.minimum(v_cruise, v_lim)
    np.testing.assert_allclose(limited, expected, atol=1e-6)
    assert np.any(limited < before_lead - 1e-6)

  def test_total_acceleration_capped_while_braking(self):
    # Rising curvature: braking overlaps the curve, so a_lat-only would exceed a_total.
    personality = log.LongitudinalPersonality.standard
    a_total_max = get_A_TOTAL_max_from_personality(personality)
    cruise_min_accel = CRUISE_MIN_ACCEL * get_cruise_min_accel_factor(personality)
    v = 25.0
    kappa_model = 0.04 * (MODEL_T_IDXS / MODEL_T_IDXS[-1])
    yaw_rates = kappa_model * v
    velocity_x = np.full(ModelConstants.IDX_N, v)
    model = make_model(velocity_x, yaw_rates)

    v_cruise = np.full(N + 1, 40.0)
    limited = apply_curvature_speed_limit(v_cruise, model, personality)

    kappa = curvature_from_model(velocity_x, yaw_rates)
    a_long = np.zeros_like(limited)
    a_long[:-1] = (limited[1:] - limited[:-1]) / T_DIFFS[1:]
    a_brake = np.clip(a_long, cruise_min_accel, 0.0)
    a_lat = limited**2 * kappa
    a_tot = np.sqrt(a_lat**2 + a_brake**2)
    assert np.all(a_tot <= a_total_max + 1e-2)

    v_lat_only = np.minimum(v_cruise, speed_limit_from_curvature(kappa, a_total_max))
    braking_in_curve = (a_brake < -1e-6) & (kappa > 1e-4)
    assert np.any(braking_in_curve)
    assert np.any(limited[braking_in_curve] < v_lat_only[braking_in_curve] - 1e-6)

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
