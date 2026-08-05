from types import SimpleNamespace

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  N, T_IDXS, T_DIFFS, CRUISE_MIN_ACCEL, ISO_LATERAL_ACCEL,
  get_A_LAT_max_from_personality, get_cruise_min_accel_factor,
  curvature_from_path_polys, speed_limit_from_curvature, apply_curvature_speed_limit,
)


def make_path(x_coeffs, y_coeffs):
  return SimpleNamespace(xCoefficients=list(x_coeffs), yCoefficients=list(y_coeffs))


class TestCurvatureSpeedLimiter(OpenpilotTestCase):
  def test_straight_path_does_not_reduce_cruise(self):
    # x = 20*t, y = 0
    path = make_path([0.0, 20.0], [0.0])
    v_cruise = np.full(N + 1, 30.0)
    limited = apply_curvature_speed_limit(v_cruise, path, log.LongitudinalPersonality.standard)
    np.testing.assert_array_equal(limited, v_cruise)

  def test_constant_radius_speed_limit(self):
    # Near t=0, parabolic path y = 0.5*κ*v^2*t^2 with x = v*t has curvature κ
    kappa = 0.01  # 1/m
    v = 20.0
    a_lat = get_A_LAT_max_from_personality(log.LongitudinalPersonality.standard)
    path = make_path([0.0, v], [0.0, 0.0, 0.5 * kappa * v * v])

    kappa_t = curvature_from_path_polys(path.xCoefficients, path.yCoefficients, np.array([0.0]))
    np.testing.assert_allclose(kappa_t[0], kappa, rtol=1e-6)

    v_lim = speed_limit_from_curvature(kappa_t, a_lat)
    np.testing.assert_allclose(v_lim[0], np.sqrt(a_lat / kappa), rtol=1e-6)

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
    # Soft quartic lateral path: curvature grows with t, so late horizon limits speed.
    # Backward prop with personality-scaled CRUISE_MIN_ACCEL must keep the profile achievable.
    personality = log.LongitudinalPersonality.standard
    cruise_min_accel = CRUISE_MIN_ACCEL * get_cruise_min_accel_factor(personality)
    v = 25.0
    path = make_path([0.0, v], [0.0, 0.0, 0.0, 0.0, 0.002])
    v_cruise = np.full(N + 1, 40.0)
    limited = apply_curvature_speed_limit(v_cruise, path, personality)

    assert np.any(limited < v_cruise)
    for i in range(N):
      max_from_next = limited[i + 1] - cruise_min_accel * T_DIFFS[i + 1]
      assert limited[i] <= max_from_next + 1e-6

    a_lat = get_A_LAT_max_from_personality(personality)
    kappa = curvature_from_path_polys(path.xCoefficients, path.yCoefficients, T_IDXS)
    v_lim_raw = speed_limit_from_curvature(kappa, a_lat)
    # If the end of the horizon is limited enough to require braking from t=0, cruise[0] drops
    v_needed_at_0 = v_lim_raw[-1]
    for i in range(N - 1, -1, -1):
      v_needed_at_0 = min(v_lim_raw[i], v_needed_at_0 - cruise_min_accel * T_DIFFS[i + 1])
    if v_needed_at_0 < v_cruise[0]:
      assert limited[0] < v_cruise[0]

  def test_cruise_min_accel_factor_ordering(self):
    relaxed = get_cruise_min_accel_factor(log.LongitudinalPersonality.relaxed)
    standard = get_cruise_min_accel_factor(log.LongitudinalPersonality.standard)
    aggressive = get_cruise_min_accel_factor(log.LongitudinalPersonality.aggressive)
    assert relaxed < standard < aggressive

  def test_zero_path_speed_no_nans(self):
    # Constant point path: ẋ=ẏ=0 → denominator would be zero without flooring
    kappa = curvature_from_path_polys([5.0], [2.0], T_IDXS)
    assert np.all(np.isfinite(kappa))
    np.testing.assert_array_equal(kappa, np.zeros_like(T_IDXS))

    path = make_path([5.0, 0.0], [2.0, 0.0])  # zero velocity, nonzero position
    v_cruise = np.full(N + 1, 25.0)
    limited = apply_curvature_speed_limit(v_cruise, path, log.LongitudinalPersonality.standard)
    assert np.all(np.isfinite(limited))
    np.testing.assert_array_equal(limited, v_cruise)

  def test_invalid_path_unchanged(self):
    v_cruise = np.full(N + 1, 25.0)

    empty = SimpleNamespace(xCoefficients=[], yCoefficients=[])
    np.testing.assert_array_equal(
      apply_curvature_speed_limit(v_cruise, empty, log.LongitudinalPersonality.standard),
      v_cruise,
    )

    short = SimpleNamespace(xCoefficients=[1.0], yCoefficients=[1.0])
    np.testing.assert_array_equal(
      apply_curvature_speed_limit(v_cruise, short, log.LongitudinalPersonality.standard),
      v_cruise,
    )

    missing = SimpleNamespace()
    np.testing.assert_array_equal(
      apply_curvature_speed_limit(v_cruise, missing, log.LongitudinalPersonality.standard),
      v_cruise,
    )
