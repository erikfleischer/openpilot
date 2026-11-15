import numpy as np
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

ACCEL_BOOST_INITIAL = 0.1
ACCEL_BOOST_MAX = 0.2
ACCEL_BOOST_RATE = 0.025
ACCEL_BOOST_PER_OVERRIDE = 0.05
ACCEL_BOOST_MIN_SPEED = 10 * CV.MPH_TO_MS
# Ramp across the confidence ball's yellow band: 0 at red (0.2), full boost at green (0.5).
CONFIDENCE_HIGH = 0.5
CONFIDENCE_MID = 0.2
CONFIDENCE_FILTER_RC = 0.5
CONFIDENCE_FILTER_INIT = -0.5
# Sentinel: set total_boost from model confidence instead of a fixed value.
ACCEL_BOOST_CONFIDENCE = object()
# Index matches AccelBoostMode. None keeps the gas-pedal accumulator.
ACCEL_BOOST_FIXED = (None, 0.0, 0.1, 0.2, ACCEL_BOOST_CONFIDENCE)


class AccelBoost:
  def __init__(self):
    self.total_boost = 0.0
    self.boost_this_override = 0.0
    self.boost_eligible = False
    self._gas_mode = True
    self._confidence_filter = FirstOrderFilter(CONFIDENCE_FILTER_INIT, CONFIDENCE_FILTER_RC, DT_MDL)

  def _confidence_boost(self, sm, enabled: bool) -> float:
    if not enabled:
      raw = CONFIDENCE_FILTER_INIT
    else:
      preds = sm['modelV2'].meta.disengagePredictions
      raw = ((1 - max(preds.brakeDisengageProbs or [1])) *
             (1 - max(preds.steerOverrideProbs or [1])))
    confidence = self._confidence_filter.update(raw)
    return float(np.interp(confidence, [CONFIDENCE_MID, CONFIDENCE_HIGH], [0.0, ACCEL_BOOST_MAX]))

  def update(self, sm, output_a_target_e2e, output_a_target_mpc, a_cruise, fixed_boost=None):
    enabled = sm['selfdriveState'].enabled
    gas_pressed = sm['carState'].gasPressed
    v_ego = sm['carState'].vEgo
    initial_boost = ACCEL_BOOST_INITIAL if sm['selfdriveState'].personality == log.LongitudinalPersonality.aggressive else 0.0
    self.total_boost = max(self.total_boost, initial_boost) if enabled else initial_boost

    speed_scale = np.interp(v_ego, [ACCEL_BOOST_MIN_SPEED, 2 * ACCEL_BOOST_MIN_SPEED], [0.0, 1.0])
    boost_scale = speed_scale * np.interp(output_a_target_e2e, [-1.0, -0.5, 5.0], [0.0, 1.0, 1.0], right=0.0)

    if fixed_boost is ACCEL_BOOST_CONFIDENCE:
      fixed_boost = self._confidence_boost(sm, enabled)

    if fixed_boost is not None:
      self.boost_this_override = 0.0
      self.boost_eligible = False
      self._gas_mode = False
      self.total_boost = 0.0 if not enabled else float(fixed_boost)
      return output_a_target_e2e + self.total_boost * boost_scale

    # A fixed value must not stick after switching back to the pedal.
    if not self._gas_mode:
      self.total_boost = 0.0
      self.boost_this_override = 0.0
      self.boost_eligible = False
    self._gas_mode = True

    model_limited = output_a_target_e2e + self.total_boost * boost_scale < min(output_a_target_mpc, a_cruise) - 0.1

    if not enabled or not gas_pressed:
      self.boost_this_override = 0.0
      # Latch eligibility before the override changes the lead plan.
      self.boost_eligible = enabled and model_limited

    if enabled and gas_pressed and self.boost_eligible:
      increase = min(ACCEL_BOOST_RATE * DT_MDL * speed_scale,
                     ACCEL_BOOST_PER_OVERRIDE - self.boost_this_override, ACCEL_BOOST_MAX - self.total_boost)
      self.total_boost += increase
      self.boost_this_override += increase

    return output_a_target_e2e + self.total_boost * boost_scale
