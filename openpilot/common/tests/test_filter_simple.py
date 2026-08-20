import cmath
import math

from openpilot.common.filter_simple import SecondOrderBesselFilter
from openpilot.common.test import OpenpilotTestCase


def _biquad_h(f_hz, fs, b0, b1, b2, a1, a2):
  z = cmath.exp(1j * 2.0 * math.pi * f_hz / fs)
  num = b0 + b1 / z + b2 / (z * z)
  den = 1.0 + a1 / z + a2 / (z * z)
  return num / den


class TestSecondOrderBesselFilter(OpenpilotTestCase):
  def setup_method(self):
    self.fs = 20.0
    self.dt = 1.0 / self.fs
    self.fc = 2.0
    self.filt = SecondOrderBesselFilter(0.0, self.fc, self.dt)

  def test_dc_gain(self):
    y = 0.0
    for _ in range(int(5.0 / self.dt)):
      y = self.filt.update(1.0)
    assert abs(y - 1.0) < 1e-6

  def test_corner_frequency(self):
    mag = abs(_biquad_h(self.fc, self.fs, self.filt.b0, self.filt.b1, self.filt.b2,
                        self.filt.a1, self.filt.a2))
    assert abs(mag - 1.0 / math.sqrt(2.0)) < 1e-3

  def test_reset(self):
    for _ in range(20):
      self.filt.update(1.0)
    self.filt.reset(-0.5)
    assert self.filt.x == -0.5
    assert abs(self.filt.update(-0.5) + 0.5) < 1e-12

  def test_nan_input_keeps_state(self):
    self.filt.reset(0.5)
    assert self.filt.update(float("nan")) == 0.5
    assert math.isfinite(self.filt.update(0.5))
