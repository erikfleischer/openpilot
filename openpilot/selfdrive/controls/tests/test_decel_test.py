import unittest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.decel_test import (
  HOLD_FRAMES,
  RAMP_FRAMES,
  DecelTest,
  DecelTestStatus,
  profile_frame,
  zero_frames,
)


class TestDecelProfile(unittest.TestCase):
  def test_from_positive(self):
    a0 = 0.2
    self.assertEqual(zero_frames(a0), 200)
    self.assertEqual(profile_frame(a0, 0), (0.2, DecelTestStatus.rampToZero))
    self.assertEqual(profile_frame(a0, 199)[1], DecelTestStatus.rampToZero)
    self.assertAlmostEqual(profile_frame(a0, 199)[0], 0.001)

    # 2 s to 0, then the 3 s hold.
    self.assertEqual(profile_frame(a0, 200), (0.0, DecelTestStatus.hold))
    self.assertEqual(profile_frame(a0, 200 + HOLD_FRAMES - 1), (0.0, DecelTestStatus.hold))

    # Ramp to -0.5 over the next 5 s. The sample at 10 s is the valley.
    down_start = 200 + HOLD_FRAMES
    self.assertEqual(profile_frame(a0, down_start), (0.0, DecelTestStatus.rampDown))
    valley = down_start + RAMP_FRAMES
    self.assertEqual(valley, 1000)
    self.assertEqual(profile_frame(a0, valley), (-0.5, DecelTestStatus.rampUp))

    # Back to 0 over 5 s. t = 15 s is the finished sample and is not applied.
    end = valley + RAMP_FRAMES
    self.assertEqual(end, 1500)
    self.assertEqual(end * DT_CTRL, 15.0)
    self.assertEqual(profile_frame(a0, end - 1)[1], DecelTestStatus.rampUp)
    self.assertEqual(profile_frame(a0, end), (0.0, DecelTestStatus.finished))

  def test_from_negative_ramps_up_to_zero(self):
    a0 = -0.4
    self.assertEqual(zero_frames(a0), 400)
    self.assertEqual(profile_frame(a0, 0), (-0.4, DecelTestStatus.rampToZero))
    self.assertAlmostEqual(profile_frame(a0, 200)[0], -0.2)
    self.assertEqual(profile_frame(a0, 400), (0.0, DecelTestStatus.hold))
    # The hold and both ramps match the positive-start schedule after reaching 0.
    self.assertEqual(profile_frame(a0, 400 + HOLD_FRAMES + RAMP_FRAMES), (-0.5, DecelTestStatus.rampUp))

  def test_already_zero_starts_at_hold(self):
    self.assertEqual(zero_frames(0.0), 0)
    self.assertEqual(profile_frame(0.0, 0), (0.0, DecelTestStatus.hold))
    self.assertEqual(profile_frame(0.0, HOLD_FRAMES), (0.0, DecelTestStatus.rampDown))

  def test_run_cancel_and_disengage(self):
    test = DecelTest()
    test.start(0.2)
    self.assertEqual(test.poll(True), 0.2)
    self.assertTrue(test.active)
    self.assertEqual(test.status, DecelTestStatus.rampToZero)
    self.assertTrue(test.publish)

    # Losing longitudinal control cancels and leaves the planner in charge.
    self.assertIsNone(test.poll(False))
    self.assertFalse(test.active)
    self.assertEqual(test.status, DecelTestStatus.cancelled)
    self.assertTrue(test.publish)
    test.publish = False
    self.assertIsNone(test.poll(True))
    self.assertFalse(test.publish)

    test.start(0.0)
    self.assertEqual(test.poll(True), 0.0)
    test.cancel()
    self.assertFalse(test.active)
    self.assertEqual(test.status, DecelTestStatus.cancelled)
    self.assertTrue(test.publish)

  def test_schedule_reaches_finished(self):
    test = DecelTest()
    test.start(0.2)
    last = None
    for _ in range(1500):
      last = test.poll(True)
      self.assertIsNotNone(last)
    self.assertIsNone(test.poll(True))
    self.assertFalse(test.active)
    self.assertEqual(test.status, DecelTestStatus.finished)
    self.assertEqual(test.accel, 0.0)
    self.assertTrue(test.publish)
