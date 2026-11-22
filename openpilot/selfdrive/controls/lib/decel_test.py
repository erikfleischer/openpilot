from enum import IntEnum

from openpilot.common.realtime import DT_CTRL

# Same slope as the 0.5 m/s^2 ramps below, so the lead-in duration is |a0| / JERK.
JERK = 0.1  # m/s^3
HOLD_S = 3.0
RAMP_S = 5.0
DECEL = -0.5  # m/s^2

HOLD_FRAMES = round(HOLD_S / DT_CTRL)
RAMP_FRAMES = round(RAMP_S / DT_CTRL)


class DecelTestStatus(IntEnum):
  rampToZero = 0
  hold = 1
  rampDown = 2
  rampUp = 3
  finished = 4
  cancelled = 5


def zero_frames(a0: float) -> int:
  return round(abs(a0) / JERK / DT_CTRL)


def profile_frame(a0: float, frame: int) -> tuple[float, DecelTestStatus]:
  """Command and segment at a 100 Hz frame index. Frame 0 is the start sample.

  The frame at the end of the schedule is `finished` with accel 0 and is not
  applied: the planner runs again on that frame.
  """
  zero = zero_frames(a0)
  hold_end = zero + HOLD_FRAMES
  down_end = hold_end + RAMP_FRAMES
  up_end = down_end + RAMP_FRAMES

  if frame >= up_end:
    return 0.0, DecelTestStatus.finished
  if frame < zero:
    return a0 * (1.0 - frame / zero), DecelTestStatus.rampToZero
  if frame < hold_end:
    return 0.0, DecelTestStatus.hold
  if frame < down_end:
    return DECEL * ((frame - hold_end) / RAMP_FRAMES), DecelTestStatus.rampDown
  return DECEL * (1.0 - (frame - down_end) / RAMP_FRAMES), DecelTestStatus.rampUp


class DecelTest:
  def __init__(self) -> None:
    self.active = False
    self.accel = 0.0
    self.status = DecelTestStatus.finished
    self.publish = False
    self._a0 = 0.0
    self._frame = 0

  def start(self, a0: float) -> None:
    self._a0 = float(a0)
    self._frame = 0
    self.active = True
    self.accel, self.status = profile_frame(self._a0, 0)
    self.publish = False

  def cancel(self) -> None:
    if not self.active:
      return
    self.active = False
    self.status = DecelTestStatus.cancelled
    self.publish = True

  def poll(self, long_active: bool) -> float | None:
    """Return the accel to apply, or None when the planner should run.

    Sets `publish` when this frame should be written to `longitudinalDecelTest`.
    Losing longActive cancels the test.
    """
    if self.active and not long_active:
      self.cancel()
      return None
    if not self.active:
      return None

    accel, status = profile_frame(self._a0, self._frame)
    self._frame += 1
    self.accel = accel
    self.status = status
    self.publish = True
    if status == DecelTestStatus.finished:
      self.active = False
      return None
    return accel
