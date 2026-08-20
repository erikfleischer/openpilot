import math


class FirstOrderFilter:
  def __init__(self, x0, rc, dt, initialized=True):
    self.x = x0
    self.dt = dt
    self.update_alpha(rc)
    self.initialized = initialized

  def update_alpha(self, rc):
    self.alpha = self.dt / (rc + self.dt)

  def update(self, x):
    if self.initialized:
      self.x = (1. - self.alpha) * self.x + self.alpha * x
    else:
      self.initialized = True
      self.x = x
    return self.x


class BounceFilter(FirstOrderFilter):
  def __init__(self, x0, rc, dt, initialized=True, bounce=2):
    self.velocity = FirstOrderFilter(0.0, 0.15, dt)
    self.bounce = bounce
    super().__init__(x0, rc, dt, initialized)

  def update(self, x):
    super().update(x)
    scale = self.dt / (1.0 / 60.0)  # tuned at 60 fps
    self.velocity.x += (x - self.x) * self.bounce * scale * self.dt
    self.velocity.update(0.0)
    if abs(self.velocity.x) < 1e-3:
      self.velocity.x = 0.0
    self.x += self.velocity.x
    return self.x


class SecondOrderBesselFilter:
  """Mag-normalized 2nd-order Bessel LPF, bilinear with frequency prewarping."""

  # Delay-normalized Bessel N=2 analog prototype |H| = -3 dB at this rad/s
  _WN_3DB = math.sqrt(1.5 * (math.sqrt(5.0) - 1.0))

  def __init__(self, x0, fc_hz, dt):
    self.dt = dt
    self._update_coeffs(fc_hz)
    self.reset(x0)

  def _update_coeffs(self, fc_hz):
    fc_hz = float(fc_hz)
    nyquist = 0.5 / self.dt
    if not (0.0 < fc_hz < nyquist):
      raise ValueError(f"fc_hz must be in (0, {nyquist}), got {fc_hz}")
    omega = (2.0 / self.dt) * math.tan(math.pi * fc_hz * self.dt)
    wn = omega / self._WN_3DB
    c0 = 3.0 * wn * wn
    c1 = 3.0 * wn
    K = 2.0 / self.dt
    a0 = K * K + c1 * K + c0
    self.b0 = c0 / a0
    self.b1 = 2.0 * c0 / a0
    self.b2 = c0 / a0
    self.a1 = (-2.0 * K * K + 2.0 * c0) / a0
    self.a2 = (K * K - c1 * K + c0) / a0

  def reset(self, x):
    x = float(x)
    w_ss = x / (1.0 + self.a1 + self.a2)
    self._w1 = w_ss
    self._w2 = w_ss
    self.x = x
    return self.x

  def update(self, x):
    x = float(x)
    w0 = x - self.a1 * self._w1 - self.a2 * self._w2
    y = self.b0 * w0 + self.b1 * self._w1 + self.b2 * self._w2
    self._w2 = self._w1
    self._w1 = w0
    self.x = y
    return self.x
