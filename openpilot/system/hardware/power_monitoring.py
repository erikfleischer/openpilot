import time
import threading

from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog

CAR_VOLTAGE_LOW_PASS_K = 0.011 # LPF gain for 45s tau (dt/tau / (dt/tau + 1))

# While driving, a battery charges completely in about 30-60 minutes
CAR_BATTERY_CAPACITY_uWh = 30e6
CAR_CHARGING_RATE_W = 45

VBATT_PAUSE_CHARGING = 11.8           # Lower limit on the LPF car battery voltage
VBATT_PAUSE_CHARGING_mV = VBATT_PAUSE_CHARGING * 1e3
MAX_TIME_OFFROAD_S = 30*3600
MIN_ON_TIME_S = 3600
DELAY_SHUTDOWN_TIME_S = 300 # Wait at least DELAY_SHUTDOWN_TIME_S seconds after offroad_time to shutdown.
VOLTAGE_SHUTDOWN_DELAY_S = 30  # Wait this long after LPF voltage drops below VBATT_PAUSE_CHARGING


class PowerMonitoring:
  def __init__(self):
    self.params = Params()
    self.last_measurement_time = None           # Used for integration delta
    self.last_save_time = 0                     # Used for saving current value in a param
    self.power_used_uWh = 0                     # Integrated power usage in uWh since going into offroad
    self.next_pulsed_measurement_time = None
    self.car_voltage_mV = 12e3                  # Low-passed version of peripheralState voltage
    self.car_voltage_instant_mV = 12e3          # Last value of peripheralState voltage
    self.integration_lock = threading.Lock()

    self._low_voltage_since: float | None = None
    self._offroad_min_voltage_mV: float | None = None
    self._last_min_voltage_save_time = 0

    # Continue the last offroad session across reboot so a recovered 12V
    # reading doesn't hide the diagnostic from the previous park.
    persisted_min = self.params.get("CarBatteryOffroadMinVoltageMv")
    if persisted_min is not None and persisted_min < VBATT_PAUSE_CHARGING_mV:
      self._offroad_min_voltage_mV = float(persisted_min)

    car_battery_capacity_uWh = self.params.get("CarBatteryCapacity") or 0

    # Reset capacity if it's low
    self.car_battery_capacity_uWh = max((CAR_BATTERY_CAPACITY_uWh / 10), car_battery_capacity_uWh)

  # Calculation tick
  def calculate(self, voltage: float | None, ignition: bool, in_car: bool = False):
    try:
      now = time.monotonic()

      # If peripheralState is None, we're probably not in a car, so we don't care
      if voltage is None:
        self._low_voltage_since = None
        with self.integration_lock:
          self.last_measurement_time = None
          self.next_pulsed_measurement_time = None
          self.power_used_uWh = 0
        return

      # Low-pass battery voltage
      self.car_voltage_instant_mV = voltage
      self.car_voltage_mV = ((voltage * CAR_VOLTAGE_LOW_PASS_K) + (self.car_voltage_mV * (1 - CAR_VOLTAGE_LOW_PASS_K)))

      # Cap the car battery power and save it in a param every 10-ish seconds
      self.car_battery_capacity_uWh = max(self.car_battery_capacity_uWh, 0)
      self.car_battery_capacity_uWh = min(self.car_battery_capacity_uWh, CAR_BATTERY_CAPACITY_uWh)
      if now - self.last_save_time >= 10:
        self.params.put("CarBatteryCapacity", int(self.car_battery_capacity_uWh))
        self.last_save_time = now

      self._update_voltage_diagnostic(now, ignition, in_car)

      # First measurement, set integration time
      with self.integration_lock:
        if self.last_measurement_time is None:
          self.last_measurement_time = now
          return

      if ignition:
        # If there is ignition, we integrate the charging rate of the car
        with self.integration_lock:
          self.power_used_uWh = 0
          integration_time_h = (now - self.last_measurement_time) / 3600
          if integration_time_h < 0:
            raise ValueError(f"Negative integration time: {integration_time_h}h")
          self.car_battery_capacity_uWh += (CAR_CHARGING_RATE_W * 1e6 * integration_time_h)
          self.last_measurement_time = now
      else:
        # Get current power draw somehow
        current_power = HARDWARE.get_current_power_draw()

        # Do the integration
        self._perform_integration(now, current_power)
    except Exception:
      cloudlog.exception("Power monitoring calculation failed")

  def _update_voltage_diagnostic(self, now: float, ignition: bool, in_car: bool) -> None:
    # USB-C / firehose: panda still reports ~5 V. Ignore it for the 12V diagnostic.
    if not in_car:
      self._low_voltage_since = None
      return

    if ignition:
      if self._offroad_min_voltage_mV is not None:
        if self._offroad_min_voltage_mV >= VBATT_PAUSE_CHARGING_mV:
          self.params.remove("CarBatteryOffroadMinVoltageMv")
        self._offroad_min_voltage_mV = None
      self._low_voltage_since = None
      return

    if self._offroad_min_voltage_mV is None:
      self._offroad_min_voltage_mV = self.car_voltage_instant_mV
    else:
      self._offroad_min_voltage_mV = min(self._offroad_min_voltage_mV, self.car_voltage_instant_mV)

    if self.car_voltage_mV < VBATT_PAUSE_CHARGING_mV:
      if self._low_voltage_since is None:
        self._low_voltage_since = now
        cloudlog.event("Car battery voltage below shutdown threshold",
                       voltage_instant_mV=self.car_voltage_instant_mV,
                       voltage_lpf_mV=self.car_voltage_mV,
                       min_voltage_mV=self._offroad_min_voltage_mV)
    else:
      self._low_voltage_since = None

    if self._offroad_min_voltage_mV < VBATT_PAUSE_CHARGING_mV:
      self.save_offroad_min_voltage(now=now)

  def save_offroad_min_voltage(self, force: bool = False, now: float | None = None) -> None:
    # check if the min voltage is below the pause charging voltage because this function called from hardwared.py
    min_v = self._offroad_min_voltage_mV
    if min_v is None or min_v >= VBATT_PAUSE_CHARGING_mV:
      return

    min_v_int = int(min_v)
    if now is None:
      now = time.monotonic()
    persisted = self.params.get("CarBatteryOffroadMinVoltageMv")
    is_new_min = persisted is None or min_v_int < persisted
    if not force and not is_new_min and now - self._last_min_voltage_save_time < 10:
      return

    self.params.put("CarBatteryOffroadMinVoltageMv", min_v_int, block=True)
    self._last_min_voltage_save_time = now

  def get_offroad_min_voltage_mV(self) -> int | None:
    live = int(self._offroad_min_voltage_mV) if self._offroad_min_voltage_mV is not None else None
    persisted = self.params.get("CarBatteryOffroadMinVoltageMv")
    values = [v for v in (live, persisted) if v is not None]
    return min(values) if values else None

  def low_voltage_diagnostic_text(self) -> str | None:
    min_v = self.get_offroad_min_voltage_mV()
    if min_v is not None and min_v < VBATT_PAUSE_CHARGING_mV:
      return f"{min_v / 1000:.1f}"
    return None

  def _perform_integration(self, t: float, current_power: float) -> None:
    with self.integration_lock:
      try:
        if self.last_measurement_time:
          integration_time_h = (t - self.last_measurement_time) / 3600
          power_used = (current_power * 1000000) * integration_time_h
          if power_used < 0:
            raise ValueError(f"Negative power used! Integration time: {integration_time_h} h Current Power: {power_used} uWh")
          self.power_used_uWh += power_used
          self.car_battery_capacity_uWh -= power_used
          self.last_measurement_time = t
      except Exception:
        cloudlog.exception("Integration failed")

  # Get the power usage
  def get_power_used(self) -> int:
    return int(self.power_used_uWh)

  def get_car_battery_capacity(self) -> int:
    return int(self.car_battery_capacity_uWh)

  # See if we need to shutdown
  def should_shutdown(self, ignition: bool, in_car: bool, offroad_timestamp: float | None, started_seen: bool):
    if offroad_timestamp is None:
      return False

    now = time.monotonic()
    offroad_time = (now - offroad_timestamp)

    low_voltage_shutdown = (self._low_voltage_since is not None and
                            (now - self._low_voltage_since) >= VOLTAGE_SHUTDOWN_DELAY_S)

    should_shutdown = False
    should_shutdown |= offroad_time > MAX_TIME_OFFROAD_S
    should_shutdown |= (self.car_battery_capacity_uWh <= 0)
    should_shutdown &= not ignition
    should_shutdown &= (not self.params.get_bool("DisablePowerDown"))
    should_shutdown &= in_car
    should_shutdown &= offroad_time > DELAY_SHUTDOWN_TIME_S
    # LV shutdown waits 30 s after voltage drops, not the 5 min offroad floor
    should_shutdown |= (low_voltage_shutdown and not ignition and
                        (not self.params.get_bool("DisablePowerDown")) and in_car)
    should_shutdown |= self.params.get_bool("ForcePowerDown")
    should_shutdown &= started_seen or (now > MIN_ON_TIME_S)
    return should_shutdown
