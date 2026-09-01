
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
from openpilot.system.hardware.power_monitoring import PowerMonitoring, CAR_BATTERY_CAPACITY_uWh, \
                                                CAR_CHARGING_RATE_W, VBATT_PAUSE_CHARGING, DELAY_SHUTDOWN_TIME_S, \
                                                VOLTAGE_SHUTDOWN_DELAY_S

# Create fake time
ssb = 0.
def mock_time_monotonic():
  global ssb
  ssb += 1.
  return ssb

def set_mock_time(value):
  global ssb
  ssb = value

TEST_DURATION_S = 50
GOOD_VOLTAGE = 12 * 1e3
VOLTAGE_BELOW_PAUSE_CHARGING = (VBATT_PAUSE_CHARGING - 1) * 1e3
USB_C_VOLTAGE = 5 * 1e3

def pm_patch(mocker, name, value, constant=False):
  if constant:
    mocker.patch(f"openpilot.system.hardware.power_monitoring.{name}", value)
  else:
    mocker.patch(f"openpilot.system.hardware.power_monitoring.{name}", return_value=value)


class TestPowerMonitoring(OpenpilotTestCase):
  def setup_method(self):
    global ssb
    ssb = 0.
    self._fixture("mocker").patch("time.monotonic", mock_time_monotonic)
    self.params = Params()

  # Test to see that it doesn't do anything when pandaState is None
  def test_panda_state_present(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(None, False)
    assert pm.get_power_used() == 0
    assert pm.get_car_battery_capacity() == (CAR_BATTERY_CAPACITY_uWh / 10)

  # Test to see that it doesn't integrate offroad when ignition is True
  def test_offroad_ignition(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(GOOD_VOLTAGE, True)
    assert pm.get_power_used() == 0

  # Test to see that it integrates with discharging battery
  def test_offroad_integration_discharging(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    expected_power_usage = ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_power_used() - expected_power_usage) < 10

  # Test to check positive integration of car_battery_capacity
  def test_car_battery_integration_onroad(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, True)
    expected_capacity = ((TEST_DURATION_S/3600) * CAR_CHARGING_RATE_W * 1e6)
    assert abs(pm.get_car_battery_capacity() - expected_capacity) < 10

  # Test to check positive integration upper limit
  def test_car_battery_integration_upper_limit(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh - 1000
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, True)
    estimated_capacity = CAR_BATTERY_CAPACITY_uWh + (CAR_CHARGING_RATE_W / 3600 * 1e6)
    assert abs(pm.get_car_battery_capacity() - estimated_capacity) < 10

  # Test to check negative integration of car_battery_capacity
  def test_car_battery_integration_offroad(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    expected_capacity = CAR_BATTERY_CAPACITY_uWh - ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_car_battery_capacity() - expected_capacity) < 10

  # Test to check negative integration lower limit
  def test_car_battery_integration_lower_limit(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 1000
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    estimated_capacity = 0 - ((1/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_car_battery_capacity() - estimated_capacity) < 10

  # Test to check policy of stopping charging after MAX_TIME_OFFROAD_S
  def test_max_time_offroad(self, mocker):
    MOCKED_MAX_OFFROAD_TIME = 3600
    POWER_DRAW = 0 # To stop shutting down for other reasons
    pm_patch(mocker, "MAX_TIME_OFFROAD_S", MOCKED_MAX_OFFROAD_TIME, constant=True)
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    start_time = ssb
    ignition = False
    set_mock_time(start_time + MOCKED_MAX_OFFROAD_TIME - 1)
    assert not pm.should_shutdown(ignition, True, start_time, False)
    set_mock_time(start_time + MOCKED_MAX_OFFROAD_TIME)
    assert pm.should_shutdown(ignition, True, start_time, False)

  def test_car_voltage(self, mocker):
    POWER_DRAW = 0  # To stop shutting down for other reasons
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    pm.car_voltage_mV = VOLTAGE_BELOW_PAUSE_CHARGING
    ignition = False
    in_car = True
    off_ts = ssb

    pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition, in_car)
    armed_at = pm._low_voltage_since
    assert armed_at is not None
    assert not pm.should_shutdown(ignition, in_car, off_ts, True)

    set_mock_time(armed_at + VOLTAGE_SHUTDOWN_DELAY_S - 2)
    assert not pm.should_shutdown(ignition, in_car, off_ts, True)

    # LV shutdown is not gated by the 5 min DELAY_SHUTDOWN_TIME_S floor
    assert (ssb - off_ts) < DELAY_SHUTDOWN_TIME_S
    set_mock_time(armed_at + VOLTAGE_SHUTDOWN_DELAY_S - 1)
    assert pm.should_shutdown(ignition, in_car, off_ts, True)

  def test_car_voltage_recovers_before_delay(self, mocker):
    POWER_DRAW = 0
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    pm.car_voltage_mV = VOLTAGE_BELOW_PAUSE_CHARGING
    off_ts = ssb

    pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, False, True)
    assert pm._low_voltage_since is not None

    pm.car_voltage_mV = GOOD_VOLTAGE
    pm.calculate(GOOD_VOLTAGE, False, True)
    assert pm._low_voltage_since is None

    set_mock_time(off_ts + VOLTAGE_SHUTDOWN_DELAY_S + 10)
    assert not pm.should_shutdown(False, True, off_ts, True)

  # Test to check policy of not stopping charging when DisablePowerDown is set
  def test_disable_power_down(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    self.params.put_bool("DisablePowerDown", True, block=True)
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    ignition = False
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
      if i % 10 == 0:
        assert not pm.should_shutdown(ignition, True, ssb, False)
    assert not pm.should_shutdown(ignition, True, ssb, False)

  # Test to check policy of not stopping charging when ignition
  def test_ignition(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    ignition = True
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
      if i % 10 == 0:
        assert not pm.should_shutdown(ignition, True, ssb, False)
    assert not pm.should_shutdown(ignition, True, ssb, False)

  # Test to check policy of not stopping charging when harness is not connected
  def test_harness_connection(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    ignition = False
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition, False)
      if i % 10 == 0:
        assert not pm.should_shutdown(ignition, False, ssb, False)
        assert pm._low_voltage_since is None
        assert self.params.get("CarBatteryOffroadMinVoltageMv") is None
    assert not pm.should_shutdown(ignition, False, ssb, False)

  def test_delay_shutdown_time(self):
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    ignition = False
    in_car = True
    offroad_timestamp = ssb
    started_seen = True
    pm.calculate(GOOD_VOLTAGE, ignition)

    set_mock_time(offroad_timestamp + DELAY_SHUTDOWN_TIME_S - 1)
    assert not pm.should_shutdown(ignition, in_car, offroad_timestamp, started_seen), \
                     f"Should not shutdown before {DELAY_SHUTDOWN_TIME_S} seconds offroad time"
    set_mock_time(offroad_timestamp + DELAY_SHUTDOWN_TIME_S)
    assert pm.should_shutdown(ignition, in_car,
                                       offroad_timestamp,
                                       started_seen), \
                    f"Should shutdown after {DELAY_SHUTDOWN_TIME_S} seconds offroad time"

  def test_min_voltage_tracking(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    pm.calculate(12.5e3, False, True)
    pm.calculate(12.2e3, False, True)
    pm.calculate(11.5e3, False, True)
    pm.calculate(11.9e3, False, True)

    assert pm.get_offroad_min_voltage_mV() == 11500
    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500
    assert pm.low_voltage_diagnostic_text() == "11.5"

  def test_min_voltage_not_persisted_when_healthy(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    pm.calculate(GOOD_VOLTAGE, False, True)
    pm.calculate(12.1e3, False, True)

    assert pm.get_offroad_min_voltage_mV() == 12000
    assert self.params.get("CarBatteryOffroadMinVoltageMv") is None
    assert pm.low_voltage_diagnostic_text() is None

  def test_ignition_clears_healthy_min(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    pm.calculate(GOOD_VOLTAGE, False, True)
    assert pm.get_offroad_min_voltage_mV() == int(GOOD_VOLTAGE)

    pm.calculate(GOOD_VOLTAGE, True, True)
    assert pm.get_offroad_min_voltage_mV() is None
    assert self.params.get("CarBatteryOffroadMinVoltageMv") is None

  def test_ignition_keeps_low_min(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    pm.calculate(11.5e3, False, True)
    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500

    pm.calculate(GOOD_VOLTAGE, True, True)
    assert pm._offroad_min_voltage_mV is None
    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500
    assert pm.get_offroad_min_voltage_mV() == 11500
    assert pm.low_voltage_diagnostic_text() == "11.5"

  def test_firehose_usb_c_does_not_trigger_diagnostic(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    pm.car_voltage_mV = USB_C_VOLTAGE
    off_ts = ssb

    for _ in range(60):
      pm.calculate(USB_C_VOLTAGE, False, False)

    assert pm._low_voltage_since is None
    assert pm.get_offroad_min_voltage_mV() is None
    assert self.params.get("CarBatteryOffroadMinVoltageMv") is None
    assert pm.low_voltage_diagnostic_text() is None
    assert not pm.should_shutdown(False, False, off_ts, True)

  def test_firehose_does_not_overwrite_in_car_diagnostic(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    pm.calculate(11.5e3, False, True)
    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500

    pm.car_voltage_mV = USB_C_VOLTAGE
    pm.calculate(USB_C_VOLTAGE, False, False)

    assert pm._low_voltage_since is None
    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500
    assert pm.get_offroad_min_voltage_mV() == 11500
    assert pm.low_voltage_diagnostic_text() == "11.5"

  def test_boot_keeps_persisted_low_when_voltage_recovered(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    self.params.put("CarBatteryOffroadMinVoltageMv", 11500, block=True)

    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    pm.calculate(GOOD_VOLTAGE, False, True)

    assert pm.get_offroad_min_voltage_mV() == 11500
    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500
    assert pm.low_voltage_diagnostic_text() == "11.5"

  def test_boot_ignition_does_not_clear_persisted_low(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    self.params.put("CarBatteryOffroadMinVoltageMv", 11500, block=True)

    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    pm.calculate(GOOD_VOLTAGE, False, True)
    pm.calculate(GOOD_VOLTAGE, True, True)

    assert self.params.get("CarBatteryOffroadMinVoltageMv") == 11500
    assert pm.low_voltage_diagnostic_text() == "11.5"

  def test_healthy_offroad_after_drive_clears_persisted_low(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    self.params.put("CarBatteryOffroadMinVoltageMv", 11500, block=True)

    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    pm.calculate(GOOD_VOLTAGE, False, True)
    pm.calculate(GOOD_VOLTAGE, True, True)
    assert pm.low_voltage_diagnostic_text() == "11.5"

    # Next offroad period is after a drive; still show the last-drive min
    pm.calculate(GOOD_VOLTAGE, False, True)
    assert pm.low_voltage_diagnostic_text() == "11.5"

    # Healthy offroad after that drive can clear the diagnostic
    pm.calculate(GOOD_VOLTAGE, True, True)
    assert self.params.get("CarBatteryOffroadMinVoltageMv") is None
    assert pm.low_voltage_diagnostic_text() is None
