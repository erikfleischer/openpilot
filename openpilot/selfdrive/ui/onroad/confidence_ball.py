import math
import pyray as rl
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.onroad.hud_renderer import UI_CONFIG
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.filter_simple import FirstOrderFilter

# Scaled from mici radius 24 @ 240p (~3.2x), then 30% smaller
STATUS_DOT_RADIUS = round(77 * 0.7)
TRAVEL_HEIGHT_FRAC = 0.62
# Match onroad border navy (#122839) with exp-button translucency
PANEL_COLOR = rl.Color(0x12, 0x28, 0x39, 0xFF)


def draw_circle_gradient(center_x: float, center_y: float, radius: int,
                         top: rl.Color, bottom: rl.Color) -> None:
  cx, cy = int(center_x), int(center_y)
  for y in range(-radius, radius + 1):
    half_w = int(math.sqrt(max(radius * radius - y * y, 0)))
    if half_w <= 0:
      continue
    t = (y + radius) / (2 * radius)
    color = rl.Color(
      int(top.r + (bottom.r - top.r) * t),
      int(top.g + (bottom.g - top.g) * t),
      int(top.b + (bottom.b - top.b) * t),
      int(top.a + (bottom.a - top.a) * t),
    )
    rl.draw_rectangle(cx - half_w, cy + y, half_w * 2, 1, color)


def draw_capsule(center_x: float, top: float, bottom: float, radius: int, color: rl.Color) -> None:
  """Vertical capsule matching a circle of `radius` traveling from top to bottom."""
  rl.draw_circle(int(center_x), int(top + radius), radius, color)
  rl.draw_circle(int(center_x), int(bottom - radius), radius, color)
  rl.draw_rectangle(int(center_x - radius), int(top + radius), radius * 2, int(bottom - top - 2 * radius), color)


class ConfidenceBall(Widget):
  def __init__(self, demo: bool = False):
    super().__init__()
    self._demo = demo
    self._confidence_filter = FirstOrderFilter(0.0, 0.5, 1 / gui_app.target_fps)

  def update_filter(self, value: float):
    self._confidence_filter.update(value)

  def _update_state(self):
    if self._demo:
      return

    # sit at bottom of travel range when disengaged
    if ui_state.status == UIStatus.DISENGAGED:
      self._confidence_filter.update(0.0)
    else:
      self._confidence_filter.update((1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs or [1])) *
                                                        (1 - max(ui_state.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs or [1])))

  def _render(self, _):
    # Match experimental mode icon column (HUD renders into the bordered content rect)
    content_x = self.rect.x + UI_BORDER_SIZE
    content_width = self.rect.width - 2 * UI_BORDER_SIZE
    content_bottom = self.rect.y + self.rect.height - UI_BORDER_SIZE
    exp_button_x = content_x + content_width - UI_CONFIG.border_size - UI_CONFIG.button_size
    panel_x = exp_button_x + (UI_CONFIG.button_size - SIDE_PANEL_WIDTH) / 2
    # Bottom margin matches driver monitoring indicator (UI_BORDER_SIZE above content bottom)
    panel_height = self.rect.height * TRAVEL_HEIGHT_FRAC
    panel_bottom = content_bottom - UI_BORDER_SIZE
    content_rect = rl.Rectangle(
      panel_x,
      panel_bottom - panel_height,
      SIDE_PANEL_WIDTH,
      panel_height,
    )

    status_dot_radius = STATUS_DOT_RADIUS
    center_x = content_rect.x + content_rect.width / 2

    # Capsule travel area sized to the confidence ball
    draw_capsule(center_x, content_rect.y, content_rect.y + content_rect.height,
                 status_dot_radius, PANEL_COLOR)

    dot_height = (1 - self._confidence_filter.x) * (content_rect.height - 2 * status_dot_radius) + status_dot_radius
    dot_height = content_rect.y + dot_height

    # confidence zones
    if ui_state.status == UIStatus.ENGAGED or self._demo:
      if self._confidence_filter.x > 0.5:
        top_dot_color = rl.Color(0, 255, 204, 255)
        bottom_dot_color = rl.Color(0, 255, 38, 255)
      elif self._confidence_filter.x > 0.2:
        top_dot_color = rl.Color(255, 200, 0, 255)
        bottom_dot_color = rl.Color(255, 115, 0, 255)
      else:
        top_dot_color = rl.Color(255, 0, 21, 255)
        bottom_dot_color = rl.Color(255, 0, 89, 255)

    elif ui_state.status == UIStatus.OVERRIDE:
      top_dot_color = rl.Color(255, 255, 255, 255)
      bottom_dot_color = rl.Color(82, 82, 82, 255)

    else:
      top_dot_color = rl.Color(50, 50, 50, 255)
      bottom_dot_color = rl.Color(13, 13, 13, 255)

    draw_circle_gradient(center_x, dot_height, status_dot_radius,
                         top_dot_color, bottom_dot_color)
