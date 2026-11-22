import raylib_nb as rl

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

BUTTON_W = 120
BUTTON_H = 46
BUTTON_FONT = 26


class DecelTestButton(Widget):
  """Starts the low-deceleration follow test. The phase is logged, not drawn."""

  def __init__(self):
    super().__init__()
    self._pm = messaging.PubMaster(['decelTestButton'])
    self._request_id = 0
    self._ate_click = False
    self._font = gui_app.font(FontWeight.BOLD)

  def ate_click(self) -> bool:
    ate, self._ate_click = self._ate_click, False
    return ate

  def _engaged(self) -> bool:
    return bool(ui_state.sm['selfdriveState'].enabled and ui_state.has_longitudinal_control)

  def _active(self) -> bool:
    sm = ui_state.sm
    return bool(sm.valid['longitudinalDecelTest'] and sm['longitudinalDecelTest'].active)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    self._ate_click = True
    if not self._active() and not self._engaged():
      return
    self._request_id += 1
    msg = messaging.new_message('decelTestButton', valid=True)
    msg.decelTestButton.requestId = self._request_id
    self._pm.send('decelTestButton', msg)

  def _render(self, rect: rl.Rectangle) -> None:
    active = self._active()
    engaged = self._engaged()
    if active:
      bg = rl.Color(255, 255, 255, 230)
      fg = rl.Color(0, 0, 0, 255)
    elif engaged:
      bg = rl.Color(0, 0, 0, 170)
      fg = rl.WHITE
    else:
      bg = rl.Color(0, 0, 0, 90)
      fg = rl.Color(255, 255, 255, 120)

    rl.draw_rectangle_rounded(rect, 0.6, 8, bg)
    text = "DECEL"
    size = measure_text_cached(self._font, text, BUTTON_FONT)
    pos = rl.Vector2(rect.x + (rect.width - size.x) / 2, rect.y + (rect.height - size.y) / 2)
    rl.draw_text_ex(self._font, text, pos, BUTTON_FONT, 0, fg)
