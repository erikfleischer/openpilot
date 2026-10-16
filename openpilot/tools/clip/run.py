#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
import threading
import queue
import multiprocessing
import itertools
import numpy as np
import tqdm
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openpilot.tools.lib.route import Route
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.framereader import FrameReader, ffprobe
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.utils import Timer
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcServer

FRAMERATE = 20
DEMO_ROUTE, DEMO_START, DEMO_END = '5beb9b58bd12b691/0000010a--a51155e496', 90, 105

# Stock Autopilot DAS_control on the autopilot party bus. Openpilot sends its own copy on bus 0.
DAS_CONTROL_ADDR = 697
AP_PARTY_BUS = 2
DAS_ACCEL_SNA = 511
DAS_SET_SPEED_SNA = 4095
KPH_TO_MS = 1.0 / 3.6
ACCEL_BAR_MIN = -4.0
ACCEL_BAR_MAX = 3.0

logger = logging.getLogger('clip')


def parse_args():
  parser = ArgumentParser(description="Direct clip renderer")
  parser.add_argument("route", nargs="?", help="Route ID (dongle/route or dongle/route/start/end)")
  parser.add_argument("-s", "--start", type=int, help="Start time in seconds")
  parser.add_argument("-e", "--end", type=int, help="End time in seconds")
  parser.add_argument("-o", "--output", default="output.mp4", help="Output file path")
  parser.add_argument("-d", "--data-dir", help="Local directory with route data")
  parser.add_argument("-t", "--title", help="Title overlay text")
  parser.add_argument("-f", "--file-size", type=float, default=9.0, help="Target file size in MB")
  parser.add_argument("-x", "--speed", type=int, default=1, help="Speed multiplier")
  parser.add_argument("--demo", action="store_true", help="Use demo route with default timing")
  parser.add_argument("--big", action="store_true", help="Use big UI (2160x1080)")
  parser.add_argument("--qcam", action="store_true", help="Use qcamera instead of fcamera")
  parser.add_argument("--windowed", action="store_true", help="Show window")
  parser.add_argument("--no-metadata", action="store_true", help="Disable metadata overlay")
  parser.add_argument("--no-time-overlay", action="store_true", help="Disable time overlay")
  args = parser.parse_args()

  if args.demo:
    args.route, args.start, args.end = args.route or DEMO_ROUTE, args.start or DEMO_START, args.end or DEMO_END
  elif not args.route:
    parser.error("route is required (or use --demo)")

  if args.route and args.route.count('/') == 3:
    parts = args.route.split('/')
    args.route, args.start, args.end = '/'.join(parts[:2]), args.start or int(parts[2]), args.end or int(parts[3])

  if args.start is None or args.end is None:
    parser.error("--start and --end are required")
  if args.end <= args.start:
    parser.error(f"end ({args.end}) must be greater than start ({args.start})")
  return args


def setup_env(output_path: str, big: bool = False, speed: int = 1, target_mb: float = 0, duration: int = 0, headless: bool = True):
  os.environ.update({"RECORD": "1", "RECORD_OUTPUT": str(Path(output_path).with_suffix(".mp4"))})
  if headless:
    os.environ["OFFSCREEN"] = "1"
  if speed > 1:
    os.environ["RECORD_SPEED"] = str(speed)
  if target_mb > 0 and duration > 0:
    os.environ["RECORD_BITRATE"] = f"{int(target_mb * 8 * 1024 / (duration / speed))}k"
  if big:
    os.environ["BIG"] = "1"


def _download_segment(path: str) -> bytes:
  with FileReader(path) as f:
    return bytes(f.read())


def _das_signal(raw: int, start: int, length: int) -> int:
  return (raw >> start) & ((1 << length) - 1)


def decode_das_control(dat) -> tuple[float, float, float] | None:
  """Return (setSpeed kph, accelMin, accelMax). SNA on any field drops the frame."""
  raw_bytes = bytes(dat)
  # accelMax occupies bits 44-52, so the payload must cover byte 6.
  if len(raw_bytes) < 7:
    return None
  raw = int.from_bytes(raw_bytes[:8], 'little')
  set_speed = _das_signal(raw, 0, 12)
  accel_min = _das_signal(raw, 35, 9)
  accel_max = _das_signal(raw, 44, 9)
  if set_speed == DAS_SET_SPEED_SNA or accel_min == DAS_ACCEL_SNA or accel_max == DAS_ACCEL_SNA:
    return None
  return set_speed * 0.1, accel_min * 0.04 - 15.0, accel_max * 0.04 - 15.0


def das_control_from_can(msg) -> tuple[float, float, float] | None:
  latest = None
  if msg.which() != 'can':
    return None
  for frame in msg.can:
    if frame.address == DAS_CONTROL_ADDR and frame.src == AP_PARTY_BUS:
      decoded = decode_das_control(frame.dat)
      if decoded is not None:
        latest = decoded
  return latest


def ap_accel_from_das(das_control: tuple[float, float, float] | None, v_ego: float | None) -> float | None:
  """Below vEgo the min edge is active; above vEgo the max edge is active."""
  if das_control is None or v_ego is None:
    return None
  set_speed_kph, accel_min, accel_max = das_control
  set_speed = set_speed_kph * KPH_TO_MS
  if set_speed < v_ego:
    return accel_min
  if set_speed > v_ego:
    return accel_max
  return 0.0


def _carry_das_control(chunks: list[dict]) -> None:
  last = None
  for chunk in chunks:
    value = chunk.get('_dasControl')
    if value is None:
      chunk['_dasControl'] = last
    else:
      last = value


def _parse_and_chunk_segment(args: tuple) -> list[dict]:
  raw_data, fps = args
  from openpilot.tools.lib.logreader import _LogFileReader
  messages = migrate_all(list(_LogFileReader("", dat=raw_data, sort_by_time=True)))
  if not messages:
    return []

  dt_ns, chunks, current, next_time = 1e9 / fps, [], {}, messages[0].logMonoTime + 1e9 / fps
  das_control = None
  for msg in messages:
    if msg.logMonoTime >= next_time:
      current['_dasControl'] = das_control
      chunks.append(current)
      current, next_time = {}, next_time + dt_ns * ((msg.logMonoTime - next_time) // dt_ns + 1)
      das_control = None
    sample = das_control_from_can(msg)
    if sample is not None:
      das_control = sample
    current[msg.which()] = msg
  if current:
    current['_dasControl'] = das_control
    chunks.append(current)
  return chunks


def load_logs_parallel(log_paths: list[str], fps: int = 20) -> list[dict]:
  num_workers = min(16, len(log_paths), (multiprocessing.cpu_count() or 1))
  logger.info(f"Downloading {len(log_paths)} segments with {num_workers} workers...")

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    futures = {pool.submit(_download_segment, path): idx for idx, path in enumerate(log_paths)}
    raw_data = {futures[f]: f.result() for f in as_completed(futures)}

  logger.info("Parsing and chunking segments...")
  with multiprocessing.Pool(num_workers) as pool:
    chunks = list(itertools.chain.from_iterable(pool.map(_parse_and_chunk_segment, [(raw_data[i], fps) for i in range(len(log_paths))])))
  _carry_das_control(chunks)
  return chunks


def patch_submaster(message_chunks, ui_state):
  # Reset started_frame so alerts render correctly (recv_frame must be >= started_frame)
  ui_state.started_frame = 0
  ui_state.started_time = time.monotonic()

  def mock_update(timeout=None):
    sm, t = ui_state.sm, time.monotonic()
    sm.updated = dict.fromkeys(sm.services, False)
    if sm.frame < len(message_chunks):
      for svc, msg in message_chunks[sm.frame].items():
        if svc in sm.data:
          sm.seen[svc] = sm.updated[svc] = sm.alive[svc] = sm.valid[svc] = True
          sm.data[svc] = getattr(msg.as_builder(), svc)
          sm.logMonoTime[svc], sm.recv_time[svc], sm.recv_frame[svc] = msg.logMonoTime, t, sm.frame
    sm.frame += 1
  ui_state.sm.update = mock_update


def get_frame_dimensions(camera_path: str) -> tuple[int, int]:
  """Get frame dimensions from a video file using ffprobe."""
  probe = ffprobe(camera_path)
  stream = probe["streams"][0]
  return stream["width"], stream["height"]


def iter_segment_frames(camera_paths, start_time, end_time, fps=20, use_qcam=False, frame_size: tuple[int, int] | None = None):
  frames_per_seg = fps * 60
  start_frame, end_frame = int(start_time * fps), int(end_time * fps)
  current_seg: int = -1
  get_frame: Callable[[int], np.ndarray] | None = None

  for global_idx in range(start_frame, end_frame):
    seg_idx, local_idx = global_idx // frames_per_seg, global_idx % frames_per_seg

    if seg_idx != current_seg:
      current_seg = seg_idx
      path = camera_paths[seg_idx] if seg_idx < len(camera_paths) else None
      if not path:
        raise RuntimeError(f"No camera file for segment {seg_idx}")

      if use_qcam:
        w, h = frame_size or get_frame_dimensions(path)
        with FileReader(path) as f:
          result = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "-", "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
                                  input=f.read(), capture_output=True)
        if result.returncode != 0:
          raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        seg_frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, w * h * 3 // 2)
        get_frame = seg_frames.__getitem__
      else:
        get_frame = FrameReader(path, pix_fmt="nv12").get

    assert get_frame is not None
    yield global_idx, get_frame(local_idx)


class FrameQueue:
  def __init__(self, camera_paths, start_time, end_time, fps=20, prefetch_count=60, use_qcam=False):
    # Probe first valid camera file for dimensions
    first_path = next((p for p in camera_paths if p), None)
    if not first_path:
      raise RuntimeError("No valid camera paths")
    self.frame_w, self.frame_h = get_frame_dimensions(first_path)

    self._queue, self._stop, self._error = queue.Queue(maxsize=prefetch_count), threading.Event(), None
    self._thread = threading.Thread(target=self._worker,
                                    args=(camera_paths, start_time, end_time, fps, use_qcam, (self.frame_w, self.frame_h)), daemon=True)
    self._thread.start()

  def _worker(self, camera_paths, start_time, end_time, fps, use_qcam, frame_size):
    try:
      for idx, data in iter_segment_frames(camera_paths, start_time, end_time, fps, use_qcam, frame_size):
        if self._stop.is_set():
          break
        self._queue.put((idx, data.tobytes()))
    except Exception as e:
      logger.exception("Decode error")
      self._error = e
    finally:
      self._queue.put(None)

  def get(self, timeout=60.0):
    if self._error:
      raise self._error
    result = self._queue.get(timeout=timeout)
    if result is None:
      raise StopIteration("No more frames")
    return result

  def stop(self):
    self._stop.set()
    while not self._queue.empty():
      try:
        self._queue.get_nowait()
      except queue.Empty:
        break
    self._thread.join(timeout=2.0)


def load_route_metadata(route):
  from openpilot.common.params import Params, UnknownKeyName
  path = next((item for item in route.log_paths() if item), None)
  if not path:
    raise Exception('error getting route metadata: cannot find any uploaded logs')
  lr = LogReader(path)
  init_data, car_params = lr.first('initData'), lr.first('carParams')

  params = Params()
  for entry in init_data.params.entries:
    try:
      params.put(entry.key, params.cpp2python(entry.key, entry.value), block=True)
    except UnknownKeyName:
      pass

  origin = init_data.gitRemote.split('/')[3] if len(init_data.gitRemote.split('/')) > 3 else 'unknown'
  return {
    'version': init_data.version, 'route': route.name.canonical_name,
    'car': car_params.carFingerprint if car_params else 'unknown', 'origin': origin,
    'branch': init_data.gitBranch, 'commit': init_data.gitCommit[:7], 'modified': str(init_data.dirty).lower(),
  }


def draw_text_box(text, x, y, size, gui_app, font, color=None, center=False):
  import pyray as rl
  from openpilot.system.ui.lib.text_measure import measure_text_cached
  box_color, text_color = rl.Color(0, 0, 0, 85), color or rl.WHITE
  text_size = measure_text_cached(font, text, size)
  text_width, text_height = int(text_size.x), int(text_size.y)
  if center:
    x = (gui_app.width - text_width) // 2
  rl.draw_rectangle(x - 8, y - 4, text_width + 16, text_height + 8, box_color)
  rl.draw_text_ex(font, text, rl.Vector2(x, y), size, 0, text_color)


def _accel_sample(sm, service: str, getter):
  if not sm.seen[service]:
    return None
  return float(getter())


def _draw_accel_bars(gui_app, font, das_control):
  import pyray as rl
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.lib.text_measure import measure_text_cached

  sm = ui_state.sm
  v_ego = _accel_sample(sm, "carState", lambda: sm["carState"].vEgo)
  series = [
    ("model", _accel_sample(sm, "modelV2", lambda: sm["modelV2"].action.desiredAcceleration), rl.Color(80, 160, 255, 230)),
    ("openpilot", _accel_sample(sm, "carControl", lambda: sm["carControl"].actuators.accel), rl.Color(128, 216, 166, 230)),
    ("AP", ap_accel_from_das(das_control, v_ego), rl.Color(255, 170, 60, 230)),
    ("actual", _accel_sample(sm, "carState", lambda: sm["carState"].aEgo), rl.Color(230, 230, 230, 230)),
  ]

  # Left third of the big frame. Tracks stay at most one third of that region's height,
  # and sit under the set-speed box (y=45, height=204).
  left_width = gui_app.width // 3
  track_h = gui_app.height // 3
  value_size, label_size = 28, 22
  bar_w = 36
  label_w = max(int(measure_text_cached(font, label, label_size).x) for label, _, _ in series)
  column_w = max(bar_w + 16, label_w + 12)
  group_w = column_w * len(series)
  origin_x = max(40, (left_width - group_w) // 2)
  value_h = int(measure_text_cached(font, "0.0", value_size).y)
  origin_y = 45 + 204 + 28
  track_top = origin_y + value_h + 8
  span = ACCEL_BAR_MAX - ACCEL_BAR_MIN

  def y_for(accel: float) -> int:
    clipped = min(max(accel, ACCEL_BAR_MIN), ACCEL_BAR_MAX)
    return int(track_top + (ACCEL_BAR_MAX - clipped) / span * track_h)

  zero_y = y_for(0.0)
  rl.draw_rectangle(origin_x - 8, track_top, group_w + 16, track_h, rl.Color(0, 0, 0, 90))
  rl.draw_line(origin_x, zero_y, origin_x + group_w, zero_y, rl.Color(255, 255, 255, 160))

  for i, (label, value, color) in enumerate(series):
    column_x = origin_x + i * column_w
    bar_x = column_x + (column_w - bar_w) // 2
    text = "--" if value is None else f"{value:.1f}"
    text_w = int(measure_text_cached(font, text, value_size).x)
    rl.draw_text_ex(font, text, rl.Vector2(column_x + (column_w - text_w) / 2, origin_y), value_size, 0, rl.WHITE)
    if value is not None:
      bar_y = y_for(value)
      top, bottom = min(bar_y, zero_y), max(bar_y, zero_y)
      rl.draw_rectangle(bar_x, top, bar_w, max(bottom - top, 2), color)
    label_w = int(measure_text_cached(font, label, label_size).x)
    rl.draw_text_ex(font, label, rl.Vector2(column_x + (column_w - label_w) / 2, track_top + track_h + 8), label_size, 0, rl.WHITE)


def render_overlays(gui_app, font, big, metadata, title, start_time, frame_idx, show_metadata, show_time, das_control=None):
  from openpilot.system.ui.lib.text_measure import measure_text_cached
  from openpilot.system.ui.lib.wrap_text import wrap_text
  metadata_size = 16 if big else 12
  title_size = 32 if big else 24
  time_size = 24 if big else 16

  # Time overlay
  time_width = 0
  if show_time:
    t = start_time + frame_idx / FRAMERATE
    time_text = f"{int(t) // 60:02d}:{int(t) % 60:02d}"
    time_width = int(measure_text_cached(font, time_text, time_size).x)
    draw_text_box(time_text, gui_app.width - time_width - 5, 0, time_size, gui_app, font)

  # Metadata overlay (first 5 seconds)
  if show_metadata and metadata and frame_idx < FRAMERATE * 5:
    m = metadata
    text = ", ".join([f"openpilot v{m['version']}", f"route: {m['route']}", f"car: {m['car']}", f"origin: {m['origin']}",
                      f"branch: {m['branch']}", f"commit: {m['commit']}", f"modified: {m['modified']}"])
    # Wrap text if too wide (leave margin on each side)
    margin = 2 * (time_width + 10 if show_time else 20)  # leave enough margin for time overlay
    max_width = gui_app.width - margin
    lines = wrap_text(font, text, metadata_size, max_width)

    # Draw wrapped metadata text
    y_offset = 6
    for line in lines:
      draw_text_box(line, 0, y_offset, metadata_size, gui_app, font, center=True)
      line_height = int(measure_text_cached(font, line, metadata_size).y) + 4
      y_offset += line_height

  # Title overlay
  if title:
    draw_text_box(title, 0, 60, title_size, gui_app, font, center=True)

  if big:
    _draw_accel_bars(gui_app, font, das_control)


def clip(route: Route, output: str, start: int, end: int, headless: bool = True, big: bool = False,
         title: str | None = None, show_metadata: bool = True, show_time: bool = True, use_qcam: bool = False):
  timer, duration = Timer(), end - start

  import pyray as rl
  if big:
    from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
  else:
    from openpilot.selfdrive.ui.mici.onroad.augmented_road_view import AugmentedRoadView
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.lib.application import gui_app, FontWeight
  timer.lap("import")

  logger.info(f"Clipping {route.name.canonical_name}, {start}s-{end}s ({duration}s)")
  seg_start, seg_end = start // 60, (end - 1) // 60 + 1
  all_chunks = load_logs_parallel(route.log_paths()[seg_start:seg_end], fps=FRAMERATE)
  timer.lap("logs")

  frame_start = (start - seg_start * 60) * FRAMERATE
  message_chunks = all_chunks[frame_start:frame_start + duration * FRAMERATE]
  if not message_chunks:
    logger.error("No messages to render")
    sys.exit(1)

  if headless:
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

  with OpenpilotPrefix(shared_download_cache=True):
    metadata = load_route_metadata(route) if show_metadata else None
    camera_paths = route.qcamera_paths() if use_qcam else route.camera_paths()
    frame_queue = FrameQueue(camera_paths, start, end, fps=FRAMERATE, use_qcam=use_qcam)

    ecamera_paths = route.ecamera_paths() if not use_qcam else []
    wide_frame_queue: FrameQueue | None = None
    if any(p for p in ecamera_paths[seg_start:seg_end] if p):
      wide_frame_queue = FrameQueue(ecamera_paths, start, end, fps=FRAMERATE)

    vipc = VisionIpcServer("camerad")
    vipc.create_buffers(VisionStreamType.VISION_STREAM_NARROW_ROAD, 4, frame_queue.frame_w, frame_queue.frame_h)
    if wide_frame_queue:
      vipc.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 4, wide_frame_queue.frame_w, wide_frame_queue.frame_h)
    vipc.start_listener()

    patch_submaster(message_chunks, ui_state)
    gui_app.init_window("clip", fps=FRAMERATE)

    road_view = AugmentedRoadView()
    road_view.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    font = gui_app.font(FontWeight.NORMAL)
    timer.lap("setup")

    frame_idx = 0
    with tqdm.tqdm(total=len(message_chunks), desc="Rendering", unit="frame") as pbar:
      for should_render, _, _ in gui_app.render():
        if frame_idx >= len(message_chunks):
          break
        _, frame_bytes = frame_queue.get()
        vipc.send(VisionStreamType.VISION_STREAM_NARROW_ROAD, frame_bytes, frame_idx, int(frame_idx * 5e7), int(frame_idx * 5e7))
        if wide_frame_queue:
          _, wide_bytes = wide_frame_queue.get()
          vipc.send(VisionStreamType.VISION_STREAM_WIDE_ROAD, wide_bytes, frame_idx, int(frame_idx * 5e7), int(frame_idx * 5e7))
        ui_state.update()
        if should_render:
          road_view.render()
          das_control = message_chunks[frame_idx].get('_dasControl')
          render_overlays(gui_app, font, big, metadata, title, start, frame_idx, show_metadata, show_time, das_control)
        frame_idx += 1
        pbar.update(1)
    timer.lap("render")

    frame_queue.stop()
    if wide_frame_queue:
      wide_frame_queue.stop()
    gui_app.close()
    timer.lap("ffmpeg")

  logger.info(f"Clip saved to: {Path(output).resolve()}")
  logger.info(f"Generated {timer.fmt(duration)}")


def main():
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s\t%(message)s")
  args = parse_args()

  headless = not args.windowed
  setup_env(args.output, big=args.big, speed=args.speed, target_mb=args.file_size, duration=args.end - args.start, headless=headless)
  clip(Route(args.route, data_dir=args.data_dir), args.output, args.start, args.end, headless,
       args.big, args.title, not args.no_metadata, not args.no_time_overlay, args.qcam)


if __name__ == "__main__":
  main()
