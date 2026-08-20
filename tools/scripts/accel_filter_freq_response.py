#!/usr/bin/env python3
"""Frequency response of modeld's desired-accel smoother vs matched Bessel LPF.

Existing filter: first-order exponential smoother in modeld
  H(s) = 1 / (1 + tau s),  tau = LONG_SMOOTH_SECONDS = 0.3
  discretized at 20 Hz with alpha = 1 - exp(-dt/tau)

Bessel N=2 and N=4 analog prototypes are scaled so |H(j 2 pi * 20)| matches
the first-order analog filter, then bilinear-discretized at 100 Hz for
jotPluggler previews.

Requires scipy and matplotlib (e.g. `.venv/bin/python tools/scripts/accel_filter_freq_response.py`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import brentq

TAU = 0.3  # LONG_SMOOTH_SECONDS
FS_MDL = 20.0
FS_CTRL = 100.0
F_MATCH = 20.0
DT_MDL = 1.0 / FS_MDL


def fo_analog_mag(f_hz: np.ndarray, tau: float = TAU) -> np.ndarray:
  return 1.0 / np.sqrt(1.0 + (2.0 * np.pi * f_hz * tau) ** 2)


def fo_discrete_ba(tau: float = TAU, dt: float = DT_MDL) -> tuple[np.ndarray, np.ndarray]:
  alpha = 1.0 - np.exp(-dt / tau)
  b = np.array([alpha])
  a = np.array([1.0, -(1.0 - alpha)])
  return b, a


def analog_eval(b: np.ndarray, a: np.ndarray, f_hz: np.ndarray | float) -> np.ndarray:
  w = 2.0 * np.pi * np.atleast_1d(np.asarray(f_hz, dtype=np.float64))
  _, h = signal.freqs(b, a, worN=w)
  return h


def analog_fc_3db(b: np.ndarray, a: np.ndarray) -> float:
  def mag_err(f):
    return np.abs(analog_eval(b, a, f)[0]) - (1.0 / np.sqrt(2.0))
  return brentq(mag_err, 1e-6, 1e3)


def analog_group_delay_dc(b: np.ndarray, a: np.ndarray, eps: float = 1e-4) -> float:
  w = np.array([0.0, eps])
  _, h = signal.freqs(b, a, worN=w)
  phase = np.unwrap(np.angle(h))
  return float(-(phase[1] - phase[0]) / (w[1] - w[0]))


def match_bessel_wn(order: int, f_match: float, target_mag: float) -> float:
  def mag_err(wn: float) -> float:
    b, a = signal.bessel(order, wn, analog=True, btype="low", output="ba", norm="mag")
    mag = np.abs(analog_eval(b, a, f_match)[0])
    return mag - target_mag
  return brentq(mag_err, 1e-3, 500.0)


def match_bessel_wn_digital(order: int, f_match: float, target_mag: float, fs: float) -> float:
  """Scale analog Wn so the bilinear 100 Hz filter matches |H| at f_match."""
  def mag_err(wn: float) -> float:
    sos = bilinear_sos(order, wn, fs)
    _, h = signal.sosfreqz(sos, worN=[f_match], fs=fs)
    return float(np.abs(h[0])) - target_mag
  return brentq(mag_err, 1e-3, 500.0)


def bilinear_sos(order: int, wn_rad: float, fs: float) -> np.ndarray:
  z, p, k = signal.bessel(order, wn_rad, analog=True, btype="low", output="zpk", norm="mag")
  zd, pd, kd = signal.bilinear_zpk(z, p, k, fs=fs)
  return signal.zpk2sos(zd, pd, kd)


def fmt_sos(sos: np.ndarray) -> str:
  rows = []
  for row in sos:
    rows.append("[" + ", ".join(f"{x:.16e}" for x in row) + "]")
  return "[\n  " + ",\n  ".join(rows) + "\n]"


def plot_bode(filters: list[dict], f_match: float, out_path: Path) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
  ax_mag, ax_phase, ax_gd = axes
  f = np.logspace(-2, np.log10(45.0), 800)

  for spec in filters:
    style = spec.get("style", {})
    if spec["kind"] == "analog":
      h = analog_eval(spec["b"], spec["a"], f)
      # analog group delay via local phase derivative
      w = 2.0 * np.pi * f
      phase = np.unwrap(np.angle(h))
      gd = -np.gradient(phase, w)
    else:
      w_norm, h = signal.freqz(spec["b"], spec["a"], worN=4096, fs=spec["fs"])
      # freqz returns Hz when fs is given
      f_d = w_norm
      nyq = spec["fs"] / 2.0
      mask = f_d < min(nyq * 0.99, f[-1])
      f_plot = f_d[mask]
      h = h[mask]
      w, gd = signal.group_delay((spec["b"], spec["a"]), w=f_plot, fs=spec["fs"])
      ax_mag.semilogx(f_plot, 20.0 * np.log10(np.maximum(np.abs(h), 1e-12)), label=spec["label"], **style)
      ax_phase.semilogx(f_plot, np.unwrap(np.angle(h, deg=True)), **style)
      ax_gd.semilogx(f_plot, gd, **style)
      continue

    ax_mag.semilogx(f, 20.0 * np.log10(np.maximum(np.abs(h), 1e-12)), label=spec["label"], **style)
    ax_phase.semilogx(f, np.unwrap(np.angle(h, deg=True)), **style)
    ax_gd.semilogx(f, gd, **style)

  ax_mag.axvline(f_match, color="k", linestyle=":", linewidth=0.8, label=f"{f_match:g} Hz match")
  ax_mag.axhline(-3.0, color="0.6", linestyle="--", linewidth=0.8)
  ax_mag.set_ylabel("Magnitude [dB]")
  ax_mag.grid(True, which="both", alpha=0.4)
  ax_mag.legend(loc="lower left", fontsize=8)
  ax_mag.set_ylim(-50, 5)

  ax_phase.set_ylabel("Phase [deg]")
  ax_phase.grid(True, which="both", alpha=0.4)

  ax_gd.set_ylabel("Group delay [s]")
  ax_gd.set_xlabel("Frequency [Hz]")
  ax_gd.grid(True, which="both", alpha=0.4)
  ax_gd.set_ylim(0, 0.5)

  fig.suptitle("Desired-acceleration smoother vs matched Bessel LPF")
  fig.tight_layout()
  fig.savefig(out_path, dpi=120)
  print(f"wrote {out_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("/tmp/accel_filter_freq_response.png"))
  parser.add_argument("--no-plot", action="store_true")
  args = parser.parse_args()

  target_mag = float(fo_analog_mag(np.array([F_MATCH]))[0])
  target_db = 20.0 * np.log10(target_mag)
  fo_b = np.array([1.0])
  fo_a = np.array([TAU, 1.0])  # H(s) = 1 / (tau s + 1)

  print("Existing first-order analog (tau=0.3 s)")
  print(f"  |H({F_MATCH:g} Hz)| = {target_mag:.6g}  ({target_db:.3f} dB)")
  print(f"  -3 dB fc           = {analog_fc_3db(fo_b, fo_a):.4f} Hz")
  print(f"  DC group delay     = {analog_group_delay_dc(fo_b, fo_a):.4f} s")
  b_d20, a_d20 = fo_discrete_ba()
  print(f"  discrete 20 Hz     alpha={b_d20[0]:.6g}")

  analog = [("FO analog", 1, fo_b, fo_a)]
  sos_by_order: dict[int, np.ndarray] = {}
  rows = [{
    "name": "FO analog",
    "order": 1,
    "fc": analog_fc_3db(fo_b, fo_a),
    "h20_db": target_db,
    "tgd0": analog_group_delay_dc(fo_b, fo_a),
    "wn": 1.0 / TAU,
  }]

  for order in (2, 4):
    wn = match_bessel_wn(order, F_MATCH, target_mag)
    b, a = signal.bessel(order, wn, analog=True, btype="low", output="ba", norm="mag")
    mag20 = np.abs(analog_eval(b, a, F_MATCH)[0])
    row = {
      "name": f"Bessel N={order} analog",
      "order": order,
      "fc": analog_fc_3db(b, a),
      "h20_db": 20.0 * np.log10(mag20),
      "tgd0": analog_group_delay_dc(b, a),
      "wn": wn,
    }
    rows.append(row)
    analog.append((row["name"], order, b, a))
    sos = bilinear_sos(order, wn, FS_CTRL)
    sos_by_order[order] = sos
    print(f"\n{row['name']}")
    print(f"  Wn (rad/s, -3 dB) = {wn:.6g}  ({wn / (2 * np.pi):.4f} Hz)")
    print(f"  -3 dB fc           = {row['fc']:.4f} Hz")
    print(f"  |H({F_MATCH:g} Hz)|       = {mag20:.6g}  ({row['h20_db']:.3f} dB)")
    print(f"  DC group delay     = {row['tgd0']:.4f} s")
    print(f"  analog-matched 100 Hz SOS (bilinear, |H(20 Hz)| warped):\n{fmt_sos(sos)}")

    wn_d = match_bessel_wn_digital(order, F_MATCH, target_mag, FS_CTRL)
    sos_d = bilinear_sos(order, wn_d, FS_CTRL)
    b_d, a_d = signal.bessel(order, wn_d, analog=True, btype="low", output="ba", norm="mag")
    _, h_d = signal.sosfreqz(sos_d, worN=[F_MATCH], fs=FS_CTRL)
    print(f"  digital-matched analog fc = {analog_fc_3db(b_d, a_d):.4f} Hz  Tgd(0)={analog_group_delay_dc(b_d, a_d):.4f} s")
    print(f"  |H_d({F_MATCH:g} Hz)|     = {np.abs(h_d[0]):.6g}  ({20.0 * np.log10(np.abs(h_d[0])):.3f} dB)")
    print(f"  jotPluggler 100 Hz SOS (match |H| at 20 Hz after bilinear):\n{fmt_sos(sos_d)}")
    sos_by_order[order] = sos_d

  print("\nCorner-frequency comparison (analog -3 dB)")
  print(f"{'filter':<22} {'order':>5} {'fc [Hz]':>10} {'|H(20Hz)| [dB]':>16} {'Tgd(0) [s]':>12}")
  for r in rows:
    print(f"{r['name']:<22} {r['order']:5d} {r['fc']:10.4f} {r['h20_db']:16.3f} {r['tgd0']:12.4f}")

  if not args.no_plot:
    plot_specs: list[dict] = [
      {"kind": "analog", "label": "FO analog τ=0.3", "b": fo_b, "a": fo_a, "style": {"color": "C0", "lw": 2}},
      {"kind": "digital", "label": "FO discrete 20 Hz", "b": b_d20, "a": a_d20, "fs": FS_MDL,
       "style": {"color": "C0", "ls": "--", "lw": 1.5}},
    ]
    colors = {2: "C1", 4: "C2"}
    for name, order, b, a in analog[1:]:
      plot_specs.append({"kind": "analog", "label": name, "b": b, "a": a,
                         "style": {"color": colors[order], "lw": 2}})
      bd, ad = signal.sos2tf(sos_by_order[order])
      plot_specs.append({"kind": "digital", "label": f"Bessel N={order} 100 Hz", "b": bd, "a": ad, "fs": FS_CTRL,
                         "style": {"color": colors[order], "ls": "--", "lw": 1.5}})
    plot_bode(plot_specs, F_MATCH, args.out)


if __name__ == "__main__":
  main()
