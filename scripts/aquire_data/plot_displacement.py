#!/usr/bin/env python3
"""
plot_displacement.py

Blade displacement from the accelerometers: integrates each sensor's acceleration twice, for ONE trial
(one motor angle at one tunnel condition = one angle_*deg.csv inside one run directory), and plots

    top panel     flapwise displacement vs physical time, S1..S4
    bottom panel  edgewise displacement vs physical time, S1..S4

Sensor layout (blade root at r = 0, R = 28 cm). Initial position of every sensor is 0 in the edgewise
direction; spanwise positions are fixed:

    S1  r/R = 0.13      S2  r/R = 0.26      S3  r/R = 0.35      S4  r/R = 0.48

Usage (from scripts/aquire_data):
    python plot_displacement.py Data/Production_10-8-26/DTU10MW_704psi_500fs --angle 170 --out Production_summaries/figures
    python plot_displacement.py Data/.../DTU10MW_704psi_500fs/angle_+170.00deg.csv
    python plot_displacement.py <run_dir> --angle 170 --t0 0.5 --t1 3 --fc 2 --flap-axis z --edge-axis x
    python plot_displacement.py <run_dir> --angle 170 --edge-axis=-x      # note the "=" for a negative sign

The first argument is either a run directory (then --angle picks the nearest requested angle) or the
angle_*.csv file itself. The run directory supplies the tunnel condition (run_metadata.json) for the title.

Method (per sensor, per axis)
    1. counts -> g (sensitivity from the CSV header) -> m/s^2
    2. resample onto a uniform time grid at the measured sample rate (the CSV time column is stamped per
       FIFO batch; utils/run_loader.py reconstructs a per-sample time, which is used here)
    3. remove the mean (the static part: gravity and the mean aerodynamic load) and high-pass at --fc
       (zero-phase Butterworth)
    4. integrate with the trapezoid rule from rest (v = 0, x = 0), high-passing the velocity and the
       displacement at --fc again, because any DC or very-low-frequency error in a double integral
       grows as t^2
    5. crop to [--t0, --t1] and, with --reference start (default), subtract the value at the first
       plotted sample so every trace starts at 0 (--reference mean centres them on their mean instead)

IMPORTANT assumptions to check
    * The accelerometer axes are the SENSOR's x, y, z, not blade directions. Which sensor axis is flapwise
      and which is edgewise depends on how the boards are mounted: set --flap-axis / --edge-axis
      (defaults z and x are a guess). Prefix with - to flip the sign.
    * Displacement is only meaningful for frequencies above --fc. Choose --fc a factor of ~3 below the
      lowest structural frequency you care about (e.g. the first flap/edge natural frequency from the
      ring-down tests); motion slower than that is removed, as is any static deflection.
    * Output is the displacement of each sensor relative to its starting position, in the direction of the
      chosen sensor axis (small-displacement, no rotation of the sensor with the blade).

Requires: numpy pandas scipy matplotlib
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from scipy.integrate import cumulative_trapezoid                  # noqa: E402
from scipy.signal import butter, sosfiltfilt                      # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils.run_loader import _discover_angle_files, load_accel_csv   # noqa: E402
from plot_angle_sweep import INK, MUTED, SURFACE, SENSOR_COLORS, style_axes   # noqa: E402

G = 9.80665                       # m/s^2
R_M = 0.28                        # blade length R (m)
R_OVER_R = (0.13, 0.26, 0.35, 0.48)           # spanwise position of S1..S4
SPAN_POS_M = tuple(x * R_M for x in R_OVER_R)  # (m); the edgewise start position is 0 for all
NUM_SENSORS = 4
INT16_MAX = 32767
SAT_LEVEL = 0.995
FALLBACK_MG_PER_LSB = 0.061       # 2g mode, used only if the CSV header has no sensitivity

ANGLE_RE = re.compile(r"angle_([+\-]?\d+\.\d+)deg\.csv$")


# --------------------------------------------------
# Trial selection
# --------------------------------------------------

def find_trial(path, angle):
    """Returns (run_dir, accel_csv_path, requested_angle_deg)."""
    p = Path(path)
    if p.is_file():
        m = ANGLE_RE.search(p.name)
        return p.parent, p, (float(m.group(1)) if m else None)
    if not p.is_dir():
        raise SystemExit(f"{p} is not a run directory or an angle_*.csv file")
    pairs = _discover_angle_files(p)
    if not pairs:
        raise SystemExit(f"No angle_*deg.csv files in {p}")
    if angle is None:
        avail = ", ".join(f"{a:g}" for a, _, _ in pairs)
        raise SystemExit(f"Give --angle (requested angles in this run: {avail})")
    a, ap, _ = min(pairs, key=lambda t: abs(t[0] - angle))
    if abs(a - angle) > 0.5:
        print(f"NOTE: no capture at {angle:g} deg; using the nearest requested angle, {a:+.2f} deg")
    return p, ap, a


def read_metadata(run_dir):
    path = Path(run_dir) / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def motor_angle_from_meta(meta, requested):
    if requested is None:
        return None
    for r in meta.get("angles") or []:
        rq, act = r.get("motor_angle_requested_deg"), r.get("motor_angle_actual_deg")
        if rq is not None and act is not None and abs(rq - requested) < 1e-6:
            return float(act)
    return None


def parse_axis(text):
    m = re.fullmatch(r"([+-]?)([xyz])", text.strip().lower())
    if not m:
        raise SystemExit(f"axis {text!r}: use x, y or z, optionally with a sign (-x)")
    return "xyz".index(m[2]), (-1.0 if m[1] == "-" else 1.0), text.strip().lower()


# --------------------------------------------------
# Signal processing
# --------------------------------------------------

def highpass(x, fs, fc, order=4):
    if fc <= 0:
        return x - x.mean(axis=0)
    sos = butter(order, fc, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def double_integrate(a, fs, fc):
    """a: (N, k) acceleration in m/s^2 on a uniform grid. Returns (velocity, displacement), zero-drift."""
    a = highpass(a - a.mean(axis=0), fs, fc)
    v = cumulative_trapezoid(a, dx=1.0 / fs, axis=0, initial=0.0)
    v = highpass(v, fs, fc)
    d = cumulative_trapezoid(v, dx=1.0 / fs, axis=0, initial=0.0)
    d = highpass(d, fs, fc)
    return v, d


def load_trial(accel_path, fs_override=None):
    """Returns (t_uniform, accel_g (N, 4 sensors, 3 axes), fs, raw int16 array)."""
    df, _ = load_accel_csv(accel_path)
    mg = df.attrs.get("mg_per_lsb")
    if mg is None:
        mg = FALLBACK_MG_PER_LSB
        print(f"  WARN: no mg/LSB in the CSV header; assuming {mg} (2g mode)")
    cols = [f"s{s}_{a}" for s in range(1, NUM_SENSORS + 1) for a in "xyz"]
    raw = df[cols].to_numpy(float)
    t = df["time"].to_numpy(float)
    if len(t) < 64:
        raise SystemExit("too few samples in this capture")
    fs = fs_override or (len(t) - 1) / (t[-1] - t[0])
    tu = t[0] + np.arange(len(t)) / fs
    acc = np.column_stack([np.interp(tu, t, raw[:, k]) for k in range(raw.shape[1])])
    acc_g = (acc * mg / 1000.0).reshape(-1, NUM_SENSORS, 3)
    return tu, acc_g, fs, raw, mg


# --------------------------------------------------
# Plot
# --------------------------------------------------

def make_figure(t, d_flap, d_edge, labels, title, flap_name, edge_name, args, out_png):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, facecolor=SURFACE,
                                   constrained_layout=True)
    for ax, d, name in ((ax1, d_flap, "Flapwise"), (ax2, d_edge, "Edgewise")):
        for s in range(NUM_SENSORS):
            ax.plot(t, d[:, s] * 1000.0, color=SENSOR_COLORS[s], lw=1.1, label=labels[s])
        ax.axhline(0, color=MUTED, lw=0.6, alpha=0.6)
        ax.set_ylabel(f"{name} displacement (mm)", color=INK)
        ax.set_title(f"{name}  (sensor {flap_name if name == 'Flapwise' else edge_name} axis)",
                     loc="left", fontsize=10, color=INK)
        style_axes(ax)
    ax1.legend(loc="upper right", ncol=NUM_SENSORS, frameon=False, fontsize=9, labelcolor=INK)
    ax2.set_xlabel("Time (s)", color=INK)
    ax2.set_xlim(t[0], t[-1])
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", help="run directory (with --angle) or the angle_*.csv file of one trial")
    ap.add_argument("--angle", type=float, help="requested motor angle (deg) of the trial, nearest is used")
    ap.add_argument("--flap-axis", default="z", help="sensor axis for flapwise motion: x, y, z, or -x ... (default z)")
    ap.add_argument("--edge-axis", default="x", help="sensor axis for edgewise motion (default x)")
    ap.add_argument("--fc", type=float, default=2.0,
                    help="high-pass cutoff (Hz) used to stop integration drift (default 2)")
    ap.add_argument("--t0", type=float, default=0.5,
                    help="first plotted time (s); the start of the capture holds filter transients (default 0.5)")
    ap.add_argument("--t1", type=float, help="last plotted time (s) (default end of capture)")
    ap.add_argument("--reference", choices=["start", "mean"], default="start",
                    help="start: traces start at 0 (default); mean: traces are centred on their mean")
    ap.add_argument("--fs", type=float, help="override the sample rate (Hz); default is measured from the CSV")
    ap.add_argument("--csv", action="store_true", help="also write the plotted displacements to a csv")
    ap.add_argument("--out", help="output folder (default <run_dir>/analysis)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    run_dir, accel_path, requested = find_trial(args.trial, args.angle)
    meta = read_metadata(run_dir)
    fi, fsign, fname = parse_axis(args.flap_axis)
    ei, esign, ename = parse_axis(args.edge_axis)
    if fi == ei:
        raise SystemExit("--flap-axis and --edge-axis must be different sensor axes")

    print(f"Trial: {accel_path}")
    t, acc_g, fs, raw, mg = load_trial(accel_path, args.fs)
    print(f"  {len(t)} samples, measured rate {fs:.1f} Hz, {mg} mg/LSB")
    if args.fc >= fs / 4:
        raise SystemExit(f"--fc {args.fc:g} Hz is too high for a {fs:.0f} Hz sample rate")
    odr = meta.get("accel_odr_hz")
    if odr and fs < 0.9 * odr:
        print(f"  WARN: measured rate {fs:.0f} Hz is below 90% of the ODR ({odr} Hz): dropped samples distort the integration")

    sat = (np.abs(raw) >= SAT_LEVEL * INT16_MAX).mean(axis=0).reshape(NUM_SENSORS, 3)
    for s in range(NUM_SENSORS):
        for j in (fi, ei):
            if sat[s, j] >= 0.01:
                print(f"  WARN: S{s + 1} {'xyz'[j]} axis was at full scale for {100 * sat[s, j]:.1f}% of samples "
                      "(clipped; displacement is underestimated)")

    # integrate every sensor for the two chosen axes
    d = {}
    for name, j, sign in (("flap", fi, fsign), ("edge", ei, esign)):
        a = sign * acc_g[:, :, j] * G                       # (N, 4) m/s^2
        _, disp = double_integrate(a, fs, args.fc)
        d[name] = disp

    t1 = args.t1 if args.t1 is not None else t[-1]
    m = (t >= args.t0) & (t <= t1)
    if m.sum() < 16:
        raise SystemExit(f"no samples in the window {args.t0:g} to {t1:g} s")
    tw = t[m]
    out = {}
    for name in ("flap", "edge"):
        w = d[name][m]
        out[name] = w - (w[0] if args.reference == "start" else w.mean(axis=0))

    labels = [f"S{s + 1}  r/R = {R_OVER_R[s]:.2f}  (r = {100 * SPAN_POS_M[s]:.1f} cm)" for s in range(NUM_SENSORS)]
    bits = [run_dir.name]
    ang = motor_angle_from_meta(meta, requested)
    if ang is not None:
        bits.append(f"motor angle {ang:.1f}°")
    elif requested is not None:
        bits.append(f"requested angle {requested:+.1f}°")
    tc = meta.get("tunnel_conditions") or {}
    if tc.get("velocity_freestream_m_s") is not None:
        bits.append(f"U = {tc['velocity_freestream_m_s']:.2f} m/s")
    bits.append(f"high-pass {args.fc:g} Hz")
    title = "   ·   ".join(bits)

    out_dir = Path(args.out) if args.out else run_dir / "analysis"
    stem = f"displacement_{accel_path.stem}"
    png = out_dir / f"{stem}.png"
    make_figure(tw, out["flap"], out["edge"], labels, title, fname, ename, args, png)
    print(f"Wrote {png}")

    for name in ("flap", "edge"):
        pp = np.ptp(out[name], axis=0) * 1000.0
        rms = out[name].std(axis=0) * 1000.0
        print(f"  {name}wise  peak-to-peak (mm): " + ", ".join(f"S{s + 1} {pp[s]:.3g}" for s in range(NUM_SENSORS))
              + "   rms (mm): " + ", ".join(f"S{s + 1} {rms[s]:.3g}" for s in range(NUM_SENSORS)))

    if args.csv:
        tab = pd.DataFrame({"time_s": tw})
        for name in ("flap", "edge"):
            for s in range(NUM_SENSORS):
                tab[f"{name}_s{s + 1}_mm"] = out[name][:, s] * 1000.0
        tab.to_csv(png.with_suffix(".csv"), index=False)
        print(f"Wrote {png.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
