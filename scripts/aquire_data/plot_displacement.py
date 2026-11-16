#!/usr/bin/env python3
"""
plot_displacement.py

Blade displacement from the accelerometers: integrates each sensor's acceleration twice, for ONE trial
(one motor angle at one tunnel condition = one angle_*deg.csv inside one run directory), and plots

    top panel     flapwise displacement vs physical time, S1..S4
    bottom panel  edgewise displacement vs physical time, S1..S4

With --animate it also writes a two-panel animation (flapwise on top, edgewise below): the four sensors
are dots on a cubic spline along the span, so the blade shape is visible as it moves.

Sensor layout (blade root at r = 0, R = 28 cm):

    S1  r/R = 0.13      S2  r/R = 0.26      S3  r/R = 0.35      S4  r/R = 0.48

Usage (from scripts/aquire_data):
    python plot_displacement.py Data/Production_10-8-26/DTU10MW_704psi_500fs --angle 170
    python plot_displacement.py Data/.../DTU10MW_704psi_500fs/angle_+170.00deg.csv
    python plot_displacement.py <run_dir> --angle 170 --t0 0.5 --t1 3 --fc 2 --flap-axis z
    python plot_displacement.py <run_dir> --angle 170 --flap-axis=-z       # note the "=" for a negative sign
    python plot_displacement.py <run_dir> --angle 170 --animate
    python plot_displacement.py <run_dir> --angle 170 --animate --anim-t0 1.0 --anim-duration 0.5 --slowdown 40

The first argument is either a run directory (then --angle picks the nearest requested angle) or the
angle_*.csv file itself. The run directory supplies the tunnel condition (run_metadata.json) for the title.

Method (per sensor, per axis)
    1. counts -> g (sensitivity from the CSV header) -> m/s^2
    2. resample onto a uniform time grid at the measured sample rate (the CSV time column is stamped per
       FIFO batch; utils/run_loader.py reconstructs a per-sample time, which is used here)
    3. remove the mean and high-pass the acceleration at --fc (zero-phase Butterworth)
    4. integrate with the trapezoid rule from rest (v = 0, x = 0), high-passing the velocity and the
       displacement at --fc again, because any DC or very-low-frequency error in a double integral
       grows as t^2
    5. crop to [--t0, --t1] (so the filter start-up transient is excluded) and, with --reference start
       (default), subtract the value at the first plotted sample so every trace starts at 0
       (--reference mean centres them on their mean instead)

IMPORTANT assumptions to check
    * The accelerometer axes are the SENSOR's x, y, z, not blade directions. Edgewise is the sensor y axis
      (default). The flapwise default is the sensor z axis, which is a guess: set --flap-axis to match how
      the boards are mounted. Prefix with - to flip the sign.
    * Displacement is only meaningful for frequencies above --fc. Choose --fc a factor of ~3 below the
      lowest structural frequency you care about (e.g. the first flap/edge natural frequency from the
      ring-down tests); slower motion, including any static deflection, is removed.
    * The high-pass is what keeps the integration from drifting; if a trace still wanders slowly, raise
      --fc.
    * Displacement falls as 1/f^2 for a given acceleration, so a ~100 Hz component that is obvious in the
      acceleration or PSD can be tiny in displacement next to the 5-30 Hz motion. Use --fc 80 --fmax 120
      (a band around it) to see it on its own.
    * Small-displacement assumption: each sensor moves along its chosen axis and does not rotate with the
      blade.
    * Animation: the blade root (r/R = 0) is treated as fixed at zero displacement and the spline runs from
      the root to S4 (r/R = 0.48). The spline is clamped (zero slope) at the root. Between the sensors it is
      an interpolation, not a measurement.

Requires: numpy pandas scipy matplotlib (animation: ffmpeg for mp4, otherwise a gif is written with Pillow)
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
from matplotlib.lines import Line2D                               # noqa: E402
from scipy.integrate import cumulative_trapezoid                  # noqa: E402
from scipy.interpolate import CubicSpline                         # noqa: E402
from scipy.signal import butter, sosfiltfilt                      # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils.run_loader import _discover_angle_files, load_accel_csv   # noqa: E402
from plot_angle_sweep import INK, MUTED, SURFACE, SENSOR_COLORS, style_axes   # noqa: E402

G = 9.80665                       # m/s^2
R_M = 0.28                        # blade length R (m)
R_OVER_R = (0.13, 0.26, 0.35, 0.48)           # spanwise position of S1..S4
SPAN_POS_M = tuple(x * R_M for x in R_OVER_R)
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
        return x 
    sos = butter(order, fc, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def lowpass(x, fs, fmax, order=4):
    sos = butter(order, fmax, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def double_integrate(a, fs, fc, fmax=None):
    """a: (N, k) acceleration (m/s^2) on a uniform grid. Returns (velocity, displacement), drift-reduced:
    the acceleration, velocity and displacement are each high-passed at fc. With fmax the acceleration is
    also low-passed at fmax, so only the band fc..fmax is integrated."""
    a = highpass(a, fs, fc)
    if fmax:
        a = lowpass(a, fs, fmax)
    v = cumulative_trapezoid(a, dx=1.0 / fs, axis=0, initial=0.0)
    v = highpass(v, fs, fc)
    d = cumulative_trapezoid(v, dx=1.0 / fs, axis=0, initial=0.0)
    d = highpass(d, fs, fc)
    return v, d


def load_trial(accel_path, fs_override=None):
    """Returns (t_uniform, accel_g (N, 4 sensors, 3 axes), fs, raw int16 array, mg_per_lsb)."""
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
# Static plot
# --------------------------------------------------

def make_figure(t, d_flap, d_edge, labels, title, flap_name, edge_name, args, out_png):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, facecolor=SURFACE,
                                   constrained_layout=True)
    for ax, d, name, axname in ((ax1, d_flap, "Flapwise", flap_name), (ax2, d_edge, "Edgewise", edge_name)):
        for s in range(NUM_SENSORS):
            ax.plot(t, d[:, s] * 1000.0, color=SENSOR_COLORS[s], lw=1.1, label=labels[s])
        ax.axhline(0, color=MUTED, lw=0.6, alpha=0.6)
        ax.set_ylabel(f"{name} displacement (mm)", color=INK)
        ax.set_title(f"{name}  (sensor {axname} axis)", loc="left", fontsize=10, color=INK)
        style_axes(ax)
    ax1.legend(loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    ax2.set_xlabel("Time (s)", color=INK)
    ax2.set_xlim(t[0], t[-1])
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------
# Animation
# --------------------------------------------------

def spline_curve(r_nodes, y_nodes, n=120):
    """Cubic spline through the root (fixed, zero slope) and the sensor displacements."""
    cs = CubicSpline(r_nodes, y_nodes, bc_type=((1, 0.0), (2, 0.0)))
    xs = np.linspace(r_nodes[0], r_nodes[-1], n)
    return xs, cs(xs)


def make_animation(tw, d_flap, d_edge, fs, title, flap_name, edge_name, args, out_base):
    from matplotlib import animation

    n = len(tw)
    t_start = args.anim_t0 if args.anim_t0 is not None else tw[0]
    i0 = min(int(np.searchsorted(tw, t_start)), n - 2)
    i1 = min(max(int(np.searchsorted(tw, t_start + args.anim_duration)), i0 + 2), n)
    step = max(1, int(round(fs / (args.fps * args.slowdown)))) * max(1, args.subsample)
    frames = np.arange(i0, i1, step)
    eff_slow = fs / (step * args.fps)            # actual slowdown after rounding the frame step to samples
    f_ok = fs / (4.0 * step)                     # >= 4 frames per cycle below this frequency
    print(f"  Animation frame spacing {1000.0 * step / fs:.2f} ms: motion up to ~{f_ok:.0f} Hz is resolved "
          "(4+ frames per cycle); faster motion will look aliased or slowed")
    print(f"  Animation: {len(frames)} frames, data {tw[i0]:.3f} to {tw[i1 - 1]:.3f} s, "
          f"{len(frames) / args.fps:.1f} s of video at {args.fps:g} fps ({eff_slow:.0f}x slower than real time)")

    r_nodes = np.concatenate([[0.0], R_OVER_R])
    ymax = 1.15 * max(np.abs(d_flap[i0:i1]).max(), np.abs(d_edge[i0:i1]).max()) * 1000.0
    ymax = ymax if ymax > 0 else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, sharey=True, facecolor=SURFACE,
                             constrained_layout=True)
    arts = []
    for ax, name, axname in ((axes[0], "Flapwise", flap_name), (axes[1], "Edgewise", edge_name)):
        ax.axhline(0, color=MUTED, lw=0.7, alpha=0.6, ls=(0, (4, 3)))
        (line,) = ax.plot(r_nodes, np.zeros_like(r_nodes), color=INK, lw=2.0, zorder=3)
        dots = ax.scatter(R_OVER_R, np.zeros(NUM_SENSORS), s=95, c=SENSOR_COLORS, edgecolor=SURFACE,
                          linewidth=1.2, zorder=5)
        ax.plot([0.0], [0.0], marker="s", ms=8, color=INK, zorder=4)         # fixed root
        ax.set_ylabel(f"{name} displacement (mm)", color=INK)
        ax.set_title(f"{name}  (sensor {axname} axis)", loc="left", fontsize=10, color=INK)
        style_axes(ax)
        arts.append((line, dots))
    axes[0].set_ylim(-ymax, ymax)
    axes[1].set_xlim(-0.02, R_OVER_R[-1] + 0.04)
    axes[1].set_xlabel("Spanwise position r/R  (root at 0)", color=INK)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=SENSOR_COLORS[s], mec=SURFACE,
                      label=f"S{s + 1}  r/R = {R_OVER_R[s]:.2f}") for s in range(NUM_SENSORS)]
    axes[0].legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    sup = fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)

    def update(k):
        i = frames[k]
        for (line, dots), d in zip(arts, (d_flap, d_edge)):
            y = d[i] * 1000.0
            xs, ys = spline_curve(r_nodes, np.concatenate([[0.0], y]))
            line.set_data(xs, ys)
            dots.set_offsets(np.column_stack([R_OVER_R, y]))
        sup.set_text(f"{title}\nt = {tw[i]:.3f} s   ({eff_slow:.0f}x slower than real time)")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    fmt = args.anim_format
    if fmt == "mp4" and not animation.writers.is_available("ffmpeg"):
        print("  NOTE: ffmpeg not found; writing a gif instead (--anim-format gif)")
        fmt = "gif"
    path = out_base.with_name(out_base.stem + "_animation").with_suffix("." + fmt)
    writer = animation.FFMpegWriter(fps=args.fps) if fmt == "mp4" else animation.PillowWriter(fps=args.fps)
    anim.save(path, writer=writer, dpi=args.anim_dpi, savefig_kwargs=dict(facecolor=SURFACE))
    plt.close(fig)
    return path


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", help="run directory (with --angle) or the angle_*.csv file of one trial")
    ap.add_argument("--angle", type=float, help="requested motor angle (deg) of the trial, nearest is used")
    ap.add_argument("--flap-axis", default="z", help="sensor axis for flapwise motion: x, y, z, or -x ... (default z)")
    ap.add_argument("--edge-axis", default="y", help="sensor axis for edgewise motion (default y)")
    ap.add_argument("--fc", type=float, default=2.0,
                    help="high-pass cutoff (Hz) applied to the acceleration to stop integration drift (default 2)")
    ap.add_argument("--fmax", type=float,
                    help="also low-pass the acceleration at this frequency (Hz), so only the band --fc..--fmax is "
                         "shown, e.g. --fc 80 --fmax 120 isolates a 100 Hz component")
    ap.add_argument("--t0", type=float, default=0.5,
                    help="first plotted/integrated time (s); the start of the capture holds filter transients (default 0.5)")
    ap.add_argument("--t1", type=float, help="last plotted/integrated time (s) (default end of capture)")
    ap.add_argument("--reference", choices=["start", "mean"], default="start",
                    help="start: traces start at 0 (default); mean: traces are centred on their mean")
    ap.add_argument("--fs", type=float, help="override the sample rate (Hz); default is measured from the CSV")
    ap.add_argument("--csv", action="store_true", help="also write the plotted displacements to a csv")
    ap.add_argument("--out", help="output folder (default <run_dir>/analysis)")
    ap.add_argument("--dpi", type=int, default=200)

    an = ap.add_argument_group("animation")
    an.add_argument("--animate", action="store_true",
                    help="also write a two-panel animation (flapwise, edgewise) with the sensors as dots on a spline")
    an.add_argument("--anim-t0", type=float, help="start time (s) of the animated segment (default: start of the window)")
    an.add_argument("--anim-duration", type=float, default=1.0,
                    help="seconds of DATA to animate (default 1.0)")
    an.add_argument("--slowdown", type=float, default=20.0,
                    help="playback is this many times slower than real time (default 20; a 1 s segment becomes "
                         "20 s of video at 30 fps)")
    an.add_argument("--fps", type=float, default=30.0, help="video frame rate (default 30)")
    an.add_argument("--subsample", type=int, default=1,
                    help="keep only every Nth animation frame (default 1 = all). N times fewer frames, so N times "
                         "faster to render and N times shorter video; the playback slowdown drops by N and "
                         "frequencies above ~1/(4 x frame spacing) are undersampled (a note is printed)")
    an.add_argument("--anim-format", choices=["mp4", "gif"], default="mp4",
                    help="mp4 needs ffmpeg (falls back to gif if it is missing)")
    an.add_argument("--anim-dpi", type=int, default=110)
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
    if args.fmax is not None and not (args.fc < args.fmax < fs / 2):
        raise SystemExit(f"--fmax must be between --fc ({args.fc:g}) and the Nyquist frequency ({fs / 2:.0f} Hz)")
    odr = meta.get("accel_odr_hz")
    if odr and fs < 0.9 * odr:
        print(f"  WARN: measured rate {fs:.0f} Hz is below 90% of the ODR ({odr} Hz): dropped samples distort the integration")

    sat = (np.abs(raw) >= SAT_LEVEL * INT16_MAX).mean(axis=0).reshape(NUM_SENSORS, 3)
    for s in range(NUM_SENSORS):
        for j in (fi, ei):
            if sat[s, j] >= 0.01:
                print(f"  WARN: S{s + 1} {'xyz'[j]} axis was at full scale for {100 * sat[s, j]:.1f}% of samples "
                      "(clipped; displacement is underestimated)")

    t1 = args.t1 if args.t1 is not None else t[-1]
    m = (t >= args.t0) & (t <= t1)
    if m.sum() < 16:
        raise SystemExit(f"no samples in the window {args.t0:g} to {t1:g} s")
    tw = t[m]

    # integrate the whole record, then crop to the window and apply the reference
    disp = {}
    for name, j, sign in (("flap", fi, fsign), ("edge", ei, esign)):
        a = sign * acc_g[:, :, j] * G                                  # (N, 4) m/s^2
        _, d = double_integrate(a, fs, args.fc, args.fmax)
        w = d[m]
        disp[name] = w - (w[0] if args.reference == "start" else w.mean(axis=0))

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
    bits.append(f"band {args.fc:g}-{args.fmax:g} Hz" if args.fmax else f"high-pass {args.fc:g} Hz")
    title = "   ·   ".join(bits)

    out_dir = Path(args.out) if args.out else run_dir / "analysis"
    png = out_dir / f"displacement_{accel_path.stem}.png"
    make_figure(tw, disp["flap"], disp["edge"], labels, title, fname, ename, args, png)
    print(f"Wrote {png}")

    for name in ("flap", "edge"):
        pp = np.ptp(disp[name], axis=0) * 1000.0
        rms = disp[name].std(axis=0) * 1000.0
        print(f"  {name}wise  peak-to-peak (mm): " + ", ".join(f"S{s + 1} {pp[s]:.3g}" for s in range(NUM_SENSORS))
              + "   rms (mm): " + ", ".join(f"S{s + 1} {rms[s]:.3g}" for s in range(NUM_SENSORS)))

    if args.csv:
        tab = pd.DataFrame({"time_s": tw})
        for name in ("flap", "edge"):
            for s in range(NUM_SENSORS):
                tab[f"{name}_s{s + 1}_mm"] = disp[name][:, s] * 1000.0
        tab.to_csv(png.with_suffix(".csv"), index=False)
        print(f"Wrote {png.with_suffix('.csv')}")

    if args.animate:
        apath = make_animation(tw, disp["flap"], disp["edge"], fs, title, fname, ename, args, png)
        print(f"Wrote {apath}")


if __name__ == "__main__":
    main()
