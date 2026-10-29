#!/usr/bin/env python3
"""
peak_frequencies.py

Vibration peak frequencies of the experiment runs, against motor angle and
against the flow, to tell structural resonances (fixed in Hz, near the
ring-down modes) from flow-driven ones (fixed in Strouhal number).

Reads the per-angle accelerometer PSDs written by extract_run_summary.py
(version 1.3+, which places samples on the ODR grid; older summaries put
every frequency ~22% low and are refused). No raw data is read.

Usage (from the repo root):
    python scripts/data_analysis/experiments/peak_frequencies.py Production_summaries --run all
    python scripts/data_analysis/experiments/peak_frequencies.py Production_summaries --run 704/800 \\
        --sensor S4 --component z --modes 62.5 176 282
    python scripts/data_analysis/experiments/peak_frequencies.py Production_summaries --run 704psi --at 170

--run takes folder names, short keys PSI/FAN[/VERSION], substrings, or all
(same rules as plot_angle_sweep.py).

Per angle, the PSD peaks are picked with accel_spectra.top_peaks (log PSD
smoothed over --smooth-hz, peaks at least --min-prom decades above their
surroundings) and the batch-gap copies of stronger peaks (+/- n x the
read-cycle rate, ~15 Hz) are dropped, keeping the --npeaks highest.

Outputs, in <summaries>/figures/peak_frequencies/ (or --out):
    peaks_<run>_<S>_<comp>.png   top: normalized variance vs motor angle;
                                 bottom: peak frequencies vs motor angle,
                                 coloured by PSD, with --modes as dashed lines.
    peaks_vs_flow_<S>_<comp>.png one point per run and peak rank at that
                                 run's variance peak (or --at angle):
                                 frequency (Hz) and Strouhal number vs U.
    peak_frequencies_<S>_<comp>.csv  every picked peak: run, angle, rank,
                                 freq_hz, strouhal, psd_g2_hz, flow conditions.

Options
    --sensor S4            accelerometer (default S4)
    --component {z,y,x,mag}   sensor axis or |a| (default z)
    --npeaks N             peaks kept per angle (default 3)
    --fmin/--fmax HZ       frequency band searched (default 5-800)
    --modes HZ [...]       ring-down mode frequencies drawn as reference lines
    --at {peak,ANGLE}      angle used per run in the flow figure (default peak)
    --chord M              average chord for St = f c / U (default: run_info.json, else 0.0138)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LogNorm                             # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "aquire_data"))
from accel_spectra import top_peaks, drop_sidebands               # noqa: E402
from plot_angle_sweep import (INK, MUTED, CAT, resolve, short_name, slug,  # noqa: E402
                              parse_sensors, style_axes, conditions_line, save)
from plot_psd import load_run                                     # noqa: E402

MIN_EXTRACTOR_VERSION = (1, 3)
DEFAULT_CHORD_M = 0.0138
CMAP = "viridis"
RANK_SIZES = [46, 22, 12, 8, 6]


# --------------------------------------------------
# Data
# --------------------------------------------------

def run_info(summ, run):
    info = json.loads((summ / run / "run_info.json").read_text())
    ver = tuple(int(v) for v in str(info.get("extractor_version", "0")).split("."))
    if ver < MIN_EXTRACTOR_VERSION:
        raise SystemExit(f"{run}: summary is extractor version {info.get('extractor_version')}, whose accel "
                         "spectra assume evenly spaced samples (~22% low). Rerun extract_run_summary.py "
                         "with --force.")
    ang = pd.read_csv(summ / run / "angle_summary.csv")
    info["accel_batch_hz"] = float(ang["accel_batch_hz"].median())
    return info


def pick_peaks(f, p, f_batch, args):
    """Indices of the args.npeaks highest peaks in [fmin, fmax], batch-gap copies removed."""
    idx = [i for i in top_peaks(f, p, fmin=args.fmin, smooth_hz=args.smooth_hz, min_prom=args.min_prom)
           if f[i] <= args.fmax]
    return drop_sidebands(f, p, idx, f_batch)[:args.npeaks]


def peak_table(run, D, info, args):
    chord = args.chord or info.get("chord_avg_m") or DEFAULT_CHORD_M
    U = info.get("velocity_m_s", np.nan)
    rows = []
    for k, th in enumerate(D["th"]):
        for rank, i in enumerate(pick_peaks(D["f"], D["P"][k], info["accel_batch_hz"], args), 1):
            rows.append(dict(run_name=run, motor_angle_deg=th, nvar=D["nv"][k], clipped=bool(D["clip"][k]),
                             rank=rank, freq_hz=D["f"][i], psd_g2_hz=D["P"][k][i],
                             strouhal=D["f"][i] * chord / U if U else np.nan,
                             velocity_m_s=U, reynolds_number=info.get("reynolds_number"),
                             cauchy_edgewise=info.get("cauchy_edgewise"),
                             accel_batch_hz=info["accel_batch_hz"]))
    return pd.DataFrame(rows)


def angle_for_run(T, at):
    """Motor angle used for a run in the flow figure: its largest unclipped variance, or the
    measured angle nearest `at`."""
    ok = T[~T["clipped"]]
    if ok.empty:
        return np.nan
    if at == "peak":
        return ok.loc[ok["nvar"].idxmax(), "motor_angle_deg"]
    th = ok["motor_angle_deg"].unique()
    return th[np.argmin(np.abs(th - float(at)))]


# --------------------------------------------------
# Figures
# --------------------------------------------------

def draw_modes(ax, modes, horizontal=True):
    for m in modes:
        (ax.axhline if horizontal else ax.axvline)(m, color=MUTED, ls="--", lw=0.9, zorder=1)


def fig_run(run, D, T, info, args, png):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 2.2]))
    for ax in (a1, a2):
        style_axes(ax)
    a1.semilogy(D["th"], D["nv"], "-o", ms=3, lw=1, color=CAT[0])
    if D["clip"].any():
        a1.plot(D["th"][D["clip"]], D["nv"][D["clip"]], "x", color=CAT[7], ms=7, label="clipped")
        a1.legend(frameon=False, fontsize=9)
    a1.set_ylabel(r"Var / $a_c^2$", color=INK)

    draw_modes(a2, args.modes)
    norm = LogNorm(vmin=T["psd_g2_hz"].quantile(0.05), vmax=T["psd_g2_hz"].max())
    for rank in sorted(T["rank"].unique(), reverse=True):
        t = T[T["rank"] == rank]
        sc = a2.scatter(t["motor_angle_deg"], t["freq_hz"], c=t["psd_g2_hz"], norm=norm, cmap=CMAP,
                        s=RANK_SIZES[min(rank, len(RANK_SIZES)) - 1], edgecolors="none", zorder=3)
    cb = fig.colorbar(sc, ax=[a1, a2], pad=0.01, fraction=0.03)
    cb.set_label(r"PSD at peak $(g^2/\mathrm{Hz})$", color=INK)
    a2.set_ylim(args.fmin, args.fmax)
    a2.set_xlabel("Motor angle (deg)", color=INK)
    a2.set_ylabel("Peak frequency (Hz)", color=INK)
    sizes = [Line2D([], [], ls="", marker="o", color=MUTED, ms=np.sqrt(RANK_SIZES[r]), label=f"peak {r + 1}")
             for r in range(args.npeaks)]
    if args.modes:
        sizes.append(Line2D([], [], color=MUTED, ls="--", label="ring-down modes"))
    a2.legend(handles=sizes, frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"{conditions_line(run, info)}\nS{args.sensor_id} {args.component}  ·  "
                 f"batch-gap copies (±n × {info['accel_batch_hz']:.1f} Hz) removed",
                 fontsize=10, color=INK)
    save(fig, png, args.dpi)


def fig_flow(T, args, png):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax in (a1, a2):
        style_axes(ax)
    draw_modes(a1, args.modes)
    for rank in range(1, args.npeaks + 1):
        t = T[T["rank"] == rank]
        kw = dict(s=RANK_SIZES[min(rank, len(RANK_SIZES)) - 1] * 1.2, color=CAT[rank - 1], zorder=3,
                  label=f"peak {rank}")
        a1.scatter(t["velocity_m_s"], t["freq_hz"], **kw)
        a2.scatter(t["velocity_m_s"], t["strouhal"], **kw)
    if args.labels:
        for _, r in T[T["rank"] == 1].iterrows():
            a1.annotate(short_name(r["run_name"]), (r["velocity_m_s"], r["freq_hz"]), fontsize=7,
                        color=MUTED, xytext=(3, 3), textcoords="offset points")
    a1.set_ylabel("Peak frequency (Hz)", color=INK)
    a2.set_ylabel(r"Strouhal number $f\,c/U$", color=INK)
    for ax in (a1, a2):
        ax.set_xlabel(r"Free-stream velocity $U$ (m/s)", color=INK)
    a1.legend(frameon=False, fontsize=9)
    at = "each run's variance peak" if args.at == "peak" else f"motor angle nearest {float(args.at):g} deg"
    fig.suptitle(f"S{args.sensor_id} {args.component} peaks at {at}.  "
                 "Structural: flat in Hz (left).  Flow-driven: flat in St (right).", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, png, args.dpi)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", help="Production_summaries folder (or the master_angles.csv inside it)")
    ap.add_argument("--run", nargs="+", default=["all"])
    ap.add_argument("--sensor", default="S4")
    ap.add_argument("--component", choices=["z", "y", "x", "mag"], default="z")
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=800.0)
    ap.add_argument("--smooth-hz", type=float, default=3.0)
    ap.add_argument("--min-prom", type=float, default=0.3, help="peak prominence, decades (default 0.3)")
    ap.add_argument("--modes", type=float, nargs="*", default=[])
    ap.add_argument("--at", default="peak")
    ap.add_argument("--chord", type=float, default=None)
    ap.add_argument("--labels", action="store_true", help="label runs in the flow figure")
    ap.add_argument("--out")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    summ = Path(args.summaries)
    summ = summ.parent if summ.is_file() else summ
    runs_csv = summ / "master_runs.csv"
    if not runs_csv.exists():
        raise SystemExit(f"{runs_csv} not found; give the Production_summaries folder")
    sensors = parse_sensors([args.sensor])
    if len(sensors) != 1:
        raise SystemExit("--sensor takes one accelerometer (S1..S4)")
    args.sensor_id = sensors[0]
    out = Path(args.out) if args.out else summ / "figures" / "peak_frequencies"
    tag = f"S{args.sensor_id}_{args.component}"

    targets, _ = resolve(sorted(pd.read_csv(runs_csv)["run_name"]), args.run)
    tables, flow = [], []
    for run in targets:
        info = run_info(summ, run)
        D = load_run(summ, run, args.sensor_id, args.component)
        T = peak_table(run, D, info, args)
        if T.empty:
            print(f"{run}: no peaks found")
            continue
        tables.append(T)
        fig_run(run, D, T, info, args, out / f"peaks_{slug(run)}_{tag}.png")
        th = angle_for_run(T, args.at)
        flow.append(T[T["motor_angle_deg"] == th])
        print(f"{run}: {T['motor_angle_deg'].nunique()} angles; at {th:+.1f} deg peaks "
              + ", ".join(f"{f:.1f}" for f in flow[-1].sort_values("rank")["freq_hz"]) + " Hz")
    if not tables:
        raise SystemExit("No peaks found in any run")
    pd.concat(tables, ignore_index=True).to_csv(out / f"peak_frequencies_{tag}.csv", index=False)
    fig_flow(pd.concat(flow, ignore_index=True), args, out / f"peaks_vs_flow_{tag}.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
