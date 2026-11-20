#!/usr/bin/env python3
"""
make_figures.py

Numbers and the summary figure for psd_analysis.tex. Run from the repo root:
    python3 reports/psd_analysis/make_figures.py

Reads the per-run peak tables that plot_psd.py wrote next to the PSD figures
(figures/psd_analysis/R*_*_frequency.csv for S1, figures/psd_analysis/S4/ for S4)
and master_runs.csv for the flow conditions. Writes figs/peak_vs_speed.pdf and
prints every number quoted in the report. Only runs flagged in a region (a
vibration peak >= 10x the run's baseline inside the window) are used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "figures" / "psd_analysis"
SENSOR_DIRS = {"S1": FIGS, "S4": FIGS / "S4"}
RUNS = ROOT / "data" / "Production_summaries" / "master_runs.csv"
OUT = Path(__file__).resolve().parent / "figs"
CHORD_M = 0.0138                       # c-bar used for St in plot_psd.py
REGIONS = {"R1": "105--135", "R2": "155--185", "R3": "195--225"}
DIRS = ("edgewise", "flapwise")
STYLE = {"R1": ("C0", "o"), "R2": ("C1", "s"), "R3": ("C2", "^")}
HIGH_HZ = 250                          # R2 peaks above this form their own group (Results)


def load(sensor):
    """Flagged runs only: one row per (region, direction, run) with the run's flow conditions."""
    mr = pd.read_csv(RUNS).set_index("run_name")
    rows = []
    for reg in REGIONS:
        for d in DIRS:
            t = pd.read_csv(SENSOR_DIRS[sensor] / f"{reg}_{d}_frequency.csv")
            t = t[t["flag"] == True].copy()                                   # noqa: E712
            t["region"], t["direction"] = reg, d
            for col, key in (("U", "velocity_m_s"), ("rho", "density_kg_m3"), ("Re", "reynolds_number"),
                             ("Ca", "cauchy_edgewise")):
                t[col] = mr.loc[t["run_name"], key].to_numpy()
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def summary(df, sensor):
    print(f"\n== {sensor}: flagged runs, peak frequency (Hz) and Strouhal number")
    print("region dir       n   f median [min-max] (IQR)        St median [min-max]   rho_s(U,f)  rho_s(U,St)")
    for (reg, d), t in df.groupby(["region", "direction"]):
        f, st = t["peak_hz"], t["peak_St"]
        rf = spearmanr(t["U"], f)[0] if len(t) > 2 else np.nan
        rs = spearmanr(t["U"], st)[0] if len(t) > 2 else np.nan
        print(f"{reg}     {d:9s} {len(t):2d}   {f.median():5.0f} [{f.min():3.0f}-{f.max():3.0f}] "
              f"({f.quantile(.25):3.0f}-{f.quantile(.75):3.0f})      {st.median():.2f} [{st.min():.2f}-{st.max():.2f}]"
              f"        {rf:+.2f}       {rs:+.2f}")


def groups(df):
    """The groupings the Results section describes, for S4."""
    print("\n== S4 groups")
    e1 = df[(df.region == "R1") & (df.direction == "edgewise") & (df.peak_hz < 200)]
    hi_p = e1[e1.rho < 100]
    print(f"R1 edgewise, below 1414 psi: {len(hi_p)} runs, f = {hi_p.peak_hz.mean():.0f} +- {hi_p.peak_hz.std():.0f} Hz "
          f"(range {hi_p.peak_hz.min():.0f}-{hi_p.peak_hz.max():.0f}) over U {hi_p.U.min():.1f}-{hi_p.U.max():.1f} m/s, "
          f"St {hi_p.peak_St.min():.2f}-{hi_p.peak_St.max():.2f}")
    lo = e1[e1.rho >= 100]
    print(f"R1 edgewise, 1414 psi: f = {', '.join(f'{v:.0f}' for v in lo.peak_hz)} Hz")
    f1 = df[(df.region == "R1") & (df.direction == "flapwise")]
    for name, sel in (("130-150 Hz", (f1.peak_hz >= 125) & (f1.peak_hz <= 150)),
                      ("< 125 Hz", f1.peak_hz < 125), ("> 200 Hz", f1.peak_hz > 200)):
        t = f1[sel]
        print(f"R1 flapwise {name}: {len(t)} runs: " +
              ", ".join(f"{r[8:]} ({u:.1f} m/s, {f:.0f} Hz)" for r, u, f in zip(t.run_name, t.U, t.peak_hz)))
    r2 = df[df.region == "R2"]
    lowr2 = r2[r2.peak_hz < HIGH_HZ]
    print(f"R2 below {HIGH_HZ} Hz: {len(lowr2)} peaks, St = {lowr2.peak_St.median():.2f} "
          f"(range {lowr2.peak_St.min():.2f}-{lowr2.peak_St.max():.2f}), f {lowr2.peak_hz.min():.0f}-{lowr2.peak_hz.max():.0f} Hz, "
          f"U {lowr2.U.min():.1f}-{lowr2.U.max():.1f} m/s; spearman(U, f) = {spearmanr(lowr2.U, lowr2.peak_hz)[0]:+.2f}")
    fit = np.polyfit(lowr2.U, lowr2.peak_hz, 1)
    print(f"   straight-line fit f = {fit[0]:.1f} U {fit[1]:+.1f}  (f = St U / c with St = 0.40 has slope {0.40 / CHORD_M:.1f})")
    hi = r2[r2.peak_hz >= HIGH_HZ]
    print(f"R2 above {HIGH_HZ} Hz: " + ", ".join(f"{r[8:]} {d[0]} ({u:.1f} m/s, {f:.0f} Hz, St {s:.2f})"
                                          for r, d, u, f, s in zip(hi.run_name, hi.direction, hi.U, hi.peak_hz, hi.peak_St)))
    r3 = df[df.region == "R3"]
    print(f"R3: {r3.run_name.nunique()} runs, f = {r3.peak_hz.mean():.0f} +- {r3.peak_hz.std():.0f} Hz "
          f"(range {r3.peak_hz.min():.0f}-{r3.peak_hz.max():.0f}) over U {r3.U.min():.1f}-{r3.U.max():.1f} m/s, "
          f"St {r3.peak_St.min():.2f}-{r3.peak_St.max():.2f}")
    same = (df.pivot_table(index=["region", "run_name"], columns="direction", values="peak_hz")
            .dropna().pipe(lambda p: (p.edgewise - p.flapwise).abs() < 3))
    print(f"edgewise and flapwise peaks within 3 Hz of each other: {int(same.sum())} of {len(same)} (region, run) pairs; "
          + ", ".join(f"{k}: {int(v.sum())}/{len(v)}" for k, v in same.groupby(level=0)))


def fig_peak_vs_speed(df):
    """Peak frequency vs U for each region, with constant-St lines f = St U / c."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True)
    u = np.linspace(2, 8.5, 50)
    for ax, d in zip(axes, DIRS):
        for st in (0.2, 0.4, 0.6):
            ax.plot(u, st * u / CHORD_M, color="0.6", ls="--", lw=0.9, zorder=1)
            ux = min(8.35, 0.97 * 320 * CHORD_M / st)          # label where the line leaves the plot
            ax.text(ux, st * ux / CHORD_M, f"St = {st:g}", fontsize=8, color="0.4", ha="right", va="bottom")
        for reg, (col, mk) in STYLE.items():
            t = df[(df.region == reg) & (df.direction == d)]
            ax.scatter(t.U, t.peak_hz, color=col, marker=mk, s=40, edgecolor="k", linewidth=0.5, zorder=3,
                       label=f"{reg} ({REGIONS[reg].replace('--', '–')}°)")
        ax.set_xlim(2, 8.5)
        ax.set_ylim(50, 320)
        ax.set_xlabel("Free-stream velocity $U$ (m/s)")
        ax.set_title(f"S4, {d}")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Peak frequency (Hz)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "peak_vs_speed.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = {s: load(s) for s in SENSOR_DIRS}
    for s, df in data.items():
        summary(df, s)
    groups(data["S4"])
    a, b = (data[s].groupby(["region", "direction"]).peak_hz.median() for s in ("S1", "S4"))
    print("\n== median peak frequency, S1 vs S4 (Hz):\n" + pd.DataFrame({"S1": a, "S4": b}).round(0).to_string())
    both = data["S1"].merge(data["S4"], on=["region", "direction", "run_name"], suffixes=("_1", "_4"))
    both["close"] = (both.peak_hz_1 - both.peak_hz_4).abs() < 5
    print(f"\n== runs flagged by both sensors: peak within 5 Hz in {int(both.close.sum())} of {len(both)}")
    for (reg, d), t in both.groupby(["region", "direction"]):
        off = t[~t.close]
        print(f"{reg} {d}: {int(t.close.sum())}/{len(t)}" + ("" if off.empty else "; differ: " + ", ".join(
            f"{r[8:]} S1 {f1:.0f} / S4 {f4:.0f}" for r, f1, f4 in zip(off.run_name, off.peak_hz_1, off.peak_hz_4))))
    clip = data["S4"].groupby("region").clipped.sum()
    print("\n== S4 flagged (region, direction) peaks taken at a clipped angle:", clip.to_dict())
    fig_peak_vs_speed(data["S4"])
    print(f"\nWrote {OUT / 'peak_vs_speed.pdf'}")


if __name__ == "__main__":
    sys.exit(main())
