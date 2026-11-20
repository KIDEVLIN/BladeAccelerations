#!/usr/bin/env python3
"""
make_figures.py

Figures and numbers for angle_check.tex. Run from the repo root:
    poetry run python reports/angle_check/make_figures.py [--before DIR]

Compares three versions of scripts/aquire_data/utils/coordinate_transforms.py,
loaded from git so the comparison is reproducible:
    1 Oct  commit 2965360 (the version that wrote the summaries)
    master commit 3e181e2 (the collaborator's original, origin/master)
    fixed  the working tree
against an independent vector construction of the rig (the reference in
tests/test_coordinate_transforms.py) and the experimentalists' sketch.

With --before pointing at a copy of data/Production_summaries made before
regenerating them, also checks that only the angle columns changed.
"""

import argparse
import importlib.util
import json
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "scripts" / "aquire_data" / "tests"
sys.path.insert(0, str(TESTS))
import render_rig as RR  # noqa: E402
from test_coordinate_transforms import rig_reference, angdiff  # noqa: E402
from test_angle_chain import production_runs, load_encoder_log, windowed_average_angle  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
SWEEP, MOUNTING = -30.0, 84.105
VERSIONS = {"1 Oct (2965360)": "2965360", "master (3e181e2)": "3e181e2", "fixed": None}
STYLE = {"1 Oct (2965360)": dict(color="#d1495b", ls="--", lw=1.6),
         "master (3e181e2)": dict(color="#2a6fdb", ls="-", lw=4.0, alpha=0.35),
         "fixed": dict(color="k", ls="-", lw=1.2)}
REGIONS = {"R1": (105, 135), "R2": (155, 185), "R3": (195, 225)}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "legend.frameon": False})


def load(commit):
    path = "scripts/aquire_data/utils/coordinate_transforms.py"
    code = (ROOT / path).read_text() if commit is None else subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader(f"ct_{commit}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(code, mod.__dict__)
    return mod


def angles(mod, motor, sweep=SWEEP, mounting=MOUNTING, U=(1.0, 0, 0)):
    r = mod.compute_frame_angles(motor, sweep, mounting, U_global=U) if "U_global" in \
        mod.compute_frame_angles.__code__.co_varnames else mod.compute_frame_angles(motor, sweep, mounting)
    return r["inclination_angle_deg_shifted"], r["angle_of_attack_deg_shifted"]


def broken(x, y, jump=180):
    """Insert NaNs where a wrapped angle jumps, so lines do not cross the plot."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = np.flatnonzero(np.abs(np.diff(y)) > jump) + 1
    return np.insert(x, k, np.nan), np.insert(y, k, np.nan)


# ---------------------------------------------------------------- sketch vs geometry
def fig_sketch(mods):
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.25], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    img = plt.imread(OUT / "sketch.jpg")
    h, w = img.shape[:2]
    ax.imshow(img[int(0.17 * h):int(0.78 * h), int(0.08 * w):int(0.80 * w)])
    ax.axis("off")
    ax.set_title("(a) Experimentalists' sketch, from above", loc="left", fontsize=10)
    sub = gs[1].subgridspec(2, 2, wspace=0.05, hspace=0.2)
    for i, m in enumerate((0.0, 90.0, -90.0, 120.0)):
        a = fig.add_subplot(sub[i // 2, i % 2])
        RR.draw_top(a, m, abs(SWEEP), 90.0 - MOUNTING, highlight=(m == 120.0))
        ic, ac = angles(mods["fixed"], m)
        a.text(-1.75, 1.75, f"code: AoA {ac:+.1f}°, incl {ic:+.1f}°", fontsize=7, va="top", color="0.35")
    fig.text(0.42, 0.93, "(b) The same cases drawn from the rig geometry; bottom right: the code's values agree",
             fontsize=10)
    fig.savefig(OUT / "sketch_vs_geometry.pdf", bbox_inches="tight")
    print("sketch cases (fixed code):", {m: tuple(round(v, 2) for v in angles(mods['fixed'], m)) for m in (0, 90, -90, 120)})


# ---------------------------------------------------------------- rig in 3D
def fig_rig3d():
    fig = plt.figure(figsize=(11, 3.4))
    for i, m in enumerate((0.0, 90.0, -90.0, 120.0)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        RR.draw_3d(ax, RR.rig_vectors(m, abs(SWEEP), 90.0 - MOUNTING), m)
        ax.set_title(f"motor {m:+g}°", loc="left", fontsize=10)
    fig.subplots_adjust(wspace=0.0, left=0, right=1)
    fig.savefig(OUT / "rig3d.pdf", bbox_inches="tight")


# ---------------------------------------------------------------- angles vs motor
def fig_vs_motor(mods):
    motor = np.arange(-180.0, 180.01, 1.0)
    ref = np.array([rig_reference(m, SWEEP, MOUNTING) for m in motor])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)
    for name, mod in mods.items():
        A = np.array([angles(mod, m) for m in motor])
        axes[0].plot(motor, A[:, 0], label=name, **STYLE[name])
        axes[1].plot(*broken(motor, A[:, 1]), label=name, **STYLE[name])
    mk = motor[::15]
    axes[0].plot(mk, ref[::15, 0], "o", ms=4, mfc="none", color="#1b9e77", label="vector reference")
    axes[1].plot(mk, ref[::15, 1], "o", ms=4, mfc="none", color="#1b9e77", label="vector reference")
    for m, inc in ((0, 0), (90, 30), (-90, -30)):
        axes[0].plot(m, inc, marker="*", ms=12, color="#f2a900", zorder=5, mec="k", mew=0.5,
                     label="sketch" if m == 0 else None)
    for ax in axes:
        ax.axvline(120, color="0.5", lw=0.7, ls=":")
        for lo, hi in REGIONS.values():
            for a, b in ((lo, hi), (lo - 360, hi - 360)):
                if b > -180 and a < 180:
                    ax.axvspan(max(a, -180), min(b, 180), color="0.94", zorder=0)
    axes[0].set_ylabel("Inclination (deg)")
    axes[1].set_ylabel("Angle of attack (deg)")
    axes[1].set_xlabel("Motor angle (deg)")
    axes[1].set_xticks(np.arange(-180, 181, 45))
    axes[0].set_yticks(np.arange(-30, 31, 15))
    axes[1].set_yticks(np.arange(-180, 181, 90))
    axes[0].legend(ncol=3, fontsize=7.5, loc="upper left")
    axes[0].text(121, 26, "120°", fontsize=7.5, color="0.4")
    fig.savefig(OUT / "angles_vs_motor.pdf")
    for name, mod in mods.items():
        i, a = angles(mod, 120.0)
        print(f"motor 120, {name:18s}: incl {i:+8.3f}  AoA {a:+8.3f}")


# ---------------------------------------------------------------- agreement
def agreement(mods):
    """Worst differences over a grid of motor x sweep x mounting (x flow)."""
    motor = np.arange(-180.0, 180.0, 2.5)
    sweeps = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
    mounts = [60.0, 75.0, 84.105, 90.0, 105.0]
    flows = [(1.0, 0, 0), (1.0, 0.3, -0.4), (0.2, -0.7, 0.5)]
    rows = []
    for name, mod in mods.items():
        for U in flows:
            if name != "fixed" and U != (1.0, 0, 0):
                continue
            wi = wa = 0.0
            for m, s, mt in itertools.product(motor, sweeps, mounts):
                i1, a1 = angles(mod, m, s, mt, U)
                i2, a2 = rig_reference(m, s, mt, U)
                wi = max(wi, abs(i1 - i2))
                if abs(i2) < 89.9:
                    wa = max(wa, abs(angdiff(a1, a2)))
            rows.append((name, U, wi, wa))
    n = len(motor) * len(sweeps) * len(mounts)
    print(f"agreement with the vector reference ({n} conditions per flow):")
    for name, U, wi, wa in rows:
        print(f"  {name:18s} flow {str(U):18s} max |d incl| {wi:9.2e}  max |d AoA| {wa:9.2e}")
    return rows, n


def fig_error(mods):
    """Difference from the vector reference along the production sweep."""
    motor = np.arange(-180.0, 180.01, 1.0)
    ref = np.array([rig_reference(m, SWEEP, MOUNTING) for m in motor])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
    for name, mod in mods.items():
        A = np.array([angles(mod, m) for m in motor])
        di = np.abs(A[:, 0] - ref[:, 0])
        da = np.abs(angdiff(A[:, 1], ref[:, 1]))
        st = {k: v for k, v in STYLE[name].items() if k != "alpha"}
        st["lw"] = min(st["lw"], 1.6)
        axes[0].semilogy(motor, np.maximum(di, 1e-16), label=name, **st)
        axes[1].semilogy(motor, np.maximum(da, 1e-16), label=name, **st)
    axes[0].set_title("Inclination", loc="left", fontsize=9)
    axes[1].set_title("Angle of attack", loc="left", fontsize=9)
    axes[0].set_ylabel("|code − vector reference| (deg)")
    for ax in axes:
        ax.set_xlabel("Motor angle (deg)")
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_ylim(1e-16, 1e3)
    axes[1].legend(fontsize=7.5, loc="center right")
    fig.savefig(OUT / "error.pdf")


# ---------------------------------------------------------------- inflow map
def fig_map(mods):
    motor = np.arange(-180.0, 180.0, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, name in zip(axes, ("1 Oct (2965360)", "fixed")):
        A = np.array([angles(mods[name], m) for m in motor])
        sc = ax.scatter(A[:, 0], A[:, 1], c=motor, cmap="twilight_shifted", vmin=-180, vmax=180, s=10)
        for lab, (lo, hi) in REGIONS.items():
            mm = (((motor - lo) % 360) <= (hi - lo))
            ax.scatter(A[mm, 0], A[mm, 1], s=40, facecolors="none", edgecolors="k", lw=0.6)
            j = np.flatnonzero(mm)[len(np.flatnonzero(mm)) // 2]
            ax.annotate(lab, (A[j, 0], A[j, 1]), xytext=(6, 4), textcoords="offset points", fontsize=8)
        i, a = angles(mods[name], 120.0)
        ax.plot(i, a, "*", ms=13, color="#f2a900", mec="k", mew=0.6, zorder=5)
        ax.annotate("120°", (i, a), xytext=(6, -10), textcoords="offset points", fontsize=8)
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_xlabel("Inclination (deg)")
        ax.set_yticks(np.arange(-180, 181, 45))
    axes[0].set_ylabel("Angle of attack (deg)")
    fig.colorbar(sc, ax=axes, label="Motor angle (deg)", ticks=np.arange(-180, 181, 90))
    fig.savefig(OUT / "inflow_map.pdf")


# ---------------------------------------------------------------- the chain on the recorded data
def fig_chain(mods):
    """Every recorded angle of the 32 production runs: (a) the angles the lab
    PC computed during acquisition (master's code) against the fixed code,
    recomputed from the same motor angle; (b) the encoder-window motor angle
    of run_loader against the one main.py recorded; (c) the recorded motor
    angle against the requested one."""
    fixed = mods["fixed"]
    rows = []
    for d in production_runs():
        md = json.loads((d / "run_metadata.json").read_text())
        f = pd.read_csv(d / "coordinate_frame_angles.csv")
        enc_df, enc_start = (load_encoder_log(d / "encoder_log.txt") if (d / "encoder_log.txt").exists()
                             else (None, None))
        for _, r in f.iterrows():
            a = fixed.compute_frame_angles(r.motor_angle_actual_deg, r.sweep_angle_deg, r.mounting_angle_deg)
            em = None
            if enc_start is not None:
                em, _, _ = windowed_average_angle(enc_df, enc_start, r.accel_t_start,
                                                  md["sample_duration_s"], md["counts_per_deg"])
            rows.append(dict(run=d.name, req=r.motor_angle_requested_deg, act=r.motor_angle_actual_deg,
                             di=abs(a["inclination_angle_deg_shifted"] - r.inclination_angle_deg_shifted),
                             da=abs(float(angdiff(a["angle_of_attack_deg"], r.angle_of_attack_deg))),
                             enc=np.nan if em is None else em - r.motor_angle_actual_deg))
    T = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), constrained_layout=True)
    ax = axes[0]
    ax.semilogy(T.req, np.maximum(T.di, 1e-16), "o", ms=2.5, color="k", label="inclination")
    ax.semilogy(T.req, np.maximum(T.da, 1e-16), "s", ms=2.5, mfc="none", color="#2a6fdb", label="angle of attack")
    ax.set_ylim(1e-16, 1e2)
    ax.set_ylabel("|lab PC − fixed code| (deg)")
    ax.set_title("(a) Angles computed during acquisition", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    ax.plot(T.req, T.enc * 1e3, "o", ms=2.5, color="k")
    ax.set_ylabel("encoder window − recorded (mdeg)")
    ax.set_title("(b) Motor angle from the encoder log", loc="left", fontsize=9)
    ax = axes[2]
    ax.plot(T.req, (T.act - T.req) * 1e3, "o", ms=2.5, color="k")
    ax.set_ylabel("recorded − requested (mdeg)")
    ax.set_title("(c) Motor angle reached", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Requested motor angle (deg)")
        ax.set_xticks(np.arange(-90, 271, 90))
    fig.savefig(OUT / "chain.pdf")
    print(f"chain: {len(T)} recorded angles in {T.run.nunique()} runs")
    print(f"  lab PC vs fixed: max |d incl| {T.di.max():.1e}, max |d AoA| {T.da.max():.1e}")
    e = T.enc.dropna()
    print(f"  encoder window vs recorded: {len(e)} angles, max |d| {e.abs().max() * 1e3:.2f} mdeg, "
          f"median {e.abs().median() * 1e3:.3f} mdeg")
    print(f"  recorded vs requested: max |d| {(T.act - T.req).abs().max() * 1e3:.2f} mdeg")


# ---------------------------------------------------------------- summaries before / after
def check_summaries(before):
    after = ROOT / "data" / "Production_summaries"
    n_runs = n_rows = 0
    worst_i = worst_a = 0.0
    other_changed = set()
    for d in sorted(p for p in after.iterdir() if p.is_dir() and (p / "angle_summary.csv").exists()):
        old = before / d.name / "angle_summary.csv"
        if not old.exists():
            continue
        A, B = pd.read_csv(old), pd.read_csv(d / "angle_summary.csv")
        assert len(A) == len(B), d.name
        n_runs += 1
        n_rows += len(A)
        worst_i = max(worst_i, np.nanmax(np.abs(B.inclination_deg + A.inclination_deg)))
        worst_a = max(worst_a, np.nanmax(np.abs(angdiff(A.aoa_deg - B.aoa_deg, 2 * MOUNTING - 90))))
        for c in A.columns:
            if c in ("inclination_deg", "aoa_deg"):
                continue
            x, y = A[c], B[c]
            if x.dtype.kind in "fi" and y.dtype.kind in "fi":
                same = np.allclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True, rtol=0, atol=0)
            else:
                same = x.astype(str).equals(y.astype(str))
            if not same:
                other_changed.add(c)
    print(f"summaries: {n_runs} runs, {n_rows} angles")
    print(f"  max |incl_new + incl_old| = {worst_i:.2e} (sign flip)")
    print(f"  max |(aoa_old - aoa_new) - {2 * MOUNTING - 90:.3f}| = {worst_a:.2e} (constant shift)")
    print(f"  other columns that changed: {sorted(other_changed) or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", type=Path, help="copy of data/Production_summaries before regenerating")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    mods = {k: load(v) for k, v in VERSIONS.items()}
    fig_sketch(mods)
    fig_rig3d()
    fig_vs_motor(mods)
    fig_error(mods)
    fig_map(mods)
    agreement(mods)
    fig_chain(mods)
    if args.before:
        check_summaries(args.before)


if __name__ == "__main__":
    main()
