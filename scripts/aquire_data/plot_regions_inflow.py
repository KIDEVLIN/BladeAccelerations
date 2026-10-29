#!/usr/bin/env python3
"""
plot_regions_inflow.py

Where the vibration regions of angle_metrics.py (motor-angle windows, default 120, 170 and 210 deg,
each +-15 deg) fall in the blade's own inflow frame: angle of attack against inclination angle.
The thin line is the full motor sweep of the experiment mount (coordinate_transforms.compute_frame_angles);
the regions are drawn over it and numbered 1, 2, 3 in order of motor angle.

    python plot_regions_inflow.py
    python plot_regions_inflow.py --out ../../data/Production_summaries/figures
    python plot_regions_inflow.py --regions 120 170 210 --halfwidth 15 --sweep 30 --mounting 5

Defaults follow tests/map_angles.py: sweep +30 deg and mounting angle (AoA at motor 0 deg, the convention of
coordinate_transforms) 5 deg. run_metadata.json records sweep -30 and mounting 84.105 deg in a different
convention; taken at face value they mirror the inclination and put motor 0 deg at AoA 84 deg. AoA runs from
-180 to 180 deg, so a region through reversed flow is drawn in two pieces, at the top and bottom. Writes
regions_inflow_aoa.png and a CSV with each region's motor, inclination and AoA ranges.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import coordinate_transforms as ct                     # noqa: E402
from angle_metrics import DEFAULT_REGIONS, DEFAULT_HALFWIDTH_DEG  # noqa: E402
from plot_angle_sweep import INK, MUTED, SURFACE, CAT, style_axes  # noqa: E402

# Font sizes (pt), large enough to read when the figure is shrunk onto a slide.
FS_TICK, FS_LABEL, FS_REGION = 22, 24, 36


AOA_MIN_DEG = -180.0                # AoA axis runs AOA_MIN_DEG .. AOA_MIN_DEG + 360


def frame(motor, sweep, mounting):
    """(inclination, AoA) in deg for an array of motor angles; inclination in [-180, 180), AoA in
    [AOA_MIN_DEG, AOA_MIN_DEG + 360)."""
    a = [ct.compute_frame_angles(float(m), sweep, mounting) for m in motor]
    aoa = np.array([x["angle_of_attack_deg"] for x in a])
    return (np.array([x["inclination_angle_deg_shifted"] for x in a]),
            (aoa - AOA_MIN_DEG) % 360 + AOA_MIN_DEG)


def plot_path(ax, x, y, **kw):
    """Line through (x, y), broken where the AoA wraps around the axis limits."""
    breaks = np.flatnonzero(np.abs(np.diff(y)) > 180) + 1
    for seg in np.split(np.arange(len(x)), breaks):
        ax.plot(x[seg], y[seg], **kw)


def upward_normal(ax, x, y, i):
    """Unit normal to the curve (x, y) at index i, in display space, on the upper side (or the outer
    side, away from inclination 0, where the curve is near vertical). Axis limits must be set first."""
    k = slice(max(i - 4, 0), i + 5)
    p = ax.transData.transform(np.c_[x[k], y[k]])
    t = p[-1] - p[0]
    nrm = np.array([-t[1], t[0]]) / np.hypot(*t)
    if abs(nrm[1]) < 0.3:
        return nrm if np.sign(nrm[0]) == np.sign(x[i] or 1) else -nrm
    return nrm if nrm[1] > 0 else -nrm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", type=float, default=[float(r) for r in DEFAULT_REGIONS])
    ap.add_argument("--halfwidth", type=float, default=DEFAULT_HALFWIDTH_DEG)
    ap.add_argument("--sweep", type=float, default=30.0, help="default +30, as in tests/map_angles.py")
    ap.add_argument("--mounting", type=float, default=5.0, help="AoA at motor 0 deg (default 5, as in "
                                                                 "tests/map_angles.py)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "Production_summaries"
                                         / "figures"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    motor = np.arange(-180.0, 180.0 + 0.25, 0.25)
    inc, aoa = frame(motor, args.sweep, args.mounting)

    fig, ax = plt.subplots(figsize=(8, 6.6), facecolor=SURFACE, constrained_layout=True)
    ax.set_xlim(-45, 45)
    ax.set_ylim(AOA_MIN_DEG - 20, AOA_MIN_DEG + 380)    # a little room beyond the +-180 ticks
    plot_path(ax, inc, aoa, color=MUTED, lw=2.5, zorder=2)

    rows = []
    for n, (c, color) in enumerate(zip(sorted(args.regions), CAT), 1):
        lo, hi = c - args.halfwidth, c + args.halfwidth
        m = np.arange(lo, hi + 0.25, 0.25)
        x, y = frame(m, args.sweep, args.mounting)
        plot_path(ax, x, y, color=color, lw=10, solid_capstyle="butt", alpha=0.9, zorder=4)
        # Number on the middle of the longest piece (a region through +-180 is drawn in two), above the
        # curve unless that runs off the top.
        piece = max(np.split(np.arange(len(y)), np.flatnonzero(np.abs(np.diff(y)) > 180) + 1), key=len)
        j = int(piece[len(piece) // 2])
        nrm = upward_normal(ax, x, y, j)
        if y[j] > AOA_MIN_DEG + 330:
            nrm = -nrm
        ax.annotate(f"R{n}", (x[j], y[j]), xytext=tuple(40 * nrm), textcoords="offset points",
                    ha="center", va="center", fontsize=FS_REGION, fontweight="bold", color=color, zorder=5)
        rows.append(dict(region=n, center_motor_deg=c, motor_lo_deg=lo, motor_hi_deg=hi,
                         inclination_min_deg=x.min(), inclination_max_deg=x.max(),
                         aoa_at_motor_lo_deg=y[0], aoa_at_motor_hi_deg=y[-1]))

    ax.set_xlabel("Inclination angle (deg)", color=INK, fontsize=FS_LABEL)
    ax.set_ylabel("Angle of attack (deg)", color=INK, fontsize=FS_LABEL)
    ax.set_yticks(np.arange(AOA_MIN_DEG, AOA_MIN_DEG + 361, 90))
    ax.set_xticks(np.arange(-30, 31, 30))
    style_axes(ax)
    ax.grid(False)
    ax.tick_params(labelsize=FS_TICK, labelcolor=INK)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "regions_inflow_aoa.png"
    fig.savefig(png, dpi=args.dpi, facecolor=SURFACE)
    pd.DataFrame(rows).round(2).to_csv(png.with_suffix(".csv"), index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print(f"Wrote {png}  (+ .csv)")


if __name__ == "__main__":
    main()
