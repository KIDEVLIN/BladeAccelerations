"""
plot_variance_compare.py

Overlay the variance diagnostics of several acquisition runs on the same
axes. One PNG per requested plot (see PLOT_NAMES):

    accel_sweep   accel magnitude variance (g^2) vs motor angle
    load_sweep    load-cell force-magnitude variance (N^2) vs motor angle
    accel_map     inclination vs angle of attack, colored by accel variance
    load_map      inclination vs angle of attack, colored by load-cell variance

Accel variance is in g^2 by default. If every run has a characteristic
acceleration (RunData.a_c_g, see normalization.py) the accel plots instead show
the nondimensional variance a*^2 = Var(a) / a_c^2, so runs at different tunnel
conditions / blades are comparable. Load-cell plots are never normalized.

Conventions
  - Sweep plots: one color per run. When several accelerometer series are
    drawn per run (sensors=[1, 2, 3, 4]), the sensor is distinguished by
    line style instead.
  - Map plots: one marker shape per run, with a single color scale shared
    across all runs so the colors are directly comparable.
  - Frame angles come from each run's own sweep / mounting angle
    (RunData.sweep_angle_deg / mounting_angle_deg, normally read from
    run_metadata.json), so runs with different blade setups land in the
    right place on the map.

`sensors` selects the accelerometer series (sensor numbers 1-4). The
default is sensor 4 only; sensors are never averaged together.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from variance_analysis import NUM_SENSORS, accel_per_sensor_variance, load_variance
from plot_variance_sweep import MG2_TO_G2, _frame_angles, _motor_angles

PLOT_NAMES = ("accel_sweep", "load_sweep", "accel_map", "load_map")

_LINESTYLES = ["-", "--", ":", "-."]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


@dataclass
class RunData:
    label: str
    captures: list
    sweep_angle_deg: Optional[float] = None
    mounting_angle_deg: Optional[float] = None
    a_c_g: Optional[float] = None   # characteristic acceleration (g); None = plot raw g^2


class _Prepared:
    """Per-run arrays, computed once and shared by every plot."""

    def __init__(self, run):
        caps = run.captures
        self.label = run.label
        self.motor = _motor_angles(caps)
        self.order = np.argsort(self.motor)
        self.incl, self.aoa = _frame_angles(
            caps, self.motor, run.sweep_angle_deg, run.mounting_angle_deg)
        # (n_angles, NUM_SENSORS), g^2 -- or a*^2 (dimensionless) when normalized
        self.accel = np.array([
            [v[s + 1] for s in range(NUM_SENSORS)]
            for v in (accel_per_sensor_variance(c.accel_df) for c in caps)
        ]) * MG2_TO_G2
        if run.a_c_g is not None:
            self.accel = self.accel / run.a_c_g ** 2
        self.load = np.array([
            np.nan if v is None else v
            for v in (load_variance(c.load_df) for c in caps)
        ])

    def accel_series(self, sensor):
        """sensor: 1-based sensor number -> (n_angles,) array."""
        return self.accel[:, sensor - 1]


def _sensor_name(sensor):
    return f"S{sensor}"


def _colors(n):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _accel_variance_label(normalized):
    return (r"Accel magnitude variance, $a^{*2}$ ($a^* = a/a_c$)" if normalized
            else r"Accel magnitude variance (g$^2$)")


def _plot_accel_sweep(runs, sensors, out_path, normalized=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, color in zip(runs, _colors(len(runs))):
        for k, sensor in enumerate(sensors):
            label = run.label if len(sensors) == 1 else f"{run.label} - {_sensor_name(sensor)}"
            ax.plot(run.motor[run.order], run.accel_series(sensor)[run.order],
                    marker="o", linestyle=_LINESTYLES[k % len(_LINESTYLES)],
                    color=color, label=label)
    ax.set_xlabel("Motor angle (deg)")
    ax.set_ylabel(_accel_variance_label(normalized))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_load_sweep(runs, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = 0
    for run, color in zip(runs, _colors(len(runs))):
        if np.all(np.isnan(run.load)):
            print(f"  NOTE: {run.label} has no load-cell data -- skipped in load_sweep")
            continue
        ax.plot(run.motor[run.order], run.load[run.order], marker="s",
                color=color, label=run.label)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xlabel("Motor angle (deg)")
    ax.set_ylabel("Load-cell magnitude variance (N$^2$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_map(runs, values_of, title, cbar_label, cmap, out_path):
    """values_of(run) -> (n_angles,) array of the variance to color by."""
    usable = []
    for run in runs:
        if run.incl is None:
            print(f"  NOTE: {run.label} has no frame angles (needs sweep/mounting "
                  f"angle) -- skipped in map")
            continue
        vals = values_of(run)
        if np.all(np.isnan(vals)):
            print(f"  NOTE: {run.label} has no data for this map -- skipped")
            continue
        usable.append((run, vals))
    if not usable:
        return False

    allvals = np.concatenate([v[~np.isnan(v)] for _, v in usable])
    lo, hi = float(allvals.min()), float(allvals.max())
    if hi <= lo:
        hi = lo + 1e-12

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = None
    for k, (run, vals) in enumerate(usable):
        sc = ax.scatter(run.incl, run.aoa, c=vals, cmap=cmap, vmin=lo, vmax=hi,
                        marker=_MARKERS[k % len(_MARKERS)], edgecolors="k",
                        linewidths=0.4, label=run.label)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Inclination angle (deg)")
    ax.set_ylabel("Angle of attack / pitch (deg)")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=cbar_label)
    # Legend shows marker shape per run only (colors encode variance, not the run)
    handles = [Line2D([], [], marker=_MARKERS[k % len(_MARKERS)], linestyle="",
                      markerfacecolor="lightgrey", markeredgecolor="k", label=run.label)
               for k, (run, _) in enumerate(usable)]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_variance_comparison(runs, plots, out_dir, sensors=(4,), map_sensor=4):
    """runs: list[RunData]. plots: iterable of names from PLOT_NAMES.
    sensors: accel series for accel_sweep (sensor numbers 1-4).
    map_sensor: 1-4, the sensor shown in accel_map.
    Returns the list of PNG paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_norm = sum(r.a_c_g is not None for r in runs)
    if 0 < n_norm < len(runs):
        raise ValueError(
            "Only some runs have a characteristic acceleration (a_c_g) -- normalized "
            "and raw variances can't share axes. Give every run one, or none.")
    normalized = n_norm == len(runs) and len(runs) > 0
    prepared = [_Prepared(r) for r in runs]
    written = []

    def done(name, ok):
        path = out_dir / f"{name}.png"
        if ok:
            written.append(path)
            print(f"  Saved {path}")
        else:
            print(f"  Skipped {name}: no usable data")

    for name in plots:
        path = out_dir / f"{name}.png"
        if name == "accel_sweep":
            _plot_accel_sweep(prepared, list(sensors), path, normalized=normalized)
            done(name, True)
        elif name == "load_sweep":
            done(name, _plot_load_sweep(prepared, path))
        elif name == "accel_map":
            done(name, _plot_map(
                prepared, lambda r: r.accel_series(map_sensor),
                f"Accel magnitude variance ({_sensor_name(map_sensor)})",
                r"Variance, $a^{*2}$" if normalized else "Variance (g$^2$)",
                "viridis", path))
        elif name == "load_map":
            done(name, _plot_map(
                prepared, lambda r: r.load,
                "Load-cell magnitude variance",
                "Variance (N$^2$)", "plasma", path))
        else:
            raise ValueError(f"Unknown plot {name!r}; choose from {PLOT_NAMES}")
    return written
