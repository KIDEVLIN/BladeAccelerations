# %% [markdown]
# explore.py -- interactive scratch file for post-processing a run.
#
# Open in Spyder (or VS Code with the Jupyter extension) and run one cell
# at a time with Ctrl+Enter ("# %%" starts a cell). Variables stay in the
# console, so the slow load cell only needs to run once per session.
#
# Reusable logic stays in run_loader.py, variance_analysis.py, etc.
# This file is just your "command window".

# %% 1. Setup (run once per session)
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Work out where this file lives (Spyder/VS Code cells sometimes lack __file__)
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
    if (HERE / "scripts" / "aquire_data").exists():
        HERE = HERE / "scripts" / "aquire_data"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Autoreload: edits to run_loader.py, variance_analysis.py, plot_*.py etc.
# take effect on the next call without restarting the kernel.
ip = get_ipython()  # noqa: F821  (None outside IPython)
if ip is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    # Uncomment for pop-up, zoomable figure windows instead of the Plots pane:
    # ip.run_line_magic("matplotlib", "qt")

from utils.run_loader import load_run
from utils.plot_timeseries import (
    plot_angle_timeseries, compute_shared_ylims,
    LOAD_FORCE_COLS, LOAD_MOMENT_COLS, _accel_magnitudes_g,
)
from variance_analysis import (
    accel_per_sensor_variance, accel_mean_variance, load_variance,
    pick_extreme_angles, ACCEL_COLS, accel_mg_per_lsb,
)
from plot_variance_sweep import plot_variance_sweep
from utils import coordinate_transforms as ct
import test_modbus as pcb

# %% 2. Config -- point at a run. Pulls settings from run_metadata.json
# when it exists (main.py writes it), otherwise uses the fallbacks below.
RUN_DIR = Path("Data/run3")          # relative to where you launch from, or absolute

FALLBACK = dict(
    counts_per_deg=694.44,
    sample_duration_s=2.0,
    sweep_angle_deg=-30.0,
    mounting_angle_deg=-5.895 + 90,
)

meta_path = RUN_DIR / "run_metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print(f"Loaded settings from {meta_path} (status: {meta.get('status')})")
else:
    meta = {}
    print("No run_metadata.json -- using FALLBACK settings")

cfg = {k: meta.get(k, v) for k, v in FALLBACK.items()}
print(cfg)

OUT_DIR = RUN_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% 3. Load the run (the slow part -- rerun only if the data changes)
captures = load_run(RUN_DIR, cfg["counts_per_deg"], cfg["sample_duration_s"])
print(f"{len(captures)} captures loaded")

# %% 3b. (optional) Cache to disk so a kernel restart doesn't need a reload.
# Like save/load of a .mat file.
CACHE = RUN_DIR / "captures.pkl"
# with open(CACHE, "wb") as f: pickle.dump(captures, f)       # save
# with open(CACHE, "rb") as f: captures = pickle.load(f)      # load

# %% 4. Quick look -- a table of every angle (your "whos" + summary)
rows = []
for c in captures:
    per_sensor = accel_per_sensor_variance(c.accel_df)
    rows.append({
        "requested_deg": c.requested_angle_deg,
        "avg_motor_deg": c.avg_motor_angle_deg,
        "n_accel": len(c.accel_df),
        "n_load": None if c.load_df is None else len(c.load_df),
        **{f"var_S{s}_mg2": v for s, v in per_sensor.items()},
        "load_var_N2": load_variance(c.load_df),
    })
summary = pd.DataFrame(rows)
summary

# %% 4b. Poke at one capture (change the index)
c = captures[0]
c.accel_df.head()
# c.accel_df.describe()
# c.load_df.head()

# %% 5. Add frame angles (inclination / angle of attack) to the summary
frames = [
    ct.compute_frame_angles(
        c.avg_motor_angle_deg if c.avg_motor_angle_deg is not None else c.requested_angle_deg,
        cfg["sweep_angle_deg"], cfg["mounting_angle_deg"])
    for c in captures
]
summary["inclination_deg"] = [f["inclination_angle_deg_shifted"] for f in frames]
summary["aoa_deg"] = [f["angle_of_attack_deg_shifted"] for f in frames]
summary

# %% 6. Variance vs motor angle, all 4 sensors (edit freely, rerun this cell only)
fig, ax = plt.subplots(figsize=(8, 5))
order = summary["requested_deg"].argsort()
x = summary["requested_deg"].to_numpy()[order]
for s in range(1, 5):
    ax.plot(x, summary[f"var_S{s}_mg2"].to_numpy()[order] * 1e-6, "o-", label=f"S{s}")
ax.set_xlabel("Motor angle (deg)")
ax.set_ylabel("Accel magnitude variance (g$^2$)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

# %% 7. Map: inclination vs AoA, colored by variance of one sensor
SENSOR = 4
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(summary["inclination_deg"], summary["aoa_deg"],
                c=summary[f"var_S{SENSOR}_mg2"] * 1e-6, cmap="viridis")
ax.set_xlabel("Inclination angle (deg)")
ax.set_ylabel("Angle of attack / pitch (deg)")
fig.colorbar(sc, ax=ax, label=f"S{SENSOR} variance (g$^2$)")
fig.tight_layout()

# %% 8. Time series for one capture (change IDX)
IDX = 26
WINDOW_S = .5
c = captures[IDX]

mags = _accel_magnitudes_g(c.accel_df)
mags = mags - mags.mean(axis=0, keepdims=True)
t = c.accel_df["time"].to_numpy()
m = t <= WINDOW_S

fig, ax = plt.subplots(figsize=(10, 4))
for s in range(mags.shape[1]):
    ax.plot(t[m], mags[m, s], lw=1, label=f"S{s + 1}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration fluctuation (g)")
ax.set_title(f"Requested angle {c.requested_angle_deg:+.1f} deg")
ax.legend(ncol=4)
fig.tight_layout()

# %% 8b. Per-component accel (a_x, a_y, a_z) for ONE sensor
def accel_components_g(accel_df, sensor):
    """(N, 3) array of [a_x, a_y, a_z] in g for one sensor (1-based, S1..S4).
    Same layout idea as _accel_magnitudes_g, which returns (N, sensor)."""
    cols = [f"s{sensor}_{axis}" for axis in "xyz"]
    return accel_df[cols].to_numpy(dtype=float) * accel_mg_per_lsb(accel_df) / 1000.0


SENSOR = 4          # which sensor to look at
IDX = 26            # which capture
WINDOW_S = .5       # time window in seconds

c = captures[IDX]
comps = accel_components_g(c.accel_df, SENSOR)
comps = comps - comps.mean(axis=0, keepdims=True)    # fluctuation about the mean
t = c.accel_df["time"].to_numpy()
m = t <= WINDOW_S

# Three stacked panels sharing one time axis: one line each, so it stays readable
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, k, label in zip(axes, range(3), ["a_x", "a_y", "a_z"]):
    ax.plot(t[m], comps[m, k], lw=1, color=f"C{k}")
    ax.set_ylabel(f"{label} (g)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"S{SENSOR} components, requested angle {c.requested_angle_deg:+.1f} deg")
fig.tight_layout()

# %% 8c. Per-component variance vs motor angle for one sensor
# (one panel, three lines -- the component version of the sweep plot)
comp_var = np.array([
    np.var(accel_components_g(c.accel_df, SENSOR), axis=0) for c in captures
])                                                    # (n_angles, 3), g^2
order = np.argsort([c.requested_angle_deg for c in captures])
angles = np.array([c.requested_angle_deg for c in captures])[order]

fig, ax = plt.subplots(figsize=(8, 5))
for k, label in enumerate(["a_x", "a_y", "a_z"]):
    ax.plot(angles, comp_var[order, k], "o-", label=label)
ax.set_xlabel("Motor angle (deg)")
ax.set_ylabel("Variance (g$^2$)")
ax.set_title(f"S{SENSOR} per-component variance")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

# %% 8b. Components (a_x, a_y, a_z) for ONE sensor
# Same layout as _accel_magnitudes_g, but returns (N, 3) in g for the
# chosen sensor instead of (N, sensors) magnitudes. Columns = x, y, z.
def accel_components_g(accel_df, sensor):
    """sensor is 1-based (S1..S4), matching the plot labels."""
    cols = [f"s{sensor}_{axis}" for axis in "xyz"]
    return accel_df[cols].to_numpy(dtype=float) * accel_mg_per_lsb(accel_df) / 1000.0


SENSOR = 4        # which sensor to look at (1-4)
IDX = 0           # which capture
WINDOW_S = 0.5
c = captures[IDX]

comp = accel_components_g(c.accel_df, SENSOR)
comp_fluct = comp - comp.mean(axis=0, keepdims=True)   # mean (incl. gravity) removed
t = c.accel_df["time"].to_numpy()
m = t <= WINDOW_S

# Three stacked panels sharing the time axis: uncluttered, and each axis
# gets its own y-scale. Use sharey=True if you want them directly comparable.
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=False)
for ax, k, name in zip(axes, range(3), ("a_x", "a_y", "a_z")):
    ax.plot(t[m], comp_fluct[m, k], lw=1)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel(f"{name} (g)")
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"S{SENSOR} components, requested angle {c.requested_angle_deg:+.1f} deg")
fig.tight_layout()

# %% 8c. Component statistics across the whole sweep (table, no plot clutter)
rows = []
for c in captures:
    comp = accel_components_g(c.accel_df, SENSOR)
    rows.append({
        "requested_deg": c.requested_angle_deg,
        **{f"mean_{n}_g": comp[:, k].mean() for k, n in enumerate("xyz")},
        **{f"var_{n}_g2": comp[:, k].var() for k, n in enumerate("xyz")},
    })
comp_stats = pd.DataFrame(rows).sort_values("requested_deg").reset_index(drop=True)
comp_stats

# %% 8d. Component variance vs motor angle (one sensor, 3 lines)
fig, ax = plt.subplots(figsize=(8, 5))
for n in "xyz":
    ax.plot(comp_stats["requested_deg"], comp_stats[f"var_{n}_g2"], "o-", label=f"a_{n}")
ax.set_xlabel("Motor angle (deg)")
ax.set_ylabel("Variance (g$^2$)")
ax.set_title(f"S{SENSOR} component variance")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

# %% 9. Spectrum of one capture (example of a new analysis -- the accel
# time column is reconstructed per row, so estimate fs from the ODR instead)
from scipy.signal import welch

FS = 1600.0   # accel ODR (Hz); change if your header says otherwise
c = captures[IDX]
mags_g = _accel_magnitudes_g(c.accel_df)

fig, ax = plt.subplots(figsize=(8, 4))
for s in range(mags_g.shape[1]):
    f, p = welch(mags_g[:, s] - mags_g[:, s].mean(), fs=FS, nperseg=512)
    ax.semilogy(f, p, label=f"S{s + 1}")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (g$^2$/Hz)")
ax.legend()
fig.tight_layout()

# %% 10. Run the existing saved-PNG plots (these save to OUT_DIR and close
# their figure, so view the PNGs rather than expecting a live window)
highest, lowest = pick_extreme_angles(captures)
ylims = compute_shared_ylims([highest, lowest], plot_window_s=0.5)
plot_angle_timeseries(highest, OUT_DIR / "timeseries_high_variance.png",
                      case_label="High", plot_window_s=0.5, ylims=ylims)
plot_angle_timeseries(lowest, OUT_DIR / "timeseries_low_variance.png",
                      case_label="Low", plot_window_s=0.5, ylims=ylims)
plot_variance_sweep(captures, OUT_DIR / "variance_sweep.png",
                    sweep_angle_deg=cfg["sweep_angle_deg"],
                    mounting_angle_deg=cfg["mounting_angle_deg"])