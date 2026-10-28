"""
plot_variance_sweep.py

Recreates utils/live_varience_plot.py's 2x2 diagnostic layout as a
single static figure for a whole run, with one change from the
original: the accelerometer SWEEP panel (variance vs motor angle)
shows all 4 individual sensors as separate overlaid series (instead of
the live plot's mean-across-sensor value), so sensor-to-sensor
disagreement is visible instead of being averaged away.

The top "map" row (inclination vs angle-of-attack, colored by
variance) needs frame angles. Pass sweep_angle_deg and
mounting_angle_deg (the fixed constants for the run -- see
main.py/SWEEP_ANGLE_DEG, MOUNTING_ANGLE_DEG in the acquisition
pipeline) and this computes the map directly with
utils/coordinate_transforms.compute_frame_angles, using each
capture's reduced average motor angle -- this no longer depends on
coordinate_frame_angles.csv (which, per run_loader.py gap #1, may not
exist). If sweep/mounting angle aren't given, it falls back to reading
inclination_angle_deg_shifted / angle_of_attack_deg_shifted off
coordinate_frame_angles.csv if that's present, and skips the map row
entirely if neither is available.

Since the map panel encodes variance as color, it shows a single
sensor (MAP_SENSOR, currently S4); the per-sensor breakdown lives in
the sweep row below it. Accel variance is shown in g^2 (converted
from the sensor's mg^2); no overall figure title (presentation use).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from variance_analysis import accel_per_sensor_variance, load_variance

_ACQUIRE_DIR = Path(__file__).resolve().parents[1] / "aquire_data"
if str(_ACQUIRE_DIR) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE_DIR))
from utils import coordinate_transforms as ct  # noqa: E402

_ACCEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

# Accel variance is computed in mg^2 (sensor units); plots show g^2.
MG2_TO_G2 = 1e-6
# Sensor used for the wind-turbine-frame accel map (1-based, S4 = 4).
MAP_SENSOR = 4


def _motor_angles(captures):
    return np.array([
        c.avg_motor_angle_deg if c.avg_motor_angle_deg is not None
        else c.requested_angle_deg
        for c in captures
    ])


def _frame_angles(captures, motor_angles, sweep_angle_deg, mounting_angle_deg):
    """Returns (inclinations, aoas) or (None, None) if unavailable."""
    if sweep_angle_deg is not None and mounting_angle_deg is not None:
        frames = [
            ct.compute_frame_angles(m, sweep_angle_deg, mounting_angle_deg)
            for m in motor_angles
        ]
        inclinations = np.array([f["inclination_angle_deg_shifted"] for f in frames])
        aoas = np.array([f["angle_of_attack_deg_shifted"] for f in frames])
        return inclinations, aoas

    if all(c.summary_row is not None and "inclination_angle_deg_shifted" in c.summary_row
           for c in captures):
        inclinations = np.array([c.summary_row["inclination_angle_deg_shifted"] for c in captures])
        aoas = np.array([c.summary_row["angle_of_attack_deg_shifted"] for c in captures])
        return inclinations, aoas

    return None, None


def plot_variance_sweep(captures, out_path, sweep_angle_deg=None, mounting_angle_deg=None,
                        a_c_g=None):
    """a_c_g: optional characteristic acceleration (g, see normalization.py).
    When given, accel variance is plotted as a*^2 = Var / a_c^2 (dimensionless)
    instead of g^2. Load-cell panels are unchanged."""
    accel_scale = MG2_TO_G2 / a_c_g ** 2 if a_c_g is not None else MG2_TO_G2
    var_label = "Variance, a*^2" if a_c_g is not None else "Variance (g^2)"
    sweep_label = ("Accel magnitude variance, a*^2" if a_c_g is not None
                   else "Accel magnitude variance (g^2)")
    motor_angles = _motor_angles(captures)
    inclinations, aoas = _frame_angles(captures, motor_angles, sweep_angle_deg, mounting_angle_deg)
    has_frame_angles = inclinations is not None
    has_load = any(c.load_df is not None for c in captures)

    nrows = 2 if has_frame_angles else 1
    ncols = 2 if has_load else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    order = np.argsort(motor_angles)

    per_sensor_var = [accel_per_sensor_variance(c.accel_df) for c in captures]
    load_var = [load_variance(c.load_df) for c in captures]

    row = 0
    if has_frame_angles:
        ax_map_accel = axes[row][0]
        map_accel_var = np.array([v[MAP_SENSOR] for v in per_sensor_var]) * accel_scale
        sc = ax_map_accel.scatter(inclinations, aoas, c=map_accel_var, cmap="viridis")
        ax_map_accel.set_xlim(-45, 45)
        ax_map_accel.set_ylim(-180, 180)
        ax_map_accel.set_xlabel("Inclination angle (deg)")
        ax_map_accel.set_ylabel("Angle of attack / pitch (deg)")
        ax_map_accel.set_title(f"Accel magnitude variance (S{MAP_SENSOR})")
        fig.colorbar(sc, ax=ax_map_accel, label=var_label)

        if has_load:
            ax_map_load = axes[row][1]
            load_var_arr = np.array([v if v is not None else np.nan for v in load_var])
            sc2 = ax_map_load.scatter(inclinations, aoas, c=load_var_arr, cmap="plasma")
            ax_map_load.set_xlim(-45, 45)
            ax_map_load.set_ylim(-180, 180)
            ax_map_load.set_xlabel("Inclination angle (deg)")
            ax_map_load.set_ylabel("Angle of attack / pitch (deg)")
            ax_map_load.set_title("Load-cell magnitude variance")
            fig.colorbar(sc2, ax=ax_map_load, label="Variance (N^2)")
        row += 1

    ax_sweep_accel = axes[row][0]
    for s in range(4):
        vals = np.array([v[s + 1] for v in per_sensor_var])[order] * accel_scale
        ax_sweep_accel.plot(motor_angles[order], vals, "o-", color=_ACCEL_COLORS[s], label=f"S{s + 1}")
    ax_sweep_accel.set_xlabel("Motor angle (deg)")
    ax_sweep_accel.set_ylabel(sweep_label)
    ax_sweep_accel.set_title("Accel variance vs motor angle -- all 4 sensors")
    ax_sweep_accel.legend(fontsize=8)

    if has_load:
        ax_sweep_load = axes[row][1]
        lv = np.array([v if v is not None else np.nan for v in load_var])[order]
        ax_sweep_load.plot(motor_angles[order], lv, "s-", color="tab:red")
        ax_sweep_load.set_xlabel("Motor angle (deg)")
        ax_sweep_load.set_ylabel("Load-cell magnitude variance (N^2)")
        ax_sweep_load.set_title("Load-cell variance vs motor angle")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")