"""
plot_timeseries.py

One organized, presentable time-series figure per angle capture:
  - accel magnitude (mg) per sensor, S1-S4 overlaid in one panel
  - load-cell forces Fx/Fy/Fz in one panel
  - load-cell moments Mx/My/Mz in one panel

Raw per-axis accel traces (12 of them) and the raw strain-gauge
(stg0-5) voltages are deliberately left out -- plotting everything
would make this unreadable; magnitude (the accel signal that matters
for the variance analysis anyway) and calibrated force/moment are what
you actually want to look at.

Two changes from the first version, per feedback:
  - Every panel plots FLUCTUATIONS -- each channel's own mean (over the
    full capture, same window the variance metric uses) is subtracted
    before plotting, rather than the raw signal.
  - Only the first `plot_window_s` seconds are shown (default 0.5 s)
    instead of the whole multi-second capture -- at 1600 Hz the accel
    panel in particular was too dense to read over the full window.

The motor angle is reported as a single reduced number in the title
(the point of the reduction requested), not plotted as its own time
series.
"""

import numpy as np
import matplotlib.pyplot as plt

from variance_analysis import ACCEL_COLS, LOAD_FORCE_COLS, MG_PER_LSB, NUM_SENSORS

LOAD_MOMENT_COLS = ["Mx", "My", "Mz"]


def _accel_magnitudes(accel_df):
    rows = accel_df[ACCEL_COLS].to_numpy(dtype=float) * MG_PER_LSB
    rows = rows.reshape(-1, NUM_SENSORS, 3)
    return np.linalg.norm(rows, axis=2)  # (N, sensor)


def plot_angle_timeseries(capture, out_path, title_prefix="", plot_window_s=0.5):
    fig, (ax_accel, ax_force, ax_moment) = plt.subplots(3, 1, figsize=(10, 8))

    # --- Accel magnitude fluctuation, one line per sensor ---
    # Mean is computed over the FULL capture (matching the window the
    # variance metric uses), then only the first plot_window_s seconds
    # are actually drawn.
    mags = _accel_magnitudes(capture.accel_df)
    mags_fluct = mags - mags.mean(axis=0, keepdims=True)
    t_accel = capture.accel_df["time"].to_numpy(dtype=float)
    mask_a = t_accel <= plot_window_s
    for s in range(NUM_SENSORS):
        ax_accel.plot(t_accel[mask_a], mags_fluct[mask_a, s], label=f"S{s + 1}", linewidth=1.0)
    ax_accel.set_ylabel("Accel magnitude fluctuation (mg)")
    ax_accel.set_xlim(0, plot_window_s)
    ax_accel.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax_accel.legend(loc="upper right", ncol=NUM_SENSORS, fontsize=8)
    ax_accel.set_title("Accelerometers (mean subtracted)")

    # --- Load cell force / moment fluctuation ---
    if capture.load_df is not None:
        t_load = capture.load_df["time"].to_numpy(dtype=float)
        mask_l = t_load <= plot_window_s

        for col in LOAD_FORCE_COLS:
            vals = capture.load_df[col].to_numpy(dtype=float)
            vals_fluct = vals - vals.mean()
            ax_force.plot(t_load[mask_l], vals_fluct[mask_l], label=col, linewidth=1.0)
        ax_force.set_ylabel("Force fluctuation (N)")
        ax_force.set_xlim(0, plot_window_s)
        ax_force.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax_force.legend(loc="upper right", fontsize=8)
        ax_force.set_title("Load cell -- forces (mean subtracted)")

        for col in LOAD_MOMENT_COLS:
            vals = capture.load_df[col].to_numpy(dtype=float)
            vals_fluct = vals - vals.mean()
            ax_moment.plot(t_load[mask_l], vals_fluct[mask_l], label=col, linewidth=1.0)
        ax_moment.set_ylabel("Moment fluctuation (N\u00b7m)")
        ax_moment.set_xlabel("Time (s, relative to capture start)")
        ax_moment.set_xlim(0, plot_window_s)
        ax_moment.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax_moment.legend(loc="upper right", fontsize=8)
        ax_moment.set_title("Load cell -- moments (mean subtracted)")
    else:
        for ax in (ax_force, ax_moment):
            ax.text(0.5, 0.5, "No load-cell data for this angle",
                    ha="center", va="center", transform=ax.transAxes)
        ax_accel.set_xlabel("Time (s, relative to capture start)")

    # --- Title: requested angle + reduced motor angle ---
    if capture.avg_motor_angle_deg is not None:
        angle_str = (
            f"avg motor angle: {capture.avg_motor_angle_deg:.2f} deg "
            f"(n={capture.motor_angle_n_samples}"
            + (f", std={capture.motor_angle_std_deg:.3f} deg)"
               if capture.motor_angle_std_deg is not None else ")")
        )
    else:
        angle_str = "avg motor angle: unavailable (see run_loader.py gap #2)"

    fig.suptitle(
        f"{title_prefix}Angle {capture.requested_angle_deg:+.2f} deg requested "
        f"-- {angle_str}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")