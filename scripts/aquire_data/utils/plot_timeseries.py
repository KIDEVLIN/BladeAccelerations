"""
plot_timeseries.py

One organized, presentable time-series figure per angle capture:
  - acceleration magnitude fluctuation (g), S1-S4 overlaid in one panel
  - load-cell force fluctuations Fx/Fy/Fz in one panel
  - load-cell moment fluctuations Mx/My/Mz in one panel

Raw per-axis accel traces (12 of them) and the raw strain-gauge
(stg0-5) voltages are deliberately left out -- plotting everything
would make this unreadable.

Presentation formatting:
  - Figure title is just "<High/Low> variance case at motor angle X";
    no per-panel titles.
  - Every panel plots FLUCTUATIONS -- each channel's own mean (over the
    full capture, same window the variance metric uses) is subtracted
    before plotting.
  - Acceleration is shown in g (converted from the sensor's mg).
  - Only the first `plot_window_s` seconds are drawn (default 0.5 s).
  - compute_shared_ylims() gives both figures identical y-axes per panel.

The motor angle is the reduced (averaged) value from run_loader, not a
time series; it appears only in the title.
"""

import numpy as np
import matplotlib.pyplot as plt

from variance_analysis import ACCEL_COLS, LOAD_FORCE_COLS, NUM_SENSORS, accel_mg_per_lsb

LOAD_MOMENT_COLS = ["Mx", "My", "Mz"]
MG_PER_G = 1000.0


def _accel_magnitudes_g(accel_df):
    rows = accel_df[ACCEL_COLS].to_numpy(dtype=float) * accel_mg_per_lsb(accel_df) / MG_PER_G
    rows = rows.reshape(-1, NUM_SENSORS, 3)
    return np.linalg.norm(rows, axis=2)  # (N, sensor), in g


def _windowed_fluctuations(capture, plot_window_s):
    """Returns {"accel": arr, "force": arr, "moment": arr} of the values
    that will actually be drawn (mean-subtracted over the full capture,
    then cut to the plot window). Load entries are None without load data."""
    out = {}
    mags = _accel_magnitudes_g(capture.accel_df)
    fluct = mags - mags.mean(axis=0, keepdims=True)
    t = capture.accel_df["time"].to_numpy(dtype=float)
    out["accel"] = fluct[t <= plot_window_s]
    if capture.load_df is None:
        out["force"] = out["moment"] = None
    else:
        tl = capture.load_df["time"].to_numpy(dtype=float)
        mask = tl <= plot_window_s
        for key, cols in (("force", LOAD_FORCE_COLS), ("moment", LOAD_MOMENT_COLS)):
            arr = capture.load_df[cols].to_numpy(dtype=float)
            out[key] = (arr - arr.mean(axis=0, keepdims=True))[mask]
    return out


def compute_shared_ylims(captures, plot_window_s=0.5, pad=0.05):
    """Common (ymin, ymax) per panel across several captures so the
    high/low variance figures can be compared by eye. Symmetric about
    zero (these are fluctuations), based on only the plotted window,
    with `pad` fractional headroom. Returns {"accel", "force", "moment"};
    a panel is None if no capture has data for it."""
    peaks = {"accel": 0.0, "force": 0.0, "moment": 0.0}
    seen = {k: False for k in peaks}
    for c in captures:
        for key, arr in _windowed_fluctuations(c, plot_window_s).items():
            if arr is not None and arr.size:
                peaks[key] = max(peaks[key], float(np.abs(arr).max()))
                seen[key] = True
    return {k: ((-peaks[k] * (1 + pad), peaks[k] * (1 + pad)) if seen[k] else None)
            for k in peaks}


def plot_angle_timeseries(capture, out_path, case_label="High", plot_window_s=0.5, ylims=None):
    """case_label: "High" or "Low" -- used in the figure title.
    ylims: optional dict from compute_shared_ylims() to force identical
    y-axes across figures; None = autoscale each panel."""
    fig, (ax_accel, ax_force, ax_moment) = plt.subplots(3, 1, figsize=(10, 8))

    # --- Acceleration magnitude fluctuation, one line per sensor ---
    # Mean is computed over the FULL capture, then only the first
    # plot_window_s seconds are drawn.
    mags = _accel_magnitudes_g(capture.accel_df)
    mags_fluct = mags - mags.mean(axis=0, keepdims=True)
    t_accel = capture.accel_df["time"].to_numpy(dtype=float)
    mask_a = t_accel <= plot_window_s
    for s in range(NUM_SENSORS):
        ax_accel.plot(t_accel[mask_a], mags_fluct[mask_a, s], label=f"S{s + 1}", linewidth=1.0)
    ax_accel.set_ylabel("Acceleration fluctuation (g)")
    ax_accel.set_xlim(0, plot_window_s)
    ax_accel.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax_accel.legend(loc="upper right", ncol=NUM_SENSORS, fontsize=8)

    # --- Load cell force / moment fluctuation ---
    if capture.load_df is not None:
        t_load = capture.load_df["time"].to_numpy(dtype=float)
        mask_l = t_load <= plot_window_s

        for col in LOAD_FORCE_COLS:
            vals = capture.load_df[col].to_numpy(dtype=float)
            ax_force.plot(t_load[mask_l], (vals - vals.mean())[mask_l], label=col, linewidth=1.0)
        ax_force.set_ylabel("Force fluctuation (N)")
        ax_force.set_xlim(0, plot_window_s)
        ax_force.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax_force.legend(loc="upper right", fontsize=8)

        for col in LOAD_MOMENT_COLS:
            vals = capture.load_df[col].to_numpy(dtype=float)
            ax_moment.plot(t_load[mask_l], (vals - vals.mean())[mask_l], label=col, linewidth=1.0)
        ax_moment.set_ylabel("Moment fluctuation (N\u00b7m)")
        ax_moment.set_xlabel("Time (s)")
        ax_moment.set_xlim(0, plot_window_s)
        ax_moment.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax_moment.legend(loc="upper right", fontsize=8)
    else:
        for ax in (ax_force, ax_moment):
            ax.text(0.5, 0.5, "No load-cell data for this angle",
                    ha="center", va="center", transform=ax.transAxes)
        ax_accel.set_xlabel("Time (s)")

    if ylims is not None:
        for ax, key in ((ax_accel, "accel"), (ax_force, "force"), (ax_moment, "moment")):
            if ylims.get(key) is not None and (key == "accel" or capture.load_df is not None):
                ax.set_ylim(*ylims[key])

    # --- Title: case + reduced motor angle (falls back to requested) ---
    if capture.avg_motor_angle_deg is not None:
        angle = capture.avg_motor_angle_deg
        print(f"  {case_label} case: avg motor angle {angle:.3f} deg "
              f"(n={capture.motor_angle_n_samples}, std={capture.motor_angle_std_deg})")
    else:
        angle = capture.requested_angle_deg
        print(f"  {case_label} case: avg motor angle unavailable, using requested "
              f"angle {angle:.2f} deg in title (see run_loader.py gap #2)")

    fig.suptitle(f"{case_label} variance case at motor angle {angle:.2f}\u00b0")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")