"""
live_plot.py

Live-updating diagnostic plots for the angle-sweep acquisition run in
main.py. Two figures are created once at the start of a run and then
updated with one new point after each motor angle finishes capturing:

  Figure 1 (map view):
      Left  axes: inclination angle (x, -45..+45 deg) vs angle of attack
                  / "pitch" (y, -180..+180 deg), colored by variance of
                  the acceleration-magnitude signal at that angle.
      Right axes: same x/y axes, colored by variance of the load-cell
                  force-magnitude signal (stays empty until load-cell
                  data is wired into the loop -- see add_point below).

  Figure 2 (sweep view):
      Variance of acceleration magnitude (left y-axis, blue) and
      load-cell force magnitude (right y-axis, red, twin axis) vs motor
      angle (x-axis).

Axis-naming note: "pitch angle" is mapped to
frame_angles["angle_of_attack_deg_shifted"] from
utils/coordinate_transforms.compute_frame_angles, since that's the
angle that sweeps the full +-180 deg range as the motor rotates
(inclination_angle is the one naturally bounded near +-45 deg). If you
meant a different field, just pass a different value into aoa_deg.

Requires matplotlib (pip install matplotlib) and a display -- if you
ever run this headless (e.g. over SSH with no X server), construct
LivePlotter(enabled=False) to no-op every call instead of crashing.
"""

import os

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class LivePlotter:
    def __init__(self, enabled=True, block=False):
        """enabled=False makes every method a no-op -- useful for headless
        runs (e.g. SSH with no display) without littering main.py with
        if-checks."""
        self.enabled = enabled and _HAS_MPL
        if not self.enabled:
            if enabled and not _HAS_MPL:
                print("  WARN: matplotlib not available, live plotting disabled "
                      "(pip install matplotlib)")
            return

        plt.ion()

        # ---- Figure 1: map view (inclination vs angle-of-attack/pitch) ----
        self.fig1, (self.ax_map_accel, self.ax_map_load) = plt.subplots(
            1, 2, figsize=(11, 5), num="Variance map"
        )
        self.fig1.suptitle("Variance vs. Blade-Frame Angles")

        for ax, title in (
            (self.ax_map_accel, "Accel magnitude variance"),
            (self.ax_map_load, "Load-cell magnitude variance"),
        ):
            ax.set_xlim(-45, 45)
            ax.set_ylim(-180, 180)
            ax.set_xlabel("Inclination angle (deg)")
            ax.set_ylabel("Angle of attack / pitch (deg)")
            ax.set_title(title)

        self.scatter_accel = self.ax_map_accel.scatter([], [], c=[], cmap="viridis")
        self.cbar_accel = self.fig1.colorbar(self.scatter_accel, ax=self.ax_map_accel)
        self.cbar_accel.set_label("Variance (mg^2)")

        self.scatter_load = self.ax_map_load.scatter([], [], c=[], cmap="plasma")
        self.cbar_load = self.fig1.colorbar(self.scatter_load, ax=self.ax_map_load)
        self.cbar_load.set_label("Variance (N^2)")

        self.fig1.tight_layout()

        # ---- Figure 2: sweep view (variance vs motor angle) ----
        self.fig2, self.ax_sweep = plt.subplots(figsize=(8, 5), num="Variance vs motor angle")
        self.fig2.suptitle("Variance vs. Motor Angle")
        self.ax_sweep.set_xlabel("Motor angle (deg)")
        self.ax_sweep.set_ylabel("Accel magnitude variance (mg^2)", color="tab:blue")
        self.ax_sweep.tick_params(axis="y", labelcolor="tab:blue")
        (self.line_accel,) = self.ax_sweep.plot([], [], "o-", color="tab:blue", label="Accel")

        self.ax_sweep_load = self.ax_sweep.twinx()
        self.ax_sweep_load.set_ylabel("Load-cell magnitude variance (N^2)", color="tab:red")
        self.ax_sweep_load.tick_params(axis="y", labelcolor="tab:red")
        (self.line_load,) = self.ax_sweep_load.plot([], [], "s--", color="tab:red", label="Load cell")

        self.fig2.tight_layout()

        # ---- Data buffers (kept so redraws are cheap set_* calls) ----
        self._motor_angles = []
        self._inclinations = []
        self._aoas = []
        self._accel_var = []
        self._load_var = []

        plt.show(block=block)

    def add_point(self, motor_angle_deg, inclination_deg, aoa_deg,
                   accel_variance, loadcell_variance=None):
        """Append one new (angle, variance) result and refresh both
        figures. Call this once per completed motor angle."""
        if not self.enabled:
            return

        self._motor_angles.append(motor_angle_deg)
        self._inclinations.append(inclination_deg)
        self._aoas.append(aoa_deg)
        self._accel_var.append(accel_variance)
        self._load_var.append(loadcell_variance)

        # --- Figure 1: accel map (always has data) ---
        offsets = np.column_stack([self._inclinations, self._aoas])
        self.scatter_accel.set_offsets(offsets)
        accel_arr = np.array(self._accel_var, dtype=float)
        self.scatter_accel.set_array(accel_arr)
        if len(accel_arr) > 0:
            lo, hi = float(accel_arr.min()), float(accel_arr.max())
            self.scatter_accel.set_clim(lo, hi if hi > lo else lo + 1e-9)

        # --- Figure 1: load-cell map (only plot points that have data) ---
        load_pts = [(x, y, v) for x, y, v in
                    zip(self._inclinations, self._aoas, self._load_var) if v is not None]
        if load_pts:
            lx, ly, lv = zip(*load_pts)
            self.scatter_load.set_offsets(np.column_stack([lx, ly]))
            lv_arr = np.array(lv, dtype=float)
            self.scatter_load.set_array(lv_arr)
            lo, hi = float(lv_arr.min()), float(lv_arr.max())
            self.scatter_load.set_clim(lo, hi if hi > lo else lo + 1e-9)

        # --- Figure 2: sweep lines, sorted by motor angle for a clean line ---
        order = np.argsort(self._motor_angles)
        ma = np.array(self._motor_angles)[order]

        av = accel_arr[order]
        self.line_accel.set_data(ma, av)
        self.ax_sweep.relim()
        self.ax_sweep.autoscale_view()

        lv_full = np.array([v if v is not None else np.nan for v in self._load_var])[order]
        if not np.all(np.isnan(lv_full)):
            self.line_load.set_data(ma, lv_full)
            self.ax_sweep_load.relim()
            self.ax_sweep_load.autoscale_view()

        for fig in (self.fig1, self.fig2):
            fig.canvas.draw_idle()
        plt.pause(0.001)  # yields to the GUI event loop so the window actually redraws

    def save(self, output_dir):
        """Save both figures as PNGs into output_dir."""
        if not self.enabled:
            return
        self.fig1.savefig(os.path.join(output_dir, "variance_map.png"), dpi=150)
        self.fig2.savefig(os.path.join(output_dir, "variance_vs_motor_angle.png"), dpi=150)

    def close(self):
        if not self.enabled:
            return
        plt.close(self.fig1)
        plt.close(self.fig2)


# ------------------------------------------------------------
# Variance helper -- lives here so the "how do we define variance of a
# 4-sensor signal" choice is documented and reusable in one place,
# rather than inline in main.py's capture loop.
# ------------------------------------------------------------


def accel_magnitude_variance(rows, mg_per_lsb):
    """Given `rows` (an iterable of the 12 raw int16 accel columns,
    s1_x..s4_z, one row per sample -- exactly what
    collect_accel_for_duration writes per CSV row) compute the variance
    of the per-sample acceleration magnitude, averaged across the 4
    onboard sensors.

    i.e. per sample: mag_i = sqrt(x_i^2 + y_i^2 + z_i^2) per sensor,
    averaged over the 4 sensors -> one magnitude value per sample ->
    variance of that time series, in mg^2.

    This is one reasonable definition (average-then-variance); an
    alternative is variance-per-sensor then averaged across sensors --
    swap the reduction order here if you'd rather track sensor-to-sensor
    disagreement instead of overall vibration level.
    """
    arr = np.asarray(rows, dtype=float) * mg_per_lsb  # (N, 12)
    arr = arr.reshape(-1, 4, 3)                        # (N, sensor, xyz)
    mags = np.linalg.norm(arr, axis=2)                 # (N, sensor)
    mean_mag = mags.mean(axis=1)                        # (N,) avg across sensors
    return float(np.var(mean_mag))


def loadcell_magnitude_variance(force_xyz_rows):
    """Placeholder for once the NI DAQ load-cell read is wired in.
    Expects rows of (fx, fy, fz) per sample (already in force units,
    e.g. N) and returns the variance of the per-sample force magnitude.
    Same average-then-variance convention as accel_magnitude_variance,
    but there's only one load cell so there's no sensor-averaging step.
    """
    arr = np.asarray(force_xyz_rows, dtype=float)  # (N, 3)
    mags = np.linalg.norm(arr, axis=1)              # (N,)
    return float(np.var(mags))