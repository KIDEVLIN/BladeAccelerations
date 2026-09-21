"""
live_plot.py

Live-updating diagnostic plots for the angle-sweep acquisition run in
main.py. A single figure is created once at the start of a run and then
updated with one new point after each motor angle finishes capturing.

Layout is controlled by two flags:
  show_loadcell=True (default): right column shows load-cell variance
      alongside the left (accelerometer) column. False drops the load-cell
      column entirely -- just the accel map (and accel sweep, if
      show_sweep is also True).
  show_sweep=True (default): bottom row shows variance vs motor angle.
      False drops that row -- just the map view on top.

  Map row (top, always shown):
      Left  axes: inclination angle (x, -45..+45 deg) vs angle of attack
                  / "pitch" (y, -180..+180 deg), colored by variance of
                  the acceleration-magnitude signal at that angle.
      Right axes (only if show_loadcell): same x/y axes, colored by
                  variance of the load-cell force-magnitude signal
                  (stays empty until load-cell data is wired into the
                  loop -- see add_point below).

  Sweep row (bottom, only if show_sweep):
      Left  axes: variance of acceleration magnitude vs motor angle.
      Right axes (only if show_loadcell): variance of load-cell force
                  magnitude vs motor angle.

Axis-naming note: "pitch angle" is mapped to
frame_angles["angle_of_attack_deg_shifted"] from
utils/coordinate_transforms.compute_frame_angles, since that's the
angle that sweeps the full +-180 deg range as the motor rotates
(inclination_angle is the one naturally bounded near +-45 deg). If you
meant a different field, just pass a different value into aoa_deg.

Usage from main.py:

    from live_plot import LivePlotter, accel_magnitude_variance

    plotter = LivePlotter()
    ...
    for i, angle in enumerate(angles):
        ...
        plotter.add_point(
            motor_angle_deg=angle,
            inclination_deg=frame_angles["inclination_angle_deg_shifted"],
            aoa_deg=frame_angles["angle_of_attack_deg_shifted"],
            accel_variance=accel_var,
            loadcell_variance=None,   # wire in once load cell is added
        )
    ...
    plotter.save(OUTPUT_DIR)   # optional, writes a PNG at the end of the run
    plotter.close()

Requires matplotlib (pip install matplotlib) and a display -- if you
ever run this headless (e.g. over SSH with no X server), construct
LivePlotter(enabled=False) to no-op every call instead of crashing.

If the window doesn't visibly update after every add_point() call (e.g.
several angles' worth of points appear at once instead of one at a
time), the active matplotlib backend may not be interactive. Uncomment
the matplotlib.use(...) line below (before pyplot is imported) to force
one, or check matplotlib.get_backend() -- "Agg" means no live window
will ever be shown regardless of plt.pause()/flush_events().
"""

import os

import numpy as np

# import matplotlib
# matplotlib.use("TkAgg")  # uncomment (and pick "QtAgg" etc. if needed) if
                            # live updates aren't showing -- must happen
                            # before matplotlib.pyplot is imported anywhere.

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class LivePlotter:
    def __init__(self, enabled=True, block=False, show_sweep=True,
                 show_loadcell=True, figsize=(9, 7), dpi=100):
        """
        enabled=False makes every method a no-op -- useful for headless
            runs (e.g. SSH with no display) without littering main.py with
            if-checks.
        show_sweep=False drops the bottom row (variance vs motor angle)
            entirely, leaving just the top map row.
        show_loadcell=False drops the right column (load-cell variance)
            entirely, leaving just the accelerometer column.
        figsize, dpi control the on-screen window size in pixels
            (figsize_inches * dpi). Shrink either if the window pops up
            too large for your screen.
        """
        self.enabled = enabled and _HAS_MPL
        self.show_sweep = show_sweep
        self.show_loadcell = show_loadcell
        if not self.enabled:
            if enabled and not _HAS_MPL:
                print("  WARN: matplotlib not available, live plotting disabled "
                      "(pip install matplotlib)")
            return

        plt.ion()

        nrows = 2 if show_sweep else 1
        ncols = 2 if show_loadcell else 1
        fig_w = figsize[0] if show_loadcell else figsize[0] * 0.55
        fig_h = figsize[1] if show_sweep else figsize[1] * 0.55
        self.fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi, num="Live diagnostics",
            squeeze=False,
        )
        self.fig.suptitle("Variance Diagnostics")
        self.ax_map_accel = axes[0][0]

        self.ax_map_accel.set_xlim(-45, 45)
        self.ax_map_accel.set_ylim(-180, 180)
        self.ax_map_accel.set_xlabel("Inclination angle (deg)")
        self.ax_map_accel.set_ylabel("Angle of attack / pitch (deg)")
        self.ax_map_accel.set_title("Accel magnitude variance")

        self.scatter_accel = self.ax_map_accel.scatter([], [], c=[], cmap="viridis")
        self.cbar_accel = self.fig.colorbar(self.scatter_accel, ax=self.ax_map_accel)
        self.cbar_accel.set_label("Variance (mg^2)")

        if self.show_loadcell:
            self.ax_map_load = axes[0][1]
            self.ax_map_load.set_xlim(-45, 45)
            self.ax_map_load.set_ylim(-180, 180)
            self.ax_map_load.set_xlabel("Inclination angle (deg)")
            self.ax_map_load.set_ylabel("Angle of attack / pitch (deg)")
            self.ax_map_load.set_title("Load-cell magnitude variance")

            self.scatter_load = self.ax_map_load.scatter([], [], c=[], cmap="plasma")
            self.cbar_load = self.fig.colorbar(self.scatter_load, ax=self.ax_map_load)
            self.cbar_load.set_label("Variance (N^2)")

        if self.show_sweep:
            self.ax_sweep_accel = axes[1][0]
            self.ax_sweep_accel.set_xlabel("Motor angle (deg)")
            self.ax_sweep_accel.set_ylabel("Accel magnitude variance ", color="tab:blue")
            self.ax_sweep_accel.tick_params(axis="y", labelcolor="tab:blue")
            self.ax_sweep_accel.set_title("Accel magnitude variance")
            (self.line_accel,) = self.ax_sweep_accel.plot([], [], "o-", color="tab:blue")

            if self.show_loadcell:
                self.ax_sweep_load = axes[1][1]
                self.ax_sweep_load.set_xlabel("Motor angle (deg)")
                self.ax_sweep_load.set_ylabel("Load-cell magnitude variance ", color="tab:red")
                self.ax_sweep_load.tick_params(axis="y", labelcolor="tab:red")
                self.ax_sweep_load.set_title("Load-cell magnitude variance")
                (self.line_load,) = self.ax_sweep_load.plot([], [], "s-", color="tab:red")

        self.fig.tight_layout()

        # ---- Data buffers (kept so redraws are cheap set_* calls) ----
        self._motor_angles = []
        self._inclinations = []
        self._aoas = []
        self._accel_var = []
        self._load_var = []

        plt.show(block=block)
        # Give the window manager a moment to actually open/paint the
        # window before the first add_point() call comes in.
        plt.pause(0.1)

    def add_point(self, motor_angle_deg, inclination_deg, aoa_deg,
                   accel_variance, loadcell_variance=None):
        """Append one new (angle, variance) result and refresh the figure.
        Call this once per completed motor angle. loadcell_variance is
        accepted (and stored) even when show_loadcell=False, so you can
        swap plotting on/off later without touching the call site -- it's
        just not drawn."""
        if not self.enabled:
            return

        self._motor_angles.append(motor_angle_deg)
        self._inclinations.append(inclination_deg)
        self._aoas.append(aoa_deg)
        self._accel_var.append(accel_variance)
        self._load_var.append(loadcell_variance)

        # --- accel map (always shown) ---
        offsets = np.column_stack([self._inclinations, self._aoas])
        self.scatter_accel.set_offsets(offsets)
        accel_arr = np.array(self._accel_var, dtype=float)
        self.scatter_accel.set_array(accel_arr)
        if len(accel_arr) > 0:
            lo, hi = float(accel_arr.min()), float(accel_arr.max())
            self.scatter_accel.set_clim(lo, hi if hi > lo else lo + 1e-9)

        # --- load-cell map (only if enabled, and only points with data) ---
        if self.show_loadcell:
            load_pts = [(x, y, v) for x, y, v in
                        zip(self._inclinations, self._aoas, self._load_var) if v is not None]
            if load_pts:
                lx, ly, lv = zip(*load_pts)
                self.scatter_load.set_offsets(np.column_stack([lx, ly]))
                lv_arr = np.array(lv, dtype=float)
                self.scatter_load.set_array(lv_arr)
                lo, hi = float(lv_arr.min()), float(lv_arr.max())
                self.scatter_load.set_clim(lo, hi if hi > lo else lo + 1e-9)

        # --- sweep plots, sorted by motor angle for a clean line ---
        if self.show_sweep:
            order = np.argsort(self._motor_angles)
            ma = np.array(self._motor_angles)[order]

            av = accel_arr[order]
            self.line_accel.set_data(ma, av)
            self.ax_sweep_accel.relim()
            self.ax_sweep_accel.autoscale_view()

            if self.show_loadcell:
                lv_full = np.array([v if v is not None else np.nan for v in self._load_var])[order]
                if not np.all(np.isnan(lv_full)):
                    self.line_load.set_data(ma, lv_full)
                    self.ax_sweep_load.relim()
                    self.ax_sweep_load.autoscale_view()

        # --- Redraw. draw_idle() queues the redraw; pause()+flush_events()
        # are both included because which one actually forces the window
        # to repaint is backend-dependent -- belt and suspenders. ---
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, output_dir):
        """Save the figure as a PNG into output_dir."""
        if not self.enabled:
            return
        self.fig.savefig(os.path.join(output_dir, "live_diagnostics.png"), dpi=150)

    def close(self):
        if not self.enabled:
            return
        plt.close(self.fig)


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