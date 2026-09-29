#!/usr/bin/env python3
"""
Post-processing pipeline entry point.

Usage:
    python main.py <run_dir> \
        --counts-per-deg 694.44 --duration 4.0 \
        --sweep-angle-deg 30.0 --mounting-angle-deg 97 \
        --plot-window-s 0.5 --out analysis

<run_dir> is a completed acquisition run directory, e.g. Data/run_010,
in the same layout scripts/aquire_data/main.py writes: one angle_*.csv
(+ optional angle_*_load.csv) per motor angle, plus encoder_log.txt and
(if present) coordinate_frame_angles.csv.

--counts-per-deg, --duration, --sweep-angle-deg, --mounting-angle-deg
must match what was actually used for this run (Motor's
counts_per_deg, main.py's SAMPLE_DURATION_S / SWEEP_ANGLE_DEG /
MOUNTING_ANGLE_DEG) -- none of these are currently persisted into the
run directory itself (see run_loader.py's module docstring, gap #3).

--sweep-angle-deg / --mounting-angle-deg are optional: without them,
the wind-turbine-frame map plot falls back to reading
coordinate_frame_angles.csv (if present) and is skipped otherwise --
everything else in the pipeline still runs.

Produces, in <run_dir>/<out>/:
    timeseries_high_variance.png   -- time series for the angle with the
                                       highest mean accel variance
    timeseries_low_variance.png    -- time series for the angle with the
                                       lowest mean accel variance
    variance_sweep.png             -- live_varience_plot-style diagnostic
                                       for the whole sweep, all 4
                                       accelerometers overlaid
"""

import argparse
from pathlib import Path

from utils.run_loader import load_run
from variance_analysis import pick_extreme_angles
from utils.plot_timeseries import plot_angle_timeseries
from plot_variance_sweep import plot_variance_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--counts-per-deg", type=float, default=694.44,
                         help="Must match the Motor's counts_per_deg for this run")
    parser.add_argument("--duration", type=float, default=4.0,
                         help="Must match SAMPLE_DURATION_S used for this run")
    parser.add_argument("--sweep-angle-deg", type=float, default=None,
                         help="Blade azimuthal/sweep angle used for this run "
                              "(main.py's SWEEP_ANGLE_DEG) -- enables the "
                              "wind-turbine-frame map plot")
    parser.add_argument("--mounting-angle-deg", type=float, default=None,
                         help="Blade mounting angle used for this run "
                              "(main.py's MOUNTING_ANGLE_DEG) -- enables the "
                              "wind-turbine-frame map plot")
    parser.add_argument("--plot-window-s", type=float, default=0.5,
                         help="Seconds of each time-series plot actually "
                              "shown (default 0.5); the full capture is still "
                              "used to compute the subtracted mean/variance")
    parser.add_argument("--out", default="analysis",
                         help="Output subfolder name, created under run_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    out_dir = run_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    captures = load_run(run_dir, args.counts_per_deg, args.duration)
    if not captures:
        raise SystemExit(f"No angle_*.csv files found in {run_dir}")
    print(f"Loaded {len(captures)} angle captures from {run_dir}")

    highest, lowest = pick_extreme_angles(captures)
    print(f"  Highest accel variance: {highest.requested_angle_deg:+.2f} deg")
    print(f"  Lowest accel variance:  {lowest.requested_angle_deg:+.2f} deg")

    plot_angle_timeseries(highest, out_dir / "timeseries_high_variance.png",
                           case_label="High",
                           plot_window_s=args.plot_window_s)
    plot_angle_timeseries(lowest, out_dir / "timeseries_low_variance.png",
                           case_label="Low",
                           plot_window_s=args.plot_window_s)
    plot_variance_sweep(captures, out_dir / "variance_sweep.png",
                         sweep_angle_deg=args.sweep_angle_deg,
                         mounting_angle_deg=args.mounting_angle_deg)

    print(f"\nDone. Plots written to {out_dir}")


if __name__ == "__main__":
    main()