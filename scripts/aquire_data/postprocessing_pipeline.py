#!/usr/bin/env python3
"""
Post-processing pipeline entry point: overlay the variance plots of one or
more acquisition runs.

Usage (from the repo root):
    python scripts/aquire_data/postprocessing_pipeline.py Data/run3 Data/run4 Data/run5
    python scripts/aquire_data/postprocessing_pipeline.py Data/run3 Data/run4 \
        --plots accel_sweep load_map --labels "rough" "smooth" \
        --out Data/comparison_rough_vs_smooth

Positional arguments are run directories (folders written by main.py, each
holding angle_*.csv, encoder_log.txt and run_metadata.json). All of them are
drawn on the same axes.

--plots chooses which variance plots to make (default: all four):
    accel_sweep   accel variance vs motor angle
    load_sweep    load-cell variance vs motor angle
    accel_map     inclination vs angle of attack, colored by accel variance
    load_map      inclination vs angle of attack, colored by load-cell variance

Accelerometer plots use the magnitude of sensor 4 only (never averaged across
sensors). --sensor N picks a different one (1-4) for both accel_sweep and
accel_map.

Run settings (counts_per_deg, sample duration, sweep angle, mounting angle)
are read from each run's run_metadata.json, so runs with different blade
setups can be mixed. The --counts-per-deg / --duration / --sweep-angle-deg /
--mounting-angle-deg flags override the metadata for EVERY run, and are
the fallback for older runs that have no run_metadata.json.

Output goes to <out>/<plot>.png. Default <out> is <run_dir>/analysis for a
single run and Data/comparison for several.
"""

import argparse
import json
from pathlib import Path

from utils.run_loader import load_run
from plot_variance_compare import PLOT_NAMES, RunData, plot_variance_comparison

DEFAULT_COUNTS_PER_DEG = 694.44
DEFAULT_DURATION_S = 4.0


def read_metadata(run_dir):
    path = Path(run_dir) / "run_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _pick(cli_value, meta, key, default=None):
    """CLI flag wins, then run_metadata.json, then the default.
    Returns (value, source) so the log shows where each number came from."""
    if cli_value is not None:
        return cli_value, "cli"
    if meta.get(key) is not None:
        return meta[key], "metadata"
    return default, "default"


def make_labels(run_dirs, labels):
    if labels:
        if len(labels) != len(run_dirs):
            raise SystemExit(f"--labels needs {len(run_dirs)} values (one per run), "
                             f"got {len(labels)}")
        return labels
    names = [Path(d).name for d in run_dirs]
    if len(set(names)) < len(names):   # same folder name under different parents
        return [str(d) for d in run_dirs]
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="run folders to overlay")
    parser.add_argument("--plots", nargs="+", default=["all"],
                        choices=list(PLOT_NAMES) + ["all"],
                        help="which variance plots to make (default: all)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="legend labels, one per run (default: folder names)")
    parser.add_argument("--sensor", type=int, default=4, choices=[1, 2, 3, 4],
                        help="accelerometer used for accel_sweep and accel_map (default 4)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--counts-per-deg", type=float, default=None,
                        help="override Motor counts_per_deg for all runs")
    parser.add_argument("--duration", type=float, default=None,
                        help="override SAMPLE_DURATION_S for all runs")
    parser.add_argument("--sweep-angle-deg", type=float, default=None,
                        help="override SWEEP_ANGLE_DEG for all runs")
    parser.add_argument("--mounting-angle-deg", type=float, default=None,
                        help="override MOUNTING_ANGLE_DEG for all runs")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            raise SystemExit(f"Run directory not found: {d}")

    plots = list(PLOT_NAMES) if "all" in args.plots else list(dict.fromkeys(args.plots))
    labels = make_labels(run_dirs, args.labels)

    runs = []
    for run_dir, label in zip(run_dirs, labels):
        meta = read_metadata(run_dir)
        counts, c_src = _pick(args.counts_per_deg, meta, "counts_per_deg", DEFAULT_COUNTS_PER_DEG)
        duration, d_src = _pick(args.duration, meta, "sample_duration_s", DEFAULT_DURATION_S)
        sweep, s_src = _pick(args.sweep_angle_deg, meta, "sweep_angle_deg")
        mount, m_src = _pick(args.mounting_angle_deg, meta, "mounting_angle_deg")

        print(f"\n[{label}] {run_dir}")
        if not meta:
            print("  NOTE: no run_metadata.json -- using CLI flags / defaults")
        elif meta.get("status") != "complete":
            print(f"  WARN: run status is {meta.get('status')!r}, data may be partial")
        print(f"  counts_per_deg={counts} ({c_src}), duration={duration}s ({d_src}), "
              f"sweep={sweep} ({s_src}), mounting={mount} ({m_src})")

        captures = load_run(run_dir, counts, duration)
        if not captures:
            raise SystemExit(f"No angle_*.csv files found in {run_dir}")
        print(f"  Loaded {len(captures)} angle captures")
        runs.append(RunData(label=label, captures=captures,
                            sweep_angle_deg=sweep, mounting_angle_deg=mount))

    out_dir = Path(args.out) if args.out else (
        run_dirs[0] / "analysis" if len(run_dirs) == 1 else Path("Data/comparison"))

    print(f"\nPlotting {', '.join(plots)} for {len(runs)} run(s)")
    plot_variance_comparison(runs, plots, out_dir, sensors=(args.sensor,),
                             map_sensor=args.sensor)
    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
