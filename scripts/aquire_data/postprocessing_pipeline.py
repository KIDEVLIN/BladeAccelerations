#!/usr/bin/env python3
"""
Post-processing pipeline entry point: overlay the variance plots of one or
more acquisition runs.

Usage (from the repo root). Put it all on one line, or continue lines with
a backtick in PowerShell (a backslash in bash):
    python scripts/aquire_data/postprocessing_pipeline.py Data/run3 Data/run4 Data/run5
    python scripts/aquire_data/postprocessing_pipeline.py Data/run3 Data/run4 --plots accel_sweep load_map --labels "rough" "smooth" --out Data/comparison

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

--normalize plots the accel variance as the nondimensional a*^2 = Var(a) / a_c^2
instead of g^2, with a_c = (1/2 rho U^2 c_avg b) / m (see normalization.py), so
runs at different tunnel conditions / blades / materials can share axes. The
five inputs are read per run from run_metadata.json (tunnel_conditions density
and free-stream velocity, plus blade_chord_avg_m / blade_span_m / blade_mass_kg
as recorded by main.py). For older runs, or to override, give them on the
command line -- one value (used for every run) or one per run, in run order:
    --chord-avg-m 0.02 --span-m 0.3 --blade-mass-kg 0.05 0.08
    (also --velocity-m-s and --density-kg-m3)
Every run needs all five or the command stops and says what is missing.
Load-cell plots are not normalized.

Output goes to <out>/<plot>.png. Default <out> is <run_dir>/analysis for a
single run and Data/comparison for several.
"""

import argparse
import json
from pathlib import Path

from utils.run_loader import load_run
from normalization import INPUTS, characteristic_acceleration_for_run
from plot_variance_compare import PLOT_NAMES, RunData, plot_variance_comparison

DEFAULT_COUNTS_PER_DEG = 694.44
DEFAULT_DURATION_S = 4.0

# normalization input -> (CLI flag, description); keys match normalization.INPUTS
NORM_FLAGS = {
    "chord_avg_m": ("--chord-avg-m", "average blade chord (m)"),
    "span_m": ("--span-m", "blade span (m)"),
    "mass_kg": ("--blade-mass-kg", "blade mass (kg)"),
    "velocity_m_s": ("--velocity-m-s", "free-stream velocity (m/s)"),
    "density_kg_m3": ("--density-kg-m3", "air density (kg/m^3)"),
}


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


def per_run_values(values, n_runs, flag):
    """Expands a CLI value list to one entry per run: None -> all None,
    one value -> repeated for every run, n values -> one each."""
    if values is None:
        return [None] * n_runs
    if len(values) == 1:
        return list(values) * n_runs
    if len(values) != n_runs:
        raise SystemExit(f"{flag} needs 1 value (all runs) or {n_runs} values "
                         f"(one per run), got {len(values)}")
    return list(values)


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
    parser.add_argument("--normalize", action="store_true",
                        help="plot accel variance as a*^2 = Var(a)/a_c^2 instead of g^2")
    for name, (flag, what) in NORM_FLAGS.items():
        parser.add_argument(flag, dest=f"norm_{name}", type=float, nargs="+", default=None,
                            help=f"{what}: one value for all runs or one per run "
                                 f"(overrides run_metadata.json; used with --normalize)")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            raise SystemExit(f"Run directory not found: {d}")

    plots = list(PLOT_NAMES) if "all" in args.plots else list(dict.fromkeys(args.plots))
    labels = make_labels(run_dirs, args.labels)
    norm_overrides = {name: per_run_values(getattr(args, f"norm_{name}"), len(run_dirs), flag)
                      for name, (flag, _) in NORM_FLAGS.items()}

    runs = []
    normalize_problems = []
    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
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

        a_c_g = None
        if args.normalize:
            a_c_g, vals, srcs, missing = characteristic_acceleration_for_run(
                meta, {name: norm_overrides[name][i] for name in INPUTS})
            if missing:
                normalize_problems.append((label, missing))
            else:
                print("  a_c = 1/2 rho U^2 c_avg b / m = {:.4g} g   ({})".format(
                    a_c_g, ", ".join(f"{k}={vals[k]:.4g} [{srcs[k]}]" for k in INPUTS)))
        runs.append(RunData(label=label, captures=captures,
                            sweep_angle_deg=sweep, mounting_angle_deg=mount, a_c_g=a_c_g))

    if normalize_problems:
        lines = [f"  {label}: missing {', '.join(NORM_FLAGS[m][0] for m in missing)}"
                 for label, missing in normalize_problems]
        raise SystemExit(
            "--normalize needs rho, U, c_avg, b and m for every run:\n" + "\n".join(lines) +
            "\nGive them with the flags shown (or record them in run_metadata.json).")

    out_dir = Path(args.out) if args.out else (
        run_dirs[0] / "analysis" if len(run_dirs) == 1 else Path("Data/comparison"))

    print(f"\nPlotting {', '.join(plots)} for {len(runs)} run(s)")
    plot_variance_comparison(runs, plots, out_dir, sensors=(args.sensor,),
                             map_sensor=args.sensor)
    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
