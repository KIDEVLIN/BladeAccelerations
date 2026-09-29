# process_data — post-processing pipeline

Reads a completed acquisition run directory (same layout `scripts/aquire_data/main.py`
writes) and produces:

1. **`timeseries_high_variance.png`** / **`timeseries_low_variance.png`** — one
   organized time-series figure each, for the angle with the highest and lowest
   mean accelerometer variance in the run. Each figure has 3 panels: accel
   magnitude (mg) per sensor overlaid, load-cell forces, load-cell moments.
   The motor angle is *not* plotted as a time series — it's reduced to a single
   averaged number and reported in the figure title, per your ask.
2. **`variance_sweep.png`** — a static recreation of `utils/live_varience_plot.py`'s
   2×2 layout for the whole run, except the accelerometer panels show all 4
   sensors as separate overlaid series instead of the live plot's mean-across-
   sensor value.

## Usage

```
cd scripts/data_pipeline
python main.py <run_dir> --counts-per-deg 694.44 --duration 4.0
```

- `--counts-per-deg` must match the `Motor(counts_per_deg=...)` value actually
  used for that run (default 694.44, the code's own default).
- `--duration` must match `SAMPLE_DURATION_S` from that run's `main.py` config
  (default 4.0, matching the current config).
- Neither is currently written into the run directory, so double-check them
  against whatever you had set when you collected `<run_dir>` — see gap #3
  below.

Output goes to `<run_dir>/analysis/` by default (`--out` to change it).

## Three things worth fixing upstream (found while writing this)

**1. `coordinate_frame_angles.csv` is never actually written.**
In the `main.py` you shared, `angle_log_path` and `angle_log_fields` are defined,
but nothing in the sweep loop opens that file or calls `writerow`. So it's
likely missing from your run directories today. This pipeline treats it as
optional — it discovers everything else from the `angle_*.csv` / `angle_*_load.csv`
filenames and their own header comments — but you lose the map-row plot
(inclination vs. angle-of-attack) and the settled-position fallback for the
motor angle without it. If you want, I can add the missing writer to `main.py`
in a follow-up.

**2. `encoder_log.txt` has no absolute time reference.**
It logs `time_sec` relative to when `Motor.start()` was called
(`perf_counter() - self._start_time`), but never logs `self._start_time`
itself. The accel/load CSVs *do* log their own absolute `t_start_perf_counter`
in their header comment (same process, same `perf_counter()` clock) — so
without the motor's absolute reference there's no way to know which encoder
samples fall inside a given capture window. `windowed_average_angle()` in
`run_loader.py` returns `None` (with a printed warning) when this is missing,
and falls back to the settled `motor_angle_actual_deg` from
`coordinate_frame_angles.csv` if that file exists.

Suggested one-line fix in `motor.py`, in `Motor.start()`:

```python
self._log_file = open(self._log_path, "w", buffering=1)
self._log_file.write(f"# start_time_perf_counter={time.perf_counter():.6f}\n")   # NEW
self._log_file.write("time_sec, encoder_counts\n")
self._start_time = time.perf_counter()
```

(Stamp it right before `self._start_time` is set, using the same call, so the
two match exactly.) This loader auto-detects that header line when present and
is fully backward compatible with logs that don't have it — both were tested.

**3. `counts_per_deg`, `SAMPLE_DURATION_S`, `SWEEP_ANGLE_DEG`, `MOUNTING_ANGLE_DEG`
aren't persisted per run.** They're config constants at the top of `main.py`,
but nothing writes their actual values into the output directory, so a run
processed months later relies on you remembering (or having recorded
elsewhere) what they were set to. Worth a line in `coordinate_frame_angles.csv`'s
header, or a small `run_config.json` written once per run, once #1 is fixed.

## Files

- `run_loader.py` — discovers and parses everything in a run directory;
  `AngleCapture` is the per-angle bundle everything else consumes.
- `variance_analysis.py` — per-sensor accel variance (kept separate, not
  averaged, per your ask), load-cell variance (reused from
  `utils/live_varience_plot.py`), and picking the high/low variance angles.
- `plot_timeseries.py` — the 3-panel per-angle figure.
- `plot_variance_sweep.py` — the whole-run diagnostic figure.
- `main.py` — CLI entry point.

Tested end-to-end against synthetic data matching the real CSV/log formats,
including: a run with no `coordinate_frame_angles.csv`, an angle missing its
load-cell file, an encoder log without the new header (old-style, falls back
correctly with a warning), and one with it (windowed averaging works: recovered
10.00° ± 0.003° against a synthetic ground truth of 10.00°).