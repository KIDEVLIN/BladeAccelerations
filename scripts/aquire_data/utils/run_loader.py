"""
run_loader.py

Loads a completed acquisition run directory (as produced by
scripts/aquire_data/main.py) for offline processing: discovers all
per-angle accelerometer + load-cell CSVs, the encoder log, and (if
present) coordinate_frame_angles.csv, and reduces the encoder's raw
time series down to one average motor angle per capture window.

NOTE -- gaps found in the current acquisition pipeline while writing
this loader (see README.md in this folder for the full writeup and a
suggested one-line patch):

  1. coordinate_frame_angles.csv is described/planned in main.py
     (angle_log_path / angle_log_fields are defined) but nothing in
     the current main.py loop actually opens that file and writes rows
     to it -- so it may not exist for a given run. This loader treats
     it as OPTIONAL and falls back to discovering everything from the
     angle_*.csv / angle_*_load.csv filenames + their own header
     comments instead.
  2. encoder_log.txt only stores time relative to when Motor.start()
     was called (perf_counter() - self._start_time); it never logs
     that absolute perf_counter() reference. The accel/load CSVs DO
     log their own absolute t_start_perf_counter in their header
     comment. Without motor's absolute reference there is no way to
     line up "which encoder samples happened during this capture
     window" -- windowed_average_angle() below returns None when this
     reference isn't available (falls back to the settled
     motor_angle_actual_deg from coordinate_frame_angles.csv, if that
     file exists). A one-line patch to motor.py (writing
     "# start_time_perf_counter=<value>" as the first line of
     encoder_log.txt) makes future runs support real windowed
     averaging; this loader auto-detects that header line when present
     and is fully backward compatible with logs that don't have it.
  3. counts_per_deg (Motor's encoder scale) is a run-time config
     constant in main.py/motor.py that isn't persisted anywhere in the
     output directory. Pass counts_per_deg into load_run() matching
     what was actually used for the run you're processing.

Accel timestamp reconstruction
-------------------------------
The accel CSV's "time" column is stamped once per FIFO batch read
(t_batch = perf_counter() - t_start at the moment Python finished
reading that batch off the PCB) and then written on EVERY row in that
batch -- so many rows share one timestamp, in the order they came off
the sensor. load_accel_csv() reconstructs a real per-row time by
assuming, within each batch, samples are evenly spaced between the
previous batch's timestamp and this batch's timestamp (the first
batch is spaced from 0). See _reconstruct_batch_times() below.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reuse constants from the acquisition pipeline instead of duplicating them.
_ACQUIRE_DIR = Path(__file__).resolve().parents[1] / "aquire_data"
if str(_ACQUIRE_DIR) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE_DIR))

_T_START_RE = re.compile(r"t_start_perf_counter=([0-9.eE+\-]+)")
_ENCODER_ABS_RE = re.compile(r"start_time_perf_counter=([0-9.eE+\-]+)")
_ANGLE_FILENAME_RE = re.compile(r"angle_([+\-]?\d+\.\d+)deg\.csv$")


def _read_header_t_start(path):
    """Reads the '# ... t_start_perf_counter=<value>' comment line that
    the accel/load CSV writers put on line 1. Returns None if absent."""
    with open(path) as f:
        first_line = f.readline()
    m = _T_START_RE.search(first_line)
    return float(m.group(1)) if m else None


def _reconstruct_batch_times(df, time_col="time"):
    """Replaces a "time" column where every row in a FIFO batch shares
    the same value (the perf_counter() reading when that whole batch
    was read off the PCB) with one timestamp per row: within each
    batch, the n samples are assumed evenly spaced across
    (previous_batch_time, this_batch_time], in their existing order --
    i.e. the last sample in a batch lands exactly at this_batch_time
    (when it was actually read), and earlier samples in the same batch
    are spaced back from there. The very first batch is spaced from 0.
    Batches are identified as runs of consecutive rows with an
    identical time value, which holds as long as the CSV is in
    acquisition order (true for every writer in this pipeline)."""
    times = df[time_col].to_numpy(dtype=float)
    n = len(times)
    new_times = np.empty(n, dtype=float)

    prev_t = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and times[j] == times[i]:
            j += 1
        batch_len = j - i
        cur_t = times[i]
        step = (cur_t - prev_t) / batch_len
        new_times[i:j] = prev_t + step * np.arange(1, batch_len + 1)
        prev_t = cur_t
        i = j

    df = df.copy()
    df[time_col] = new_times
    return df


def load_accel_csv(path):
    """Returns (DataFrame, t_start_perf_counter_or_None). The "time"
    column is reconstructed to one timestamp per row -- see
    _reconstruct_batch_times() / the module docstring."""
    t_start = _read_header_t_start(path)
    df = pd.read_csv(path, comment="#")
    df = _reconstruct_batch_times(df)
    return df, t_start


def load_load_csv(path):
    """Returns (DataFrame, t_start_perf_counter_or_None)."""
    t_start = _read_header_t_start(path)
    df = pd.read_csv(path, comment="#")
    return df, t_start


def load_encoder_log(path):
    """Returns (DataFrame[time_sec, encoder_counts], abs_start_or_None).
    Backward compatible with encoder logs that don't have the absolute-
    time header line (abs_start comes back None in that case, and the
    file parses exactly as it does today)."""
    with open(path) as f:
        first_line = f.readline()
    abs_start = None
    skiprows = 0
    if first_line.strip().startswith("#"):
        m = _ENCODER_ABS_RE.search(first_line)
        abs_start = float(m.group(1)) if m else None
        skiprows = 1
    df = pd.read_csv(path, skiprows=skiprows)
    df.columns = [c.strip() for c in df.columns]
    return df, abs_start


def windowed_average_angle(encoder_df, abs_start, window_start, duration_s, counts_per_deg):
    """Average encoder angle (deg) over [window_start, window_start+duration_s],
    in the same perf_counter() clock as the accel/load t_start values.
    Returns (mean_deg_or_None, n_samples_used, std_deg_or_None)."""
    if abs_start is None or window_start is None:
        return None, 0, None
    t_abs = abs_start + encoder_df["time_sec"].to_numpy(dtype=float)
    mask = (t_abs >= window_start) & (t_abs <= window_start + duration_s)
    counts = encoder_df["encoder_counts"].to_numpy(dtype=float)[mask]
    if len(counts) == 0:
        return None, 0, None
    degrees = counts / counts_per_deg
    return float(degrees.mean()), int(len(counts)), float(degrees.std())


@dataclass
class AngleCapture:
    requested_angle_deg: float
    accel_path: Path
    load_path: Optional[Path]
    accel_df: pd.DataFrame
    load_df: Optional[pd.DataFrame]
    accel_t_start: Optional[float]
    load_t_start: Optional[float]
    avg_motor_angle_deg: Optional[float]
    motor_angle_n_samples: int
    motor_angle_std_deg: Optional[float]
    duration_s: float
    summary_row: Optional[pd.Series] = None  # coordinate_frame_angles.csv row, if matched


def _discover_angle_files(run_dir):
    """Finds angle_*deg.csv (accel) files, pairing each with its
    ..._load.csv sibling if present. Sorted by the angle value in the
    filename (the requested angle, not the settled one)."""
    run_dir = Path(run_dir)
    pairs = []
    for accel_path in sorted(run_dir.glob("angle_*deg.csv")):
        if "_load" in accel_path.stem:
            continue
        m = _ANGLE_FILENAME_RE.search(accel_path.name)
        if not m:
            continue
        angle_val = float(m.group(1))
        load_path = run_dir / accel_path.name.replace("deg.csv", "deg_load.csv")
        pairs.append((angle_val, accel_path, load_path if load_path.exists() else None))
    pairs.sort(key=lambda p: p[0])
    return pairs


def load_run(run_dir, counts_per_deg, duration_s):
    """Loads every angle capture in run_dir. duration_s must match
    SAMPLE_DURATION_S used for the acquisition run (not persisted
    anywhere in the output -- see module docstring, gap #3)."""
    run_dir = Path(run_dir)

    encoder_path = run_dir / "encoder_log.txt"
    encoder_df, encoder_abs_start = (
        load_encoder_log(encoder_path) if encoder_path.exists() else (None, None)
    )
    if encoder_path.exists() and encoder_abs_start is None:
        print(
            "  WARN: encoder_log.txt has no absolute time reference "
            "(start_time_perf_counter) -- windowed motor-angle averaging "
            "will be skipped for every angle. See run_loader.py's module "
            "docstring (gap #2) for the one-line motor.py patch that fixes "
            "this for future runs."
        )

    summary_path = run_dir / "coordinate_frame_angles.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    if summary_df is None:
        print(
            "  NOTE: coordinate_frame_angles.csv not found -- falling back "
            "to angle_*.csv filenames/headers only (frame angles and the "
            "settled motor_angle_actual_deg fallback won't be available). "
            "See run_loader.py's module docstring (gap #1)."
        )

    captures = []
    for angle_val, accel_path, load_path in _discover_angle_files(run_dir):
        accel_df, accel_t_start = load_accel_csv(accel_path)
        load_df, load_t_start = (
            load_load_csv(load_path) if load_path is not None else (None, None)
        )

        if encoder_df is not None:
            avg_angle, n_samples, std_angle = windowed_average_angle(
                encoder_df, encoder_abs_start, accel_t_start, duration_s, counts_per_deg
            )
        else:
            avg_angle, n_samples, std_angle = None, 0, None

        summary_row = None
        if summary_df is not None and "motor_angle_requested_deg" in summary_df.columns:
            match = summary_df[
                np.isclose(summary_df["motor_angle_requested_deg"], angle_val, atol=1e-6)
            ]
            if len(match):
                summary_row = match.iloc[0]
                if avg_angle is None and "motor_angle_actual_deg" in summary_row:
                    # Fallback: settled position at capture start, not a
                    # true window average, but better than nothing.
                    avg_angle = float(summary_row["motor_angle_actual_deg"])

        captures.append(AngleCapture(
            requested_angle_deg=angle_val,
            accel_path=accel_path,
            load_path=load_path,
            accel_df=accel_df,
            load_df=load_df,
            accel_t_start=accel_t_start,
            load_t_start=load_t_start,
            avg_motor_angle_deg=avg_angle,
            motor_angle_n_samples=n_samples,
            motor_angle_std_deg=std_angle,
            duration_s=duration_s,
            summary_row=summary_row,
        ))

    return captures