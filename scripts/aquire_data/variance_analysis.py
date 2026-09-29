"""
variance_analysis.py

Per-sensor variance metrics for one run's angle captures, and picking
representative "high variance" / "low variance" angles out of a sweep
for the example time-series plots.
"""

import sys
from pathlib import Path

import numpy as np

_ACQUIRE_DIR = Path(__file__).resolve().parents[1] / "aquire_data"
if str(_ACQUIRE_DIR) not in sys.path:
    sys.path.insert(0, str(_ACQUIRE_DIR))

from test_modbus import MG_PER_LSB, NUM_SENSORS  # noqa: E402
from utils.live_variance_plot import loadcell_magnitude_variance  # noqa: E402

ACCEL_COLS = [f"s{s}_{axis}" for s in range(1, NUM_SENSORS + 1) for axis in "xyz"]
LOAD_FORCE_COLS = ["Fx", "Fy", "Fz"]


def accel_per_sensor_variance(accel_df):
    """Returns {1: var1, ..., NUM_SENSORS: varN} -- variance (mg^2) of
    each sensor's own acceleration-magnitude signal, kept separate
    (no averaging across sensors, unlike
    live_varience_plot.accel_magnitude_variance)."""
    rows = accel_df[ACCEL_COLS].to_numpy(dtype=float)
    arr = rows * MG_PER_LSB
    arr = arr.reshape(-1, NUM_SENSORS, 3)
    mags = np.linalg.norm(arr, axis=2)  # (N, sensor)
    return {i + 1: float(np.var(mags[:, i])) for i in range(NUM_SENSORS)}


def accel_mean_variance(accel_df):
    """Mean-across-sensors variance -- same convention main.py already
    logs as accel_variance. Used here only to rank angles for the
    high/low example plots."""
    per_sensor = accel_per_sensor_variance(accel_df)
    return float(np.mean(list(per_sensor.values())))


def load_variance(load_df):
    if load_df is None:
        return None
    force_xyz = load_df[LOAD_FORCE_COLS].to_numpy(dtype=float)
    return loadcell_magnitude_variance(force_xyz)


def pick_extreme_angles(captures):
    """Returns (highest_variance_capture, lowest_variance_capture),
    ranked by mean-across-sensor accel variance."""
    scored = [(accel_mean_variance(c.accel_df), c) for c in captures]
    scored.sort(key=lambda pair: pair[0])
    lowest = scored[0][1]
    highest = scored[-1][1]
    return highest, lowest