"""
load_cell.py

Adapted from fun_read_load.py (Girgus Sedky / Keller Morrison) to add a
duration-bounded, synchronized-start capture function that main.py's
sweep pipeline can run in a background thread alongside the accelerometer
capture (collect_accel_for_duration in main.py).

Original standalone helpers (acquire_load, calibrate_load, OUTPUT_LABELS)
are unchanged and still usable on their own:

    from load_cell import acquire_load, calibrate_load, OUTPUT_LABELS
    data, t = acquire_load("Dev11", 1000.0, 10.0)
    F = calibrate_load(data)

New for the sweep pipeline:

    from load_cell import collect_load_for_duration
    n_samples, t_start = collect_load_for_duration(
        "Dev11", 1000.0, duration_s=5.0, csv_path="angle_+05.00deg_load.csv",
        start_event=start_gate,
    )

`start_event` is anything with a no-arg `.wait()` method -- a
threading.Event or a threading.Barrier both work. Passing the SAME
threading.Barrier(2) object to this function and to
main.collect_accel_for_duration lets both captures release at (as close
as possible to) the same instant: each side finishes its own
setup/arming, calls start_gate.wait(), and both are released together as
soon as the second one arrives -- no guessed sleep needed in the caller.

Timing note (software sync, ms-level jitter -- see project notes):
    t_start (a time.perf_counter() value) is captured immediately after
    the DAQ task is started (right after the shared start_event/barrier
    releases). Every sample's "time" column in the output CSV is
    relative to that same t_start. main.py logs both this load-cell
    t_start and the accelerometer's t_start to coordinate_frame_angles.csv
    so the two streams can be aligned precisely in post-processing --
    the two t_start values typically differ by well under the polling
    interval, but recording both keeps that offset explicit instead of
    assumed.
"""

import csv
import time

import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration


N_CHANNELS = 6  # AI0 through AI5, see column mapping below

# Column order in `data` (AI channel each column comes from):
#   col 0 = STG0 <- AI0  (SGO1/SGR1)
#   col 1 = STG1 <- AI1  (SGO2/SGR2)
#   col 2 = STG2 <- AI2  (SGO3/SGR3)
#   col 3 = STG3 <- AI3  (SGO4/SGR4)
#   col 4 = STG4 <- AI4  (SGO5/SGR5)
#   col 5 = STG5 <- AI5  (SGO6/SGR6)
# All channels are wired as DIFFERENTIAL inputs.

CALIBRATION_MATRIX = np.array([
    [ 0.01441, -0.03561,  0.06162,  3.17245, -0.02691, -3.05961],
    [ 0.00040, -3.56010,  0.06360,  1.77360,  0.01050,  1.81183],
    [ 5.19068, -0.13715,  5.18041, -0.21654,  5.31873, -0.02642],
    [-0.00123, -0.03754,  0.07456,  0.01569, -0.07386,  0.01950],
    [-0.08462,  0.00250,  0.04108, -0.03531,  0.04421,  0.03225],
    [ 0.00055, -0.04511, -0.00133, -0.04620, -0.00074, -0.04466],
])

# Static offset correction determined alongside CALIBRATION_MATRIX.
ZERO_OFFSET = np.array([[-3.161689], [3.691926], [-4.383389],
                         [0.159664], [0.157204], [-0.014721]])

OUTPUT_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


# --------------------------------------------------
# Standalone helpers (unchanged behavior from fun_read_load.py)
# --------------------------------------------------


def acquire_load(device_name, sample_rate, duration_s):
    """
    Sets up the DAQ task and reads all 6 channels for duration_s seconds.
    Returns:
        data : (n_samples, 6) matrix, one row per sample, columns = AI0..AI5
        t    : (n_samples,) timestamps in seconds
    """
    n_samples = int(round(sample_rate * duration_s))

    with nidaqmx.Task() as task:
        for ch in range(N_CHANNELS):
            task.ai_channels.add_ai_voltage_chan(
                f"{device_name}/ai{ch}",
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-10.0,
                max_val=10.0,
            )
        task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=n_samples)
        raw = task.read(number_of_samples_per_channel=n_samples,
                         timeout=duration_s + 10.0)

    data = np.array(raw).T  # (n_samples, 6)
    t = np.arange(n_samples) / sample_rate
    return data, t


def calibrate_load(data, debug=False):
    """
    data : (n_samples, 6) matrix of STG voltages.
    Returns F : (6, n_samples) matrix of [Fx, Fy, Fz, Mx, My, Mz] per sample.
    """
    raw = CALIBRATION_MATRIX @ data.T
    if debug:
        print("DEBUG pre-correction sample:", raw[:, 0])
    corrected = raw - ZERO_OFFSET
    if debug:
        print("DEBUG post-correction sample:", corrected[:, 0])
    return corrected


# --------------------------------------------------
# Duration-bounded, synchronized-start capture for the sweep pipeline
# --------------------------------------------------


def collect_load_for_duration(device_name, sample_rate, duration_s, csv_path,
                               start_event=None):
    """Capture load-cell data for duration_s seconds and write it to
    csv_path, releasing (starting the DAQ clock) at the same instant as a
    concurrent accelerometer capture when start_event is shared between
    the two.

    start_event: optional object with a no-arg .wait() method (a
    threading.Event or threading.Barrier). All DAQ task setup/config
    happens BEFORE this call so it's fast; the task is only actually
    started (task.start()) after start_event.wait() returns, and t_start
    is stamped immediately after that -- mirroring
    main.collect_accel_for_duration's arm-then-wait-then-go pattern so
    both captures' zero references land as close together as possible.

    Returns (n_samples, t_start) where t_start is the time.perf_counter()
    value at which DAQ acquisition actually began (also written into the
    CSV header for reference).
    """
    n_samples = int(round(sample_rate * duration_s))

    with nidaqmx.Task() as task:
        for ch in range(N_CHANNELS):
            task.ai_channels.add_ai_voltage_chan(
                f"{device_name}/ai{ch}",
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-10.0,
                max_val=10.0,
            )
        task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=n_samples)

        if start_event is not None:
            start_event.wait()

        task.start()
        t_start = time.perf_counter()

        raw = task.read(number_of_samples_per_channel=n_samples,
                         timeout=duration_s + 10.0)
        task.stop()

    data = np.array(raw).T  # (n_samples, 6)
    F = calibrate_load(data)  # (6, n_samples)
    t = np.arange(n_samples) / sample_rate  # relative to t_start

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        f.write(f"# NI load cell, {sample_rate:.1f} Hz, "
                 f"t_start_perf_counter={t_start:.6f}\n")
        writer.writerow(
            ["sample", "time"]
            + [f"stg{i}" for i in range(N_CHANNELS)]
            + OUTPUT_LABELS
        )
        for i in range(n_samples):
            writer.writerow(
                [i, f"{t[i]:.6f}"] + data[i, :].tolist() + F[:, i].tolist()
            )

    elapsed = time.perf_counter() - t_start
    rate = n_samples / elapsed if elapsed > 0 else 0
    print(f"  Load cell: captured {n_samples} samples in {elapsed:.2f}s "
          f"({rate:.0f}/s)")

    return n_samples, t_start


# --------------------------------------------------
# CLI entry point (unchanged, for standalone testing off the DAQ)
# --------------------------------------------------


def main():
    """
    Default daq is Dev11, the default sample rate is 1000hz and the
    default duration is 10s.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="Dev11")
    parser.add_argument("--rate", type=float, default=1000.0)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    data, t = acquire_load(args.device, args.rate, args.duration)
    F = calibrate_load(data, debug=True)

    print("\nAverages:")
    for label, val in zip(OUTPUT_LABELS, F.mean(axis=1)):
        print(f"  {label}: {val: .6f}")


if __name__ == "__main__":
    main()
