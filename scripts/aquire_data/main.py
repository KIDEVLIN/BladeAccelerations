"""
Data acquisition pipeline (Step 2):
    Load angles file
        -> for each angle:
             move motor to angle
             wait for settle
             capture accelerometer AND load cell data simultaneously,
                 released at the same instant via a shared threading.Barrier
             save each sensor's data to <output_dir>/angle_<value>deg*.csv
        -> repeat until all angles done

Synchronization: a fresh threading.Barrier(2) is created per angle. Each
capture function does all of its own setup/arming first (modbus sampling
arm for the accelerometer; DAQ task + channel config for the load cell),
then calls start_gate.wait(). Whichever finishes setup first blocks until
the other arrives, and both are released together -- this is tighter than
a fixed guessed sleep and needs no coordination logic in this file beyond
creating the barrier and starting the two threads. Each capture stamps
its own t_start (time.perf_counter(), taken immediately after release) so
the exact offset between the two streams' zero references is known and
logged, not assumed. This is software-level sync -- expect low
millisecond-level jitter between the two t_start values. A hardware
trigger is the planned upgrade path for sub-millisecond sync.

Each run also writes coordinate_frame_angles.csv into OUTPUT_DIR, mapping
every tested motor angle (actual, settled encoder value where available)
to the resulting inclination angle and angle of attack in the blade
frame, using SWEEP_ANGLE_DEG / MOUNTING_ANGLE_DEG set below and the
transforms in utils/coordinate_transforms.py.

Requirements:
    pip install minimalmodbus pyserial pandas openpyxl nidaqmx numpy
"""

import csv
import os
import threading
import time
from pathlib import Path

import pandas as pd

from motor import Motor
from utils import coordinate_transforms as ct
import test_modbus as pcb
import load_cell as lc
from utils.live_varience_plot import LivePlotter, accel_magnitude_variance

# --------------------------------------------------
# FOR YOU TO UPDATE BEFORE RUNS
# --------------------------------------------------

MOTOR_PORT = "COM5"
MOTOR_BAUD = 115200

PCB_PORT = "COM6"

LOADCELL_DEVICE = "Dev11"          # NI DAQ device name (same box motor.py's trigger line lives on)
LOADCELL_SAMPLE_RATE_HZ = 1000.0   # hardware-timed analog sample rate

ANGLES_FILE = "scripts/aquire_data/test_angles.xlsx"     # .xlsx, .csv, or .txt (one angle per line / row)
OUTPUT_DIR = "Data/run_001"     # created if it doesn't exist; nested under Data/ so it's easy to gitignore

SAMPLE_DURATION_S = 5.0         # how long to capture accel + load data at each angle
SETTLE_TIME_S = 1.0             # pause after move, before capture starts

# Coordinate-frame constants (see utils/coordinate_transforms.py).
# These describe the fixed physical setup for this run (as opposed to
# motor_angle, which is swept from ANGLES_FILE) -- set them here before
# starting a run.
#   SWEEP_ANGLE_DEG:    blade azimuthal/sweep orientation for this run
#   MOUNTING_ANGLE_DEG: fixed blade mounting angle for this run
SWEEP_ANGLE_DEG = 0.0
MOUNTING_ANGLE_DEG = 0.0

# --------------------------------------------------
# Angle list loading
# --------------------------------------------------


def load_angles(path):
    """Load a list of target angles (deg) from .xlsx, .csv, or .txt.

    Excel/csv: looks for a column with 'angle' in its name (case
    insensitive); falls back to the first column if none is found.
    txt: one numeric value per line, blank lines ignored.
    """
    path = Path(path)

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        with open(path) as f:
            return [float(line.strip()) for line in f if line.strip()]

    angle_cols = [c for c in df.columns if "angle" in str(c).lower()]
    col = angle_cols[0] if angle_cols else df.columns[0]
    return df[col].astype(float).tolist()


# --------------------------------------------------
# Accelerometer capture (bounded by duration, not Ctrl+C)
# --------------------------------------------------


def collect_accel_for_duration(instr, duration_s, csv_path, start_event=None):
    """Capture accelerometer FIFO data for duration_s seconds and write it
    to csv_path. Mirrors test_modbus.cmd_stream's raw-serial loop, but
    stops after a fixed duration instead of waiting for KeyboardInterrupt.

    start_event: optional object with a no-arg .wait() method (a
    threading.Event or threading.Barrier). Sampling is armed on the
    sensor first, then this call blocks on start_event.wait() immediately
    before the capture loop begins -- so a caller running this in a
    thread alongside a load-cell thread (sharing the same barrier/event)
    releases both at the same instant for a synchronized start.

    Returns (total_samples, t_start) where t_start is the
    time.perf_counter() value at which the capture loop actually began
    (also written into the CSV header for reference).
    """
    ser = instr.serial
    addr = instr.address

    # Arm sampling
    pcb.write_register_retry(instr, pcb.REG_CMD, pcb.CMD_RESET_BUF, functioncode=6)
    time.sleep(0.05)
    pcb.write_register_retry(instr, pcb.REG_CMD, pcb.CMD_START, functioncode=6)
    time.sleep(0.1)

    status = pcb.read_register_retry(instr, pcb.REG_STATUS, functioncode=4)
    if not (status & 0x0100):
        raise RuntimeError("Accelerometer sampling did not start")

    odr = pcb.read_register_retry(instr, pcb.REG_ODR, functioncode=4)

    if start_event is not None:
        start_event.wait()

    total_samples = 0
    t_start = time.perf_counter()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        f.write(f"# LIS2DS12 2g mode, {pcb.MG_PER_LSB} mg/LSB, raw int16, "
                 f"ODR={odr} Hz, t_start_perf_counter={t_start:.6f}\n")
        writer.writerow(
            ["sample", "time",
             "s1_x", "s1_y", "s1_z",
             "s2_x", "s2_y", "s2_z",
             "s3_x", "s3_y", "s3_z",
             "s4_x", "s4_y", "s4_z"]
        )

        while (time.perf_counter() - t_start) < duration_s:
            avail = pcb.read_register_raw(ser, addr, pcb.REG_AVAIL)
            if avail == 0:
                time.sleep(0.001)
                continue

            sets_to_read = min(avail, pcb.BULK_MAX_SETS)
            num_regs = sets_to_read * pcb.REGS_PER_SET
            regs = pcb.bulk_read_raw(ser, addr, pcb.REG_BULK, num_regs)
            t_batch = time.perf_counter() - t_start

            signed = [v - 65536 if v >= 32768 else v for v in regs]
            for s in range(sets_to_read):
                base = s * pcb.REGS_PER_SET
                writer.writerow(
                    [total_samples, f"{t_batch:.4f}"] + signed[base:base + pcb.REGS_PER_SET]
                )
                total_samples += 1

    pcb.write_register_retry(instr, pcb.REG_CMD, pcb.CMD_STOP, functioncode=6)

    elapsed = time.perf_counter() - t_start
    rate = total_samples / elapsed if elapsed > 0 else 0
    print(f"  Accel: captured {total_samples} samples in {elapsed:.2f}s ({rate:.0f}/s, ODR={odr} Hz)")
    return total_samples, t_start


# --------------------------------------------------
# Simultaneous accel + load cell capture
# --------------------------------------------------


def collect_synchronized(instr, duration_s, accel_csv_path, load_csv_path):
    """Runs collect_accel_for_duration and collect_load_for_duration in
    parallel threads, released together via a shared threading.Barrier(2).

    Returns a dict with per-sensor sample counts and t_start values:
        {
          "accel_samples": int, "accel_t_start": float,
          "load_samples": int,  "load_t_start": float,
        }
    Any exception raised inside either worker thread is re-raised here
    (after both threads have finished) so a sensor failure doesn't get
    silently swallowed mid-sweep.
    """
    start_gate = threading.Barrier(2)
    result = {}
    errors = []

    def _run_accel():
        try:
            n, t0 = collect_accel_for_duration(
                instr, duration_s, accel_csv_path, start_event=start_gate
            )
            result["accel_samples"], result["accel_t_start"] = n, t0
        except Exception as e:
            errors.append(("accel", e))
            # Make sure the load-cell thread isn't left waiting forever
            # on a barrier that will now never trip normally.
            start_gate.abort()

    def _run_load():
        try:
            n, t0 = lc.collect_load_for_duration(
                LOADCELL_DEVICE, LOADCELL_SAMPLE_RATE_HZ, duration_s,
                load_csv_path, start_event=start_gate,
            )
            result["load_samples"], result["load_t_start"] = n, t0
        except Exception as e:
            errors.append(("load", e))
            start_gate.abort()

    t_accel = threading.Thread(target=_run_accel)
    t_load = threading.Thread(target=_run_load)
    t_accel.start()
    t_load.start()
    t_accel.join()
    t_load.join()

    if errors:
        names = ", ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"Synchronized capture failed ({names})")

    offset_ms = (result["load_t_start"] - result["accel_t_start"]) * 1000.0
    print(f"  Sync offset (load - accel t_start): {offset_ms:+.3f} ms")

    return result


# --------------------------------------------------
# Main sweep
# --------------------------------------------------


def main():
    angles = load_angles(ANGLES_FILE)
    print(f"Loaded {len(angles)} target angles: {angles}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Keep the encoder log inside the run's output folder instead of
    # wherever the script happened to be launched from.
    encoder_log_path = os.path.join(OUTPUT_DIR, "encoder_log.txt")
    motor = Motor(port=MOTOR_PORT, baud=MOTOR_BAUD, log_file=encoder_log_path)
    instr = pcb.connect(PCB_PORT)
    pcb.set_low_latency(instr.serial)

    print(
        f"Coordinate-frame constants for this run: "
        f"sweep_angle={SWEEP_ANGLE_DEG:.2f} deg, mounting_angle={MOUNTING_ANGLE_DEG:.2f} deg"
    )

    angle_log_path = os.path.join(OUTPUT_DIR, "coordinate_frame_angles.csv")
    angle_log_fields = [
        "index", "motor_angle_requested_deg", "motor_angle_actual_deg",
        "sweep_angle_deg", "mounting_angle_deg",
        "inclination_angle_deg", "inclination_angle_deg_shifted",
        "angle_of_attack_deg", "angle_of_attack_deg_shifted",
        "accel_csv_path", "load_csv_path",
        "accel_t_start", "load_t_start", "sync_offset_ms",
    ]

    motor.start()
    try:
        with open(angle_log_path, "w", newline="") as angle_log_f:
            angle_log_writer = csv.DictWriter(angle_log_f, fieldnames=angle_log_fields)
            angle_log_writer.writeheader()

            for i, angle in enumerate(angles):
                print(f"\n=== Angle {i + 1}/{len(angles)}: {angle:+.2f} deg ===")

                motor.move_to_angle(angle)
                time.sleep(SETTLE_TIME_S)

                actual = motor.position()
                if actual is not None:
                    print(f"  Motor settled at {actual:.2f} deg")

                # Prefer the actual settled encoder position over the
                # requested angle for the coordinate-frame calculation;
                # fall back to the requested angle if no reading landed.
                motor_angle_for_transform = actual if actual is not None else angle

                frame_angles = ct.compute_frame_angles(
                    motor_angle_deg=motor_angle_for_transform,
                    sweep_angle_deg=SWEEP_ANGLE_DEG,
                    mounting_angle_deg=MOUNTING_ANGLE_DEG,
                )

                accel_csv_path = os.path.join(OUTPUT_DIR, f"angle_{angle:+07.2f}deg_accel.csv")
                load_csv_path = os.path.join(OUTPUT_DIR, f"angle_{angle:+07.2f}deg_load.csv")

                sync_result = collect_synchronized(
                    instr, SAMPLE_DURATION_S, accel_csv_path, load_csv_path
                )
                sync_offset_ms = (
                    sync_result["load_t_start"] - sync_result["accel_t_start"]
                ) * 1000.0

                angle_log_writer.writerow({
                    "index": i,
                    "motor_angle_requested_deg": angle,
                    "motor_angle_actual_deg": actual,
                    "sweep_angle_deg": SWEEP_ANGLE_DEG,
                    "mounting_angle_deg": MOUNTING_ANGLE_DEG,
                    "inclination_angle_deg": frame_angles["inclination_angle_deg"],
                    "inclination_angle_deg_shifted": frame_angles["inclination_angle_deg_shifted"],
                    "angle_of_attack_deg": frame_angles["angle_of_attack_deg"],
                    "angle_of_attack_deg_shifted": frame_angles["angle_of_attack_deg_shifted"],
                    "accel_csv_path": accel_csv_path,
                    "load_csv_path": load_csv_path,
                    "accel_t_start": sync_result["accel_t_start"],
                    "load_t_start": sync_result["load_t_start"],
                    "sync_offset_ms": sync_offset_ms,
                })
                angle_log_f.flush()

        print("\nAll angles complete.")
        print(f"Coordinate-frame angle log written to {angle_log_path}")
    finally:
        motor.stop()


if __name__ == "__main__":
    main()
