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

Tunnel conditions (pitot-static, temperature, static pressure) are
sampled continuously in the background for the whole sweep -- not
per-angle -- via wind_tunnel_conditions.TunnelConditionsCollector,
mirroring how Motor's encoder-polling thread runs for the life of the
rig rather than being re-armed each angle. The run-averaged conditions
(density, viscosity, freestream velocity, Reynolds number, etc.) are
written once at the end to Data/wind_tunnel/tunnel_conditions_summary.csv.

Requirements:
    pip install minimalmodbus pyserial pandas openpyxl nidaqmx numpy matplotlib
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
import wind_tunnel_conditions as wtc
from utils.live_varience_plot import LivePlotter, accel_magnitude_variance

# --------------------------------------------------
# FOR YOU TO UPDATE BEFORE RUNS
# --------------------------------------------------

MOTOR_PORT = "COM11"
MOTOR_BAUD = 115200

PCB_PORT = "COM12"

LOADCELL_DEVICE = "Dev11"          # NI DAQ device name (same box motor.py's trigger line lives on)
LOADCELL_SAMPLE_RATE_HZ = 1000.0   # hardware-timed analog sample rate
RUN_LOAD_CELL = False               # False = accelerometer-only run, load cell skipped entirely

RUN_TUNNEL_CONDITIONS = True       # False = skip wind tunnel conditions entirely
AIRFOIL_CHORD_M = 0.19             # Reynolds number length scale -- set per experiment

ANGLES_FILE = "scripts/aquire_data/test_angles.xlsx"     # .xlsx, .csv, or .txt (one angle per line / row)
OUTPUT_DIR = "Data/run_009"     # created if it doesn't exist; nested under Data/ so it's easy to gitignore

SAMPLE_DURATION_S = 4         # how long to capture accel + load data at each angle
SETTLE_TIME_S = 2             # pause after move, before capture starts

# Coordinate-frame constants (see utils/coordinate_transforms.py).
# These describe the fixed physical setup for this run (as opposed to
# motor_angle, which is swept from ANGLES_FILE) -- set them here before
# starting a run.
#   SWEEP_ANGLE_DEG:    blade azimuthal/sweep orientation for this run
#   MOUNTING_ANGLE_DEG: fixed blade mounting angle for this run
SWEEP_ANGLE_DEG = 30.0
MOUNTING_ANGLE_DEG = 0.0

SETTLE_TOLERANCE_DEG = 0.1
SETTLE_TIMEOUT_S = 10.0
POST_SETTLE_S = 0.2

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


def collect_accel_for_duration(instr, duration_s, csv_path, start_event=None, on_poll=None):
    """Capture accelerometer FIFO data for duration_s seconds and write it
    to csv_path. Mirrors test_modbus.cmd_stream's raw-serial loop, but
    stops after a fixed duration instead of waiting for KeyboardInterrupt.

    start_event: optional object with a no-arg .wait() method (a
    threading.Event or threading.Barrier). Sampling is armed on the
    sensor first, then this call blocks on start_event.wait() immediately
    before the capture loop begins -- so a caller running this in a
    thread alongside a load-cell thread (sharing the same barrier/event)
    releases both at the same instant for a synchronized start.

    Returns (total_samples, t_start, accel_variance) where t_start is the
    time.perf_counter() value at which the capture loop actually began
    (also written into the CSV header for reference), and accel_variance
    is the variance of the acceleration magnitude over the capture
    window (see live_varience_plot.accel_magnitude_variance), or 0.0 if
    no samples were captured.
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
    all_rows = []  # accumulated for the post-capture variance calc below
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
        last_pump = time.perf_counter()
        while (time.perf_counter() - t_start) < duration_s:
            avail = pcb.read_register_raw(ser, addr, pcb.REG_AVAIL)
            if avail == 0:
                time.sleep(0.001)
            else:
                sets_to_read = min(avail, pcb.BULK_MAX_SETS)
                num_regs = sets_to_read * pcb.REGS_PER_SET
                regs = pcb.bulk_read_raw(ser, addr, pcb.REG_BULK, num_regs)
                t_batch = time.perf_counter() - t_start

                signed = [v - 65536 if v >= 32768 else v for v in regs]
                for s in range(sets_to_read):
                    base = s * pcb.REGS_PER_SET
                    row = signed[base:base + pcb.REGS_PER_SET]
                    writer.writerow([total_samples, f"{t_batch:.4f}"] + row)
                    all_rows.append(row)
                    total_samples += 1
            if on_poll is not None and (time.perf_counter() - last_pump) > 0.1:  # NEW, throttled
                on_poll()
                last_pump = time.perf_counter()

    pcb.write_register_retry(instr, pcb.REG_CMD, pcb.CMD_STOP, functioncode=6)

    elapsed = time.perf_counter() - t_start
    rate = total_samples / elapsed if elapsed > 0 else 0
    print(f"  Accel: captured {total_samples} samples in {elapsed:.2f}s ({rate:.0f}/s, ODR={odr} Hz)")

    accel_var = accel_magnitude_variance(all_rows, pcb.MG_PER_LSB) if all_rows else 0.0
    return total_samples, t_start, accel_var


# --------------------------------------------------
# Simultaneous accel + load cell capture
# --------------------------------------------------


def collect_synchronized(instr, duration_s, accel_csv_path, load_csv_path=None):
    """Runs collect_accel_for_duration and (if load_csv_path is given)
    collect_load_for_duration in parallel threads, released together via
    a shared threading.Barrier sized to the number of threads actually
    running.

    load_csv_path=None (or RUN_LOAD_CELL=False upstream) skips the load
    cell entirely -- only the accelerometer thread runs, against a
    Barrier(1) so it releases itself immediately instead of waiting on a
    party that will never show up.

    Returns a dict with per-sensor sample counts, t_start values, and
    the accelerometer variance:
        {
          "accel_samples": int, "accel_t_start": float,
          "accel_variance": float,
          # present only when load_csv_path was given:
          "load_samples": int,  "load_t_start": float,
        }
    Any exception raised inside a worker thread is re-raised here (after
    all threads have finished) so a sensor failure doesn't get silently
    swallowed mid-sweep.
    """
    run_load = load_csv_path is not None
    start_gate = threading.Barrier(2 if run_load else 1)
    result = {}
    errors = []

    def _run_accel():
        try:
            n, t0, accel_var = collect_accel_for_duration(
                instr, duration_s, accel_csv_path, start_event=start_gate
            )
            result["accel_samples"], result["accel_t_start"] = n, t0
            result["accel_variance"] = accel_var
        except Exception as e:
            errors.append(("accel", e))
            # Make sure a still-waiting load-cell thread (if any) isn't
            # left hanging forever on a barrier that will now never trip
            # normally.
            start_gate.abort()

    threads = [threading.Thread(target=_run_accel)]

    if run_load:
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

        threads.append(threading.Thread(target=_run_load))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        names = ", ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"Synchronized capture failed ({names})")

    if run_load:
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

    encoder_log_path = os.path.join(OUTPUT_DIR, "encoder_log.txt")
    motor = Motor(port=MOTOR_PORT, baud=MOTOR_BAUD, log_file=encoder_log_path)
    # plotter = LivePlotter(show_sweep=False, figsize=(8, 8), dpi=100)
    plotter = LivePlotter(show_loadcell=False)
    instr = pcb.connect(PCB_PORT)
    pcb.set_low_latency(instr.serial)

    tunnel_collector = wtc.TunnelConditionsCollector(length_scale_m=AIRFOIL_CHORD_M) if RUN_TUNNEL_CONDITIONS else None

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
        "accel_variance",
    ]


    motor.start()
    try:
        for i, angle in enumerate(angles):
            print(f"\n=== Angle {i + 1}/{len(angles)}: {angle:+.2f} deg ===")

            try:
                actual = motor.move_to_angle(
                    angle, wait=True,
                    tolerance_deg=SETTLE_TOLERANCE_DEG,
                    timeout_s=SETTLE_TIMEOUT_S,
                    on_poll=plotter.pump,
                )
            except TimeoutError as e:
                print(f"  WARN: {e} -- proceeding with last known position")
                actual = motor.position()

            time.sleep(POST_SETTLE_S)

            csv_path = os.path.join(OUTPUT_DIR, f"angle_{angle:+07.2f}deg.csv")
            collect_accel_for_duration(instr, SAMPLE_DURATION_S, csv_path, on_poll=plotter.pump)

        print("\nAll angles complete.")
    finally:
        try:
            motor.home()
        except Exception as e:
            print(f"  WARN: failed to home motor: {e}")
        motor.stop()


if __name__ == "__main__":
    main()