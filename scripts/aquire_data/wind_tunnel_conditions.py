"""
wind_tunnel_conditions.py

Python port of the LabVIEW 2015 wind-tunnel-conditions VIs
("Windtunnel_conditions.vi" live readout, and "Tunnel_conditions.vi"
DAQ1/DAQ2 file-saving version), plus the post-processing calibration
constants that used to live in load_tunnel_conditions.m /
read_tunnel_conditions(). All of that now happens in one place -- no
LabVIEW binary file + MATLAB parsing round-trip needed.

Reads 4 analog-input channels on a NI DAQ device:
    ch0: freestream pitot-static dP   (psi transducer, "2.0 psid")
    ch1: traverse pitot-static dP     (uncalibrated -- see note below)
    ch2: tunnel temperature
    ch3: tunnel static/fluid pressure

...converts to engineering units, and computes air density (virial-type
compressibility correction), viscosity (Sutherland's law), freestream
velocity, and Reynolds number.

Calibration constants below (P_fluid_sense, T_fluid_sense,
P_freestream_sense, toPa) are taken directly from
load_tunnel_conditions.m -- reproduced here, not re-derived. The
traverse channel (ch1) has no known calibration; raw volts are recorded
for reference but not scaled into a physical quantity. If/when you have
its sense, fill in TRAVERSE_DP_SENSE_V_PER_PSI (mirroring
FREESTREAM_DP_SENSE_V_PER_PSI) and extend compute_conditions().

Requirements:
    pip install nidaqmx
"""

import csv
import math
import os
import threading
import time
from dataclasses import dataclass, fields

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# --------------------------------------------------
# FOR YOU TO UPDATE BEFORE RUNS
# --------------------------------------------------

# NOTE: Dev11 is also used by the load cell (6 AI differential channels,
# ai0-ai5 per load_cell.py). ai6/ai7 are assumed free here -- verify
# against your actual load cell channel list and device's AI count
# before running both concurrently. Running two DAQmx tasks on
# different channels of the same multifunction board is normally fine,
# but on boards with a single multiplexed ADC the two tasks' samples
# are time-interleaved by the driver rather than truly simultaneous --
# not a correctness problem for slowly-varying tunnel conditions, but
# worth knowing.
DAQ_DEVICE = "Dev11"
CHANNEL_FREESTREAM_DP = f"{DAQ_DEVICE}/ai6"    # TODO: confirm - avoid load cell's ai0-ai5
CHANNEL_TRAVERSE_DP = f"{DAQ_DEVICE}/ai7"      # TODO: confirm
CHANNEL_STATIC_PRESSURE = f"{DAQ_DEVICE}/ai8"  # TODO: confirm
CHANNEL_TEMPERATURE = f"{DAQ_DEVICE}/ai9"      # TODO: confirm

AI_TERMINAL_CONFIG = TerminalConfiguration.RSE
AI_MIN_V = -10.0
AI_MAX_V = 10.0

SAMPLING_RATE_HZ = 1000.0   # matches "Sample Rate DAQ 1 [Hz]" in the original VI
NUMBER_OF_SAMPLES = 100     # samples per batch, averaged into one WindTunnelConditions reading

# --------------------------------------------------
# Calibration constants, from load_tunnel_conditions.m (read_tunnel_conditions)
# --------------------------------------------------

TO_PA = 6894.75729  # 1 psi = 6894.75729 Pa

STATIC_PRESSURE_SENSE_PSI_PER_V = 350.0   # P_fluid_sense
STATIC_PRESSURE_OFFSET_PSI = 0.0          # P_fluid_offset

TEMP_SENSE_DEGF_PER_V = 20.0              # T_fluid_sense ("100 F / 5 V" transducer)
TEMP_OFFSET_DEGF = 0.0                    # T_fluid_offset

FREESTREAM_DP_SENSE_V_PER_PSI = 5.0908    # P_freestream_sense (note: V per psi, not psi per V)
FREESTREAM_PT_NOMINAL_PSID = 2.0          # sanity label only -- the "2.0 psid" transducer option

# Traverse channel (ch1): no calibration constant available yet. Leave
# as None -- compute_conditions() will skip scaling it and just carry
# the raw volts through.
TRAVERSE_DP_SENSE_V_PER_PSI = None

# --------------------------------------------------
# Length scale for Reynolds number
# --------------------------------------------------
# This depends on the experiment (e.g. airfoil chord) and is meant to
# be set per-run from main.py rather than hardcoded here -- see
# AIRFOIL_CHORD_M in main.py. This default (0.19 m) matches
# Airfoil.c from load_tunnel_conditions.m, kept only as a fallback for
# standalone use of this module.
DEFAULT_LENGTH_SCALE_M = 0.19

OUTPUT_DIR = "Data/wind_tunnel"
SUMMARY_CSV_NAME = "tunnel_conditions_summary.csv"


# --------------------------------------------------
# Physics (formulas extracted from the LabVIEW Formula nodes / ZSI.m)
# --------------------------------------------------


@dataclass
class WindTunnelConditions:
    timestamp: float
    temp_c: float
    temp_k: float
    static_pressure_pa: float
    static_pressure_atm: float
    density_kg_m3: float
    dynamic_viscosity: float
    kinematic_viscosity: float
    freestream_dp_v: float
    freestream_dp_pa: float
    traverse_dp_v: float
    velocity_freestream_m_s: float
    reynolds_number: float


def _z_correction(temp_k: float, press_atm: float) -> float:
    """Air compressibility factor Z -- 3-term virial expansion in
    (Press_atm - 1), each coefficient a polynomial in TempK. Reproduced
    verbatim from the decompiled LabVIEW Formula nodes (this is what
    ZSI.m almost certainly implements internally)."""
    z1 = 3.1753e-5 + (-1.7155e-7) * temp_k + (2.4630e-10) * temp_k ** 2
    z2 = -9.5378e-3 + (5.1986e-5) * temp_k + (-7.0621e-8) * temp_k ** 2
    z3 = (6.3764e-7 + (-6.4678e-9) * temp_k + (2.1880e-11) * temp_k ** 2
          + (-2.4691e-14) * temp_k ** 3)
    return 1.0 + z1 * (press_atm - 1) + z2 * (press_atm - 1) ** 2 + z3 * (press_atm - 1) ** 3


def compute_conditions(freestream_dp_v, traverse_dp_v, static_pressure_v, temperature_v,
                        length_scale_m=DEFAULT_LENGTH_SCALE_M, timestamp=None) -> WindTunnelConditions:
    """Converts the 4 raw channel means (volts) into engineering
    conditions, following load_tunnel_conditions.m's
    read_tunnel_conditions() exactly.
    """
    if timestamp is None:
        timestamp = time.time()

    # --- Temperature: volts -> degF -> degC (matches the .m script,
    # which does NOT add 273.15 here -- Kelvin is computed separately
    # below for the physics that need an absolute temperature) ---
    temp_f = temperature_v * TEMP_SENSE_DEGF_PER_V + TEMP_OFFSET_DEGF
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    temp_k = temp_c + 273.15

    # --- Static/fluid pressure: volts -> Pa (absolute) ---
    static_pressure_pa = (static_pressure_v * STATIC_PRESSURE_SENSE_PSI_PER_V
                           + STATIC_PRESSURE_OFFSET_PSI) * TO_PA
    static_pressure_atm = static_pressure_pa / 101325.0

    # --- Density (with compressibility correction) + viscosity ---
    z = _z_correction(temp_k, static_pressure_atm)
    density = static_pressure_pa / (temp_k * z * 287.1)
    dynamic_viscosity = 1.458e-6 * (temp_k ** 1.5) / (110.4 + temp_k)
    kinematic_viscosity = dynamic_viscosity / density

    # --- Freestream dynamic pressure -> velocity ---
    # Note the inverted units: sense is V/psi, so divide (not multiply).
    freestream_dp_psi = freestream_dp_v / FREESTREAM_DP_SENSE_V_PER_PSI
    freestream_dp_pa = freestream_dp_psi * TO_PA
    if freestream_dp_pa < 0:
        # Mirrors the .m script setting negative freestream readings to
        # NaN before sqrt -- treat as an invalid sample rather than
        # raising, so a background collector can just skip averaging it.
        velocity_freestream = float("nan")
    else:
        velocity_freestream = math.sqrt(2 * freestream_dp_pa / density)

    # --- Reynolds number ---
    reynolds = density * length_scale_m * velocity_freestream / dynamic_viscosity

    return WindTunnelConditions(
        timestamp=timestamp,
        temp_c=temp_c,
        temp_k=temp_k,
        static_pressure_pa=static_pressure_pa,
        static_pressure_atm=static_pressure_atm,
        density_kg_m3=density,
        dynamic_viscosity=dynamic_viscosity,
        kinematic_viscosity=kinematic_viscosity,
        freestream_dp_v=freestream_dp_v,
        freestream_dp_pa=freestream_dp_pa,
        traverse_dp_v=traverse_dp_v,
        velocity_freestream_m_s=velocity_freestream,
        reynolds_number=reynolds,
    )


# --------------------------------------------------
# Single-shot read (handy for a quick standalone check)
# --------------------------------------------------


def read_raw_voltages(num_samples=NUMBER_OF_SAMPLES, sample_rate=SAMPLING_RATE_HZ):
    """Read all 4 channels, num_samples each, return per-channel means in
    order: (freestream_dp_v, traverse_dp_v, static_pressure_v, temperature_v).
    """
    with nidaqmx.Task() as task:
        for chan in (CHANNEL_FREESTREAM_DP, CHANNEL_TRAVERSE_DP,
                     CHANNEL_STATIC_PRESSURE, CHANNEL_TEMPERATURE):
            task.ai_channels.add_ai_voltage_chan(
                chan, terminal_config=AI_TERMINAL_CONFIG,
                min_val=AI_MIN_V, max_val=AI_MAX_V,
            )
        task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=num_samples)
        data = task.read(number_of_samples_per_channel=num_samples)

    means = [sum(ch) / len(ch) for ch in data]
    return tuple(means)


def read_conditions_once(length_scale_m=DEFAULT_LENGTH_SCALE_M) -> WindTunnelConditions:
    """One poll-average-compute cycle."""
    freestream_v, traverse_v, static_v, temp_v = read_raw_voltages()
    return compute_conditions(freestream_v, traverse_v, static_v, temp_v,
                               length_scale_m=length_scale_m)


# --------------------------------------------------
# Background collector -- runs for the whole angle sweep, mirrors
# motor.Motor's background-polling-thread pattern (arm, start, lock-
# guarded latest reading, stop -> summary)
# --------------------------------------------------


_AVERAGE_FIELDS = [f.name for f in fields(WindTunnelConditions) if f.name != "timestamp"]


class TunnelConditionsCollector:
    """Continuously samples the 4 tunnel-condition channels in a
    background thread for the duration of a run (e.g. the whole motor
    angle sweep in main.py), buffering one WindTunnelConditions reading
    per batch. Call start() once at the beginning of the sweep and
    stop() at the end to get the run-averaged conditions -- mirrors how
    Motor's encoder-polling thread runs for the life of the rig rather
    than being re-armed per angle.
    """

    def __init__(self, length_scale_m=DEFAULT_LENGTH_SCALE_M,
                 num_samples=NUMBER_OF_SAMPLES, sample_rate=SAMPLING_RATE_HZ):
        self.length_scale_m = length_scale_m
        self.num_samples = num_samples
        self.sample_rate = sample_rate

        self._task = None
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._readings = []  # list[WindTunnelConditions], guarded by _lock

    def start(self):
        self._task = nidaqmx.Task()
        for chan in (CHANNEL_FREESTREAM_DP, CHANNEL_TRAVERSE_DP,
                     CHANNEL_STATIC_PRESSURE, CHANNEL_TEMPERATURE):
            self._task.ai_channels.add_ai_voltage_chan(
                chan, terminal_config=AI_TERMINAL_CONFIG,
                min_val=AI_MIN_V, max_val=AI_MAX_V,
            )
        self._task.timing.cfg_samp_clk_timing(
            self.sample_rate, sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.num_samples * 4,
        )
        self._task.start()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("  Tunnel conditions: background collection started.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                data = self._task.read(number_of_samples_per_channel=self.num_samples)
            except Exception as e:
                print(f"  WARN: tunnel conditions read failed: {e}")
                time.sleep(0.1)
                continue
            means = [sum(ch) / len(ch) for ch in data]
            freestream_v, traverse_v, static_v, temp_v = means
            reading = compute_conditions(freestream_v, traverse_v, static_v, temp_v,
                                          length_scale_m=self.length_scale_m)
            with self._lock:
                self._readings.append(reading)

    def get_latest(self):
        """Returns the most recent WindTunnelConditions reading, or None
        if nothing has landed yet. Never blocks on DAQ I/O."""
        with self._lock:
            return self._readings[-1] if self._readings else None

    def stop(self):
        """Stops collection, closes the DAQ task, and returns a summary
        dict: the mean of every numeric field across all buffered
        readings (NaN velocity/Reynolds samples, from invalid negative
        freestream pressure, are excluded from those two means), plus
        sample_count and the length scale used.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        if self._task is not None:
            self._task.close()
            self._task = None

        with self._lock:
            readings = list(self._readings)

        print(f"  Tunnel conditions: background collection stopped ({len(readings)} batches).")

        summary = {"sample_count": len(readings), "length_scale_m": self.length_scale_m}
        for field_name in _AVERAGE_FIELDS:
            vals = [getattr(r, field_name) for r in readings]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            summary[field_name] = (sum(vals) / len(vals)) if vals else float("nan")
        return summary


def save_summary(summary: dict, output_dir=OUTPUT_DIR, filename=SUMMARY_CSV_NAME):
    """Writes a one-row CSV of the averaged tunnel conditions for the run."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"  Tunnel conditions summary written to {path}")
    return path


if __name__ == "__main__":
    # Quick standalone check: a few seconds of background collection.
    collector = TunnelConditionsCollector()
    collector.start()
    time.sleep(5)
    result = collector.stop()
    print(result)
    save_summary(result)