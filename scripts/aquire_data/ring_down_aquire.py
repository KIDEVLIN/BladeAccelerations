#!/usr/bin/env python3
"""
ring_down_acquire.py

Free-decay ("flick test") acquisition for the blade's structural dynamics
(natural frequencies, damping ratios, mode-shape ratios). Uses the same four
accelerometers and load cell as main.py, captured simultaneously through
main.collect_synchronized, so file formats, headers, units and calibration are
identical to the sweep runs. No motor and no tunnel conditions are touched.

Workflow
    1. (optional, on by default) a "rest" capture with the blade untouched: this
       is the sensor noise floor, used later for SNR and for deciding how far
       the ring-down can be trusted.
    2. Repeated trials. At the prompt press Enter, hold still while the capture
       arms (~2 s of quiet record before the flick), flick the blade once, and let
       it ring until the capture ends. Each trial is checked immediately
       (onset time, peak, saturation, did it decay, dropped samples) and you
       choose to keep or redo it.
    3. Type a word at the prompt to switch the excitation label, e.g.
       "flap", "edge", "flap_small", "flap_large", "tors". Labels are free text and
       become part of the file names, so directions and amplitudes can be
       analysed as separate groups later.

Usage (from scripts/aquire_data):
    python ring_down_acquire.py --out Data/Modal/DTU10MW_still_air --label flap
    python ring_down_acquire.py --out Data/Modal/DTU10MW_still_air --fs 4g --duration 30
    python ring_down_acquire.py --out Data/Modal/DTU10MW_still_air --skip-rest   # resume, no new rest

Re-running with the same --out resumes: trial numbers continue per label and
run_metadata.json is extended, never overwritten.

Output (in --out):
    rest_001.csv / rest_001_load.csv          noise-floor capture
    ring_<label>_<NNN>.csv / ..._load.csv     one flick trial (accel + load cell)
    run_metadata.json                         settings + per-trial QC (experiment = "ring_down")

Names deliberately do not match angle_*deg.csv, so the sweep pipeline
(extract_run_summary.py / run_loader.py) ignores these folders.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main as acq               # reuse capture + config (same setup as the sweep runs)
import test_modbus as pcb

# --------------------------------------------------
# FOR YOU TO UPDATE BEFORE RUNS
# --------------------------------------------------

OUTPUT_DIR = "Data/DTU10MW_rigid4k_ringdown"
DEFAULT_LABEL = "flap"
ACCEL_FS = "8g"            # a flick can be large; lower ranges give finer resolution for the late, small decay
ACCEL_ODR_HZ = 1600
DURATION_S = 20.0          # must be long enough for the ring-down to reach the noise floor
REST_DURATION_S = 10.0
PRE_FLICK_QUIET_S = 2.0    # asked of the operator; QC warns if the flick comes sooner
RUN_LOAD_CELL = True

# QC thresholds
RATE_LOW_FRACTION = 0.9    # measured accel rate below this fraction of ODR = dropped samples
ONSET_SNR = 6.0            # onset = first 0.1 s window with rms > ONSET_SNR * noise floor
DECAY_SNR = 3.0            # "decayed" = last second rms below DECAY_SNR * noise floor
MIN_PEAK_SNR = 20.0        # ring-down peak should clear the noise floor by at least this
LOAD_CLIP_LEVEL = 0.99     # fraction of the +-10 V DAQ range
LOAD_CLIP_WARN_FRACTION = 0.01


def clean_label(s):
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip()).strip("_")
    return s or DEFAULT_LABEL


def stem_for(label, k):
    return f"rest_{k:03d}" if label == "rest" else f"ring_{label}_{k:03d}"


def next_index(out_dir, label):
    pat = re.compile(rf"^{'rest' if label == 'rest' else 'ring_' + re.escape(label)}_(\d+)\.csv$")
    nums = [int(m[1]) for p in Path(out_dir).glob("*.csv") if (m := pat.match(p.name))]
    return max(nums, default=0) + 1


# --------------------------------------------------
# Quick per-trial QC (full analysis comes in the processing pipeline)
# --------------------------------------------------

def _window_rms(x, w):
    return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="valid"))


def qc_trial(accel_path, load_path, sat, is_rest, fs_name, odr_hz):
    warnings = []
    df = pd.read_csv(accel_path, comment="#")
    n = len(df)
    qc = {"n_samples": n}
    if n < 200:
        return dict(qc, ok=False, warnings=["too few accelerometer samples"])

    mg = pcb.FS_MG_PER_LSB[pcb.FS_CHOICES[fs_name]]
    t_end = float(df["time"].iloc[-1])
    rate = n / t_end if t_end > 0 else np.nan
    qc["accel_rate_hz"] = rate
    if rate < RATE_LOW_FRACTION * odr_hz:
        warnings.append(f"accel rate {rate:.0f} Hz < {RATE_LOW_FRACTION:.0%} of ODR (dropped samples?)")

    raw = df[acq.ACCEL_COMPONENTS].to_numpy(float)
    dev_g = (raw - np.median(raw, axis=0)) * mg / 1000.0       # static (gravity) part removed
    dev = np.linalg.norm(dev_g, axis=1)
    w = max(1, int(0.1 * rate))
    rms = _window_rms(dev, w)
    noise = float(np.percentile(rms, 10))                      # quietest 10% of windows
    qc["noise_rms_g"] = noise
    qc["noise_rms_mg_per_axis"] = float(noise * 1000.0 / np.sqrt(dev_g.shape[1]))

    if sat["flagged"]:
        warnings.append(f"saturated: {acq.format_saturation(sat['flagged'])} -> use a higher --fs")
    qc["peak_g"] = sat["peak_g"]

    if not is_rest:
        thr = ONSET_SNR * noise
        above = np.where(rms > thr)[0]
        if len(above) == 0:
            warnings.append("no flick detected (signal never exceeded the noise floor)")
            qc["onset_s"] = None
        else:
            onset = float(above[0] / rate)
            qc["onset_s"] = onset
            if onset < PRE_FLICK_QUIET_S - 0.5:
                warnings.append(f"flick at {onset:.2f} s: less than {PRE_FLICK_QUIET_S:.0f} s of quiet baseline")
            peak_rms = float(rms.max())
            qc["peak_rms_g"] = peak_rms
            qc["peak_snr"] = peak_rms / noise if noise > 0 else np.nan
            if qc["peak_snr"] < MIN_PEAK_SNR:
                warnings.append(f"weak flick (peak/noise = {qc['peak_snr']:.0f}); hit harder or move closer to the tip")
            tail = float(rms[-int(rate):].mean()) if len(rms) > rate else float(rms[-1])
            qc["tail_rms_g"] = tail
            qc["decayed"] = bool(tail < DECAY_SNR * noise)
            if not qc["decayed"]:
                warnings.append("still ringing at the end of the capture -> raise --duration (or the ring-down is cut short)")
            # crude decay time: last time the envelope was above 10x noise
            hot = np.where(rms > 10 * noise)[0]
            qc["t_above_10x_noise_s"] = float(hot[-1] / rate) if len(hot) else None

    if load_path is not None:
        if not Path(load_path).exists():
            warnings.append("load-cell file missing")
        else:
            ld = pd.read_csv(load_path, comment="#")
            F = ld[["Fx", "Fy", "Fz"]].to_numpy(float)
            qc["load_peak_dev_N"] = float(np.abs(F - np.median(F, axis=0)).max())
            stg = np.abs(ld[acq.lc.__dict__.get("STG_COLS", [f"stg{i}" for i in range(6)])].to_numpy(float))
            lfrac = float((stg >= LOAD_CLIP_LEVEL * 10.0).mean(axis=0).max())
            qc["load_clip_max_frac"] = lfrac
            if lfrac >= LOAD_CLIP_WARN_FRACTION:
                warnings.append("load-cell channel near the DAQ rail (clipping)")

    qc["warnings"] = warnings
    qc["ok"] = not warnings
    return qc


def print_qc(qc, is_rest):
    if is_rest:
        print(f"  Rest noise floor: {qc.get('noise_rms_g', float('nan')) * 1000:.2f} mg rms (all 12 axes), "
              f"{qc.get('noise_rms_mg_per_axis', float('nan')):.2f} mg per axis")
    else:
        bits = []
        if qc.get("onset_s") is not None:
            bits.append(f"flick at {qc['onset_s']:.2f} s")
        if "peak_g" in qc:
            bits.append(f"peak {qc['peak_g']:.2f} g")
        if "peak_snr" in qc:
            bits.append(f"peak/noise {qc['peak_snr']:.0f}")
        if qc.get("t_above_10x_noise_s") is not None:
            bits.append(f"above 10x noise until {qc['t_above_10x_noise_s']:.1f} s")
        if "load_peak_dev_N" in qc:
            bits.append(f"load peak dev {qc['load_peak_dev_N']:.2f} N")
        print("  QC: " + ", ".join(bits))
    for w in qc.get("warnings", []):
        print(f"  WARN: {w}")
    if qc.get("ok"):
        print("  QC passed.")


# --------------------------------------------------
# Session
# --------------------------------------------------

def run(args):
    fs_name, odr_hz = args.fs, args.odr
    acq.check_accel_mode(fs_name, odr_hz)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = str(out / "run_metadata.json")

    if os.path.exists(meta_path):
        meta = json.loads(Path(meta_path).read_text())
        meta["status"] = "in_progress"
        meta.setdefault("sessions", []).append(datetime.now().isoformat(timespec="seconds"))
        print(f"Resuming {out} ({len(meta.get('trials', []))} trials already recorded)")
    else:
        meta = {
            "experiment": "ring_down",
            "run_dir": str(out),
            "status": "in_progress",
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "sessions": [datetime.now().isoformat(timespec="seconds")],
            "note": args.note,
            "accel_fs": fs_name,
            "accel_odr_hz": odr_hz,
            "accel_mg_per_lsb": pcb.FS_MG_PER_LSB[pcb.FS_CHOICES[fs_name]],
            "loadcell_sample_rate_hz": acq.LOADCELL_SAMPLE_RATE_HZ,
            "run_load_cell": RUN_LOAD_CELL,
            "duration_s": args.duration,
            "rest_duration_s": REST_DURATION_S,
            "excitation": "manual flick at blade tip",
            "mounting_angle_deg": acq.MOUNTING_ANGLE_DEG,
            "sweep_angle_deg": acq.SWEEP_ANGLE_DEG,
            "trials": [],
        }
    if meta.get("accel_fs") != fs_name or meta.get("accel_odr_hz") != odr_hz:
        raise SystemExit(f"{out} was recorded with {meta.get('accel_fs')}/{meta.get('accel_odr_hz')} Hz; "
                         f"refusing to mix in {fs_name}/{odr_hz} Hz. Use a new --out.")
    acq.write_run_metadata(meta_path, meta)

    instr = pcb.connect(acq.PCB_PORT)
    pcb.set_low_latency(instr.serial)

    def capture(label):
        is_rest = label == "rest"
        k = next_index(out, label)
        stem = stem_for(label, k)
        accel_csv = str(out / f"{stem}.csv")
        load_csv = str(out / f"{stem}_load.csv") if RUN_LOAD_CELL else None
        dur = REST_DURATION_S if is_rest else args.duration
        time.sleep(1.0)                                   # let the keypress / hand motion die out
        if is_rest:
            print(f"\a  RECORDING {dur:.0f} s -- do NOT touch the blade.")
        else:
            print(f"\a  RECORDING {dur:.0f} s -- hold still ~{PRE_FLICK_QUIET_S:.0f} s, then flick once and let it ring.")
        res = acq.collect_synchronized(instr, dur, accel_csv, load_csv_path=load_csv,
                                       fs=fs_name, odr_hz=odr_hz)
        print("  Capture done.")
        qc = qc_trial(accel_csv, load_csv, res["accel_saturation"], is_rest, fs_name, odr_hz)
        print_qc(qc, is_rest)
        rec = {
            "label": label, "trial": k, "is_rest": is_rest, "duration_s": dur,
            "accel_csv": accel_csv, "load_csv": load_csv,
            "accel_t_start": res["accel_t_start"], "load_t_start": res.get("load_t_start"),
            "sync_offset_ms": ((res["load_t_start"] - res["accel_t_start"]) * 1000.0
                               if "load_t_start" in res else None),
            "accel_samples": res["accel_samples"], "load_samples": res.get("load_samples"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "qc": qc,
        }
        return rec, [accel_csv] + ([load_csv] if load_csv else [])

    def confirm(rec, files):
        """Keep (Enter) or redo (r) the trial just captured. Returns True if kept."""
        ans = input("  Enter = keep, r = redo (discard this trial) > ").strip().lower()
        if ans == "r":
            for p in files:
                try:
                    os.remove(p)
                except OSError:
                    pass
            print("  Discarded.")
            return False
        meta["trials"].append(rec)
        acq.write_run_metadata(meta_path, meta)
        return True

    completed = False
    try:
        have_rest = any(t.get("is_rest") for t in meta["trials"])
        if not args.skip_rest and not have_rest:
            print("\n=== Noise-floor capture (blade at rest) ===")
            input("Blade untouched, tunnel fan OFF. Press Enter to record the rest capture... ")
            while True:
                rec, files = capture("rest")
                if confirm(rec, files):
                    break

        label = clean_label(args.label)
        while True:
            n_kept = sum(1 for t in meta["trials"] if t["label"] == label)
            cmd = input(f"\n[{label}] {n_kept} kept | Enter = record, <name> = switch label, q = quit > ").strip()
            if cmd.lower() == "q":
                break
            if cmd:
                label = clean_label(cmd)
                print(f"  Label is now '{label}'.")
                continue
            rec, files = capture(label)
            confirm(rec, files)
        completed = True
    finally:
        try:
            pcb.write_register_retry(instr, pcb.REG_CMD, pcb.CMD_STOP, functioncode=6)
        except Exception:
            pass
        meta["status"] = "complete" if completed else "aborted"
        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        acq.write_run_metadata(meta_path, meta)
        counts = {}
        for t in meta["trials"]:
            counts[t["label"]] = counts.get(t["label"], 0) + 1
        print("\nKept trials per label: " + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none"))
        print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Blade ring-down (flick test) acquisition",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUTPUT_DIR, help="output folder (resumes if it exists)")
    ap.add_argument("--label", default=DEFAULT_LABEL, help="starting excitation label (e.g. flap, edge)")
    ap.add_argument("--fs", choices=list(pcb.FS_CHOICES), default=ACCEL_FS)
    ap.add_argument("--odr", type=int, choices=pcb.VALID_ODR, default=ACCEL_ODR_HZ)
    ap.add_argument("--duration", type=float, default=DURATION_S, help="seconds per flick trial")
    ap.add_argument("--skip-rest", action="store_true", help="do not take the rest/noise-floor capture")
    ap.add_argument("--note", default="", help="free text stored in run_metadata.json (setup, fan state, ...)")
    run(ap.parse_args())