#!/usr/bin/env python3
"""
extract_run_summary.py

Stage 1 + 2 of the production-campaign pipeline.

Stage 1 (slow, once per run): read one acquisition run directory ONE ANGLE AT A
TIME (never the whole run in memory) and reduce every angle to a single row of
summary numbers, plus small Welch spectra. Raw data is never modified.

Stage 2 (seconds): merge every run's summary into two master tables:
    master_angles.csv   one row per (run, angle)
    master_runs.csv     one row per run (Re, Ca_e, max normalized variance, ...)

Output layout (default --out Production_summaries/):
    <out>/<run_name>/angle_summary.csv
    <out>/<run_name>/spectra.npz
    <out>/<run_name>/run_info.json
    <out>/master_angles.csv
    <out>/master_runs.csv

Usage (from scripts/aquire_data, so utils/ is importable):
    python extract_run_summary.py --root "D:/GE_blade/Production_10-8-26" --out Production_summaries
    python extract_run_summary.py "D:/.../DTU10MW_704psi_800fs_v2" --out Production_summaries --force
    python extract_run_summary.py --catalog-only --out Production_summaries

Runs whose summary already exists (same EXTRACTOR_VERSION) are skipped unless
--force is given, so adding run 16 does not reprocess runs 1-15.

Normalization (per run, from MEASURED conditions in run_metadata.json):
    F_c   = 0.5 * rho * U^2 * c_avg * b            (characteristic force, N)
    a_c   = F_c / m                                (m/s^2; /9.80665 -> g)
    accel normalized variance  = Var(|a|)[g^2] / a_c[g]^2
    load  normalized variance  = Var(|F|)[N^2] / F_c^2
    Re    = rho * U * c_avg / mu
    Ca_e  = rho * U^2 * b^4 / B_e

Requires: numpy pandas scipy
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils.run_loader import (  # noqa: E402
    _discover_angle_files, load_accel_csv, load_load_csv,
    load_encoder_log, windowed_average_angle,
)
from utils import coordinate_transforms as ct  # noqa: E402

EXTRACTOR_VERSION = "1.0"

G = 9.80665
NUM_SENSORS = 4
ACCEL_COLS = [f"s{s}_{a}" for s in range(1, NUM_SENSORS + 1) for a in "xyz"]
FORCE_COLS = ["Fx", "Fy", "Fz"]
STG_COLS = [f"stg{i}" for i in range(6)]
TO_PSI = 1.0 / 6894.75729
ONE_ATM_PA = 101325.01

# Clipping (same rule as main.py's analyze_saturation)
INT16_MAX = 32767
ACCEL_SAT_LEVEL = 0.995
ACCEL_SAT_WARN_FRACTION = 0.01
# NI DAQ input range is +-10 V (load_cell.py); strain-gauge channels near the rail clip
DAQ_RANGE_V = 10.0
DAQ_CLIP_LEVEL = 0.99

# Blade constants (model scale, 318:1)
DEFAULTS = dict(chord_avg_m=0.0138, span_m=0.28, mass_kg=0.0267, B_e=3.47e8)

NAME_RE = re.compile(r"^(?P<blade>.+?)_(?P<psi>\d+)psi_(?P<fs>\d+)fs(?:_v(?P<v>\d+))?$")


# --------------------------------------------------
# Run-level info
# --------------------------------------------------

def parse_run_name(name):
    m = NAME_RE.match(name)
    if not m:
        return dict(blade=name, nominal_psi=np.nan, fan_setting=np.nan, version=1)
    return dict(blade=m["blade"], nominal_psi=int(m["psi"]), fan_setting=int(m["fs"]),
                version=int(m["v"]) if m["v"] else 1)


def build_run_info(run_dir, c):
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tc = meta.get("tunnel_conditions") or {}

    def num(x):
        try:
            x = float(x)
            return x if math.isfinite(x) else np.nan
        except (TypeError, ValueError):
            return np.nan

    rho = num(tc.get("density_kg_m3"))
    U = num(tc.get("velocity_freestream_m_s"))
    mu = num(tc.get("dynamic_viscosity"))
    p_abs = num(tc.get("static_pressure_pa"))

    F_c = 0.5 * rho * U ** 2 * c["chord_avg_m"] * c["span_m"]
    a_c_g = F_c / c["mass_kg"] / G
    info = dict(
        run_name=run_dir.name, run_dir=str(run_dir), extractor_version=EXTRACTOR_VERSION,
        **parse_run_name(run_dir.name),
        meta_status=meta.get("status"),
        density_kg_m3=rho, velocity_m_s=U, dynamic_viscosity=mu,
        temp_c=num(tc.get("temp_c")),
        static_pressure_abs_pa=p_abs,
        static_pressure_gauge_psi=(p_abs - ONE_ATM_PA) * TO_PSI if math.isfinite(p_abs) else np.nan,
        reynolds_number=rho * U * c["chord_avg_m"] / mu,
        cauchy_edgewise=rho * U ** 2 * c["span_m"] ** 4 / c["B_e"],
        F_c_N=F_c, a_c_g=a_c_g,
        chord_avg_m=c["chord_avg_m"], span_m=c["span_m"], mass_kg=c["mass_kg"], B_e=c["B_e"],
        sweep_angle_deg=meta.get("sweep_angle_deg"),
        mounting_angle_deg=meta.get("mounting_angle_deg"),
        counts_per_deg=meta.get("counts_per_deg") or 694.44,
        sample_duration_s=meta.get("sample_duration_s"),
        accel_fs=meta.get("accel_fs"), accel_odr_hz=meta.get("accel_odr_hz"),
        accel_mg_per_lsb=meta.get("accel_mg_per_lsb"),
        loadcell_sample_rate_hz=meta.get("loadcell_sample_rate_hz") or 2000.0,
    )
    if not (math.isfinite(rho) and math.isfinite(U)):
        print(f"  WARN [{run_dir.name}]: no measured rho/U in run_metadata.json -> "
              f"normalized quantities, Re and Ca_e will be NaN")
    return info, meta


# --------------------------------------------------
# One angle
# --------------------------------------------------

def spectrum_peak(f, p, min_hz):
    mask = f >= min_hz
    if not mask.any():
        return np.nan, np.nan
    k = np.argmax(p[mask])
    return float(f[mask][k]), float(p[mask][k])


def analyze_angle(i, angle_val, accel_path, load_path, info, meta_rows, enc, args):
    accel_df, t_acc0 = load_accel_csv(accel_path)
    mg = accel_df.attrs.get("mg_per_lsb") or info["accel_mg_per_lsb"] or 0.061
    raw = accel_df[ACCEL_COLS].to_numpy(dtype=float)
    t = accel_df["time"].to_numpy(dtype=float)
    n = len(t)
    duration = float(t[-1]) if n else 0.0
    fs_acc = n / duration if duration > 0 else np.nan
    row = dict(index=i, requested_angle_deg=angle_val, accel_file=accel_path.name,
               accel_n=n, accel_rate_hz=fs_acc, accel_mg_per_lsb=mg)
    notes = []

    # ---- clipping (full capture, before any trimming) ----
    at_max = np.abs(raw) >= ACCEL_SAT_LEVEL * INT16_MAX
    frac = at_max.mean(axis=0) if n else np.zeros(len(ACCEL_COLS))
    flagged = {c: f for c, f in zip(ACCEL_COLS, frac) if f >= ACCEL_SAT_WARN_FRACTION}
    row["accel_clip_max_frac"] = float(frac.max()) if n else np.nan
    row["accel_clipped"] = bool(flagged)
    row["accel_clip_components"] = ";".join(f"{c}={100 * f:.1f}%" for c, f in flagged.items())
    row["accel_peak_g"] = float(np.abs(raw).max() * mg / 1000.0) if n else np.nan
    row["accel_touched_max"] = bool((frac > 0).any())   # any sample at all (below warn threshold)

    odr = info["accel_odr_hz"]
    row["accel_rate_low"] = bool(odr and fs_acc < 0.9 * odr)
    if row["accel_rate_low"]:
        notes.append(f"measured accel rate {fs_acc:.0f} Hz < 90% of ODR {odr} Hz (dropped samples?)")

    # ---- accel variance / spectra ----
    keep = t >= args.trim_s
    mags = np.linalg.norm((raw * mg / 1000.0).reshape(-1, NUM_SENSORS, 3), axis=2)  # (N,4) in g
    a_c = info["a_c_g"]
    spec = {}
    nper = min(args.nperseg, int(keep.sum()))
    psd_acc = []
    for s in range(NUM_SENSORS):
        x = mags[keep, s]
        v = float(np.var(x)) if len(x) else np.nan
        row[f"var_g2_s{s + 1}"] = v
        row[f"nvar_s{s + 1}"] = v / a_c ** 2 if math.isfinite(a_c) and a_c > 0 else np.nan
        if len(x) >= 16 and math.isfinite(fs_acc):
            f, p = welch(x - x.mean(), fs=fs_acc, nperseg=nper)
            pk_f, pk_p = spectrum_peak(f, p, args.min_peak_hz)
            psd_acc.append(p)
        else:
            f, pk_f, pk_p = None, np.nan, np.nan
        row[f"peak_hz_s{s + 1}"] = pk_f
        row[f"peak_psd_s{s + 1}"] = pk_p
    if psd_acc:
        spec[f"f_accel_{i}"] = f
        spec[f"psd_accel_{i}"] = np.array(psd_acc)
    row["nvar_max_sensor"] = float(np.nanmax([row[f"nvar_s{s}"] for s in range(1, 5)]))
    row["nvar_max_sensor_which"] = int(np.nanargmax([row[f"nvar_s{s}"] for s in range(1, 5)]) + 1)

    # ---- load cell ----
    row["load_file"] = load_path.name if load_path else None
    row.update(load_n=0, var_N2=np.nan, nvar_load=np.nan, load_clipped=False,
               load_clip_max_frac=np.nan, load_peak_hz=np.nan)
    if load_path is not None:
        load_df, t_load0 = load_load_csv(load_path)
        tl = load_df["time"].to_numpy(dtype=float)
        keep_l = tl >= args.trim_s
        F = load_df[FORCE_COLS].to_numpy(dtype=float)
        Fmag = np.linalg.norm(F, axis=1)
        row["load_n"] = len(tl)
        row["var_N2"] = float(np.var(Fmag[keep_l])) if keep_l.any() else np.nan
        F_c = info["F_c_N"]
        row["nvar_load"] = row["var_N2"] / F_c ** 2 if math.isfinite(F_c) and F_c > 0 else np.nan
        for col in FORCE_COLS + ["Mx", "My", "Mz"]:
            row[f"var_{col}"] = float(np.var(load_df[col].to_numpy(dtype=float)[keep_l]))
        stg = np.abs(load_df[STG_COLS].to_numpy(dtype=float))
        lfrac = (stg >= DAQ_CLIP_LEVEL * DAQ_RANGE_V).mean(axis=0)
        row["load_clip_max_frac"] = float(lfrac.max())
        row["load_clipped"] = bool((lfrac >= ACCEL_SAT_WARN_FRACTION).any())
        fs_l = info["loadcell_sample_rate_hz"]
        xl = Fmag[keep_l] - Fmag[keep_l].mean()
        if len(xl) >= 16:
            f, p = welch(xl, fs=fs_l, nperseg=min(args.nperseg, len(xl)))
            row["load_peak_hz"], _ = spectrum_peak(f, p, args.min_peak_hz)
            spec[f"f_load_{i}"], spec[f"psd_load_{i}"] = f, p
    else:
        notes.append("no load-cell file")

    # ---- motor angle: encoder, never the commanded value ----
    ang, src, std = None, "unavailable", np.nan
    for r in meta_rows:
        rq = r.get("motor_angle_requested_deg")
        if rq is not None and abs(rq - angle_val) < 1e-6 and r.get("motor_angle_actual_deg") is not None:
            ang, src = float(r["motor_angle_actual_deg"]), "metadata"
            std = r.get("motor_angle_std_deg") if r.get("motor_angle_std_deg") is not None else np.nan
            break
    if ang is None and enc is not None and enc[1] is not None:
        mean, ns, sd = windowed_average_angle(enc[0], enc[1], t_acc0, duration, info["counts_per_deg"])
        if mean is not None:
            ang, src, std = mean, "encoder_window_mean", sd
    row.update(motor_angle_deg=ang if ang is not None else np.nan,
               motor_angle_std_deg=std, motor_angle_source=src)
    if ang is None:
        notes.append("no encoder angle")

    # ---- frame angles ----
    row.update(inclination_deg=np.nan, aoa_deg=np.nan)
    sw, mt = info["sweep_angle_deg"], info["mounting_angle_deg"]
    if ang is not None and sw is not None and mt is not None:
        fr = ct.compute_frame_angles(ang, sw, mt)
        row["inclination_deg"] = fr["inclination_angle_deg_shifted"]
        row["aoa_deg"] = fr["angle_of_attack_deg_shifted"]

    row["notes"] = "; ".join(notes)
    return row, spec


# --------------------------------------------------
# One run
# --------------------------------------------------

def process_run(run_dir, out_root, args, consts):
    out_dir = out_root / run_dir.name
    info_path = out_dir / "run_info.json"
    if info_path.exists() and not args.force:
        old = json.loads(info_path.read_text())
        if old.get("extractor_version") == EXTRACTOR_VERSION and old.get("complete"):
            print(f"[{run_dir.name}] up to date, skipping")
            return
    print(f"\n[{run_dir.name}]")
    info, meta = build_run_info(run_dir, consts)
    pairs = _discover_angle_files(run_dir)
    if not pairs:
        print("  no angle_*.csv files, skipping")
        return
    enc = None
    enc_path = run_dir / "encoder_log.txt"
    if enc_path.exists():
        enc = load_encoder_log(enc_path)
    meta_rows = meta.get("angles") or []

    rows, spectra = [], {}
    for i, (angle_val, accel_path, load_path) in enumerate(pairs):
        row, spec = analyze_angle(i, angle_val, accel_path, load_path, info, meta_rows, enc, args)
        rows.append(row)
        spectra.update(spec)
        flag = " CLIPPED" if row["accel_clipped"] else ""
        print(f"  {i + 1:2d}/{len(pairs)} {angle_val:+7.2f} deg  nvar_max={row['nvar_max_sensor']:.3g}"
              f"  peak_g={row['accel_peak_g']:.2f}{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "run_name", run_dir.name)
    df.to_csv(out_dir / "angle_summary.csv", index=False)
    np.savez_compressed(out_dir / "spectra.npz",
                        requested_angles=np.array([r["requested_angle_deg"] for r in rows]),
                        **spectra)
    info["n_angles"] = len(rows)
    info["n_accel_clipped"] = int(df["accel_clipped"].sum())
    info["complete"] = True
    info_path.write_text(json.dumps(info, indent=2, default=str))
    print(f"  wrote {out_dir}  ({info['n_accel_clipped']} angle(s) flagged as clipped)")


# --------------------------------------------------
# Catalog (stage 2)
# --------------------------------------------------

def build_catalog(out_root):
    angle_frames, run_rows = [], []
    for info_path in sorted(out_root.glob("*/run_info.json")):
        info = json.loads(info_path.read_text())
        df = pd.read_csv(info_path.parent / "angle_summary.csv")
        for k in ("blade", "nominal_psi", "fan_setting", "version", "density_kg_m3",
                  "velocity_m_s", "reynolds_number", "cauchy_edgewise",
                  "static_pressure_gauge_psi", "F_c_N", "a_c_g"):
            df[k] = info.get(k)
        angle_frames.append(df)

        ok = ~df["accel_clipped"].astype(bool)
        nv = df["nvar_max_sensor"]
        row = {k: info.get(k) for k in (
            "run_name", "blade", "nominal_psi", "fan_setting", "version", "density_kg_m3",
            "velocity_m_s", "static_pressure_gauge_psi", "temp_c", "reynolds_number",
            "cauchy_edgewise", "F_c_N", "a_c_g", "accel_fs", "meta_status", "n_angles")}
        row.update(
            nvar_max=float(nv.max()),
            nvar_max_angle_deg=float(df.loc[nv.idxmax(), "motor_angle_deg"]),
            nvar_p90=float(nv.quantile(0.9)),
            nvar_median=float(nv.median()),
            nvar_max_unclipped=float(nv[ok].max()) if ok.any() else np.nan,
            nvar_load_max=float(df["nvar_load"].max()),
            n_accel_clipped=int((~ok).sum()),
            n_load_clipped=int(df["load_clipped"].astype(bool).sum()),
        )
        run_rows.append(row)
    if not angle_frames:
        print("No summaries found to catalog.")
        return
    pd.concat(angle_frames, ignore_index=True).to_csv(out_root / "master_angles.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out_root / "master_runs.csv", index=False)
    print(f"\nCatalog: {len(run_rows)} runs -> {out_root / 'master_runs.csv'}, "
          f"{out_root / 'master_angles.csv'}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", help="run directories to process")
    ap.add_argument("--root", help="folder containing run directories (all subfolders processed)")
    ap.add_argument("--out", default="Production_summaries", help="local output folder")
    ap.add_argument("--force", action="store_true", help="reprocess runs already summarized")
    ap.add_argument("--catalog-only", action="store_true", help="only rebuild the master tables")
    ap.add_argument("--trim-s", type=float, default=0.0,
                    help="ignore the first N seconds of each capture for variance/spectra")
    ap.add_argument("--nperseg", type=int, default=2048, help="Welch segment length")
    ap.add_argument("--min-peak-hz", type=float, default=5.0,
                    help="lowest frequency considered when picking a spectral peak")
    ap.add_argument("--chord-avg-m", type=float, default=DEFAULTS["chord_avg_m"])
    ap.add_argument("--span-m", type=float, default=DEFAULTS["span_m"])
    ap.add_argument("--mass-kg", type=float, default=DEFAULTS["mass_kg"])
    ap.add_argument("--B-e", type=float, default=DEFAULTS["B_e"],
                    help="average edgewise bending stiffness for Ca_e (default 3.47e8)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    consts = dict(chord_avg_m=args.chord_avg_m, span_m=args.span_m,
                  mass_kg=args.mass_kg, B_e=args.B_e)

    if not args.catalog_only:
        dirs = [Path(d) for d in args.run_dirs]
        if args.root:
            dirs += sorted(p for p in Path(args.root).iterdir()
                           if p.is_dir() and (any(p.glob("angle_*deg.csv")) or (p / "run_metadata.json").exists()))
        if not dirs:
            raise SystemExit("Give run directories or --root")
        for d in dirs:
            try:
                process_run(d, out_root, args, consts)
            except Exception as e:   # one bad run must not stop the batch
                print(f"  ERROR in {d.name}: {type(e).__name__}: {e}")
    build_catalog(out_root)


if __name__ == "__main__":
    main()