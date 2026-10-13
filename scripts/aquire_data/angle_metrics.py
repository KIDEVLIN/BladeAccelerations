#!/usr/bin/env python3
"""
angle_metrics.py

Run-level vibration metrics from the per-angle table, built to be insensitive to how
densely angles were sampled and to cope with sweeps that do not cover the full 360 deg.
Works only on master_angles.csv (written by extract_run_summary.py); raw data is never read.

Usage (from scripts/aquire_data):
    python angle_metrics.py Production_summaries/master_angles.csv
    python angle_metrics.py Production_summaries/master_angles.csv --k 10 --decimate 10
    python angle_metrics.py Production_summaries/master_angles.csv --baseline 0.001   # fixed floor
    python angle_metrics.py Production_summaries/master_angles.csv --baseline-file baselines.csv
    python angle_metrics.py Production_summaries/master_angles.csv --sensor S4

Output: master_metrics.csv next to master_angles.csv (one row per run). With --sensor S4 the output is
master_metrics_S4.csv (the form plot_design_space.py --sensor S4 looks for).

Per-angle quantity: nvar_max_sensor = worst-sensor normalized acceleration variance
(Var(|a|)/a_c^2), or with --sensor S1..S4 that one accelerometer's nvar_s1..nvar_s4. All metrics below,
including the baseline, are then computed from that sensor alone, and the clipped-angle count refers
to that sensor. The curve of that quantity vs motor angle is treated as piecewise linear
between measured angles. Segments longer than --max-gap (default 20 deg) are treated as NOT
measured, so a targeted sweep is never interpolated across a hole.

Metrics
  baseline            quiet level b. Default: angle-weighted median over the covered angles.
                      Use --baseline p25 for a lower floor, or a number to fix b for all runs
                      (recommended when sweeps are partial and the median is not representative).
                      --baseline-file lets you give each run its own fixed value, e.g. copied from a
                      full-sweep run at the same conditions (columns: run_name,baseline).
  excess_dose_deg     integral of max(0, f - b) d(theta), units nvar*deg. Exact for the
                      piecewise-linear curve, independent of sampling density. For a partial sweep
                      it is a LOWER BOUND on the full-sweep value.
  excess_mean_full    excess_dose_deg / 360. Excess variance averaged over the whole circle:
                      exact for a full sweep, a lower bound for a partial one. This is the
                      primary cross-run comparison.
  excess_mean_covered excess_dose_deg / covered range (what a partial sweep looked like on average;
                      biased high when the sweep targeted peaks)
  vibrating_width_deg degrees of angle where f > k*b (default k=10); n_bands and bands list the
                      contiguous vibrating angle ranges
  nvar_mean_covered   plain angle-averaged variance over the covered range
  nvar_p90_weighted   90th percentile weighted by the angular span each sample represents
  nvar_max(+angle)    peak sample, with flags: peak_at_edge (peak is the first/last measured
                      angle, so the true peak may lie outside the sweep) and peak_maybe_unresolved
                      (a gap >= 10 deg next to the peak, so a narrow peak could be cut)
  coverage_deg / coverage_frac / max_gap_deg / full_sweep / baseline_reliable

--decimate STEP re-computes the dose, width and mean using only angles on a STEP-degree grid
(same baseline) and reports the ratio to the full-density value, a direct test of sampling
sensitivity. Ratios near 1 mean the metric does not depend on the extra angles.

Requires: numpy pandas
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PEAK_GAP_WARN_DEG = 10.0


# --------------------------------------------------
# Core numerics
# --------------------------------------------------

def clipped_sensors(s):
    """Sensor numbers named in an accel_clip_components string such as 's2_x=3.0%;s2_z=1.2%'."""
    if not isinstance(s, str) or not s:
        return set()
    return {int(x) for x in re.findall(r"s(\d)_", s)}


def prep(g, col="nvar_max_sensor"):
    """Sorted angles and values; repeated angles are averaged."""
    g = g.dropna(subset=["motor_angle_deg", col])
    d = pd.DataFrame({"t": g["motor_angle_deg"].round(2).to_numpy(float),
                      "f": g[col].to_numpy(float)})
    d = d.groupby("t", as_index=False)["f"].mean().sort_values("t")
    return d["t"].to_numpy(), d["f"].to_numpy()


def segments(t, f, max_gap):
    h = np.diff(t)
    ok = (h > 0) & (h <= max_gap)
    return t[:-1][ok], h[ok], f[:-1][ok], f[1:][ok]


def sample_weights(t, max_gap):
    """Angular span each sample represents: half the gap to each neighbour (gap capped)."""
    h = np.minimum(np.diff(t), max_gap)
    w = np.zeros_like(t, dtype=float)
    w[:-1] += h / 2
    w[1:] += h / 2
    return w


def weighted_quantile(v, w, q):
    if len(v) == 1 or w.sum() <= 0:
        return float(np.quantile(v, q))
    o = np.argsort(v)
    v, w = v[o], w[o]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, v))


def positive_area(y0, y1, h):
    """Exact area of max(0, y) for a straight line from y0 to y1 over width h."""
    both = (y0 >= 0) & (y1 >= 0)
    cross = y0 * y1 < 0
    yp, yn = np.maximum(y0, y1), np.minimum(y0, y1)
    area = np.where(both, (y0 + y1) / 2 * h, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        area = np.where(cross, h * yp ** 2 / (2 * (yp - yn)), area)
    return float(np.nansum(area))


def length_above(y0, y1, h):
    both = (y0 >= 0) & (y1 >= 0)
    cross = y0 * y1 < 0
    yp, yn = np.maximum(y0, y1), np.minimum(y0, y1)
    ln = np.where(both, h, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.where(cross, h * yp / (yp - yn), ln)
    return float(np.nansum(ln))


def bands_above(x0, h, y0, y1):
    """Contiguous angle intervals where the piecewise-linear curve is above zero."""
    iv = []
    for a, hh, u, v in zip(x0, h, y0, y1):
        if u >= 0 and v >= 0:
            iv.append((a, a + hh))
        elif u > 0 > v:
            iv.append((a, a + hh * u / (u - v)))
        elif u < 0 < v:
            iv.append((a + hh * (-u) / (v - u), a + hh))
    iv = [x for x in iv if x[1] - x[0] > 1e-9]
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def curve_metrics(t, f, b, thr, max_gap):
    x0, h, f0, f1 = segments(t, f, max_gap)
    cov = float(h.sum())
    if cov <= 0:
        return dict(coverage_deg=0.0, excess_dose_deg=np.nan, nvar_mean_covered=np.nan,
                    vibrating_width_deg=np.nan, bands=[])
    return dict(
        coverage_deg=cov,
        excess_dose_deg=positive_area(f0 - b, f1 - b, h),
        nvar_mean_covered=float(((f0 + f1) / 2 * h).sum() / cov),
        vibrating_width_deg=length_above(f0 - thr, f1 - thr, h),
        bands=bands_above(x0, h, f0 - thr, f1 - thr),
    )


# --------------------------------------------------
# One run
# --------------------------------------------------

def run_metrics(run, g, args):
    t, f = prep(g, args.value_col)
    row = dict(run_name=run, n_angles=len(t))
    if len(t) < 2:
        row["note"] = "fewer than 2 usable angles"
        return row
    w = sample_weights(t, args.max_gap)
    h = np.diff(t)
    covered = float(h[(h > 0) & (h <= args.max_gap)].sum())
    row.update(theta_min=t[0], theta_max=t[-1], coverage_deg=covered,
               coverage_frac=covered / args.sweep_deg, max_gap_deg=float(h.max()))
    row["full_sweep"] = row["coverage_frac"] >= args.full_frac

    # baseline (a per-run value from --baseline-file wins over --baseline)
    if run in args.baseline_map:
        b, rel = float(args.baseline_map[run]), True
    elif args.baseline == "median":
        b, rel = weighted_quantile(f, w, 0.5), row["coverage_frac"] >= args.min_cover
    elif args.baseline == "p25":
        b, rel = weighted_quantile(f, w, 0.25), row["coverage_frac"] >= args.min_cover
    else:
        b, rel = float(args.baseline), True
    thr = args.k * b
    row.update(baseline=b, baseline_reliable=bool(rel), threshold=thr)

    m = curve_metrics(t, f, b, thr, args.max_gap)
    bands = m.pop("bands")
    dose = m["excess_dose_deg"]
    row.update(
        excess_dose_deg=dose,
        excess_mean_full=dose / args.sweep_deg,
        excess_mean_covered=dose / covered if covered > 0 else np.nan,
        nvar_mean_covered=m["nvar_mean_covered"],
        nvar_p90_weighted=weighted_quantile(f, w, 0.9),
        vibrating_width_deg=m["vibrating_width_deg"],
        vibrating_frac_covered=m["vibrating_width_deg"] / covered if covered > 0 else np.nan,
        n_bands=len(bands),
        bands=";".join(f"{s:.0f}–{e:.0f}" for s, e in bands),
    )

    ip = int(np.argmax(f))
    gl = t[ip] - t[ip - 1] if ip > 0 else np.nan
    gr = t[ip + 1] - t[ip] if ip < len(t) - 1 else np.nan
    pg = np.nanmax([gl, gr])
    row.update(nvar_max=float(f[ip]), nvar_max_angle_deg=float(t[ip]),
               peak_at_edge=bool(ip in (0, len(t) - 1)), peak_nbr_gap_deg=float(pg),
               peak_maybe_unresolved=bool(pg >= PEAK_GAP_WARN_DEG))

    if args.sensor_id:                     # clipping of the chosen sensor only
        if "accel_clip_components" in g.columns:
            row["n_clipped_angles"] = int(sum(args.sensor_id in clipped_sensors(c)
                                              for c in g["accel_clip_components"]))
    else:
        for col, name in (("accel_clipped", "n_clipped_angles"), ("accel_touched_max", "n_touched_max_angles")):
            if col in g.columns:
                row[name] = int(g[col].astype(bool).sum())

    # sampling-sensitivity test on a coarser grid
    if args.decimate:
        keep = np.abs(t / args.decimate - np.round(t / args.decimate)) * args.decimate < 0.25
        row["dec_angles_removed"] = int((~keep).sum())
        if keep.sum() >= 2:
            md = curve_metrics(t[keep], f[keep], b, thr, args.max_gap)
            dd = md["excess_dose_deg"]
            for key, val, full in (("excess_mean_full", dd / args.sweep_deg, row["excess_mean_full"]),
                                   ("vibrating_width_deg", md["vibrating_width_deg"], row["vibrating_width_deg"]),
                                   ("nvar_mean_covered", md["nvar_mean_covered"], row["nvar_mean_covered"])):
                row[f"dec_ratio_{key}"] = val / full if full else np.nan
        row["dec_nvar_max_ratio"] = float(f[keep].max() / f.max()) if keep.any() else np.nan
    return row


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master_angles", help="path to master_angles.csv")
    ap.add_argument("--out", help="output csv (default: master_metrics.csv, or master_metrics_S4.csv with --sensor S4)")
    ap.add_argument("--sensor", help="use one accelerometer (S1..S4) instead of the worst of the four")
    ap.add_argument("--baseline", default="median",
                    help="median (default), p25, or a number for a fixed quiet level")
    ap.add_argument("--baseline-file",
                    help="csv with columns run_name,baseline giving a fixed quiet level per run "
                         "(e.g. taken from a full-sweep run at the same conditions)")
    ap.add_argument("--k", type=float, default=10.0, help="vibrating threshold = k * baseline")
    ap.add_argument("--max-gap", type=float, default=20.0,
                    help="angle gaps longer than this (deg) are treated as unmeasured")
    ap.add_argument("--sweep-deg", type=float, default=360.0, help="reference sweep (full circle)")
    ap.add_argument("--full-frac", type=float, default=0.95, help="coverage fraction counted as a full sweep")
    ap.add_argument("--min-cover", type=float, default=0.75,
                    help="below this coverage the run's own median/p25 baseline is flagged unreliable")
    ap.add_argument("--decimate", type=float, help="also test sensitivity using only a STEP-degree grid")
    args = ap.parse_args()
    if args.baseline not in ("median", "p25"):
        args.baseline = float(args.baseline)

    args.sensor_id, args.value_col = None, "nvar_max_sensor"
    if args.sensor:
        m = re.fullmatch(r"[sS]?([1-4])", args.sensor.strip())
        if not m:
            raise SystemExit(f"--sensor {args.sensor!r}: use S1 to S4")
        args.sensor_id = int(m[1])
        args.value_col = f"nvar_s{args.sensor_id}"

    args.baseline_map = {}
    if args.baseline_file:
        bf = pd.read_csv(args.baseline_file)
        args.baseline_map = dict(zip(bf["run_name"], bf["baseline"]))

    src = Path(args.master_angles)
    df = pd.read_csv(src)
    need = {"run_name", "motor_angle_deg", args.value_col}
    if not need <= set(df.columns):
        raise SystemExit(f"{src} is missing columns: {sorted(need - set(df.columns))}")

    rows = [run_metrics(run, g, args) for run, g in df.groupby("run_name", sort=True)]
    out = pd.DataFrame(rows)
    default_name = f"master_metrics_S{args.sensor_id}.csv" if args.sensor_id else "master_metrics.csv"
    out_path = Path(args.out) if args.out else src.parent / default_name
    out.to_csv(out_path, index=False)

    show = ["run_name", "coverage_deg", "nvar_max", "nvar_max_angle_deg", "nvar_p90_weighted",
            "excess_mean_full", "vibrating_width_deg", "n_bands", "bands"]
    print(out.sort_values("excess_mean_full", ascending=False)[[c for c in show if c in out]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    flags = []
    for _, r in out.iterrows():
        f = []
        if not r.get("full_sweep", True):
            f.append(f"partial sweep ({r['coverage_deg']:.0f} deg): excess_mean_full is a lower bound")
        if not r.get("baseline_reliable", True):
            f.append("own-median baseline unreliable, consider --baseline <number>")
        if r.get("peak_at_edge", False):
            f.append("peak at the edge of the sweep, true peak may be outside")
        if r.get("peak_maybe_unresolved", False):
            f.append(f"gap {r['peak_nbr_gap_deg']:.0f} deg next to the peak, narrow peak may be unresolved")
        if f:
            flags.append(f"  {r['run_name']}: " + "; ".join(f))
    if flags:
        print("\nFlags:\n" + "\n".join(flags))

    if args.decimate:
        cols = [c for c in out.columns if c.startswith("dec_")]
        print(f"\nSampling sensitivity (grid {args.decimate:g} deg; ratio to full density, 1 = no change):")
        print(out[["run_name"] + cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    full = out[out.get("full_sweep", False) == True]   # noqa: E712
    cols = ["nvar_max", "nvar_p90_weighted", "excess_mean_full", "vibrating_width_deg"]
    if len(full) >= 4:
        print("\nRank agreement between metrics (Spearman, full-sweep runs only):")
        print(full[cols].corr(method="spearman").round(2).to_string())
    else:
        print("\n(rank agreement needs at least 4 full-sweep runs)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()