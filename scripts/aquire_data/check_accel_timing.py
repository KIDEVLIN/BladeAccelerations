#!/usr/bin/env python3
"""
check_accel_timing.py

Checks, on our own raw files, how the accelerometer samples are really
timed, before any data is reprocessed. Read-only: raw data is never changed.

It answers three questions (see utils/accel_timing.py for the method):
  1. Are samples lost?   Every read full (84 sets)? Delivered rate vs sensor rate.
  2. What is the sensor rate fs, and are the 84 rows of a read consecutive
     samples with gaps between reads, or evenly spread between reads (the
     original assumption)? Measured from the data alone: mode frequency in
     cycles/sample inside reads + read-to-read phase in real time. No vendor
     value and no video needed.
  3. Which read time anchors each block (lag)? lag 0 = the read's own time
     stamp (postmortem assumption). Only identifiable where read intervals
     vary, i.e. production captures.

Usage (from scripts/aquire_data):
    python check_accel_timing.py "D:/GE_blade/Production_10-8-26" "D:/GE_blade/<ring-down folder>" --out timing_check
    python check_accel_timing.py "D:/GE_blade/Production_10-8-26/DTU10MW_704psi_800fs_v2" --max-est-per-run 20

Inputs may be roots (searched recursively), run folders or single CSVs. Accel
CSVs are recognised by their '# LIS2DS12' header line; *_load.csv is skipped.

Outputs in --out:
    read_stats.csv        one row per capture: reads, interval, delivered rate, loss
    timing_estimates.csv  one row per capture analysed for fs / lag
    timing_summary.txt    the verdict (also printed)
    timing_check.png      read intervals, coherence vs fs, old vs gap-aware spectrum
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils import accel_timing as at  # noqa: E402

ACCEL_COLS = [f"s{s}_{a}" for s in range(1, 5) for a in "xyz"]
BLOCK = 84
FS_GRID = np.arange(1150.0, 1700.0, 0.25)
LAGS = range(0, 16)
MIN_PROMINENCE = 30.0          # in-read line must stand this far above the median to be used for fs / lag


# --------------------------------------------------
# Discovery
# --------------------------------------------------

def is_accel_csv(p):
    if p.suffix.lower() != ".csv" or p.stem.endswith("_load"):
        return False
    try:
        with open(p) as f:
            return f.readline().startswith("# LIS2DS12")
    except OSError:
        return False


def discover(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            cands = [p]
        else:
            cands = sorted(p.rglob("*.csv"))
        out += [c for c in cands if is_accel_csv(c)]
    return sorted(set(out))


def kind_of(p):
    n = p.name.lower()
    if n.startswith("ring_"):
        return "ringdown"
    if n.startswith("rest_"):
        return "rest"
    return "production"


# --------------------------------------------------
# Per capture
# --------------------------------------------------

def stats_row(p, fs_nominal):
    bt = pd.read_csv(p, comment="#", usecols=["time"])["time"].to_numpy(float)
    st = at.read_stats(at.find_reads(bt), fs_nominal, BLOCK)
    return dict(file=str(p), run=p.parent.name, kind=kind_of(p), **st)


def estimate_row(p):
    df = pd.read_csv(p, comment="#")
    bt = df["time"].to_numpy(float)
    reads = at.find_reads(bt)
    vals = df[ACCEL_COLS].to_numpy(float)
    Xs = [at._full_blocks(vals[:, j], reads, BLOCK)[0] for j in range(len(ACCEL_COLS))]
    if Xs[0].shape[0] < 20:
        return None, None
    ch, c, prom = at.dominant_block_mode(Xs)
    row = dict(file=str(p), run=p.parent.name, kind=kind_of(p), channel=ACCEL_COLS[ch],
               c_cycles_per_sample=c, prominence=prom)
    if prom < MIN_PROMINENCE:
        row["used"] = False
        return row, None
    dc = 0.004
    band = ((c - dc) * FS_GRID[-1], (c + dc) * FS_GRID[0])
    e = at.estimate_timing(vals[:, ch], bt, band, fs_grid=FS_GRID, lags=LAGS, block_len=BLOCK)
    row.update(used=True, fs_hz=e.fs_hz, f0_hz=e.f0_hz, lag=e.lag, coherence=e.coherence,
               coherence_lag0=e.coherence_by_lag[0], coherence_lag1=e.coherence_by_lag[1],
               delivered_rate_hz=e.delivered_rate_hz, coherence_at_delivered=e.coherence_at_delivered,
               runner_up_fs_hz=e.runner_up_fs_hz, runner_up_coherence=e.runner_up_coherence,
               n_pairs=e.n_pairs)
    return row, e


# --------------------------------------------------
# Verdict
# --------------------------------------------------

def summarize(stats, est_rows, combined, fs_nominal, lobes_by_kind):
    L = []
    L.append("Accelerometer timing check")
    L.append("=" * 60)
    for kind, g in stats.groupby("kind"):
        L.append(f"\n[{kind}] {len(g)} captures, {g.n_reads.sum():,} reads")
        L.append(f"  reads returning the full {BLOCK} sets : {100 * g.frac_reads_full.mean():.2f}% (mean over captures)")
        L.append(f"  time between reads, median          : {g.read_dt_median_ms.median():.1f} ms "
                 f"(captures' 5-95%: {g.read_dt_p05_ms.median():.1f}-{g.read_dt_p95_ms.median():.1f} ms)")
        L.append(f"  delivered rate, median              : {g.delivered_rate_hz.median():.0f} rows/s")

    L.append("\nSensor rate and block anchoring, measured from the data")
    L.append("-" * 60)
    fs_best = None
    for kind, (fs, lag, coh, n, coh_del, lag_scores) in combined.items():
        if n == 0:
            L.append(f"[{kind}] no capture had a strong enough single mode (prominence >= {MIN_PROMINENCE:g})")
            continue
        dlv = stats.loc[stats.kind == kind, "delivered_rate_hz"].median()
        L.append(f"[{kind}] {n} captures combined")
        L.append(f"  best sensor rate fs   : {fs:.1f} Hz   (coherence {coh:.3f})")
        L.append(f"  if fs = delivered rate: {dlv:.0f} Hz  (coherence {coh_del:.3f})  <- the old even-spacing view")
        top = sorted(lag_scores.items(), key=lambda kv: -kv[1])[:3]
        L.append("  best lags             : " + ", ".join(f"lag {k}: {v:.3f}" for k, v in top))
        lobes = lobes_by_kind.get(kind, [])
        if len(lobes) > 1:
            L.append("  fs AMBIGUOUS from these captures alone (near-equal candidates: "
                     + ", ".join(f"{f:.0f} Hz ({r:.2f})" for f, r in lobes) + "); take fs from the ring-downs")
        if kind == "ringdown" or (fs_best is None and len(lobes) <= 1):
            fs_best = fs
        if abs(fs - dlv) < 0.03 * dlv and coh_del > coh - 0.05:
            L.append("  => samples look CONTINUOUS and evenly spread between reads; the old timing is right.")
        else:
            L.append(f"  => rows of one read are consecutive samples at ~{fs:.0f} Hz with GAPS between reads;")
            L.append("     even spacing is wrong for spectra and timing. Use the gap-aware time base.")
    if est_rows is not None and len(est_rows):
        u = est_rows[est_rows.used == True]  # noqa: E712
        if len(u):
            L.append(f"\nPer-capture fs (captures used: {len(u)}): median {u.fs_hz.median():.1f} Hz, "
                     f"IQR {u.fs_hz.quantile(.25):.1f}-{u.fs_hz.quantile(.75):.1f} Hz")
    L.append(f"\nNominal fs used for the loss figures above: {fs_nominal:.0f} Hz")
    if fs_best is None:
        L.append("\nNo unambiguous fs from this data: add ring-down captures, or pass the folder of one.")
    lost = 1 - stats.delivered_rate_hz.median() / (fs_best or fs_nominal)
    L.append(f"Implied fraction of samples lost: {100 * lost:.1f}%")
    return "\n".join(L), fs_best


# --------------------------------------------------
# Figure
# --------------------------------------------------

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def figure(stats, combined_curves, combined, example, fs_fig, lag_fig, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # (a) delivered rate per capture
    a = ax[0, 0]
    for k, col in (("production", BLUE), ("ringdown", ORANGE)):
        g = stats[stats.kind == k]
        if len(g):
            a.hist(g.delivered_rate_hz, bins=40, color=col, alpha=0.85, label=f"{k} ({len(g)})")
    if fs_fig:
        a.axvline(fs_fig, color=INK, lw=1, ls="--")
        a.text(fs_fig, a.get_ylim()[1] * 0.95, f" measured fs {fs_fig:.0f} Hz", color=INK, va="top")
    a.set(xlabel="Delivered rows per second (per capture)", ylabel="Captures",
          title="(a) Rows in the files vs sensor rate")
    a.legend(frameon=False)

    # (b) read intervals
    a = ax[0, 1]
    a.hist(stats.read_dt_median_ms, bins=40, color=BLUE)
    if fs_fig:
        a.axvline(1e3 * BLOCK / fs_fig, color=INK, lw=1, ls="--")
        a.text(1e3 * BLOCK / fs_fig, a.get_ylim()[1] * 0.95, f" {BLOCK} sets take {1e3 * BLOCK / fs_fig:.0f} ms",
               color=INK, va="top")
    a.set(xlabel="Median time between reads (ms, per capture)", ylabel="Captures",
          title="(b) Reads come slower than the sensors fill 84 sets")

    # (c) coherence vs fs
    a = ax[1, 0]
    for (k, curves), col in zip(combined_curves.items(), (BLUE, ORANGE)):
        if not curves:
            continue
        lag = combined[k][1]
        a.plot(FS_GRID, curves[lag], color=col, lw=1.5, label=f"{k}, lag {lag}")
        dlv = stats.loc[stats.kind == k, "delivered_rate_hz"].median()
        a.axvline(dlv, color=col, lw=1, ls=":")
    a.axvspan(FS_GRID[0], stats.delivered_rate_hz.min(), color=GRID, alpha=0.8, lw=0)
    a.text(FS_GRID[0] + 5, -0.95, "fs below the\ndelivered rate:\nimpossible", color=MUTED, fontsize=8)
    a.set(xlabel="Candidate sensor rate fs (Hz)", ylabel="Read-to-read phase coherence",
          title="(c) Which sample rate explains the data (dotted: delivered rate)", ylim=(-1.05, 1.05))
    a.legend(frameon=False, loc="lower right")

    # (d) example spectrum
    a = ax[1, 1]
    if example is not None:
        p, ch = example
        df = pd.read_csv(p, comment="#")
        bt = df["time"].to_numpy(float)
        x = df[ch].to_numpy(float)
        reads = at.find_reads(bt)
        dlv = at.read_stats(reads)["delivered_rate_hz"]
        fe, pe = welch(x - x.mean(), fs=dlv, nperseg=min(2048, len(x)))
        slots, _ = at.gap_aware_slots(reads, fs_fig, lag_fig)
        fg, pg, _ = at.masked_welch(x, slots, fs_fig, nperseg=2048)
        a.semilogy(fe, pe, color=ORANGE, lw=1, label=f"old: evenly spaced at {dlv:.0f}/s")
        a.semilogy(fg, pg, color=BLUE, lw=1, label=f"gap-aware: fs {fs_fig:.0f} Hz, lag {lag_fig}")
        a.set_xlim(0, min(400, fs_fig / 2))
        a.set(title=f"(d) {Path(p).parent.name}/{Path(p).name}, {ch}")
        a.legend(frameon=False)
    a.set(xlabel="Frequency (Hz)", ylabel="PSD (raw counts$^2$/Hz)")
    fig.savefig(out_png, dpi=150)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="roots, run folders or accel CSVs")
    ap.add_argument("--out", default="timing_check")
    ap.add_argument("--fs-nominal", type=float, default=at.DEFAULT_SENSOR_FS_HZ,
                    help="only used for the loss figures in read_stats.csv")
    ap.add_argument("--max-est-per-run", type=int, default=8,
                    help="captures per run folder analysed for fs/lag (evenly spread; ring-downs: all)")
    ap.add_argument("--stats-only", action="store_true", help="skip the fs/lag estimate (fast)")
    args = ap.parse_args()

    files = discover(args.paths)
    if not files:
        sys.exit("no accelerometer CSVs found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} accelerometer captures")

    rows = []
    for i, p in enumerate(files):
        try:
            rows.append(stats_row(p, args.fs_nominal))
        except Exception as ex:  # one bad file must not stop the check
            print(f"  skip {p}: {ex}")
        if (i + 1) % 100 == 0:
            print(f"  read stats {i + 1}/{len(files)}")
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "read_stats.csv", index=False)

    est_rows, ests = [], {}
    if not args.stats_only:
        by_run = {}
        for p in files:
            by_run.setdefault(p.parent, []).append(p)
        todo = []
        for run, ps in by_run.items():
            ps = [p for p in ps if kind_of(p) != "rest"]
            if ps and kind_of(ps[0]) == "production" and len(ps) > args.max_est_per_run:
                ps = [ps[j] for j in np.linspace(0, len(ps) - 1, args.max_est_per_run).round().astype(int)]
            todo += ps
        print(f"estimating fs / lag on {len(todo)} captures")
        for i, p in enumerate(todo):
            try:
                r, e = estimate_row(p)
            except Exception as ex:
                print(f"  skip {p}: {ex}")
                continue
            if r is not None:
                est_rows.append(r)
            if e is not None:
                ests.setdefault(kind_of(p), []).append((p, r["channel"], e))
            if (i + 1) % 20 == 0:
                print(f"  estimated {i + 1}/{len(todo)}")
    est_df = pd.DataFrame(est_rows)
    if len(est_df):
        est_df.to_csv(out / "timing_estimates.csv", index=False)

    combined, curves_by_kind = {}, {}
    for kind in ("ringdown", "production"):
        es = [e for _, _, e in ests.get(kind, [])]
        if not es:
            combined[kind] = (np.nan, None, np.nan, 0, np.nan, {})
            curves_by_kind[kind] = {}
            continue
        fs, lag, coh, curves = at.combine_estimates(es)
        W = sum(e.weight for e in es)
        coh_del = sum(e.coherence_at_delivered * e.weight for e in es) / W
        combined[kind] = (fs, lag, coh, len(es), coh_del, {k: float(v.max()) for k, v in curves.items()})
        curves_by_kind[kind] = curves

    lobes_by_kind = {}
    for kind, curves in curves_by_kind.items():
        if curves:
            fs_min = 0.98 * stats.loc[stats.kind == kind, "delivered_rate_hz"].max()
            lobes_by_kind[kind] = at.competing_lobes(FS_GRID, curves[combined[kind][1]], fs_min)
    text, fs_best = summarize(stats, est_df if len(est_df) else None, combined, args.fs_nominal, lobes_by_kind)
    (out / "timing_summary.txt").write_text(text)
    print("\n" + text)

    # figure: fs from ring-downs if available; lag from production if available
    fs_fig = fs_best or args.fs_nominal
    lag_fig = combined["production"][1] if combined["production"][3] else (combined["ringdown"][1] or 0)
    example = None
    for kind in ("ringdown", "production"):
        if ests.get(kind):
            p, ch, _ = max(ests[kind], key=lambda t: t[2].weight)
            example = (p, ch)
            break
    figure(stats, curves_by_kind, combined, example, fs_fig, lag_fig, out / "timing_check.png")
    print(f"\nwrote {out}/read_stats.csv, timing_estimates.csv, timing_summary.txt, timing_check.png")


if __name__ == "__main__":
    main()
