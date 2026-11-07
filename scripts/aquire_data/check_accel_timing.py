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
MIN_PROMINENCE = 30.0          # auto mode: in-read line must stand this far above the median
MIN_PAIRS_CAPTURE = 10         # a capture's own fs is listed only above this many clean read pairs
MIN_PAIRS_VERDICT = 100        # a group needs this many clean read pairs in total for a verdict
MIN_COHERENCE = 0.6            # ... and a best coherence at least this high
MIN_MARGIN = 0.3               # ... beating the other hypothesis by at least this much
MIN_LAG_MARGIN = 0.03          # best lag must beat the next one by this much to be called identified


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


def estimate_row(p, mode_hz=None):
    """mode_hz: (lo, hi) TRUE frequency range of the mode to use, or None to
    take the strongest line. The range is converted to cycles/sample over
    every fs in FS_GRID, so it does not presume a sample rate."""
    df = pd.read_csv(p, comment="#")
    bt = df["time"].to_numpy(float)
    reads = at.find_reads(bt)
    vals = df[ACCEL_COLS].to_numpy(float)
    Xs = [at._full_blocks(vals[:, j], reads, BLOCK)[0] for j in range(len(ACCEL_COLS))]
    if Xs[0].shape[0] < 20:
        return None, None
    if mode_hz is None:
        ch, c, prom = at.dominant_block_mode(Xs)
    else:
        ch, c, prom = at.dominant_block_mode(Xs, c_min=mode_hz[0] / FS_GRID[-1], c_max=mode_hz[1] / FS_GRID[0])
    label = "auto" if mode_hz is None else f"{mode_hz[0]:g}-{mode_hz[1]:g} Hz"
    row = dict(file=str(p), run=p.parent.name, kind=kind_of(p), mode=label, channel=ACCEL_COLS[ch],
               c_cycles_per_sample=c, prominence=prom)
    if mode_hz is None and prom < MIN_PROMINENCE:
        row["used"] = False
        return row, None
    if mode_hz is None:
        dc = 0.004
        band = ((c - dc) * FS_GRID[-1], (c + dc) * FS_GRID[0])
    else:
        band = mode_hz
    e = at.estimate_timing(vals[:, ch], bt, band, fs_grid=FS_GRID, lags=LAGS, block_len=BLOCK)
    row.update(used=e.n_pairs > 0, n_pairs=e.n_pairs, fs_hz=e.fs_hz, f0_hz=e.f0_hz, lag=e.lag,
               coherence=e.coherence, coherence_lag0=e.coherence_by_lag[0], coherence_lag1=e.coherence_by_lag[1],
               delivered_rate_hz=e.delivered_rate_hz, coherence_at_delivered=e.coherence_at_delivered,
               runner_up_fs_hz=e.runner_up_fs_hz, runner_up_coherence=e.runner_up_coherence)
    return row, (e if e.n_pairs > 0 else None)


# --------------------------------------------------
# Verdict
# --------------------------------------------------

def group_verdict(fs, coh, coh_del, dlv, n_pairs, lobes):
    """'gaps', 'even' or 'inconclusive', with the reason."""
    if n_pairs < MIN_PAIRS_VERDICT:
        return "inconclusive", f"only {n_pairs} clean read pairs (need {MIN_PAIRS_VERDICT})"
    if max(coh, coh_del) < MIN_COHERENCE:
        return "inconclusive", (f"no hypothesis fits well (best {max(coh, coh_del):.3f} < {MIN_COHERENCE}); "
                                "for ring-downs run --buffer: the first ~3-4 s have no gaps, which this check does not model")
    if coh_del >= coh - 0.05 or abs(fs - dlv) < 0.03 * dlv:
        if coh_del >= MIN_COHERENCE:
            return "even", "the delivered rate explains the read-to-read phase"
    if coh - coh_del >= MIN_MARGIN:
        if len(lobes) > 1:
            return "gaps", "even spacing ruled out, but fs ambiguous between alias candidates"
        return "gaps", f"fs = {fs:.0f} Hz beats the delivered rate by {coh - coh_del:.2f} in coherence"
    return "inconclusive", f"margin over the delivered rate only {coh - coh_del:.2f} (need {MIN_MARGIN})"


def summarize(stats, est_rows, groups, fs_nominal):
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
    fs_best, lag_best = None, None
    for key, G in groups.items():
        kind, mode = key
        if G["n_captures"] == 0:
            L.append(f"[{kind}, mode {mode}] no capture had a usable mode")
            continue
        dlv = stats.loc[stats.kind == kind, "delivered_rate_hz"].median()
        L.append(f"[{kind}, mode {mode}] {G['n_captures']} captures, {G['n_pairs']} clean read pairs")
        L.append(f"  best sensor rate fs   : {G['fs']:.1f} Hz   (coherence {G['coh']:.3f})")
        L.append(f"  if fs = delivered rate: {dlv:.0f} Hz  (coherence {G['coh_del']:.3f})  <- the old even-spacing view")
        u = est_rows[(est_rows.kind == kind) & (est_rows["mode"] == mode) & (est_rows.n_pairs >= MIN_PAIRS_CAPTURE)] \
            if est_rows is not None and "n_pairs" in est_rows else []
        if len(u) >= 3:
            se = (u.fs_hz.quantile(.75) - u.fs_hz.quantile(.25)) / 1.35 / np.sqrt(len(u))
            L.append(f"  per-capture fs        : median {u.fs_hz.median():.1f} Hz, IQR {u.fs_hz.quantile(.25):.1f}-"
                     f"{u.fs_hz.quantile(.75):.1f} Hz over {len(u)} captures with >= {MIN_PAIRS_CAPTURE} pairs "
                     f"(~+-{se:.1f} Hz on the median)")
        top = sorted(G["lag_scores"].items(), key=lambda kv: -kv[1])[:3]
        L.append("  best lags             : " + ", ".join(f"lag {k}: {v:.3f}" for k, v in top))
        lag_ok = len(top) > 1 and top[0][1] - top[1][1] >= MIN_LAG_MARGIN
        if len(G["lobes"]) > 1:
            L.append("  fs alias candidates   : " + ", ".join(f"{f:.0f} Hz ({r:.2f})" for f, r in G["lobes"]))
        verdict, why = group_verdict(G["fs"], G["coh"], G["coh_del"], dlv, G["n_pairs"], G["lobes"])
        G["verdict"] = verdict
        if verdict == "gaps":
            L.append(f"  => GAPS: rows of one read are consecutive samples at ~{G['fs']:.0f} Hz ({why}).")
            if len(G["lobes"]) <= 1 and (fs_best is None or kind == "ringdown"):
                fs_best = G["fs"]
            if lag_ok:
                L.append(f"     lag {top[0][0]} identified (beats lag {top[1][0]} by {top[0][1] - top[1][1]:.3f})")
                if lag_best is None:
                    lag_best = top[0][0]
            else:
                L.append(f"     lag NOT identified (top lags within {MIN_LAG_MARGIN}); needs a higher mode or more jitter")
        elif verdict == "even":
            L.append(f"  => EVEN spacing is consistent with this data ({why}).")
        else:
            L.append(f"  => INCONCLUSIVE: {why}.")
    L.append(f"\nNominal fs used for the loss figures above: {fs_nominal:.0f} Hz")
    if fs_best is None:
        L.append("No unambiguous fs from this data. For ring-downs try:  --mode-hz 55 70  (first flap mode)")
    else:
        lost = 1 - stats.delivered_rate_hz.median() / fs_best
        L.append(f"Implied fraction of samples lost (production, at fs = {fs_best:.0f} Hz): "
                 f"{100 * (1 - stats.loc[stats.kind == 'production', 'delivered_rate_hz'].median() / fs_best):.1f}%"
                 if (stats.kind == "production").any() else f"Implied fraction lost: {100 * lost:.1f}%")
    return "\n".join(L), fs_best, lag_best


# --------------------------------------------------
# Figure
# --------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e6e5e1"


def figure(stats, groups, example, fs_fig, lag_fig, out_png):
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
        a.text(fs_fig, a.get_ylim()[1] * 0.95, f" fs {fs_fig:.0f} Hz", color=INK, va="top", ha="right")
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
    cols = iter((BLUE, ORANGE, AQUA, YELLOW))
    for (kind, mode), G in groups.items():
        if not G.get("curves"):
            continue
        col = next(cols, MUTED)
        a.plot(FS_GRID, G["curves"][G["lag"]], color=col, lw=1.5,
               label=f"{kind}, {mode}, lag {G['lag']} ({G['n_pairs']} pairs)")
        dlv = stats.loc[stats.kind == kind, "delivered_rate_hz"].median()
        a.axvline(dlv, color=col, lw=1, ls=":")
    a.axvspan(FS_GRID[0], stats.delivered_rate_hz.min(), color=GRID, alpha=0.8, lw=0)
    a.text(FS_GRID[0] + 5, -0.9, "fs below the\ndelivered rate:\nimpossible", color=MUTED, fontsize=8)
    a.set(xlabel="Candidate sensor rate fs (Hz)", ylabel="Read-to-read phase coherence",
          title="(c) Which sample rate explains the data (dotted: delivered rate)", ylim=(-1.45, 1.05))
    a.legend(frameon=False, loc="lower right", fontsize=8)

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
# Ring-buffer analysis (--buffer)
# --------------------------------------------------

BUF_FS_GRID = np.arange(1460.0, 1530.01, 0.5)
BUF_K_GRID = np.arange(300.0, 1200.0, 2.0)
TIME_BINS = np.arange(0.0, 10.5, 0.5)


def buffer_analysis(files, mode_hz, capacity=1024, fs_nominal=at.DEFAULT_SENSOR_FS_HZ):
    """Per capture: boundary contiguity over time, split contiguity, and the
    overwrite-model fit. Uses the channel with the most in-band power."""
    band_c = (mode_hz[0] / FS_GRID[-1], mode_hz[1] / FS_GRID[0])
    bnd, split_sum, split_n, prof, pm_prof, fits = [], 0.0, 0, 0.0, 0.0, []
    W = 0
    for p in files:
        df = pd.read_csv(p, comment="#")
        bt = df["time"].to_numpy(float)
        reads = at.find_reads(bt)
        vals = df[ACCEL_COLS].to_numpy(float)
        Xs = [at._full_blocks(vals[:, j], reads, BLOCK)[0] for j in range(len(ACCEL_COLS))]
        if Xs[0].shape[0] < 20:
            continue
        ch, _, _ = at.dominant_block_mode(Xs, c_min=band_c[0], c_max=band_c[1])
        bp = at.block_phases(vals[:, ch], bt, band_c, BLOCK)
        b, t, sc = at.boundary_contiguity(bp, BLOCK)
        # latest possible overflow onset: buffer empty at t = 0 (START can only make it earlier)
        occ_min = fs_nominal * reads.t - BLOCK * np.arange(reads.n)
        on = np.flatnonzero(occ_min >= capacity)
        t_on_max = float(reads.t[on[0]]) if on.size else np.nan
        bnd.append(pd.DataFrame(dict(file=p.name, channel=ACCEL_COLS[ch], read=b, t=t, score=sc,
                                     after_latest_onset=t >= t_on_max)))
        if bp["keep"].sum() >= 5:
            ks, sp, nb = at.split_contiguity(bp)
            split_sum = split_sum + sp * nb
            split_n += nb
        fit = at.fit_overwrite_model(bp, BUF_FS_GRID, BUF_K_GRID, BLOCK)
        if fit["n_pairs"]:
            prof = prof + fit["profile"] * fit["n_pairs"]
            pm_prof = pm_prof + fit["postmortem_profile"] * fit["n_pairs"]
            W += fit["n_pairs"]
            fits.append(dict(file=p.name, channel=ACCEL_COLS[ch], c=bp["c"], n_pairs=fit["n_pairs"],
                             fs_hz=fit["fs"], K=fit["K"], coherence=fit["coherence"],
                             postmortem=fit["postmortem"], no_loss=fit["no_loss"], t_onset_latest=t_on_max))
    bnd = pd.concat(bnd, ignore_index=True) if bnd else pd.DataFrame()
    return dict(boundary=bnd, ks=np.arange(12, 73), split=(split_sum / split_n) if split_n else None,
                split_n=split_n, fits=pd.DataFrame(fits),
                profile=(prof / W) if W else None, pm_profile=(pm_prof / W) if W else None, n_pairs=W)


def buffer_summary(B, mode_hz):
    L = ["", f"Ring-buffer check (ring-downs, mode {mode_hz[0]:g}-{mode_hz[1]:g} Hz)", "-" * 60]
    bd = B["boundary"]
    if not len(bd):
        return "\n".join(L + ["no usable captures"]), None
    L.append(f"Read-boundary contiguity (1 = no samples lost between two reads; ~0 or below = lost), "
             f"{len(bd)} clean read pairs:")
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 99)):
        m = bd.t.between(lo, hi, inclusive="left")
        if m.sum():
            L.append(f"  t = {lo}-{hi if hi < 99 else 'end'} s : median {bd.score[m].median():+.2f}  "
                     f"(fraction > 0.9: {np.mean(bd.score[m] > 0.9):.2f}, n = {m.sum()})")
    late = bd.score[bd.after_latest_onset]
    early = bd.score[bd.t < 2.0]
    if B["split"] is not None:
        k = int(B["ks"][np.argmin(B["split"])])
        L.append(f"In-read contiguity: min {B['split'].min():.2f} at row {k} "
                 f"(median {np.median(B['split']):.2f}; ~1 everywhere = reads unbroken), {B['split_n']} reads")
    verdict = None
    if len(late) >= 20:
        lost_between = np.median(late) < 0.5
        split_gap = B["split"] is not None and B["split"].min() < 0.7
        if lost_between and not split_gap:
            verdict = "overwrite"
            L.append("=> After the buffer fills, samples are lost BETWEEN reads and each read is unbroken")
            L.append("   ('overwrite' policy). Gap-aware timing applies after the overflow onset only.")
        elif split_gap and not lost_between:
            verdict = "drop_newest"
            L.append(f"=> After the buffer fills, each read has a gap INSIDE it (near row {k}) and reads join")
            L.append("   up end to end ('drop newest' policy). The gap-aware timing in the pipeline does NOT fit.")
        elif not lost_between and not split_gap:
            verdict = "no_loss"
            L.append("=> No sign of lost samples even after the buffer must have filled: check fs / capacity.")
        else:
            L.append("=> Mixed signs (gaps both between and inside reads): inconclusive.")
    else:
        L.append(f"=> Too few clean read pairs after the latest possible overflow onset ({len(late)}).")
    if len(early) >= 10:
        L.append(f"Before 2 s (buffer cannot be full yet): median {np.median(early):+.2f} "
                 f"-> {'reads join up, no loss' if np.median(early) > 0.8 else 'UNEXPECTED: loss already'}")
    if B["profile"] is not None:
        j = int(np.argmax(B["profile"]))
        L.append(f"Overwrite-model fit over all captures: fs = {BUF_FS_GRID[j]:.1f} Hz, coherence "
                 f"{B['profile'][j]:.3f}; postmortem model (gaps from t = 0): {B['pm_profile'].max():.3f}")
        fits = B["fits"][B["fits"].n_pairs >= MIN_PAIRS_CAPTURE]
        if len(fits) >= 3:
            L.append(f"  per capture (>= {MIN_PAIRS_CAPTURE} pairs, n = {len(fits)}): fs median {fits.fs_hz.median():.1f} Hz, "
                     f"IQR {fits.fs_hz.quantile(.25):.1f}-{fits.fs_hz.quantile(.75):.1f}; "
                     f"K median {fits.K.median():.0f} (~ capacity - fs * START lead)")
    return "\n".join(L), verdict


def buffer_figure(B, mode_hz, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    bd = B["boundary"]
    a = ax[0]
    if len(bd):
        a.scatter(bd.t, bd.score, s=8, color=BLUE, alpha=0.35, lw=0)
        mid = 0.5 * (TIME_BINS[1:] + TIME_BINS[:-1])
        med = [bd.score[bd.t.between(lo, hi, inclusive="left")].median() for lo, hi in zip(TIME_BINS[:-1], TIME_BINS[1:])]
        a.plot(mid, med, color=INK, lw=1.5, label="median per 0.5 s")
        ton = B["fits"].t_onset_latest.median() if len(B["fits"]) else np.nan
        if np.isfinite(ton):
            a.axvline(ton, color=ORANGE, lw=1, ls="--")
            a.text(ton, -1.08, " buffer full by here\n (even if empty at t = 0)", color=ORANGE, fontsize=8, va="bottom")
        a.legend(frameon=False, loc="upper right")
    a.set(xlabel="Time in capture (s)", ylabel="Read-boundary contiguity", ylim=(-1.1, 1.1),
          title="(a) Do consecutive reads join up? (1 = no loss)")
    a = ax[1]
    if B["split"] is not None:
        a.plot(B["ks"], B["split"], color=BLUE, lw=1.5)
    a.set(xlabel="Row inside the read where it is split", ylabel="In-read contiguity", ylim=(-1.1, 1.1),
          title="(b) Is each read unbroken? (dip = gap inside)")
    a = ax[2]
    if B["profile"] is not None:
        a.plot(BUF_FS_GRID, B["profile"], color=BLUE, lw=1.5, label="buffer fills, then overwrites")
        a.plot(BUF_FS_GRID, B["pm_profile"], color=ORANGE, lw=1.5, label="postmortem: gaps from t = 0")
        a.legend(frameon=False, loc="lower right")
    a.set(xlabel="Sensor rate fs (Hz)", ylabel="Read-to-read phase coherence", ylim=(-1.1, 1.1),
          title=f"(c) Sample rate under each model ({B['n_pairs']} pairs)")
    fig.suptitle(f"Ring-buffer check, ring-downs, mode {mode_hz[0]:g}-{mode_hz[1]:g} Hz", color=INK)
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
                    help="captures per production run folder analysed for fs/lag (evenly spread; ring-downs: all)")
    ap.add_argument("--mode-hz", type=float, nargs=2, action="append", metavar=("LO", "HI"),
                    help="true frequency range of the mode to use (repeatable), e.g. --mode-hz 55 70 for the "
                         "first flap mode. Any fs in 1150-1700 Hz is allowed for, so no sample rate is assumed. "
                         "Default: the strongest line in each capture")
    ap.add_argument("--skip-production", action="store_true", help="read stats only for production captures")
    ap.add_argument("--stats-only", action="store_true", help="skip the fs/lag estimate (fast)")
    ap.add_argument("--buffer", action="store_true",
                    help="ring-buffer check on the ring-downs (needs --mode-hz; uses the first band): when do "
                         "losses start, between or inside reads, and fs under the fill-then-overwrite model")
    ap.add_argument("--capacity", type=int, default=1024, help="ring buffer size in sample-sets (AF10R1 UG)")
    args = ap.parse_args()
    modes = [tuple(m) for m in args.mode_hz] if args.mode_hz else [None]

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
            if args.skip_production:
                ps = [p for p in ps if kind_of(p) != "production"]
            if ps and kind_of(ps[0]) == "production" and len(ps) > args.max_est_per_run:
                ps = [ps[j] for j in np.linspace(0, len(ps) - 1, args.max_est_per_run).round().astype(int)]
            todo += ps
        print(f"estimating fs / lag on {len(todo)} captures x {len(modes)} mode(s)")
        for i, p in enumerate(todo):
            for m in modes:
                try:
                    r, e = estimate_row(p, m)
                except Exception as ex:
                    print(f"  skip {p}: {ex}")
                    continue
                if r is not None:
                    est_rows.append(r)
                if e is not None:
                    ests.setdefault((kind_of(p), r["mode"]), []).append((p, r["channel"], e))
            if (i + 1) % 20 == 0:
                print(f"  estimated {i + 1}/{len(todo)}")
    est_df = pd.DataFrame(est_rows)
    if len(est_df):
        est_df.to_csv(out / "timing_estimates.csv", index=False)

    groups = {}
    for key, lst in ests.items():
        es = [e for _, _, e in lst]
        fs, lag, coh, curves = at.combine_estimates(es)
        W = sum(e.weight for e in es)
        coh_del = sum(e.coherence_at_delivered * e.weight for e in es) / W
        fs_min = 0.98 * stats.loc[stats.kind == key[0], "delivered_rate_hz"].max()
        groups[key] = dict(fs=fs, lag=lag, coh=coh, coh_del=coh_del, curves=curves, n_captures=len(es),
                           n_pairs=int(W), lag_scores={k: float(v.max()) for k, v in curves.items()},
                           lobes=at.competing_lobes(FS_GRID, curves[lag], fs_min))
    if len(est_df):
        for key in set(zip(est_df.kind, est_df["mode"])) - set(groups):     # requested but nothing usable
            groups[key] = dict(n_captures=0, curves={})
    groups = dict(sorted(groups.items(), key=lambda kv: (kv[0][0] != "ringdown", kv[0][1])))

    text, fs_best, lag_best = summarize(stats, est_df if len(est_df) else None, groups, args.fs_nominal)
    if args.buffer:
        if not args.mode_hz:
            sys.exit("--buffer needs --mode-hz LO HI (e.g. 55 70)")
        ring = [p for p in files if kind_of(p) == "ringdown"]
        print(f"ring-buffer check on {len(ring)} ring-downs")
        B = buffer_analysis(ring, tuple(args.mode_hz[0]), args.capacity, args.fs_nominal)
        btxt, _ = buffer_summary(B, tuple(args.mode_hz[0]))
        text += "\n" + btxt
        if len(B["boundary"]):
            B["boundary"].to_csv(out / "buffer_boundaries.csv", index=False)
            B["fits"].to_csv(out / "buffer_fits.csv", index=False)
            buffer_figure(B, tuple(args.mode_hz[0]), out / "buffer_check.png")
    (out / "timing_summary.txt").write_text(text)
    print("\n" + text)

    fs_fig = fs_best or args.fs_nominal
    lag_fig = lag_best if lag_best is not None else at.BLOCK_LAG
    example = None
    for key, lst in ests.items():
        if key[0] == "ringdown" or example is None:
            p, ch, _ = max(lst, key=lambda t: t[2].weight)
            example = (p, ch)
    figure(stats, groups, example, fs_fig, lag_fig, out / "timing_check.png")
    print(f"\nwrote {out}/read_stats.csv, timing_estimates.csv, timing_summary.txt, timing_check.png")


if __name__ == "__main__":
    main()
