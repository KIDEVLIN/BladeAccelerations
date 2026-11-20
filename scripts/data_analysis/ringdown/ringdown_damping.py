#!/usr/bin/env python3
"""
ringdown_damping.py

Damping ratio of the first flap and first edge mode from the tip-flick
ring-downs, by the per-read envelope method.

The accelerometer delivers its samples in reads of 84 consecutive samples
with a gap of ~16 samples between reads (see accel_spectra.py), so methods
that assume a continuous signal (peak-to-peak log decrement, band-pass +
Hilbert envelope) are biased. Instead, within each read the signal is fitted
by least squares, at the samples' true times on the measured 1493 Hz grid,
with a constant plus a sine and cosine at each of a few known mode
frequencies. Fitting the neighbouring modes together keeps them out of the
amplitude of the mode of interest even when they are only ~30 Hz apart (a
read lasts 56 ms). The amplitude of the mode in each read, A(t), then decays
as exp(-zeta * omega * t); a straight-line fit of ln A against the read time
gives zeta = -slope / (2 pi f).

Reads used in the fit, per flick: after the read with the largest amplitude,
none with a clipped sample on the fitted channel, and only while A stays
above --snr times the noise floor (the same fit applied to the at-rest
capture). Flicks with fewer than --min-reads such reads are reported but
left out of the summary.

Usage:
    python3 ringdown_damping.py <run_dir> [<run_dir> ...] [--out analysis_damping]

Writes, in <run_dir>/<out>/:
    damping_trials.csv   one row per flick and mode: zeta, f, reads used, fit R^2,
                         zeta from the first and second half of the decay
    damping_summary.csv  median and spread of zeta per mode
    damping.png          example envelope fits and zeta per flick
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from accel_spectra import MEASURED_ODR_HZ                         # noqa: E402
import plot_ringdown_psd as P                                     # noqa: E402

SENSOR = "s4"
CLIP_RAW = 32700          # |raw int16| at or above this counts as clipped (full scale 32767)
# (excitation, channel, mode name): the mode is the strongest peak of that channel's median PSD in BAND
MODES = [("flap", f"{SENSOR}_z", "1st flap"), ("edge", f"{SENSOR}_y", "1st edge")]
BAND = {"flap": (40, 75), "edge": (75, 130)}


def read_trial(run_dir, md, tr):
    """(DataFrame: accel in g with 'time' on the 1493 Hz grid and 'read' index; raw ints) of one capture."""
    csv = run_dir / Path(tr["accel_csv"].replace("\\", "/")).name
    acc, _ = P.read_accel(csv, md["accel_mg_per_lsb"], MEASURED_ODR_HZ)
    raw = pd.read_csv(csv, comment="#")
    t = raw["time"].to_numpy()
    acc["read"] = np.cumsum(np.r_[True, np.diff(t) > 0]) - 1
    return acc, raw


def mode_freqs(run_dir):
    """Frequencies to fit in each read, from the run's median-PSD peaks (ringdown_top_peaks.csv, made by
    plot_ringdown_psd.py): the mode of interest first, then the other strong peaks of that channel."""
    top = pd.read_csv(run_dir / "analysis_psd" / "ringdown_top_peaks.csv")
    out = {}
    for label, ch, name in MODES:
        t = top[(top.label == label) & (top.channel == ch)].sort_values("rank")
        lo, hi = BAND[label]
        main = t[(t.peak_hz > lo) & (t.peak_hz < hi)].peak_hz.iloc[0]
        others = [f for f in t.peak_hz if abs(f - main) > 10 and f < 400][:2]
        out[label] = [float(main)] + [float(f) for f in others]
    return out


def read_amplitudes(acc, raw, ch, freqs):
    """Per read: (time of read centre, amplitude of freqs[0], clipped?). Least squares at the true times."""
    rows = []
    x_all, t_all = acc[ch].to_numpy(), acc["time"].to_numpy()
    clip = (raw[ch].abs() >= CLIP_RAW).to_numpy()
    for r, idx in acc.groupby("read").indices.items():
        if len(idx) < 40:
            continue
        t, x = t_all[idx], x_all[idx]
        cols = [np.ones_like(t)]
        for f in freqs:
            cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), x, rcond=None)
        rows.append((t.mean(), float(np.hypot(coef[1], coef[2])), bool(clip[idx].any())))
    return np.array(rows, dtype=float)


def fit_decay(env, f, onset, noise, snr, min_reads):
    """zeta from ln A vs t over the usable reads; also from the first and second half of them."""
    t, a, clipped = env[:, 0], env[:, 1], env[:, 2].astype(bool)
    after = t >= onset
    if not after.any():
        return None
    i0 = np.flatnonzero(after)[np.argmax(a[after])]
    use = []
    for i in range(i0, len(t)):
        if a[i] < snr * noise:
            break
        if not clipped[i]:
            use.append(i)
    use = np.array(use)
    res = dict(n_reads=len(use), noise=noise, peak_amp=a[i0])
    if len(use) < min_reads:
        return res
    tt, ll = t[use], np.log(a[use])
    p = np.polyfit(tt, ll, 1)
    r2 = 1 - np.sum((ll - np.polyval(p, tt)) ** 2) / np.sum((ll - ll.mean()) ** 2)
    half = len(use) // 2
    z1 = -np.polyfit(tt[:half], ll[:half], 1)[0] / (2 * np.pi * f) if half >= 3 else np.nan
    z2 = -np.polyfit(tt[half:], ll[half:], 1)[0] / (2 * np.pi * f) if len(use) - half >= 3 else np.nan
    res.update(zeta=-p[0] / (2 * np.pi * f), r2=r2, t_start=tt[0] - onset, t_end=tt[-1] - onset,
               zeta_early=z1, zeta_late=z2, fit=(tt, p), use=use)
    return res


def analyse(run_dir, args):
    md = json.loads((run_dir / "run_metadata.json").read_text())
    freqs = mode_freqs(run_dir)
    rest = next(tr for tr in md["trials"] if tr["is_rest"])
    racc, rraw = read_trial(run_dir, md, rest)
    rows, examples = [], {}
    for label, ch, name in MODES:
        fs = freqs[label]
        noise = float(np.median(read_amplitudes(racc, rraw, ch, fs)[:, 1]))
        for tr in md["trials"]:
            if tr["label"] != label:
                continue
            acc, raw = read_trial(run_dir, md, tr)
            env = read_amplitudes(acc, raw, ch, fs)
            res = fit_decay(env, fs[0], tr["qc"]["onset_s"], noise, args.snr, args.min_reads)
            if res is None:
                continue
            rows.append(dict(run=run_dir.name, mode=name, channel=ch, trial=tr["trial"], f_hz=fs[0],
                             fitted_with_hz=" ".join(f"{f:.1f}" for f in fs[1:]),
                             **{k: v for k, v in res.items() if k not in ("fit", "use")}))
            if "zeta" in res and name not in examples:
                examples[name] = (tr["trial"], env, res, tr["qc"]["onset_s"], noise)
    return pd.DataFrame(rows), examples


def figure(df, examples, out_png, args):
    names = [m for _, _, m in MODES]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(5 * (len(names) + 1), 3.8), constrained_layout=True)
    for ax, name in zip(axes, names):
        if name not in examples:
            ax.set_visible(False)
            continue
        trial, env, res, onset, noise = examples[name]
        t, a, clip = env[:, 0] - onset, env[:, 1], env[:, 2].astype(bool)
        ax.semilogy(t, a, ".", color="0.6", label="each read")
        ax.semilogy(t[res["use"]], a[res["use"]], "o", color="C0", ms=5, label="used in fit")
        ax.semilogy(t[clip], a[clip], "x", color="C3", label="clipped read")
        tt, p = res["fit"]
        ax.semilogy(tt - onset, np.exp(np.polyval(p, tt)), color="C1", lw=1.5,
                    label=rf"fit: $\zeta$ = {res['zeta'] * 100:.2f}%")
        ax.axhline(noise * args.snr, color="0.4", ls=":", lw=1, label=f"{args.snr:g} x noise floor")
        ax.set_xlim(-0.3, max(1.5, res["t_end"] + 0.3))
        ax.set_xlabel("Time after flick onset (s)")
        ax.set_ylabel("Mode amplitude in read (g)")
        ax.set_title(f"{name}, flick {trial}", fontsize=10)
        ax.legend(fontsize=7)
    ax = axes[-1]
    ok = df.dropna(subset=["zeta"]) if "zeta" in df else df.iloc[:0]
    for i, name in enumerate(names):
        z = ok[ok["mode"] == name]["zeta"] * 100
        ax.scatter(np.full(len(z), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(z)), z, s=18,
                   color=f"C{i}", alpha=0.8)
        if len(z):
            ax.hlines(z.median(), i - 0.25, i + 0.25, color="k", lw=2)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(r"Damping ratio $\zeta$ (%)")
    ax.set_title("Each flick (bar: median)", fontsize=10)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="analysis_damping")
    ap.add_argument("--snr", type=float, default=5.0, help="use reads while amplitude > SNR x noise (default 5)")
    ap.add_argument("--min-reads", type=int, default=5, help="minimum reads for a fit (default 5)")
    args = ap.parse_args()
    for rd in map(Path, args.run_dirs):
        df, examples = analyse(rd, args)
        out = rd / args.out
        out.mkdir(exist_ok=True)
        df.to_csv(out / "damping_trials.csv", index=False)
        ok = df.dropna(subset=["zeta"])
        summ = ok.groupby("mode").agg(f_hz=("f_hz", "first"), flicks=("zeta", "size"),
                                      flicks_total=("trial", "size"),
                                      zeta_median=("zeta", "median"),
                                      zeta_q25=("zeta", lambda z: z.quantile(.25)),
                                      zeta_q75=("zeta", lambda z: z.quantile(.75)),
                                      reads_median=("n_reads", "median"), r2_median=("r2", "median"),
                                      zeta_early_median=("zeta_early", "median"),
                                      zeta_late_median=("zeta_late", "median"))
        summ["flicks_total"] = df.groupby("mode").size()
        summ.to_csv(out / "damping_summary.csv")
        print(f"\n== {rd.name}")
        print(summ.assign(**{c: summ[c] * 100 for c in summ if c.startswith("zeta")}).round(3).to_string())
        figure(df, examples, out / "damping.png", args)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
