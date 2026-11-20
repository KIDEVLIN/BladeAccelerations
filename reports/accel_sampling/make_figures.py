#!/usr/bin/env python3
"""
make_figures.py

Figures for accel_sampling.tex. Run from the repo root:
    python3 reports/accel_sampling/make_figures.py

Reads data/DTU10MW_rigid4k_ringdown (accelerometer ringdowns) and the
beam_tracker result for the flapwise high-speed video, and writes PDFs
into reports/accel_sampling/figs/. Prints the numbers quoted in the text.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis" / "ringdown"))
import plot_ringdown_psd as P  # noqa: E402

RUN = ROOT / "data" / "DTU10MW_rigid4k_ringdown"
VIDEO = Path.home() / "packages/beam_tracker/results/flapwise_4/tip_timeseries.csv"
OUT = Path(__file__).resolve().parent / "figs"
ODR = 1600.0
BULK = 84
NPERSEG = 2048
CHANNEL = "s4_z"
FGRID = np.arange(0, 800, 0.25)


def batches(df):
    t = df["time"].to_numpy()
    first = np.flatnonzero(np.r_[True, np.diff(t) > 0])
    n = np.diff(np.r_[first, len(t)])
    return t, first, n


def video_spectrum():
    d = pd.read_csv(VIDEO)
    t, y = d["time_s"].to_numpy(), d["tip_deflection_px"].to_numpy()
    fs = 1 / np.median(np.diff(t))
    nfft = 1 << 18
    Y = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    k = np.argmax(Y * (f > 5))
    return t, y, fs, f, Y / Y[k], f[k]


def main():
    OUT.mkdir(exist_ok=True)
    md = json.loads((RUN / "run_metadata.json").read_text())
    mg = md["accel_mg_per_lsb"]

    sizes, dts, rates = [], [], []
    psd = {"delivered": [], "concat": [], "grid": []}
    example = None
    for tr in md["trials"]:
        csv = RUN / Path(tr["accel_csv"].replace("\\", "/")).name
        df = pd.read_csv(csv, comment="#")
        t, first, n = batches(df)
        sizes += n[:-1].tolist()
        dts += np.diff(t[first]).tolist()
        acc, rate = P.read_accel(csv, mg, ODR)
        rates.append(rate)
        if tr["label"] != "flap":
            continue
        if example is None:
            example = (csv.stem, t[first], first)
        on = tr["qc"]["onset_s"]
        x = acc[CHANNEL].to_numpy()
        m = np.repeat(t[first], n) >= on - P.ONSET_PAD_S
        # (1) evenly spaced at the delivered rate (extract_run_summary.py)
        f, p = P.masked_welch(x[m], rate, NPERSEG)
        psd["delivered"].append(np.interp(FGRID, f, p))
        # (2) evenly spaced at the ODR, batches joined end to end (explore.py)
        f, p = P.masked_welch(x[m], ODR, NPERSEG)
        psd["concat"].append(np.interp(FGRID, f, p))
        # (3) ODR grid with the dropped samples as masked gaps (plot_ringdown_psd.py)
        f, p = P.trial_psd(acc[[CHANNEL, "grid", "time"]], ODR, on, NPERSEG)
        psd["grid"].append(np.interp(FGRID, f, p[CHANNEL]))

    dts = np.array(dts)
    f_batch = 1 / dts.mean()
    print(f"files: {len(md['trials'])}, batch sizes: {sorted(set(sizes))}")
    print(f"batch interval: mean {dts.mean() * 1e3:.2f} ms, median {np.median(dts) * 1e3:.2f}, "
          f"std {dts.std() * 1e3:.2f}; batch rate {f_batch:.2f} Hz")
    print(f"time to fill one batch at ODR: {BULK / ODR * 1e3:.1f} ms")
    print(f"delivered rate: mean {np.mean(rates):.1f}, range {np.min(rates):.1f}-{np.max(rates):.1f} Hz; "
          f"dropped per batch {dts.mean() * ODR - BULK:.1f} ({1 - np.mean(rates) / ODR:.1%})")

    tv, yv, fsv, fv, Yv, f_video = video_spectrum()
    print(f"video: {fsv:.0f} fps, {len(tv)} frames, first flap mode {f_video:.2f} Hz")

    med = {k: np.median(v, axis=0) for k, v in psd.items()}
    band = (FGRID > 20) & (FGRID < 120)
    pk = {k: FGRID[np.argmax(v * band)] for k, v in med.items()}
    for k, v in pk.items():
        print(f"{k:9s} {CHANNEL} flap peak 20-120 Hz: {v:.2f} Hz")
    print(f"predicted alias, delivered: {f_video - f_batch:.2f} Hz; "
          f"concat: {f_video * dts.mean() * ODR / BULK - ODR / BULK:.2f} Hz")

    with plt.rc_context(P._latex_rc()):
        # Fig 1: batch structure
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.9))
        a1.hist(dts * 1e3, bins=np.arange(50, 90, 0.5), color="C0")
        a1.axvline(BULK / ODR * 1e3, color="C3", ls="--",
                   label=rf"84 samples at 1600 Hz ({BULK / ODR * 1e3:.1f} ms)")
        a1.axvline(dts.mean() * 1e3, color="k", ls=":",
                   label=rf"mean read cycle ({dts.mean() * 1e3:.1f} ms)")
        a1.set_xlabel("Time between batch reads (ms)")
        a1.set_ylabel("Count (all 61 files)")
        a1.legend(fontsize=9, loc="center right")
        name, tb, fi = example
        a2.plot(tb, fi, ".", ms=3, color="C0", label="samples received")
        a2.plot(tb, (tb - tb[0]) * ODR + fi[0], color="C3", ls="--",
                label="samples produced at 1600 Hz")
        a2.set_xlabel("Time (s)")
        a2.set_ylabel("Cumulative samples")
        a2.set_title(name.replace("_", r"\_"), fontsize=10)
        a2.legend(fontsize=9)
        for a in (a1, a2):
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "batches.pdf")
        plt.close(fig)

        # Fig 2: video reference
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.7))
        a1.plot(tv, yv, lw=0.7)
        a1.set_xlabel("Time (s)")
        a1.set_ylabel("Tip deflection (px)")
        a2.semilogy(fv, Yv, lw=0.8)
        a2.axvline(f_video, color="C3", ls="--", lw=0.8)
        a2.text(f_video + 8, 0.5, rf"{f_video:.1f} Hz", color="C3")
        a2.set_xlim(0, 400)
        a2.set_ylim(1e-4, 2)
        a2.set_xlabel("Frequency (Hz)")
        a2.set_ylabel("Normalised amplitude")
        for a in (a1, a2):
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "video.pdf")
        plt.close(fig)

        # Fig 3: the three timing methods
        labels = {"delivered": r"(1) even spacing at delivered rate ($\approx$1250 Hz)",
                  "concat": r"(2) even spacing at 1600 Hz, no gaps",
                  "grid": r"(3) 1600 Hz within batch, gaps masked"}
        fig, axs = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
        for ax, (k, lab) in zip(axs, labels.items()):
            ax.semilogy(FGRID, med[k], color="C0", lw=0.9)
            ax.axvline(f_video, color="C3", ls="--", lw=0.9, label=rf"video {f_video:.1f} Hz")
            ax.plot(pk[k], med[k][np.argmax(med[k] * band)], "v", color="k", ms=6)
            ax.text(pk[k] + 3, med[k][np.argmax(med[k] * band)], rf"{pk[k]:.1f} Hz", fontsize=10)
            ax.set_title(lab, fontsize=11)
            ax.set_ylabel(P.PSD_LABEL)
            ax.set_xlim(0, 250)
            ax.grid(alpha=0.3, which="both")
        for j in (-1, 1, 2):
            axs[2].axvline(f_video + j * f_batch, color="0.5", ls=":", lw=0.8)
        axs[2].plot([], [], color="0.5", ls=":", label=rf"video $\pm\, n \times {f_batch:.1f}$ Hz")
        axs[0].legend(fontsize=9, loc="upper right")
        axs[2].legend(fontsize=9, loc="upper right")
        axs[2].set_xlabel("Frequency (Hz)")
        fig.tight_layout()
        fig.savefig(OUT / "methods.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
