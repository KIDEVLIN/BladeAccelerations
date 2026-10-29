#!/usr/bin/env python3
"""
plot_ringdown_psd.py

Power spectral density of the accelerometer signals from a ring-down run
(experiment == "ring_down" in run_metadata.json: rest_001.csv plus
ring_flap_NNN.csv / ring_edge_NNN.csv tip-flick trials).

Usage:
    python3 plot_ringdown_psd.py <run_dir> [--out analysis_psd]
                                 [--nperseg 2048] [--fmax 800]

Produces, in <run_dir>/<out>/:
    psd_flap.png, psd_edge.png -- one panel per sensor axis (edgewise y,
                                  flapwise z; x is not plotted),
                                  one line per PLOT_SENSORS (s4): median PSD
                                  across all trials of that excitation,
                                  with the rest PSD as the noise floor.
    psd_trials/<trial>.png     -- same layout for each individual trial.
    ringdown_peaks.csv         -- dominant PSD peak frequency per trial,
                                  sensor and axis.
    ringdown_top_peaks.csv     -- top --npeaks peaks of each y/z median
                                  PSD per excitation, batch-gap copies of
                                  stronger peaks dropped (also marked with
                                  triangles on psd_flap/psd_edge.png).

Each trial's PSD is computed from the excitation onset (qc.onset_s in the
metadata) to the end of the capture, so the pre-flick quiet section does
not dilute the ring-down. Welch's method (Hann window, 50% overlap),
implemented in numpy, one-sided, in g^2/Hz.

Sample timing: the accel CSV timestamps are stamped per FIFO batch (many
samples share one time value), and the delivered rate (~1250 Hz) is below
the configured ODR (1600 Hz). Every batch is BULK_MAX_SETS (84) samples,
i.e. the reader never catches up and the MCU buffer overruns, dropping
~23 samples between batches (not logged). So samples are placed on the
1/ODR grid: contiguous within a batch, with the batch's last sample at
its read timestamp, and the dropped samples left as gaps that the Welch
estimate masks out. Checked against the high-speed video of a flapwise
ringdown (beam_tracker, 4000 fps): first flap mode 62.7 Hz on video,
~62-63 Hz here; assuming evenly spaced samples at the delivered rate gave
47 Hz.

Two artifacts remain. The gaps repeat every read cycle (~67 ms), so each
strong peak has weaker copies ~15 Hz either side (e.g. a 62 Hz mode shows
up again near 47 and 77 Hz) -- check a peak isn't 15 Hz from a stronger
one before calling it a mode. And the batch anchor is only as good as the
read timestamp (~1 ms), which smears peaks more at high frequency.

The load-cell captures are not used: every trial in the first ring-down
run is flagged as clipped at the DAQ rail.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accel_spectra import (MEASURED_ODR_HZ, batch_grid, batch_rate_hz,  # noqa: E402
                           drop_sidebands, gridded, masked_welch, top_peaks)

SENSORS = ["s1", "s2", "s3", "s4"]
AXES = ["x", "y", "z"]
# Leave a little pre-onset signal so the initial transient is captured.
ONSET_PAD_S = 0.05
# Axes whose median PSDs get top-peak picking (see accel_spectra.top_peaks).
PEAK_AXES = ["y", "z"]
# Axes drawn in the PSD figures, with their blade direction (from the
# ringdowns: edge flicks peak in y, flap flicks in z; x is spanwise and
# barely moves).
PLOT_AXES = {"y": "Edgewise", "z": "Flapwise"}
# Sensors drawn in the PSD figures (s4, nearest the tip, moves the most);
# peaks and CSVs still cover all of SENSORS.
PLOT_SENSORS = ["s4"]


def _latex_rc():
    """LaTeX-typeset text: real LaTeX (text.usetex) when a `latex` binary
    is on PATH, otherwise matplotlib's built-in Computer Modern mathtext,
    which looks the same without needing a TeX install."""
    if shutil.which("latex"):
        return {"text.usetex": True, "font.family": "serif", "font.size": 12}
    return {"mathtext.fontset": "cm", "font.family": "serif",
            "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True,
            "font.size": 12}


PSD_LABEL = r"PSD $(g^2/\mathrm{Hz})$"


def read_accel(path, mg_per_lsb, odr):
    """Returns (DataFrame of accel in g, delivered rate in Hz). Each row
    gets an integer `grid` index on the 1/odr sample grid (see
    accel_spectra.batch_grid) and `time` = grid / odr. The read-cycle
    rate (spacing of the gap copies) is in accel.attrs["batch_hz"]."""
    df = pd.read_csv(path, comment="#")
    t = df["time"].to_numpy()
    first = np.flatnonzero(np.r_[True, np.diff(t) > 0])
    rate = np.polyfit(t[first], first, 1)[0]
    grid = batch_grid(t, odr)
    cols = [f"{s}_{a}" for s in SENSORS for a in AXES]
    accel = df[cols].astype(float) * mg_per_lsb * 1e-3
    accel["grid"] = grid
    accel["time"] = grid / odr
    accel.attrs["batch_hz"] = batch_rate_hz(t)
    return accel, rate


def trial_psd(accel, odr, onset_s, nperseg):
    """Returns (freqs, {column: psd}) from onset to end of capture, with
    the dropped samples between batches left as masked gaps."""
    if onset_s is not None:
        accel = accel[accel["time"] >= onset_s - ONSET_PAD_S]
    grid = accel["grid"].to_numpy()
    psds = {}
    for col in accel.columns.drop(["grid", "time"]):
        x, mask = gridded(accel[col].to_numpy(), grid)
        f, psds[col] = masked_welch(x, odr, nperseg, mask)
    return f, psds


def plot_psd_grid(f, psds, out_path, rest=None, fmax=None, spread=None,
                  peaks=None):
    """One panel per axis, one line per sensor. `rest` is a (f, psds)
    noise floor; `spread` is {col: (lo, hi)} drawn as a shaded band;
    `peaks` is {col: [index, ...]} marked on the line."""
    with plt.rc_context(_latex_rc()):
        fig, axes = plt.subplots(1, len(PLOT_AXES), figsize=(5 * len(PLOT_AXES), 4),
                                 sharey=True)
        for ax, a in zip(axes, PLOT_AXES):
            for s in PLOT_SENSORS:
                col = f"{s}_{a}"
                color = f"C{SENSORS.index(s)}"
                if spread is not None:
                    ax.fill_between(f, *spread[col], color=color, alpha=0.15, lw=0)
                ax.semilogy(f, psds[col], color=color, lw=1, label=s)
                if peaks is not None and col in peaks:
                    k = peaks[col]
                    ax.plot(f[k], psds[col][k], "v", color=color, ms=6)
                    for kk in k:
                        ax.annotate(f"{f[kk]:.0f} Hz", (f[kk], psds[col][kk]),
                                    xytext=(5, 4), textcoords="offset points",
                                    fontsize=9, color=color)
            if rest is not None:
                ax.semilogy(rest[0], np.mean([rest[1][f"{s}_{a}"] for s in PLOT_SENSORS], axis=0),
                            color="0.5", ls="--", lw=1, label="rest")
            ax.set_title(f"{PLOT_AXES[a]} (${a}$)")
            ax.set_xlabel(r"Frequency (Hz)")
            ax.set_xlim(0, fmax or f[-1])
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel(PSD_LABEL)
        axes[-1].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default="analysis_psd",
                        help="Output folder, relative to run_dir (default: analysis_psd)")
    parser.add_argument("--nperseg", type=int, default=2048,
                        help="Welch segment length in samples (default: 2048)")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Upper frequency limit of the plots (default: Nyquist)")
    parser.add_argument("--npeaks", type=int, default=4,
                        help="Peaks to pick per y/z channel from the median PSD (default: 4)")
    parser.add_argument("--odr", type=float, default=MEASURED_ODR_HZ,
                        help="Actual accel sample rate for the sample grid "
                             f"(default: measured {MEASURED_ODR_HZ:g} Hz, not the "
                             "configured ODR in the metadata)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.out
    (out_dir / "psd_trials").mkdir(parents=True, exist_ok=True)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    mg_per_lsb = metadata["accel_mg_per_lsb"]
    odr = args.odr

    rest = None
    by_label = {}
    batch_hz = {}
    peaks = []
    top = []
    for trial in metadata["trials"]:
        # Metadata paths are Windows paths from the acquisition PC.
        csv = run_dir / Path(trial["accel_csv"].replace("\\", "/")).name
        accel, rate = read_accel(csv, mg_per_lsb, odr)
        onset = None if trial["is_rest"] else trial["qc"].get("onset_s")
        f, psds = trial_psd(accel, odr, onset, args.nperseg)
        name = csv.stem
        print(f"{name}: delivered {rate:.1f} Hz of {odr:.0f} Hz ODR, onset {onset}")

        if trial["is_rest"]:
            rest = (f, psds)
            continue
        by_label.setdefault(trial["label"], []).append((name, f, psds))
        batch_hz.setdefault(trial["label"], []).append(accel.attrs["batch_hz"])
        for col, p in psds.items():
            k = np.argmax(p[1:]) + 1  # skip DC
            peaks.append({"trial": name, "label": trial["label"],
                          "trial_num": trial["trial"], "rate_hz": rate,
                          "channel": col, "peak_hz": f[k], "peak_psd_g2_hz": p[k]})

    # Per-trial figures need the rest floor, so draw them after the loop.
    for label, trials in by_label.items():
        for name, f, psds in trials:
            plot_psd_grid(f, psds, out_dir / "psd_trials" / f"{name}.png",
                          rest=rest, fmax=args.fmax)

        # All trials share the ODR grid; interp only guards a short trial
        # whose nperseg was clipped.
        f0 = trials[0][1]
        cols = trials[0][2].keys()
        stack = {c: np.stack([np.interp(f0, f, p[c]) for _, f, p in trials]) for c in cols}
        median = {c: np.median(v, axis=0) for c, v in stack.items()}
        spread = {c: np.percentile(v, [25, 75], axis=0) for c, v in stack.items()}
        # Drop the batch-gap copies (+/- n * read-cycle rate) of stronger peaks.
        f_batch = np.median(batch_hz[label])
        picked = {c: drop_sidebands(f0, median[c], top_peaks(f0, median[c]), f_batch)[:args.npeaks]
                  for c in (f"{s}_{a}" for s in SENSORS for a in PEAK_AXES)}
        for col, idx in picked.items():
            for rank, k in enumerate(idx, 1):
                top.append({"label": label, "channel": col, "rank": rank,
                            "peak_hz": f0[k], "peak_psd_g2_hz": median[col][k]})
        plot_psd_grid(f0, median, out_dir / f"psd_{label}.png",
                      rest=rest, fmax=args.fmax, spread=spread, peaks=picked)
        print(f"psd_{label}.png: median of {len(trials)} trials (band: 25-75th pct)")

    peaks = pd.DataFrame(peaks)
    peaks.to_csv(out_dir / "ringdown_peaks.csv", index=False)
    print("\nDominant peak frequency (Hz), median across trials:")
    print(peaks.pivot_table(index="channel", columns="label", values="peak_hz",
                            aggfunc="median").round(1).to_string())

    top = pd.DataFrame(top)
    top.to_csv(out_dir / "ringdown_top_peaks.csv", index=False)
    print(f"\nTop {args.npeaks} peaks (Hz) of the median PSD, by PSD height:")
    print(top.pivot_table(index=["label", "channel"], columns="rank",
                          values="peak_hz").round(1).to_string())
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
