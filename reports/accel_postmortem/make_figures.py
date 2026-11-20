#!/usr/bin/env python3
"""
make_figures.py

Timing evidence for accel_postmortem.tex. Run from the repo root:
    python3 reports/accel_postmortem/make_figures.py

Reads every accelerometer capture of the production campaign
(data/Production_10-8-26/<run>/angle_*deg.csv) and of the two ring-down runs.
In these files the `time` column is the moment the host finished each bulk
read, shared by every sample of that read, so it gives the size of every read
and the time between reads. Writes figs/read_cycle.pdf and prints every
number quoted in the report.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis" / "ringdown"))
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))
import plot_ringdown_psd as P                                     # noqa: E402
from accel_spectra import masked_welch                            # noqa: E402

BOLTED = ROOT / "data" / "DTU10MW_rigid4k_ringdown"      # bolted run; its first flap mode matches the video
VIDEO = Path.home() / "packages/beam_tracker/results/flapwise_4/tip_timeseries.csv"
CHANNEL = "s4_z"                                          # tip sensor, flapwise axis
NPERSEG = 2048
PROD = ROOT / "data" / "Production_10-8-26"
RINGDOWNS = [ROOT / "data" / "DTU10MW_rigid4k_ringdown", ROOT / "data" / "DTU10MW_rigid4k_ringdown_NOBOLT"]
OUT = Path(__file__).resolve().parent / "figs"

ODR_HZ = 1493.0            # sensor rate measured by the vendor (AF10R1 user guide, 5-min test)
MAX_SETS = 84              # firmware limit per bulk read
BYTES_PER_SET = 24         # 4 sensors x 3 axes x int16
BAUD = 921600              # 8N1: 10 bits per byte
RESPONSE_OVERHEAD = 5      # Modbus address, function, byte count, CRC


def batches(csv):
    """(sizes, gaps between reads in s, delivered rate in Hz) of one capture."""
    t = pd.read_csv(csv, comment="#", usecols=["time"])["time"].to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(t) > 0])
    sizes = np.diff(np.r_[starts, len(t)])
    bt = t[starts]
    # first and last read can be partial (start-up, end of capture): drop them
    return sizes[1:-1], np.diff(bt)[1:], (len(t) - sizes[0]) / (bt[-1] - bt[0])


def collect(files):
    sizes, gaps, rates = [], [], []
    for f in files:
        s, g, r = batches(f)
        sizes.append(s), gaps.append(g), rates.append(r)
    return np.concatenate(sizes), np.concatenate(gaps) * 1e3, np.array(rates)


def report(name, sizes, gaps, rates, nfiles):
    print(f"\n== {name}: {nfiles} captures, {len(sizes)} reads")
    print(f"reads returning exactly {MAX_SETS} sets: {np.mean(sizes == MAX_SETS) * 100:.3f}% "
          f"(sizes seen: {sorted(set(sizes.tolist()))})")
    print(f"time between reads (ms): median {np.median(gaps):.1f}, 5-95% {np.percentile(gaps, 5):.1f}-"
          f"{np.percentile(gaps, 95):.1f}, 99% {np.percentile(gaps, 99):.1f}")
    print(f"delivered rate (sets/s): median {np.median(rates):.0f}, range {rates.min():.0f}-{rates.max():.0f}; "
          f"kept {np.median(rates) / ODR_HZ * 100:.1f}% of {ODR_HZ:g}, lost {100 - np.median(rates) / ODR_HZ * 100:.1f}%")
    produced = np.median(gaps) / 1e3 * ODR_HZ
    print(f"sets produced per read cycle: {produced:.1f}; kept {MAX_SETS}; lost {produced - MAX_SETS:.1f}")


def video_peak():
    """(f, normalized amplitude spectrum, first flap mode) of the tracked tip deflection."""
    d = pd.read_csv(VIDEO)
    t, y = d["time_s"].to_numpy(), d["tip_deflection_px"].to_numpy()
    fs = 1 / np.median(np.diff(t))
    nfft = 1 << 18
    Y = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    k = np.argmax(Y * (f > 5))
    print(f"video: {fs:.0f} frames/s, {len(t)} frames, first flap mode {f[k]:.2f} Hz")
    return f, Y / Y[k], f[k]


def ringdown_spectra():
    """Median PSD of the flap flicks (tip sensor, flapwise) two ways: samples taken as evenly spaced at the
    delivered rate (the old method), and gap-aware on the 1493 Hz grid (the corrected method)."""
    md = json.loads((BOLTED / "run_metadata.json").read_text())
    fgrid = np.arange(0, 300, 0.25)
    even, gap, rates, cycles = [], [], [], []
    for tr in md["trials"]:
        if tr["label"] != "flap":
            continue
        csv = BOLTED / Path(tr["accel_csv"].replace("\\", "/")).name
        acc, rate = P.read_accel(csv, md["accel_mg_per_lsb"], ODR_HZ)
        rates.append(rate)
        cycles.append(1 / acc.attrs["batch_hz"])
        on = tr["qc"]["onset_s"]
        t_read = pd.read_csv(csv, comment="#", usecols=["time"])["time"].to_numpy()
        x = acc[CHANNEL].to_numpy()[t_read >= on - P.ONSET_PAD_S]
        f, p = masked_welch(x, rate, NPERSEG)                                   # old: evenly spaced
        even.append(np.interp(fgrid, f, p))
        f, p = P.trial_psd(acc[[CHANNEL, "grid", "time"]], ODR_HZ, on, NPERSEG)  # corrected: gap-aware
        gap.append(np.interp(fgrid, f, p[CHANNEL]))
    even, gap = np.median(even, axis=0), np.median(gap, axis=0)
    band = (fgrid > 30) & (fgrid < 100)
    f_even, f_gap = fgrid[band][np.argmax(even[band])], fgrid[band][np.argmax(gap[band])]
    f_cycle = 1 / np.median(cycles)
    print(f"\n== bolted ring-down, {len(rates)} flap flicks, {CHANNEL}")
    print(f"first flap mode: evenly spaced at the delivered rate {f_even:.2f} Hz; gap-aware at {ODR_HZ:g} Hz "
          f"{f_gap:.2f} Hz; read rate {f_cycle:.2f} Hz")

    def near(fc, w=1.5):
        m = np.abs(fgrid - fc) <= w
        return gap[m].max()
    for k in (-1, 1, 2):
        print(f"  copy at f1 {'+' if k > 0 else '-'} {abs(k)} x {f_cycle:.1f} = {f_gap + k * f_cycle:.1f} Hz: "
              f"{near(f_gap + k * f_cycle) / near(f_gap) * 100:.1f}% of the mode's PSD")
    return fgrid, even, gap, f_even, f_gap, f_cycle


def phase_test(f_mode):
    """Phase change of the first flap mode from one read to the next, against the change expected if the
    84 samples of consecutive reads were continuous, or separated by the measured read interval."""
    md = json.loads((BOLTED / "run_metadata.json").read_text())
    steps, intervals = [], []
    for tr in md["trials"]:
        if tr["label"] != "flap":
            continue
        df = pd.read_csv(BOLTED / Path(tr["accel_csv"].replace("\\", "/")).name, comment="#")
        t, x = df["time"].to_numpy(), df[CHANNEL].to_numpy(float)
        starts = np.flatnonzero(np.r_[True, np.diff(t) > 0])
        n = np.diff(np.r_[starts, len(t)])
        z, tb = [], []
        for i0, k in zip(starts, n):
            if k != MAX_SETS:
                continue
            seg = x[i0:i0 + k] - x[i0:i0 + k].mean()
            z.append(np.sum(seg * np.exp(-2j * np.pi * f_mode * np.arange(k) / ODR_HZ)))
            tb.append(t[i0 + k - 1] - t[0])
        z, tb = np.array(z), np.array(tb)
        good = np.abs(z) > 0.2 * np.abs(z).max()                  # only reads where the mode is ringing
        d = np.angle(z[1:] * np.conj(z[:-1]))
        ok = good[1:] & good[:-1]
        steps += list(zip(tb[1:][ok], d[ok]))
        intervals += list(np.diff(tb))
    steps = np.array(steps)
    wrap = lambda a: float(np.angle(np.exp(1j * a)))
    dt = np.median(intervals)
    pred_cont = wrap(2 * np.pi * f_mode * MAX_SETS / ODR_HZ)
    pred_gap = wrap(2 * np.pi * f_mode * dt)
    print(f"\n== phase change per read at {f_mode:.1f} Hz: continuous {pred_cont:+.2f} rad, "
          f"{dt * 1e3:.1f} ms apart {pred_gap:+.2f} rad")
    rows = []
    for lo, hi in ((0, 4), (4, 6), (6, 10)):
        m = (steps[:, 0] >= lo) & (steps[:, 0] < hi)
        if m.sum():
            c = np.mean(np.exp(1j * steps[m, 1]))
            rows.append((lo, hi, int(m.sum()), np.angle(c)))
            print(f"  {lo}-{hi} s into the capture: {m.sum()} read pairs, measured {np.angle(c):+.2f} rad "
                  f"(coherence {abs(c):.2f})")
    return pred_cont, pred_gap, rows


def fig_evidence(fv, Yv, f_video, fgrid, even, gap, f_even, f_gap, f_cycle):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]
    ax.plot(fv, Yv, color="k", lw=1)
    ax.axvline(f_video, color="C3", ls="--", lw=1)
    ax.text(f_video + 4, 0.9, f"{f_video:.1f} Hz", color="C3", fontsize=9)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title("High-speed video, tip deflection", fontsize=10)
    ax = axes[1]
    ax.semilogy(fgrid, even, color="0.55", lw=1.2, label=f"samples as evenly spaced: {f_even:.1f} Hz")
    ax.semilogy(fgrid, gap, color="C0", lw=1.2, label=f"gaps kept, 1493 Hz: {f_gap:.1f} Hz")
    ax.axvline(f_video, color="C3", ls="--", lw=1, label=f"video: {f_video:.1f} Hz")
    for k in (-1, 1, 2):
        ax.axvline(f_gap + k * f_cycle, color="C0", ls=":", lw=0.9)
    ax.plot([], [], color="C0", ls=":", label=rf"copies at $\pm${f_cycle:.1f} Hz (read rate)")
    ax.set_xlim(0, 150)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD (g$^2$/Hz)")
    ax.set_title("Accelerometer, 30 flap-flick ring-downs (tip sensor, flapwise)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "evidence.pdf")


def main():
    OUT.mkdir(exist_ok=True)
    fv, Yv, f_video = video_peak()
    fgrid, even, gap, f_even, f_gap, f_cycle = ringdown_spectra()
    fig_evidence(fv, Yv, f_video, fgrid, even, gap, f_even, f_gap, f_cycle)
    phase_test(f_video)
    print()
    fill_ms = MAX_SETS / ODR_HZ * 1e3
    wire_ms = (MAX_SETS * BYTES_PER_SET + RESPONSE_OVERHEAD) * 10 / BAUD * 1e3
    bus = ODR_HZ * BYTES_PER_SET * 10 / BAUD
    print(f"time for the sensors to produce {MAX_SETS} sets: {fill_ms:.1f} ms")
    print(f"time to send one {MAX_SETS}-set response at {BAUD} baud: {wire_ms:.1f} ms")
    print(f"serial link load at the full rate: {bus * 100:.0f}% ({ODR_HZ * BYTES_PER_SET / 1e3:.1f} kB/s of "
          f"{BAUD / 10 / 1e3:.1f} kB/s)")

    pfiles = sorted(PROD.glob("*/angle_*deg.csv"))
    ps, pg, pr = collect(pfiles)
    report(f"production ({len({f.parent for f in pfiles})} runs)", ps, pg, pr, len(pfiles))
    print(f"host time per read cycle beyond the transfer: {np.median(pg) - wire_ms:.0f} ms")

    rfiles = sorted(f for d in RINGDOWNS for f in d.glob("ring_*_[0-9][0-9][0-9].csv"))
    rs, rg, rr = collect(rfiles)
    report("ring-down (2 runs)", rs, rg, rr, len(rfiles))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.hist(pg, bins=np.arange(0, 100, 1), color="C0", label="production reads")
    for x, txt, col in ((wire_ms, "transfer time\nof one read", "C2"),
                        (fill_ms, f"sensors produce\n{MAX_SETS} sets", "C3")):
        ax.axvline(x, color=col, ls="--", lw=1.2)
        ax.text(x - 1, ax.get_ylim()[1] * 0.95, txt, color=col, ha="right", va="top", fontsize=8)
    ax.set_xlabel("Time between reads (ms)")
    ax.set_ylabel("Number of reads")
    ax.set_title(f"{len(pg):,} reads, every one returning {MAX_SETS} sets", fontsize=10)
    ax = axes[1]
    ax.hist(pr, bins=np.arange(1150, 1520, 5), color="C0", label="production captures")
    ax.hist(rr, bins=np.arange(1150, 1520, 5), color="C1", alpha=0.8, label="ring-down captures")
    ax.axvline(ODR_HZ, color="C3", ls="--", lw=1.2)
    ax.text(ODR_HZ - 3, ax.get_ylim()[1] * 0.95, "sensor rate\n1493/s", color="C3", ha="right", va="top", fontsize=8)
    ax.set_xlabel("Delivered samples per second")
    ax.set_ylabel("Number of captures")
    ax.legend(fontsize=8, loc="center left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "read_cycle.pdf")
    print(f"\nWrote {OUT / 'read_cycle.pdf'}")


if __name__ == "__main__":
    main()
