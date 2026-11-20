#!/usr/bin/env python3
"""
make_figures.py

Figures for case_120deg.tex: run DTU10MW_385psi_800fs (385 psi, fan 800,
6.9 m/s), motor angle 120 deg. Run from the repo root:
    poetry run python reports/case_385psi_800fs_120deg/make_figures.py

Accelerometer samples are placed on the measured 1493 Hz grid with the
dropped samples masked (accel_spectra), as in the ring-down report. Writes
PDFs into figs/ and prints the numbers quoted in the text.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis" / "ringdown"))
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))
import plot_ringdown_psd as P  # noqa: E402
from accel_spectra import MEASURED_ODR_HZ, drop_sidebands, gridded, masked_welch, top_peaks  # noqa: E402

RUN = "DTU10MW_385psi_800fs"
RAW = ROOT / "data" / "Production_10-8-26" / RUN
SUMM = ROOT / "data" / "Production_summaries" / RUN / "angle_summary.csv"
OUT = Path(__file__).resolve().parent / "figs"
ANGLE = 120.0
QUIET = 70.0           # quietest angle of the sweep (lowest tip variance)
NPERSEG = 2048
MG = 0.488
COL = {"y": "C0", "z": "C3"}
AXNAME = {"y": "Edgewise $y$", "z": "Flapwise $z$"}
# Ring-down (no-bolt) peaks: mount, edge1/flap2 pair; and the ~110 Hz peak of
# the fast runs at their quiet angles (ringdown report, Sec. 5).
RD = {85: "mount", 131: "", 138: "edge1/flap2 pair"}
R1 = (105, 135)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.2, "legend.frameon": False})


def fname(a):
    return RAW / f"angle_{a:+07.2f}deg.csv"


def load(a):
    acc, _ = P.read_accel(fname(a), MG, MEASURED_ODR_HZ)
    return acc


def psd(acc, col, nperseg=NPERSEG):
    x, m = gridded(acc[col].to_numpy(), acc["grid"].to_numpy())
    return masked_welch(x, MEASURED_ODR_HZ, nperseg, m)


def band_rms(f, p, fc, w=4.0):
    s = (f >= fc - w) & (f <= fc + w)
    return np.sqrt(np.trapezoid(p[s], f[s]))


def peaks(f, p, fb, n=4):
    return [(round(float(f[i]), 1), float(p[i])) for i in drop_sidebands(f, p, top_peaks(f, p), fb)[:n]]


def mark_rd(ax):
    for fr, lab in RD.items():
        ax.axvline(fr, color="0.55", ls=":", lw=0.9)
    ax.text(85, 1.02, "85", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="0.4")
    ax.text(134.5, 1.02, "131/138", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="0.4")


# ---------------------------------------------------------------- sweep
def fig_sweep(S):
    S = S.sort_values("requested_angle_deg")
    a = S["requested_angle_deg"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, constrained_layout=True)
    ax = axes[0]
    for c in "yz":
        ax.plot(a, np.sqrt(S[f"var_g2_s4_{c}"]), "-o", ms=3, color=COL[c], label=f"S4 {AXNAME[c]}")
    ax.set_ylabel("Tip RMS acceleration (g)")
    ax.legend(loc="upper left")
    ax2 = axes[1]
    ax2.plot(a, np.sqrt(S["var_Fy"]), "-o", ms=3, color="0.3", label="$F_y$")
    ax2.plot(a, np.sqrt(S["var_Fx"]), "-o", ms=3, color="0.65", label="$F_x$")
    ax2.set_ylabel("Load-cell RMS force (N)")
    ax2.set_xlabel("Motor angle (deg)")
    ax2.legend(loc="upper left")
    for x in axes:
        x.axvspan(*R1, color="0.93", zorder=0)
        x.axvline(ANGLE, color="k", lw=0.8, ls="--")
    axes[0].text(ANGLE + 2, 0.93, "120°", transform=axes[0].get_xaxis_transform(), fontsize=8)
    axes[0].text(R1[0] + 1, 0.80, "R1", transform=axes[0].get_xaxis_transform(), fontsize=8, color="0.4")
    fig.savefig(OUT / "sweep.pdf")
    r = S.set_index("requested_angle_deg")
    print("sweep: tip rms y,z at 120:", np.sqrt(r.loc[ANGLE, ["var_g2_s4_y", "var_g2_s4_z"]]).round(2).tolist(),
          " median all angles:", np.sqrt(S[["var_g2_s4_y", "var_g2_s4_z"]].median()).round(2).tolist(),
          " at 70:", np.sqrt(r.loc[QUIET, ["var_g2_s4_y", "var_g2_s4_z"]]).round(2).tolist())
    print("sweep: Fy rms 120 / median:", np.sqrt(r.loc[ANGLE, "var_Fy"]).round(3), np.sqrt(S["var_Fy"].median()).round(3),
          " Fx:", np.sqrt(r.loc[ANGLE, "var_Fx"]).round(3), np.sqrt(S["var_Fx"].median()).round(3))
    # Flow angles from angle_summary.csv: correct only for summaries regenerated after the
    # coordinate_transforms fix of 12 Nov 2026 (expected at 120 deg: AoA +129.6, incl +25.7).
    print("sweep: aoa, inclination at 120:", r.loc[ANGLE, ["aoa_deg", "inclination_deg"]].round(1).tolist())


# ---------------------------------------------------------------- time
def fig_time(acc):
    t = acc["time"].to_numpy() - acc["time"].iloc[0]
    fig = plt.figure(figsize=(7, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :])
    for c in "yz":
        x = acc[f"s4_{c}"].to_numpy()
        x = x - x.mean()
        k = (t // 0.5).astype(int)
        r = pd.Series(x).groupby(k).std()
        ax.plot(r.index * 0.5 + 0.25, r.values, color=COL[c], label=f"S4 {AXNAME[c]}")
        print(f"time: S4 {c} 0.5 s RMS min/median/max:", np.round([r.min(), r.median(), r.max()], 2))
    ax.set_ylim(0, None)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS over 0.5 s (g)")
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("(a) Amplitude through the 60 s capture", loc="left", fontsize=9)

    # one read batch (84 contiguous samples, 56 ms), middle of the capture
    g = acc["grid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(g) > 1])
    b0 = starts[len(starts) // 2]
    b1 = starts[len(starts) // 2 + 1]
    seg = acc.iloc[b0:b1]
    ts = (seg["time"] - seg["time"].iloc[0]) * 1e3
    ax = fig.add_subplot(gs[1, :2])
    for c in "yz":
        ax.plot(ts, seg[f"s4_{c}"] - acc[f"s4_{c}"].mean(), color=COL[c], label=AXNAME[c])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("S4 acceleration (g)")
    ax.set_title(f"(b) One unbroken read, {len(seg)} samples", loc="left", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(seg["s4_y"] - acc["s4_y"].mean(), seg["s4_z"] - acc["s4_z"].mean(), color="0.3", lw=0.9)
    ax.set_xlabel("$y$ (g)")
    ax.set_ylabel("$z$ (g)")
    ax.set_aspect("equal", "datalim")
    ax.set_title("(c) Tip acceleration, $y$ vs $z$", loc="left", fontsize=9)
    fig.savefig(OUT / "time.pdf")


# ---------------------------------------------------------------- spectra
def fig_spectra(acc, quiet):
    fb = acc.attrs["batch_hz"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True, constrained_layout=True)
    for ax, c in zip(axes, "yz"):
        f, p = psd(acc, f"s4_{c}")
        fq, pq = psd(quiet, f"s4_{c}")
        f1, p1 = psd(acc, f"s1_{c}")
        ax.semilogy(fq, pq, color="0.6", lw=0.9, label="S4, 70° (quiet)")
        ax.semilogy(f1, p1, color=COL[c], lw=0.8, alpha=0.5, label="S1 (root), 120°")
        ax.semilogy(f, p, color=COL[c], label="S4 (tip), 120°")
        mark_rd(ax)
        ax.set_xlim(0, 400)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(AXNAME[c], loc="left", fontsize=9, pad=12)
        print(f"spectra: S4 {c} 120 peaks:", [(fr, f'{v:.2e}') for fr, v in peaks(f, p, fb)])
        print(f"spectra: S4 {c} 70  peaks:", [(fr, f'{v:.2e}') for fr, v in peaks(fq, pq, quiet.attrs['batch_hz'])])
        print(f"spectra: S1 {c} 120 peaks:", [(fr, f'{v:.2e}') for fr, v in peaks(f1, p1, fb)])
        tot = np.sqrt(np.trapezoid(p[(f > 5)], f[f > 5]))
        print(f"spectra: S4 {c} share of RMS in 113+-4 / 143+-6 / 220+-6:",
              [round(band_rms(f, p, fc, w) ** 2 / tot ** 2, 2) for fc, w in ((113, 4), (143, 6), (221, 6))])
    axes[0].set_ylabel("PSD (g$^2$/Hz)")
    axes[0].set_ylim(1e-6, 1)
    axes[1].legend(loc="upper right", fontsize=7.5)
    fig.savefig(OUT / "spectra.pdf")


# ---------------------------------------------------------------- spectrogram
def fig_spectrogram(acc):
    t0 = acc["time"].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True, constrained_layout=True)
    W = 2.0
    for ax, c in zip(axes, "yz"):
        rows, tc = [], []
        for k in range(int(60 // W)):
            s = acc[(acc["time"] - t0 >= k * W) & (acc["time"] - t0 < (k + 1) * W)]
            f, p = psd(s, f"s4_{c}", nperseg=1024)
            rows.append(p)
            tc.append((k + 0.5) * W)
        Z = np.log10(np.array(rows).T)
        sel = f <= 300
        im = ax.pcolormesh(tc, f[sel], Z[sel], shading="nearest", cmap="Blues", vmin=-5, vmax=0)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"S4 {AXNAME[c]}", loc="left", fontsize=9)
        ax.grid(False)
        pk = f[sel][np.argmax(np.array(rows)[:, sel][:, (f[sel] > 90) & (f[sel] < 180)], axis=1)
                    + np.flatnonzero((f[sel] > 90) & (f[sel] < 180))[0]]
        print(f"spectrogram: S4 {c} 2 s peak (90-180 Hz) min/median/max:", pk.min().round(1), np.median(pk).round(1), pk.max().round(1))
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes, label="log$_{10}$ PSD (g$^2$/Hz)", shrink=0.8)
    fig.savefig(OUT / "spectrogram.pdf")


# ---------------------------------------------------------------- neighbours
def fig_angles(S):
    """S4 spectra at the angles either side of 120 deg, plus the peak
    frequency (90-180 Hz) at every angle 80-160 deg (printed)."""
    angles = sorted(a for a in S["requested_angle_deg"] if 80 <= a <= 160)
    data = {a: load(a) for a in angles}
    show = [100, 110, 115, 120, 125, 130]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True, constrained_layout=True)
    for ax, c in zip(axes, "yz"):
        pk = []
        for a in angles:
            f, p = psd(data[a], f"s4_{c}")
            s = (f > 90) & (f < 180)
            pk.append(f[s][np.argmax(p[s])])
            if a in show:
                k = show.index(a)
                ax.semilogy(f, p, color=cmap(k / (len(show) - 1) * 0.9), lw=1.6 if a == ANGLE else 0.9,
                            label=f"{a:.0f}°", zorder=3 if a == ANGLE else 2)
        mark_rd(ax)
        ax.set_xlim(0, 300)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"S4 {AXNAME[c]}", loc="left", fontsize=9, pad=12)
        print(f"angles: S4 {c} peak 90-180 Hz by angle:", {a: float(round(v, 1)) for a, v in zip(angles, pk)})
    axes[0].set_ylabel("PSD (g$^2$/Hz)")
    axes[0].set_ylim(1e-5, 1)
    axes[1].legend(fontsize=7.5, loc="upper right", title="Motor angle", title_fontsize=7.5, ncol=2)
    fig.savefig(OUT / "angles.pdf")


# ---------------------------------------------------------------- span + load cell
def fig_span_load(acc, quiet_load):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), constrained_layout=True)
    ax = axes[0]
    sens = [1, 2, 3, 4]
    for c, fc, w, ls in (("y", 113, 4, "-"), ("z", 143, 6, "-"), ("z", 105, 4, "--")):
        amp = []
        for s in sens:
            f, p = psd(acc, f"s{s}_{c}")
            amp.append(band_rms(f, p, fc, w))
        ax.plot(sens, amp, ls, marker="o", ms=4, color=COL[c], label=f"{c} at {fc} Hz")
        print(f"span: {c} at {fc} Hz band RMS (g) S1..S4:", np.round(amp, 2), " S1/S4:", round(amp[0] / amp[-1], 2))
    ax.set_xticks(sens, ["S1\nroot", "S2", "S3", "S4\ntip"])
    ax.set_ylabel("RMS in band (g)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5)
    ax.set_title("(a) Along the span, 120°", loc="left", fontsize=9)

    ax = axes[1]
    for a, L, col, lw in ((QUIET, quiet_load, "0.6", 0.9), (ANGLE, pd.read_csv(fname(ANGLE).with_name(f"angle_{ANGLE:+07.2f}deg_load.csv"), comment="#"), "k", 1.1)):
        for comp, ls in (("Fy", "-"),):
            f, p = welch(L[comp].to_numpy() - L[comp].mean(), fs=2000.0, nperseg=4096)
            ax.semilogy(f, p, ls, color=col, lw=lw, label=f"$F_y$, {a:.0f}°")
            s = (f > 50) & (f < 300)
            i = np.argsort(p[s])[::-1][:1]
            print(f"load: {comp} at {a}: top peak 50-300 Hz", f[s][i].round(1), " PSD", p[s][i])
        f, p = welch(L["Fx"].to_numpy() - L["Fx"].mean(), fs=2000.0, nperseg=4096)
        s = (f > 50) & (f < 300)
        print(f"load: Fx at {a}: top peak 50-300 Hz", f[s][np.argmax(p[s])].round(1))
    mark_rd(ax)
    ax.set_xlim(0, 400)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (N$^2$/Hz)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("(b) Load cell at the root", loc="left", fontsize=9, pad=12)
    fig.savefig(OUT / "span_load.pdf")


def coherence(acc):
    """y-z coherence at the tip at the two peaks: is it one motion or two?"""
    from scipy.signal import coherence as coh
    g = acc["grid"].to_numpy()
    # Coherence of the gridded signals (gaps zeroed; they lower both spectra alike).
    xy, m = gridded(acc["s4_y"].to_numpy(), g)
    xz, _ = gridded(acc["s4_z"].to_numpy(), g)
    f, C = coh(xy, xz, fs=MEASURED_ODR_HZ, nperseg=NPERSEG)
    for fc in (113, 143, 221):
        i = np.argmin(abs(f - fc))
        print(f"coherence S4 y-z at {fc} Hz: {C[i - 2:i + 3].max():.2f}")


def main():
    OUT.mkdir(exist_ok=True)
    S = pd.read_csv(SUMM)
    acc = load(ANGLE)
    quiet = load(QUIET)
    print("capture:", round(acc["time"].iloc[-1] - acc["time"].iloc[0], 1), "s; batch rate", round(acc.attrs["batch_hz"], 2), "Hz")
    for c in ("s4_y", "s4_z"):
        print(f"{c}: std {acc[c].std():.2f} g, max |a| {acc[c].abs().max():.2f} g")
    fig_sweep(S)
    fig_time(acc)
    fig_spectra(acc, quiet)
    fig_spectrogram(acc)
    fig_angles(S)
    fig_span_load(acc, pd.read_csv(fname(QUIET).with_name(f"angle_{QUIET:+07.2f}deg_load.csv"), comment="#"))
    coherence(acc)


if __name__ == "__main__":
    main()
