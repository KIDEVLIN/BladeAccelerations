#!/usr/bin/env python3
"""
ring_down_analysis.py

Analysis of ONE ring-down (flick) trial recorded by ring_down_aquire.py: ring_<label>_<NNN>.csv.
Writes four figures (and two csv tables):

  1. <stem>_signals.png   acceleration vs time, three panels (sensor x, y, z), all four accelerometers
                          overlaid (or only the ones given with --sensor). The flick is detected
                          automatically; an exponential decay is fitted to the ring-down envelope of each
                          sensor and axis, and the damping ratio zeta is in the legend.
  2. <stem>_psd.png       power spectral density of the ring-down on log-log axes, same three panels, with
                          the pre-flick noise floor (grey dashed) and the identified modes (numbered).
  3. <stem>_fft.png       the FFT amplitude spectrum (g) of the ring-down, same panels: just the spectrum
                          of x, y, z, with S1..S4 overlaid. Taken from the flick until the fitted ring-downs
                          reach the noise floor (--fft-window T to set it), rectangular window, linear
                          amplitude (--fft-yscale log), frequency axis linear from 0 to --fmax.
  4. <stem>_modes.png     mode shapes: the normalized amplitude of the identified modes along the span
                          (r/R = 0.13, 0.26, 0.35, 0.48), one panel per sensor axis. Always uses all four
                          sensors, whatever --sensor is.

Usage (from scripts/aquire_data):
    python ring_down_analysis.py Data/Modal/DTU10MW_still_air --label flap            # first kept trial
    python ring_down_analysis.py Data/Modal/DTU10MW_still_air --label flap --trial 3
    python ring_down_analysis.py Data/Modal/DTU10MW_still_air/ring_flap_003.csv
    python ring_down_analysis.py <folder> --label edge --sensor 4                    # one accelerometer
    python ring_down_analysis.py <folder> --label flap --component y                 # one sensor axis
    python ring_down_analysis.py <folder> --label flap --sensor 2 4 --component y z
    python ring_down_analysis.py <folder> --label flap --fmin 10 --fmax 300 --nmodes 4

The first argument is the ring-down folder (then --label / --trial pick the file) or the ring_*.csv file.

--sensor (1-4, or S1-S4, several allowed) and --component (x y z, several allowed) restrict what figures 1-3
show, and the file names say so: _4 / _24 (the sensor numbers) is appended when fewer than four sensors are shown, and _y / _yz
when fewer than three axes are shown (e.g. ring_flap_001_psd_4_y.png). --component also decides which axes
are used for the flick detection and for finding the modes, and which panels the mode-shape figure shows.

Accelerometer processing: identical to the sweep pipeline. The CSV time column is stamped once per FIFO
batch, so utils/run_loader.load_accel_csv reconstructs a time for every sample, and the data are then
interpolated onto a uniform grid at the sensor ODR (from the CSV header) before any filtering, spectra or
fits. Stretches where the reconstructed sample spacing is more than 3x nominal (dropped samples / late
reads) are counted and reported. The static (gravity) part is removed with the median of the quiet
pre-flick record.

Flick detection: like the quick QC in ring_down_aquire.py, but the noise level and the static offset come
from the first --quiet-s seconds of the record (default 1 s; the acquisition script asks for ~2 s of stillness
before the flick). The deviation over the selected axes (all 12 by default) is windowed (0.1 s rms) and the
flick is the end of the first window above 6x the quiet noise level, so the window already contains the flick.

--onset T overrides it.
--flick_t overrides the length of the flick in seconds (default 0.5 s) for the PSD and mode shape calculations. 

Damping ratio (per sensor and axis)
    * f_n of an axis = the highest peak of the summed ring-down PSD of the four sensors on that axis, within
      --fmin..--fmax (or --freq to force one value).
    * the acceleration is band-passed around f_n (+/- --bw x f_n, at least 1 Hz) and its Hilbert envelope
      taken, smoothed over one cycle.
    * ln(envelope) is fitted with a straight line from the envelope peak after the flick until it first
      drops below --fit-snr x the pre-flick envelope level (or the record ends): envelope ~ exp(-sigma t).
    * zeta = sigma / (2 pi f_n)  (viscous damping, small-damping form).
    Dashed lines in figure 1 show the fitted envelope. Fits are skipped (legend says why) if the envelope
    peak is under 10x the noise or fewer than 3 cycles are available. If two modes sit close together in one
    axis, the envelope beats and zeta is unreliable: check the R^2 in the csv/console and narrow --bw.

Mode shapes (all four sensors): modes are the peaks of the PSD summed over all sensors and axes (--nmodes
strongest, at least --min-sep Hz apart). For each mode and axis the amplitude at each sensor is the square
root of the noise-corrected PSD integrated over the mode peak, signed by the phase of the cross-spectrum
with the strongest sensor on that axis (in phase +, out of phase -). Hollow dots mean low coherence with that
reference sensor (< 0.7), i.e. not a clean single mode there. By default each mode is normalized by its
largest value over all axes, so a flap-dominated mode shows little in the edge panel (--mode-norm axis
normalizes every panel separately). The curve between sensors is a cubic spline with the blade root fixed
at zero (zero slope): it only joins the dots, it is not a measurement.

Requires: numpy pandas scipy matplotlib
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from scipy.interpolate import CubicSpline                         # noqa: E402
from scipy.ndimage import uniform_filter1d                        # noqa: E402
from scipy.signal import butter, csd, find_peaks, hilbert, sosfiltfilt, welch   # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils.run_loader import load_accel_csv                       # noqa: E402
from plot_angle_sweep import INK, MUTED, SURFACE, CAT, SENSOR_COLORS, style_axes, parse_sensors   # noqa: E402

NS = 4
AXES = "xyz"
R_OVER_R = (0.13, 0.26, 0.35, 0.48)       # spanwise position of S1..S4 (R = 28 cm)
INT16_MAX = 32767
SAT_LEVEL = 0.995
FALLBACK_MG_PER_LSB = 0.061
GREY = "#b5b4ae"

ONSET_SNR = 6.0           # same rule as ring_down_aquire.py's QC
ONSET_WIN_S = 0.1
MIN_PEAK_SNR = 10.0       # envelope peak must clear the noise by this to attempt a fit
MIN_CYCLES = 3
COH_OK = 0.7

RING_RE = re.compile(r"^ring_(?P<label>.+)_(?P<k>\d+)\.csv$")


# --------------------------------------------------
# Trial selection and loading
# --------------------------------------------------

def find_trial(path, label, k):
    p = Path(path)
    if p.is_file():
        return p.parent, p
    if not p.is_dir():
        raise SystemExit(f"{p} is not a folder or a ring_*.csv file")
    found = {}
    for f in sorted(p.glob("ring_*.csv")):
        m = RING_RE.match(f.name)
        if m:
            found.setdefault(m["label"], {})[int(m["k"])] = f
    if not found:
        raise SystemExit(f"No ring_<label>_<NNN>.csv files in {p}")
    summary = "; ".join(f"{lb}: {sorted(v)}" for lb, v in found.items())
    if label is None:
        if len(found) > 1:
            raise SystemExit(f"Give --label. Available (label: trial numbers): {summary}")
        label = next(iter(found))
    if label not in found:
        raise SystemExit(f"No trials labelled {label!r}. Available (label: trial numbers): {summary}")
    trials = found[label]
    k = k if k is not None else min(trials)
    if k not in trials:
        raise SystemExit(f"{label!r} has no trial {k}; available: {sorted(trials)}")
    return p, trials[k]


def read_meta(run_dir):
    path = Path(run_dir) / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_header_odr(path):
    with open(path) as f:
        m = re.search(r"ODR=(\d+)", f.readline())
    return int(m.group(1)) if m else None


def load_trial(path, fs_override, meta):
    """Per-sample times come from run_loader (batch-stamped times spread evenly across each batch); the
    data are then interpolated onto a uniform grid at the ODR so filters, spectra and fits can assume a
    constant sample rate. Returns (t, D (N, 4 sensors, 3 axes) in g, info dict).
    
    FIX FIX FIX get rid of interpolation"""
    df, _ = load_accel_csv(path)
    mg = df.attrs.get("mg_per_lsb")
    if mg is None:
        mg = FALLBACK_MG_PER_LSB
        print(f"  WARN: no mg/LSB in the CSV header; assuming {mg} (2g mode)")
    cols = [f"s{s}_{a}" for s in range(1, NS + 1) for a in AXES]
    raw = df[cols].to_numpy(float)
    t = df["time"].to_numpy(float)
    if len(t) < 512:
        raise SystemExit("too few samples in this capture")
    odr = read_header_odr(path) or meta.get("accel_odr_hz")
    measured = (len(t) - 1) / (t[-1] - t[0])
    fs = float(fs_override or odr or measured)
    dt = np.diff(t)
    nominal = 1.0 / fs
    slow = dt > 3.0 * nominal
    info = dict(fs=fs, odr=odr, measured=measured, n=len(t), mg=mg,
                frac_slow=float(slow.mean()), max_dt_ms=float(dt.max() * 1000.0),
                missing_frac=float(max(0.0, 1.0 - len(t) / (fs * (t[-1] - t[0])))))
    
    
    tu = np.arange(t[0], t[-1], nominal)
    acc = np.column_stack([np.interp(tu, t, raw[:, k]) for k in range(raw.shape[1])])
    D = (acc * mg / 1000.0).reshape(-1, NS, 3)
    sat = (np.abs(raw) >= SAT_LEVEL * INT16_MAX).mean(axis=0).reshape(NS, 3)
    info["sat"] = sat
    return tu, D, info


# --------------------------------------------------
# Flick detection
# --------------------------------------------------

def detect_flick(t, D, fs, comps=(0, 1, 2), quiet_s=1.0):
    """Onset time (s) of the flick, or None. comps: axis indices (0, 1, 2 = x, y, z) used for detection.
    The noise level and the static (gravity) offset come from the first quiet_s seconds of the record."""
    flat = D[:, :, list(comps)].reshape(len(D), -1)
    nq = int(quiet_s * fs)
    flat = flat - np.median(flat[:nq], axis=0)         # remove gravity / static offset using the quiet start
    mag = np.linalg.norm(flat, axis=1)
    w = max(1, int(ONSET_WIN_S * fs))                  # window length in samples
    rms = np.sqrt(np.convolve(mag ** 2, np.ones(w) / w, mode="valid"))
    noise = float(np.median(rms[:max(1, nq - w)]))     # windows lying entirely inside the quiet start
    above = np.where(rms > ONSET_SNR * noise)[0]
    if noise <= 0 or len(above) == 0:
        return None
    return float(t[above[0] + w - 1])                  # end of the first window that contains the flick


# --------------------------------------------------
# Spectra, modes
# --------------------------------------------------

def psd(x, fs, nperseg):
    n = min(nperseg, len(x))
    return welch(x, fs=fs, nperseg=n, axis=0)           # detrend='constant': mean removed per segment


def find_modes(f, Ptot, fmin, fmax, nmodes, min_sep):
    idx = np.where((f >= fmin) & (f <= fmax))[0]
    if len(idx) < 5:
        return np.array([])
    y = np.log10(np.maximum(uniform_filter1d(Ptot, 3, mode="nearest"), 1e-30))
    df = f[1] - f[0]
    pk, _ = find_peaks(y[idx], prominence=0.3, distance=max(1, int(round(min_sep / df))))
    pk = idx[pk]
    top = pk[np.argsort(-Ptot[pk])][:nmodes]
    return np.sort(f[top])


def mode_shapes(Dpost, Pn_interp, f, fs, nperseg, modes):
    """Returns (shape (K, 3, 4) signed amplitudes, coh (K, 3, 4), phase_deg (K, 3, 4), ref (K, 3))."""
    n = min(nperseg, len(Dpost))
    df = f[1] - f[0]
    K = len(modes)
    shape = np.zeros((K, 3, NS))
    coh = np.zeros((K, 3, NS))
    phase = np.zeros((K, 3, NS))
    ref = np.zeros((K, 3), int)
    for j in range(3):
        x = Dpost[:, :, j]
        _, P = welch(x, fs=fs, nperseg=n, axis=0)                      # (nf, 4)
        C = {}
        for i in range(NS):
            for r in range(NS):
                if i != r:
                    C[(i, r)] = csd(x[:, i], x[:, r], fs=fs, nperseg=n)[1]
        for k, fk in enumerate(modes):
            h = max(1.5, 0.05 * fk)
            b = (f >= fk - h) & (f <= fk + h)
            Praw = P[b].sum(axis=0) * df
            Pnoise = Pn_interp[b][:, :, j].sum(axis=0) * df if Pn_interp is not None else 0.0
            amp = np.sqrt(np.maximum(Praw - Pnoise, 0.0))
            r = int(np.argmax(amp))
            ref[k, j] = r
            for i in range(NS):
                if i == r:
                    cc, ph, sg = 1.0, 0.0, 1.0
                else:
                    c = C[(i, r)][b].sum() * df
                    cc = float(np.abs(c) ** 2 / (Praw[i] * Praw[r])) if Praw[i] > 0 and Praw[r] > 0 else 0.0
                    ph = float(np.degrees(np.angle(c)))
                    sg = 1.0 if c.real >= 0 else -1.0
                shape[k, j, i], coh[k, j, i], phase[k, j, i] = sg * amp[i], cc, ph
    return shape, coh, phase, ref


# --------------------------------------------------
# Exponential-decay fit
# --------------------------------------------------

def fit_decay(x, t, fs, fn, onset, args):
    out = dict(fn=fn, zeta=np.nan, sigma=np.nan, r2=np.nan, t_peak=np.nan, t_end=np.nan,
               n_cycles=np.nan, peak_snr=np.nan, intercept=np.nan, status="")
    hw = max(1.0, args.bw * fn)
    lo, hi = max(0.5, fn - hw), min(fn + hw, 0.45 * fs)
    if hi <= lo:
        out["status"] = "band invalid"
        return out
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    w = max(3, int(round(fs / fn)))
    envs = uniform_filter1d(env, w, mode="nearest")

    pre = (t > t[0] + 0.15) & (t < onset - 0.15)
    noise = float(np.median(envs[pre])) if pre.sum() >= max(int(0.3 * fs), 4 * w) else float(np.percentile(envs, 10))
    end_lim = len(t) - int(0.05 * fs) - 1
    win = np.where((t >= onset) & (t <= onset + 1.0))[0]
    if len(win) == 0 or noise <= 0:
        out["status"] = "no data after flick"
        return out
    ip = int(win[np.argmax(envs[win])])
    out["peak_snr"] = float(envs[ip] / noise)
    if out["peak_snr"] < MIN_PEAK_SNR:
        out["status"] = f"weak signal (peak/noise {out['peak_snr']:.1f})"
        return out
    below = np.where(envs[ip:end_lim] < args.fit_snr * noise)[0]
    ie = ip + int(below[0]) if len(below) else end_lim
    n_cyc = (t[ie] - t[ip]) * fn
    out.update(t_peak=float(t[ip]), t_end=float(t[ie]), n_cycles=float(n_cyc))
    if ie - ip < 10 or n_cyc < MIN_CYCLES:
        out["status"] = f"decay too short ({n_cyc:.1f} cycles)"
        return out
    tt, yy = t[ip:ie + 1], np.log(envs[ip:ie + 1])
    slope, icpt = np.polyfit(tt, yy, 1)
    if slope >= 0:
        out["status"] = "no decay"
        return out
    ss_res = float(np.sum((yy - (slope * tt + icpt)) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    out.update(sigma=float(-slope), zeta=float(-slope / (2 * np.pi * fn)), intercept=float(icpt),
               r2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan, status="ok")
    return out


# --------------------------------------------------
# Figures
# --------------------------------------------------

def save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def sensor_label(s, fit):
    if np.isfinite(fit["zeta"]):
        return f"S{s}   ζ = {fit['zeta']:.4f}"
    return f"S{s}   ζ n/a  ({fit['status']})"


def make_axes(n):
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 + 2.45 * n), sharex=True, facecolor=SURFACE,
                             constrained_layout=True, squeeze=False)
    return fig, list(axes[:, 0])


def plot_signals(t, D, sensors, comps, onset, fits, fn_axis, title, args, out_png, t_flick):
    fig, axes = make_axes(len(comps))
    for j, ax in zip(comps, axes):
        for s in sensors:
            i = s - 1
            fit = fits[(i, j)]
            ax.plot(t, D[:, i, j], color=SENSOR_COLORS[i], lw=0.7, label=sensor_label(s, fit), zorder=3, ls ='none', marker = '.', markersize = 2)
            #if np.isfinite(fit["zeta"]):
            #    m = (t >= fit["t_peak"]) & (t <= fit["t_end"])
            #    e = np.exp(fit["intercept"] - fit["sigma"] * t[m])
            #    ax.plot(t[m], e, color=SENSOR_COLORS[i], lw=1.2, ls=(0, (4, 2)), zorder=4)
            #    ax.plot(t[m], -e, color=SENSOR_COLORS[i], lw=1.2, ls=(0, (4, 2)), zorder=4)
        ax.axvline(onset, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=2)
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5)
        ax.set_ylabel("Acceleration (g)", color=INK)
        ax.set_title(f"sensor {AXES[j]} axis   ·   f$_n$ = {fn_axis[j]:.1f} Hz", loc="left", fontsize=10, color=INK)
        ax.legend(loc="upper right", ncol=2 if len(sensors) > 1 else 1, frameon=False, fontsize=9, labelcolor=INK)
        style_axes(ax)
    axes[0].annotate("flick", (onset, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -12),
                     textcoords="offset points", fontsize=8.5, color=MUTED)
    axes[-1].set_xlabel("Time (s)", color=INK)
    axes[-1].set_xlim(t_flick, args.tmax if args.tmax else t[-1])
    #axes[-1].set_xlim(t[0], args.tmax if args.tmax else t[-1])
    fig.suptitle(title + "      dashed = fitted decay envelope", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, out_png, args.dpi)


def mode_lines(ax, modes):
    for k, fk in enumerate(modes):
        ax.axvline(fk, color=INK, lw=0.8, ls=":", alpha=0.6, zorder=1)
        ax.text(fk, 1.0, str(k + 1), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK)


def mode_legend_handle(modes):
    return Line2D([], [], color=INK, lw=0.9, ls=":",
                  label="modes: " + ", ".join(f"{k + 1}: {fk:.1f} Hz" for k, fk in enumerate(modes)))


def plot_psd(f, P, fn_f, Pn, sensors, comps, modes, fn_axis, title, args, fs, out_png):
    fig, axes = make_axes(len(comps))
    pos = f > 0
    for j, ax in zip(comps, axes):
        for s in sensors:
            i = s - 1
            if Pn is not None:
                pn = fn_f > 0
                ax.plot(fn_f[pn], Pn[pn, i, j], color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.plot(f[pos], P[pos, i, j], color=SENSOR_COLORS[i], lw=1.0, label=f"S{s}", zorder=3)
        mode_lines(ax, modes)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("PSD (g$^2$/Hz)", color=INK)
        ax.set_title(f"sensor {AXES[j]} axis   ·   dominant peak {fn_axis[j]:.1f} Hz", loc="left", fontsize=10,
                     color=INK, pad=12)
        style_axes(ax)
        ax.grid(True, which="minor", color="#efeeea", linewidth=0.5, zorder=0)
    handles = [Line2D([], [], color=SENSOR_COLORS[s - 1], lw=1.6, label=f"S{s}") for s in sensors]
    if Pn is not None:
        handles.append(Line2D([], [], color=GREY, lw=1.2, ls=(0, (3, 2)), label="pre-flick noise"))
    if len(modes):
        handles.append(mode_legend_handle(modes))
    axes[0].legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    axes[-1].set_xlabel("Frequency (Hz)", color=INK)
    axes[-1].set_xlim(max(1.0, args.fmin / 3), fs / 2)
    fig.suptitle(title + "      PSD of the ring-down (flick to end of record)", x=0.01, ha="left", fontsize=11,
                 color=INK)
    save(fig, out_png, args.dpi)


def amplitude_spectrum(x, fs):
    """Single-sided FFT amplitude (g) of x (N, ...), mean removed, rectangular window (a taper would suppress
    the ring-down, which sits at the start of the window), zero-padded to at least 4 s for a smooth curve. A
    steady sinusoid of amplitude A gives a peak of height A; a decaying one gives less, the longer the window
    the lower the peak."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(max(n, 4 * fs))))
    X = np.fft.rfft(x - x.mean(axis=0), n=nfft, axis=0)
    return np.fft.rfftfreq(nfft, 1.0 / fs), 2.0 * np.abs(X) / n


def plot_fft(f, A, sensors, comps, modes, fn_axis, title, args, fs, out_png):
    fig, axes = make_axes(len(comps))
    fmax = min(args.fmax, fs / 2)
    idx = [s - 1 for s in sensors]
    band = (f >= max(args.fmin / 3, f[1])) & (f <= fmax)
    for j, ax in zip(comps, axes):
        for s in sensors:
            ax.plot(f, A[:, s - 1, j], color=SENSOR_COLORS[s - 1], lw=0.9, label=f"S{s}", zorder=3)
        mode_lines(ax, modes)
        ax.set_xlim(0, fmax)
        if args.fft_yscale == "log":
            ax.set_yscale("log")
            vals = A[band][:, idx, j]
            vals = vals[vals > 0]
            if vals.size:
                ax.set_ylim(vals.min() / 2, vals.max() * 3)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel("FFT amplitude (g)", color=INK)
        ax.set_title(f"sensor {AXES[j]} axis   ·   dominant peak {fn_axis[j]:.1f} Hz", loc="left", fontsize=10,
                     color=INK, pad=12)
        style_axes(ax)
    handles = [Line2D([], [], color=SENSOR_COLORS[s - 1], lw=1.6, label=f"S{s}") for s in sensors]
    if len(modes):
        handles.append(mode_legend_handle(modes))
    axes[0].legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    axes[-1].set_xlabel("Frequency (Hz)", color=INK)
    fig.suptitle(title + "      FFT amplitude spectrum of the ring-down", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, out_png, args.dpi)


def plot_modes(shape, coh, modes, comps, title, args, out_png):
    K = len(modes)
    if args.mode_norm == "mode":
        scale = np.abs(shape).reshape(K, -1).max(axis=1)[:, None, None] * np.ones((K, 3, 1))
    else:
        scale = np.abs(shape).max(axis=2, keepdims=True)
    norm = shape / np.where(scale > 0, scale, 1.0)

    n = len(comps)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 0.5, 4.9), sharey=True, facecolor=SURFACE,
                             constrained_layout=True, squeeze=False)
    axes = list(axes[0])
    xs = np.array([0.0] + list(R_OVER_R))
    xx = np.linspace(0, xs[-1], 120)
    for j, ax in zip(comps, axes):
        ax.axhline(0, color=MUTED, lw=0.7, alpha=0.6, ls=(0, (4, 3)))
        for k, fk in enumerate(modes):
            col = CAT[k % len(CAT)]
            ys = np.concatenate([[0.0], norm[k, j]])
            ax.plot(xx, CubicSpline(xs, ys, bc_type=((1, 0.0), (2, 0.0)))(xx), color=col, lw=1.4, alpha=0.85,
                    zorder=3)
            ax.scatter(R_OVER_R, norm[k, j], s=75, facecolors=[col if c >= COH_OK else "none" for c in coh[k, j]],
                       edgecolors=col, linewidths=1.6, zorder=5)
        ax.plot([0.0], [0.0], marker="s", ms=7, color=INK, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(["root"] + [f"S{s + 1}\n{r:.2f}" for s, r in enumerate(R_OVER_R)])
        ax.set_xlim(-0.02, xs[-1] + 0.04)
        ax.set_title(f"sensor {AXES[j]} axis", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Spanwise position r/R", color=INK)
        style_axes(ax)
    axes[0].set_ylabel("Normalized mode shape" + (" (per mode)" if args.mode_norm == "mode" else " (per axis)"),
                       color=INK)
    axes[0].set_ylim(-1.2, 1.2)
    handles = [Line2D([], [], color=CAT[k % len(CAT)], marker="o", ms=7, lw=1.4, label=f"mode {k + 1}   {fk:.1f} Hz")
               for k, fk in enumerate(modes)]
    handles.append(Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=MUTED, mew=1.4,
                          label=f"hollow: coherence < {COH_OK:g}"))
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, out_png, args.dpi)
    return norm


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", help="ring-down folder (with --label, --trial) or one ring_<label>_<NNN>.csv file")
    ap.add_argument("--label", help="excitation label of the trial (e.g. flap, edge); needed if several exist")
    ap.add_argument("--trial-number", "--trial", dest="k", type=int, help="trial number (default: the first kept)")
    ap.add_argument("--sensor", nargs="+",
                    help="accelerometer(s) for figures 1-3: 4, S4 or 1 4 (default all four); added to the file names")
    ap.add_argument("--component", nargs="+", choices=list(AXES),
                    help="sensor axis/axes to show and use, e.g. y or x z (default x y z); added to the file names")
    ap.add_argument("--quiet-s", type=float, default=1.0,
                    help="seconds at the start of the record that are quiet: noise level and static offset for "
                         "the flick detection (default 1)")
    ap.add_argument("--onset", type=float, help="flick time (s) if the automatic detection is wrong")
    ap.add_argument("--fmin", type=float, default=5.0, help="lowest frequency searched for peaks (Hz, default 5)")
    ap.add_argument("--fmax", type=float, default=400.0, help="highest frequency searched for peaks (Hz, default 400)")
    ap.add_argument("--freq", type=float, help="force f_n (Hz) for the decay fits of every axis")
    ap.add_argument("--bw", type=float, default=0.3,
                    help="decay-fit band half-width as a fraction of f_n (default 0.3, at least 1 Hz)")
    ap.add_argument("--fit-snr", type=float, default=3.0,
                    help="stop the decay fit when the envelope falls below this x the noise level (default 3)")
    ap.add_argument("--nperseg", type=int, default=2048, help="Welch segment length (default 2048)")
    ap.add_argument("--fft-window", type=float,
                    help="seconds after the flick to include in the FFT figure (default: until the fitted "
                         "ring-downs reach the noise floor)")
    ap.add_argument("--fft-yscale", choices=["log", "linear"], default="linear", help="FFT amplitude axis (default linear)")
    ap.add_argument("--nmodes", type=int, default=3, help="number of modes for the mode-shape figure (default 3)")
    ap.add_argument("--min-sep", type=float, default=3.0, help="minimum spacing of modes (Hz, default 3)")
    ap.add_argument("--mode-norm", choices=["mode", "axis"], default="mode",
                    help="mode: each mode scaled by its largest value over all axes (default); axis: per panel")
    ap.add_argument("--tmax", type=float, help="right limit (s) of the time axis in figure 1")
    ap.add_argument("--fs", type=float, help="override the sample rate (Hz); default is the ODR from the CSV header")
    ap.add_argument("--out", help="output folder (default <folder>/analysis)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--flick_t", type = float, default = 0.5, help = "time of flick in seconds")
    args = ap.parse_args()

    run_dir, path = find_trial(args.trial, args.label, args.k)
    meta = read_meta(run_dir)
    sensors = parse_sensors(args.sensor)
    comps = [j for j, a in enumerate(AXES) if args.component is None or a in args.component]
    tag_s = "" if len(sensors) == NS else "_" + "".join(str(s) for s in sensors)
    tag_c = "" if len(comps) == 3 else "_" + "".join(AXES[j] for j in comps)
    print(f"Trial: {path}")
    t, D, info = load_trial(path, args.fs, meta)
    fs = info["fs"]
    print(f"  {info['n']} samples; grid {fs:g} Hz (ODR {info['odr']}, measured {info['measured']:.1f} Hz); "
          f"{info['mg']} mg/LSB")
    if info["measured"] < 0.9 * fs:
        print(f"  WARN: measured rate {info['measured']:.0f} Hz is below 90% of {fs:g} Hz "
              f"(~{100 * info['missing_frac']:.1f}% of samples missing); gaps are interpolated across")
    if info["frac_slow"] > 0.005:
        print(f"  WARN: {100 * info['frac_slow']:.1f}% of sample intervals are more than 3x nominal "
              f"(longest {info['max_dt_ms']:.0f} ms): late reads / gaps; spectra above a few tens of Hz may be affected")
    for s in range(NS):
        bad = [AXES[j] for j in range(3) if info["sat"][s, j] >= 0.01]
        if bad:
            print(f"  WARN: S{s + 1} {', '.join(bad)} at full scale for >= 1% of samples; use a higher --fs range "
                  "(clipped peaks bias the damping and mode shapes)")

    onset = args.onset if args.onset is not None else detect_flick(t, D, fs, comps, args.quiet_s)
    if onset is None:
        raise SystemExit("No flick detected (the signal never rose above the noise floor); give --onset T")
    print(f"  Flick at {onset:.3f} s{' (manual)' if args.onset is not None else ''}")

    # static (gravity) part from the quiet pre-flick record
    pre = (t > t[0] + 0.1) & (t < onset - 0.15)
    base = np.median(D[pre], axis=0) if pre.sum() >= int(0.3 * fs) else np.median(D, axis=0)
    if pre.sum() < int(0.3 * fs):
        print("  NOTE: less than 0.3 s of quiet before the flick; baseline and noise floor use the whole record")
    D = D - base
    post = t >= onset #boolean mask the same dimension as t, True for samples after the flick onset
    # spectra of the ring-down, and the pre-flick noise floor
    flick_length = args.flick_t if args.flick_t is not None else 0.5
    #f,P = psd(D[post][:int(flick_length * fs)], fs, args.nperseg) #saves the frequency and the power spectral density of the post-flick data
    f, P = psd(D[post][:int(flick_length * fs)], fs, np.sum([D[post][:int(flick_length * fs)]])) #saves the frequency and the power spectral density of the post-flick data
    Pn, fn_f = None, None
    if pre.sum() >= 256:
        # shorter segments than the ring-down PSD: the quiet record is short, and more averages give a
        # smooth noise-floor estimate (it is interpolated onto the ring-down frequency grid below)
        fn_f, Pn = psd(D[pre], fs, min(args.nperseg, max(256, int(pre.sum()) // 6)))
    Pn_interp = None
    if Pn is not None:
        Pn_interp = np.stack([np.stack([np.interp(f, fn_f, Pn[:, i, j]) for j in range(3)], axis=-1)
                              for i in range(NS)], axis=1)                      # (nf, 4, 3)

    band = (f >= args.fmin) & (f <= args.fmax)
    fn_axis = []
    for j in range(3):
        if args.freq:
            fn_axis.append(args.freq)
        else:
            ps = uniform_filter1d(P[:, :, j].sum(axis=1), 3, mode="nearest")
            fn_axis.append(float(f[band][np.argmax(ps[band])]))

    # decay fits for every sensor/axis
    fits = {}
    for j in range(3):
        for i in range(NS):
            fits[(i, j)] = fit_decay(D[:, i, j], t, fs, fn_axis[j], onset, args)
    print("\n  Damping (exponential fit of the ring-down envelope):")
    print("    sensor axis   f_n (Hz)   zeta      sigma (1/s)  R^2    cycles  peak/noise  note")
    rows = []
    for i in [s - 1 for s in sensors]:
        for j in comps:
            q = fits[(i, j)]
            note = "" if q["status"] == "ok" else q["status"]
            print(f"    S{i + 1}     {AXES[j]}    {q['fn']:8.2f}   {q['zeta']:.4f}   {q['sigma']:9.3f}   "
                  f"{q['r2']:.3f}  {q['n_cycles']:6.1f}  {q['peak_snr']:9.1f}  {note}")
            rows.append(dict(sensor=f"S{i + 1}", axis=AXES[j], **{k: v for k, v in q.items() if k != "intercept"}))

    # modes and mode shapes (always all four sensors)
    modes = find_modes(f, P[:, :, comps].sum(axis=(1, 2)), args.fmin, args.fmax, args.nmodes, args.min_sep)
    if len(modes):
        print("\n  Modes: " + ", ".join(f"{k + 1}: {fk:.1f} Hz" for k, fk in enumerate(modes)))
    else:
        print("\n  No PSD peaks found for the mode shapes (try a wider --fmin/--fmax)")

    out_dir = Path(args.out) if args.out else run_dir / "analysis"
    stem = path.stem
    bits = [stem, f"flick at {onset:.2f} s", f"{fs:g} Hz"]
    title = "   ·   ".join(bits)
    tag = tag_s + tag_c                                  # figures 1-3 depend on both choices

    sig_png = out_dir / f"{stem}_signals{tag}.png"
    plot_signals(t, D, sensors, comps, onset, fits, fn_axis, title, args, sig_png, onset)
    psd_png = out_dir / f"{stem}_psd{tag}.png"
    plot_psd(f, P, fn_f, Pn, sensors, comps, modes, fn_axis, title, args, fs, psd_png)
    if args.fft_window:
        t_stop = onset + args.fft_window
    else:                                       # until the selected ring-downs reached the noise floor
        ends = [fits[(s - 1, j)]["t_end"] for s in sensors for j in comps if np.isfinite(fits[(s - 1, j)]["t_end"])]
        t_stop = max(ends) if ends else t[-1]
    fm = post & (t <= t_stop)
    print(f"  FFT window: {onset:.2f} to {min(t_stop, t[-1]):.2f} s ({fm.sum() / fs:.2f} s)")
    f_fft, A = amplitude_spectrum(D[fm], fs)
    fft_png = out_dir / f"{stem}_fft{tag}.png"
    plot_fft(f_fft, A, sensors, comps, modes, fn_axis, title, args, fs, fft_png)
    print(f"\nWrote {sig_png}\n      {psd_png}\n      {fft_png}")
    damp_csv = out_dir / f"{stem}_damping{tag}.csv"
    pd.DataFrame(rows).to_csv(damp_csv, index=False)

    modes_csv = None
    if len(modes):
        shape, coh, phase, ref = mode_shapes(D[post][:int(flick_length * fs)], Pn_interp, f, fs, args.nperseg, modes)
        mode_png = out_dir / f"{stem}_modes{tag_c}.png"
        norm = plot_modes(shape, coh, modes, comps, title + f"      mode shapes ({args.mode_norm}-normalized)", args,
                          mode_png)
        print(f"      {mode_png}")
        mrows = []
        for k, fk in enumerate(modes):
            pw = (shape[k] ** 2).sum(axis=1)
            part = pw / pw.sum() if pw.sum() > 0 else pw
            print(f"\n  Mode {k + 1}  {fk:.1f} Hz   share of power: "
                  + "  ".join(f"{AXES[j]} {100 * part[j]:.0f}%" for j in range(3)))
            for j in comps:
                print(f"    {AXES[j]}: " + "  ".join(f"S{i + 1} {norm[k, j, i]:+.2f}" for i in range(NS))
                      + f"   (ref S{ref[k, j] + 1}, min coherence {coh[k, j].min():.2f})")
                for i in range(NS):
                    mrows.append(dict(mode=k + 1, freq_hz=fk, axis=AXES[j], sensor=f"S{i + 1}",
                                      r_over_R=R_OVER_R[i], shape_normalized=norm[k, j, i],
                                      shape_signed_g=shape[k, j, i], coherence=coh[k, j, i],
                                      phase_deg=phase[k, j, i], axis_power_share=part[j]))
        modes_csv = out_dir / f"{stem}_modes{tag_c}.csv"
        pd.DataFrame(mrows).to_csv(modes_csv, index=False)
    print(f"\nTables: {damp_csv}" + (f", {modes_csv}" if modes_csv else ""))

if __name__ == "__main__":
    main()
