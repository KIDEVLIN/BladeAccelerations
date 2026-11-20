"""
accel_spectra.py

Spectral tools for the LIS2DS12 accelerometer captures, shared by the
ring-down analysis and the experiment pipeline.

Sample timing. The capture loop reads at most BULK_MAX_SETS (84) samples
per Modbus read and stamps every row of a batch with the time the read
finished. A read cycle takes ~67 ms while the sensor fills 84 slots in
52.5 ms at 1600 Hz, so the board's buffer overruns and ~23 samples are
lost between batches (~1250 Hz delivered). The overrun count is not
logged. Treating the delivered samples as evenly spaced puts every
frequency ~22% low (62.7 Hz flap mode on high-speed video -> 47.5 Hz).

So each sample is placed on the 1/odr grid instead: contiguous within its
batch, with the batch's last sample at the read timestamp, and the lost
samples left as gaps that masked_welch() ignores. See
reports/accel_sampling/ for the evidence. Two caveats:
  - The gaps repeat every read cycle, so each strong peak has weaker
    copies at +/- n * batch_rate_hz() (~14.9 Hz). drop_sidebands()
    removes them from a peak list.
  - The sensors' RC oscillator runs at ~1493 Hz, not the configured
    1600 Hz (MEASURED_ODR_HZ; see reports/pcb_manual_notes/). Sharp peak
    frequencies come from the batch timestamps and barely move with odr;
    higher modes and broad humps scale with it.
"""

import re

import numpy as np

_ODR_RE = re.compile(r"ODR=([0-9.]+)\s*Hz")

# Actual LIS2DS12 sample rate at the configured 1600 Hz ODR: measured by the
# vendor guide over 5 min, +-1 Hz, all four sensors (reports/pcb_manual_notes/).
# Use this for the sample grid, not the configured ODR in the metadata/header.
MEASURED_ODR_HZ = 1493.0

# Peak picking defaults (see top_peaks)
PEAK_FMIN_HZ = 5.0
PEAK_SMOOTH_HZ = 3.0
PEAK_MIN_PROM_DEC = 0.3


def read_header_odr(path):
    """ODR in Hz from the '# LIS2DS12 ..., ODR=<x> Hz, ...' first line of an
    accel CSV, or None if absent."""
    with open(path) as f:
        m = _ODR_RE.search(f.readline())
    return float(m.group(1)) if m else None


def _batch_starts(batch_time):
    return np.flatnonzero(np.r_[True, np.diff(batch_time) > 0])


def batch_grid(batch_time, odr):
    """Integer slot on the 1/odr sample grid for each row, from the raw
    per-batch timestamps (one shared value per batch, in acquisition
    order). A batch that would overlap the previous one (read cycle
    shorter than the batch) is pushed to start right after it."""
    batch_time = np.asarray(batch_time, dtype=float)
    first = _batch_starts(batch_time)
    grid = np.empty(len(batch_time), dtype=int)
    prev_end = -1
    for a, b in zip(first, np.r_[first[1:], len(batch_time)]):
        start = max(int(round(batch_time[a] * odr)) - (b - a - 1), prev_end + 1)
        grid[a:b] = start + np.arange(b - a)
        prev_end = grid[b - 1]
    return grid


def batch_rate_hz(batch_time):
    """Mean read-cycle rate (Hz): the spacing of the gap sidebands."""
    t = np.asarray(batch_time, dtype=float)[_batch_starts(batch_time)]
    return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else np.nan


def gridded(values, grid):
    """Places values (n,) on the contiguous slots grid[0]..grid[-1].
    Returns (x, mask): x is zero and mask 0 in the empty slots."""
    idx = np.asarray(grid) - grid[0]
    x = np.zeros(idx[-1] + 1)
    mask = np.zeros(idx[-1] + 1)
    x[idx] = values
    mask[idx] = 1
    return x, mask


def masked_welch(x, fs, nperseg, mask=None):
    """One-sided Welch PSD (Hann, 50% overlap, per-segment mean removal).
    `mask` (1 = sample present, 0 = dropped) zeroes the gaps and scales
    each segment by its own windowed sample count, so a gappy segment is
    still in units^2/Hz. Without a mask this is the ordinary estimate."""
    if mask is None:
        mask = np.ones(len(x))
    nperseg = min(nperseg, len(x))
    step = nperseg // 2
    win = np.hanning(nperseg)
    starts = range(0, len(x) - nperseg + 1, step)
    segs = np.stack([x[i:i + nperseg] for i in starts])
    m = np.stack([mask[i:i + nperseg] for i in starts])
    keep = m.sum(axis=1) > 0
    segs, m = segs[keep], m[keep]
    mean = np.sum(segs * m, axis=1, keepdims=True) / m.sum(axis=1, keepdims=True)
    segs = (segs - mean) * m * win
    scale = 1.0 / (fs * np.sum((m * win) ** 2, axis=1, keepdims=True))
    p = np.mean(np.abs(np.fft.rfft(segs, axis=1)) ** 2 * scale, axis=0)
    p[1:-1 if nperseg % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(nperseg, 1.0 / fs), p


def gridded_psd(values, grid, odr, nperseg):
    """masked_welch() of values sampled on `grid` (see batch_grid)."""
    x, mask = gridded(values, grid)
    return masked_welch(x, odr, nperseg, mask)


def _prominence(y, i):
    """Peak height above the higher of the two minima between the peak
    and the nearest higher point on each side (or the edge)."""
    l = i
    while l > 0 and y[l - 1] <= y[i]:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r + 1] <= y[i]:
        r += 1
    return y[i] - max(y[l:i + 1].min(), y[i:r + 1].min())


def top_peaks(f, p, n=None, fmin=PEAK_FMIN_HZ, smooth_hz=PEAK_SMOOTH_HZ,
              min_prom=PEAK_MIN_PROM_DEC):
    """Indices of the n highest PSD peaks (all if n is None), highest
    first. Peaks are found on the log PSD smoothed over `smooth_hz` and
    must stand `min_prom` decades above their surroundings, so bin-to-bin
    noise on one hump is not counted as separate peaks."""
    k = max(1, int(round(smooth_hz / (f[1] - f[0]))) | 1)
    y = np.convolve(np.log10(np.maximum(p, 1e-300)), np.ones(k) / k, mode="same")
    idx = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    # Drop peaks the smoothing window runs off the edge for.
    idx = idx[(f[idx] >= fmin) & (idx > k) & (idx < len(f) - k)]
    idx = [i for i in idx if _prominence(y, i) >= min_prom]
    return sorted(idx, key=lambda i: p[i], reverse=True)[:n]


def drop_sidebands(f, p, idx, f_batch, n_max=3, tol_hz=None, max_ratio=0.2):
    """Removes from `idx` every peak within tol_hz of (a stronger kept
    peak) +/- k * f_batch, k = 1..n_max, AND weaker than max_ratio times
    that peak: the batch-gap copies of a mode. The default tolerance, 20%
    of f_batch (~3 Hz), allows for peak positions wandering by a few bins
    on single captures.

    The ratio test keeps real modes that happen to sit at a copy position.
    The 84-on / 16-off read pattern puts each copy at ~3% of its parent's
    PSD (|sinc|^2 of the duty cycle; 2-5% measured on the 62 Hz ring-down
    mode), so a peak above 20% of its parent cannot be a copy of it. Without
    this test the bolted 146 Hz edge mode (75% of the 117 Hz peak, 37% of
    the 162 Hz one) was dropped. Returns the kept indices, highest first."""
    if tol_hz is None:
        tol_hz = max(2 * (f[1] - f[0]), 0.2 * f_batch)
    kept = []
    for i in sorted(idx, key=lambda i: p[i], reverse=True):
        is_copy = np.isfinite(f_batch) and any(
            abs(f[i] - (f[j] + s * k * f_batch)) <= tol_hz and p[i] < max_ratio * p[j]
            for j in kept for k in range(1, n_max + 1) for s in (-1, 1))
        if not is_copy:
            kept.append(i)
    return kept
