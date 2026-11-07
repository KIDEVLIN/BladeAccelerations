"""
accel_timing.py

Gap-aware timing for AF10R1 accelerometer captures, and the diagnostics used
to decide what that timing is.

The problem
-----------
main.collect_accel_for_duration() reads the PCB in bulk reads of at most 84
sample-sets (BULK_MAX_SETS) and stamps every row of a read with one time, the
moment the read finished (the CSV "time" column). In the October 2026 data
every read returned the full 84 sets and reads came every ~62-79 ms, while the
sensors produce ~103 sets in that time. So ~19 sets per read never reach the
file (see reports: accel_postmortem, I. Upfal, 3 Nov 2026).

What that means for timing:
  * the 84 rows of one read are CONSECUTIVE sensor samples, spaced 1/fs with
    fs the real sensor rate (~1493 Hz, not the 1600 Hz ODR setting);
  * between reads there is a GAP of missing samples;
  * spreading each read evenly between read times (the old run_loader
    reconstruction, "even" below) stretches every read and shifts every
    frequency low by delivered_rate/fs (~19%) and adds sideband copies.

Model used here
---------------
Read b holds n_b consecutive samples. Its LAST sample is placed at the time
stamp of read b-lag (lag = 0: the read's own stamp, which is what the
postmortem assumes) and the earlier samples at 1/fs steps back from there:

    slot of sample k of read b = round(T[b-lag] * fs) - (n_b - 1 - k)

Slots are integers on a grid of spacing 1/fs; slots that hold no sample are
the gaps. If a read would overlap the previous one (time-stamp jitter) it is
pushed forward to start right after it, and the count is reported.

A constant delay between "sample taken" and "read stamped" (transfer time,
or a deeper buffer) does not matter for spectra, only for phase against the
load cell. Which read anchors a block (lag) does matter for high frequencies
when read intervals vary, so estimate_timing() measures it from the data
rather than assuming it.

Independent check of fs and of the gaps (estimate_timing)
---------------------------------------------------------
For a signal dominated by one mode (ring-down, or a strong vibration peak):
  1. Inside each read the samples are consecutive, so a sinusoid fit gives
     the mode frequency in CYCLES PER SAMPLE, c = f0 / fs. This uses no time
     stamps at all.
  2. From one read to the next, the mode phase must advance by
     2*pi*f0*(time between the two blocks). The time between blocks comes from
     the read time stamps (perf_counter), so scanning f0 in HERTZ and the lag
     and picking the most coherent combination gives f0 in Hz using no
     assumed sample rate.
  3. fs = f0 / c.
If the samples were really continuous and evenly spread between reads (the
original assumption), step 3 returns the delivered rate (~1205-1250 Hz). If
they are consecutive at the sensor rate with gaps, it returns ~1493 Hz.

Requires: numpy, scipy
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import get_window

DEFAULT_SENSOR_FS_HZ = 1493.0   # vendor measurement at ODR=1600 (AF10R1 UG sec 5.1); check with estimate_timing()

# --------------------------------------------------
# Timing used by the processing pipeline (run_loader.load_accel_csv).
# UPDATE these from check_accel_timing.py's timing_summary.txt:
#   SENSOR_FS_HZ <- "best sensor rate fs" of the ring-downs
#   BLOCK_LAG    <- best lag of the production captures (0 = postmortem assumption)
# Only valid for captures taken at ODR = 1600 Hz.
# --------------------------------------------------
SENSOR_FS_HZ = DEFAULT_SENSOR_FS_HZ
BLOCK_LAG = 0
BUFFER_CAPACITY = 1024          # AF10R1 ring buffer, sample-sets (UG 1.1)

# What the Nov 2026 checks found on the Oct 2026 ring-downs (check_accel_timing.py --buffer):
#   * each read is an unbroken run of consecutive samples (in-read contiguity 0.98-0.99 at every row)
#   * after ~3.8 s, samples are lost BETWEEN reads (boundary score -0.42, as predicted for ~16 lost/read)
#   * before that, reads join up: the ring buffer is still filling (RESET_BUF + START just before t = 0)
#   * fs 1482-1491 Hz from the data, consistent with the UG's 1493 Hz (5-min COUNTER measurement)
#   * "fill then overwrite" fits better than "gaps from t = 0" (0.72 vs 0.56); not ~1, so each read's
#     placement after overflow is uncertain by up to ~1.5 ms (probably USB delay before the stamp)
# So the gap-aware time base holds only once the buffer has overflowed; spectra skip the reads before
# latest_overflow_onset(). Samples after overflow are ~BUFFER_CAPACITY / fs (~0.69 s) older than their stamp.


# --------------------------------------------------
# Reads (batches)
# --------------------------------------------------

@dataclass
class Reads:
    start: np.ndarray      # first row index of each read
    length: np.ndarray     # rows in each read
    t: np.ndarray          # read time stamp (s, the CSV "time" value)

    @property
    def n(self):
        return len(self.start)


def find_reads(batch_time):
    """Groups consecutive rows that share one time stamp into reads.
    batch_time: the RAW CSV "time" column (one value per read, repeated)."""
    bt = np.asarray(batch_time, dtype=float)
    if bt.size == 0:
        return Reads(np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    edges = np.flatnonzero(np.diff(bt) != 0) + 1
    start = np.concatenate([[0], edges])
    length = np.diff(np.concatenate([start, [bt.size]]))
    return Reads(start, length, bt[start])


def read_stats(reads, fs=DEFAULT_SENSOR_FS_HZ, max_sets=84):
    """Summary numbers that show whether samples are being lost."""
    if reads.n < 2:
        return dict(n_reads=reads.n)
    dt = np.diff(reads.t)
    dur = reads.t[-1] - reads.t[0]
    delivered = reads.length[1:].sum() / dur if dur > 0 else np.nan
    return dict(
        n_reads=int(reads.n),
        n_rows=int(reads.length.sum()),
        frac_reads_full=float(np.mean(reads.length[1:] >= max_sets)),
        read_len_median=float(np.median(reads.length)),
        read_dt_median_ms=float(1e3 * np.median(dt)),
        read_dt_p05_ms=float(1e3 * np.percentile(dt, 5)),
        read_dt_p95_ms=float(1e3 * np.percentile(dt, 95)),
        delivered_rate_hz=float(delivered),
        read_rate_hz=float(1.0 / np.median(dt)),
        lost_frac_at_fs=float(1.0 - delivered / fs) if fs else np.nan,
        sets_produced_per_read=float(np.median(dt) * fs),
    )


# --------------------------------------------------
# Time bases
# --------------------------------------------------

def even_times(reads):
    """The OLD reconstruction (run_loader before Nov 2026): each read's rows
    evenly spread over (previous read time, this read time]; first read from 0.
    Kept only for comparison."""
    out = np.empty(int(reads.length.sum()))
    prev = 0.0
    for s, n, t in zip(reads.start, reads.length, reads.t):
        out[s:s + n] = prev + (t - prev) / n * np.arange(1, n + 1)
        prev = t
    return out


def _anchor_times(reads, lag):
    """Time stamp that anchors the last sample of each read: T[b - lag].
    The first `lag` reads have no earlier stamp; extrapolate back by the
    median read interval."""
    t = reads.t
    if lag == 0 or reads.n == 0:
        return t.copy()
    dt = np.median(np.diff(t)) if reads.n > 1 else 0.0
    head = t[0] - dt * np.arange(lag, 0, -1)
    return np.concatenate([head, t])[: reads.n]


def gap_aware_slots(reads, fs=DEFAULT_SENSOR_FS_HZ, lag=0):
    """Integer grid slot (spacing 1/fs) for every row. Returns (slots, info).
    time of a row = slots / fs (seconds on the same clock as the CSV time)."""
    anchor = _anchor_times(reads, lag)
    end = np.rint(anchor * fs).astype(np.int64)
    n_rows = int(reads.length.sum())
    slots = np.empty(n_rows, dtype=np.int64)
    prev_end = None
    overlaps = 0
    gaps = []
    for b in range(reads.n):
        n = int(reads.length[b])
        first = end[b] - (n - 1)
        if prev_end is not None:
            if first <= prev_end:
                overlaps += 1
                first = prev_end + 1
            gaps.append(first - prev_end - 1)
        s = reads.start[b]
        slots[s:s + n] = first + np.arange(n)
        prev_end = first + n - 1
    gaps = np.asarray(gaps)
    info = dict(fs_hz=fs, lag=lag, overlaps=overlaps,
                gap_median=float(np.median(gaps)) if gaps.size else np.nan,
                coverage=float(n_rows / (slots[-1] - slots[0] + 1)) if n_rows else np.nan)
    return slots, info


def gap_aware_times(batch_time, fs=DEFAULT_SENSOR_FS_HZ, lag=0):
    """Convenience: raw CSV time column -> (per-row time in s, slots, info)."""
    reads = find_reads(batch_time)
    slots, info = gap_aware_slots(reads, fs, lag)
    return slots / fs, slots, info


# --------------------------------------------------
# Spectra on a grid with gaps
# --------------------------------------------------

def masked_welch(x, slots, fs, nperseg=2048, noverlap=None, window="hann", min_coverage=0.5):
    """Welch PSD (one-sided, density, constant detrend per segment) of samples
    sitting on an integer grid with missing slots.

    x: (N,) or (N, C) samples; slots: (N,) increasing integer grid positions.
    Missing slots are zero after removing the segment mean of the present
    samples, and each segment is normalised by the window energy over the
    PRESENT samples only, so a white signal keeps its level and the integral
    of the PSD still equals the variance. With no gaps this equals
    scipy.signal.welch(..., detrend="constant").

    The regular gap pattern still convolves every peak with the gap pattern's
    spectrum: expect weak copies at +-k * read rate around strong peaks.

    Returns f (nf,), psd (nf,) or (C, nf), n_segments used.
    """
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    slots = np.asarray(slots, dtype=np.int64)
    s0 = slots[0]
    L = int(slots[-1] - s0 + 1)
    grid = np.zeros((L, x.shape[1]))
    mask = np.zeros(L, dtype=bool)
    grid[slots - s0] = x
    mask[slots - s0] = True

    nperseg = int(min(nperseg, L))
    step = nperseg - (nperseg // 2 if noverlap is None else int(noverlap))
    w = get_window(window, nperseg)
    nf = nperseg // 2 + 1
    acc = np.zeros((x.shape[1], nf))
    used = 0
    for a in range(0, L - nperseg + 1, step):
        m = mask[a:a + nperseg]
        if m.mean() < min_coverage:
            continue
        seg = grid[a:a + nperseg]
        mu = seg[m].mean(axis=0)
        seg = (seg - mu) * m[:, None]
        X = np.fft.rfft(seg * w[:, None], axis=0)
        p = (np.abs(X) ** 2).T / (fs * np.sum((w * m) ** 2))
        if nperseg % 2 == 0:
            p[:, 1:-1] *= 2
        else:
            p[:, 1:] *= 2
        acc += p
        used += 1
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    psd = acc / used if used else np.full_like(acc, np.nan)
    return f, (psd[0] if squeeze else psd), used


def with_gap_breaks(t, y, fs):
    """For line plots: insert NaN where consecutive times are further apart
    than 1.5/fs, so lines are not drawn across the gaps between reads.
    y may be (N,) or (N, C)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    brk = np.flatnonzero(np.diff(t) > 1.5 / fs) + 1
    return np.insert(t, brk, np.nan), np.insert(y, brk, np.nan, axis=0)


def sideband_frequencies(f_peak, read_rate_hz, orders=2):
    """Where gap-induced copies of a peak at f_peak are expected."""
    k = np.arange(-orders, orders + 1)
    k = k[k != 0]
    return f_peak + k * read_rate_hz


# --------------------------------------------------
# Timing diagnostics
# --------------------------------------------------

@dataclass
class TimingEstimate:
    c_cycles_per_sample: float        # mode frequency inside a read, cycles/sample (no time stamps used)
    fs_hz: float                      # sensor rate that best explains the read-to-read phase
    f0_hz: float                      # = c * fs
    lag: int                          # read whose stamp anchors each block (0 = its own stamp)
    coherence: float                  # -1..1, signed read-to-read phase agreement at (fs, lag); 1 = perfect
    coherence_by_lag: dict = field(default_factory=dict)   # best coherence for each lag
    runner_up_fs_hz: float = np.nan   # best fs on a different alias lobe (ambiguity check)
    runner_up_coherence: float = np.nan
    n_pairs: int = 0                  # consecutive read pairs used
    fs_grid: np.ndarray = None
    curves: dict = None               # lag -> coherence over fs_grid (sum these across captures)
    weight: float = 0.0               # = n_pairs (for combining captures)
    delivered_rate_hz: float = np.nan # rows per second in the file
    coherence_at_delivered: float = np.nan  # best-lag coherence if fs were the delivered rate (old even-spacing view)


def _full_blocks(x, reads, block_len):
    """(nb, block_len) matrix of the reads with exactly block_len rows, and
    their read indices."""
    idx = np.flatnonzero(reads.length == block_len)
    X = np.stack([x[reads.start[b]:reads.start[b] + block_len] for b in idx]) if idx.size else \
        np.zeros((0, block_len))
    return X, idx


def _block_fit(X, c):
    """Least-squares a*cos + b*sin (+ mean) at c cycles/sample in each row of X.
    Returns (complex amplitude per block, phase at the block's first sample;
    r2 = fraction of the block's variance the sinusoid explains)."""
    n = X.shape[1]
    k = np.arange(n)
    B = np.column_stack([np.cos(2 * np.pi * c * k), np.sin(2 * np.pi * c * k), np.ones(n)])
    coef, *_ = np.linalg.lstsq(B, X.T, rcond=None)
    resid = X.T - B @ coef
    tot = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1.0 - (resid ** 2).sum(axis=0) / np.where(tot > 0, tot, np.inf)
    return coef[0] - 1j * coef[1], r2       # A cos(wk + phi) = Re(A e^{i phi} e^{iwk})


def _clean_blocks(A, r2, min_r2=0.5, noise_k=0.0):
    """Blocks where the mode clearly stands out: the sinusoid explains at
    least min_r2 of the block's variance. This rejects noise-only reads
    (r2 ~ 0.02 for 84 samples) and the reads right after a flick, where many
    modes ring at once. noise_k > 0 additionally requires the amplitude to
    exceed noise_k times the median block amplitude (off by default: it
    fails for steady vibrations, where every read has the mode)."""
    keep = r2 >= min_r2
    if noise_k > 0:
        keep &= np.abs(A) > noise_k * np.median(np.abs(A))
    return keep


def in_read_frequency(X, c_lo, c_hi, n_grid=2000, normalize=False):
    """Mode frequency in cycles/sample from the power of a block-wise sinusoid
    fit, summed over blocks (refined with a parabola). Uses no time stamps.
    normalize: give every block equal weight (so a few huge blocks cannot dominate)."""
    cs = np.linspace(c_lo, c_hi, n_grid)
    n = X.shape[1]
    k = np.arange(n)
    Xc = X - X.mean(axis=1, keepdims=True)
    if normalize:
        Xc = Xc / np.sqrt((Xc ** 2).sum(axis=1, keepdims=True) + 1e-30)
    E = np.exp(-2j * np.pi * np.outer(k, cs))              # (n, ng)
    P = (np.abs(Xc @ E) ** 2).sum(axis=0)
    j = int(np.argmax(P))
    if 0 < j < n_grid - 1:
        y0, y1, y2 = np.log(P[j - 1:j + 2])
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        return float(cs[j] + d * (cs[1] - cs[0])), cs, P
    return float(cs[j]), cs, P


def dominant_block_mode(Xs, c_min=0.02, c_max=0.45, n_grid=1500):
    """Strongest spectral line across channels, measured inside reads only
    (block-averaged periodogram, no time stamps, no sample rate).
    Xs: list of (nb, block_len) block matrices, one per channel.
    Returns (channel index, c cycles/sample, prominence = peak / median power)."""
    cs = np.linspace(c_min, c_max, n_grid)
    best = (None, np.nan, 0.0)
    for ch, X in enumerate(Xs):
        n = X.shape[1]
        Xc = (X - X.mean(axis=1, keepdims=True)) * np.hanning(n)
        P = (np.abs(Xc @ np.exp(-2j * np.pi * np.outer(np.arange(n), cs))) ** 2).sum(axis=0)
        j = int(np.argmax(P))
        prom = float(P[j] / np.median(P))
        if prom > best[2]:
            best = (ch, float(cs[j]), prom)
    return best


def _pairs(keep, idx):
    """Indices (into the block list) of consecutive reads that are both clean."""
    consec = np.flatnonzero(np.diff(idx) == 1)
    return consec[keep[consec] & keep[consec + 1]]


def _signed_coherence(A, keep, idx, reads, f, lag):
    """Mean of cos(dphi_measured - 2*pi*f*dT) over consecutive clean reads,
    for each frequency f (array); every pair counts equally. dT = time
    between the two blocks under the given lag. Signed: unlike
    |mean exp(i...)|, a constant phase error counts against the model, which
    is what makes it sensitive to f when read intervals barely vary."""
    anchor = _anchor_times(reads, lag)
    start_t = anchor[idx]                                   # block start up to a constant (blocks all one length)
    consec = _pairs(keep, idx)
    if consec.size == 0:
        return np.zeros_like(f), 0
    dphi = np.angle(A[consec + 1] * np.conj(A[consec]))
    dT = start_t[consec + 1] - start_t[consec]
    R = np.cos(dphi[None, :] - 2 * np.pi * f[:, None] * dT[None, :]).mean(axis=1)
    return R, int(consec.size)


def estimate_timing(x, batch_time, band_hz, fs_grid=None, lags=range(0, 16), block_len=84,
                    min_r2=0.5, noise_k=0.0):
    """Estimate the sensor rate and how blocks are anchored, from one capture
    whose signal x is dominated by one mode whose TRUE frequency lies in
    band_hz.

    Step 1 finds the mode in cycles/sample inside reads (no time stamps).
    Step 2 scores every candidate fs in fs_grid (and lag): with f = c * fs,
    the phase must advance by 2*pi*f*dT between consecutive blocks.

    fs_grid defaults to 1150-1700 Hz, wide enough to include the delivered
    rate (~1200-1250 Hz), so the "evenly spaced" hypothesis wins if it is the
    right one. Candidate fs values repeat every read_rate/c Hz (alias
    lobes); runner_up_* says how clearly the best lobe wins. Low-frequency
    modes (ring-downs) give widely spaced lobes, hence an unambiguous fs.
    Lag is only identifiable when read intervals vary.

    Only "clean" reads are used (see _clean_blocks): the mode well above
    the noise and explaining most of the read. n_pairs says how many
    consecutive clean reads the result rests on; below ~50 it is weak."""
    x = np.asarray(x, dtype=float)
    reads = find_reads(batch_time)
    X, idx = _full_blocks(x, reads, block_len)
    if len(idx) < 10:
        raise ValueError(f"only {len(idx)} reads of {block_len} rows; nothing to fit")
    if fs_grid is None:
        fs_grid = np.arange(1150.0, 1700.0, 0.25)
    fs_grid = np.asarray(fs_grid, dtype=float)

    c_lo, c_hi = band_hz[0] / fs_grid[-1], band_hz[1] / fs_grid[0]
    c, _, _ = in_read_frequency(X, c_lo, c_hi)
    A, r2 = _block_fit(X, c)
    keep = _clean_blocks(A, r2, min_r2, noise_k)
    if keep.sum() >= 5:                                     # refine c on the clean reads only, equal weight
        c, _, _ = in_read_frequency(X[keep], c_lo, c_hi, normalize=True)
        A, r2 = _block_fit(X, c)
        keep = _clean_blocks(A, r2, min_r2, noise_k)

    # the sensors cannot produce fewer samples than the file holds: fs >= delivered rate
    delivered = reads.length[1:].sum() / (reads.t[-1] - reads.t[0])
    allowed = fs_grid >= 0.98 * delivered

    curves, by_lag = {}, {}
    best = None
    n_pairs = 0
    for lag in lags:
        R, n_pairs = _signed_coherence(A, keep, idx, reads, c * fs_grid, lag)
        curves[int(lag)] = R
        Ra = np.where(allowed, R, -np.inf)
        j = int(np.argmax(Ra))
        by_lag[int(lag)] = float(R[j])
        if best is None or R[j] > best[2]:
            best = (int(lag), float(fs_grid[j]), float(R[j]))
    lag, fs_best, coh = best

    rr = 1.0 / np.median(np.diff(reads.t))
    lobe = rr / c                                           # fs spacing between alias lobes
    far = allowed & (np.abs(fs_grid - fs_best) > 0.5 * lobe)
    R = curves[lag]
    ru = (float(fs_grid[far][np.argmax(R[far])]), float(R[far].max())) if far.any() else (np.nan, np.nan)

    coh_del = max(float(_signed_coherence(A, keep, idx, reads, np.array([c * delivered]), lg)[0][0])
                  for lg in lags)

    return TimingEstimate(c_cycles_per_sample=c, fs_hz=fs_best, f0_hz=c * fs_best, lag=lag, coherence=coh,
                          coherence_by_lag=by_lag, runner_up_fs_hz=ru[0], runner_up_coherence=ru[1],
                          n_pairs=n_pairs, fs_grid=fs_grid, curves=curves, weight=float(n_pairs),
                          delivered_rate_hz=float(delivered), coherence_at_delivered=coh_del)


def combine_estimates(estimates):
    """Weighted sum of coherence curves from several captures (same fs_grid
    and lags), restricted to fs >= the largest delivered rate.
    Returns (fs_best, lag_best, coherence, {lag: curve})."""
    est = [e for e in estimates if e is not None and e.weight > 0]
    if not est:
        return np.nan, None, np.nan, {}
    grid = est[0].fs_grid
    allowed = grid >= 0.98 * max(e.delivered_rate_hz for e in est)
    W = sum(e.weight for e in est)
    curves = {lag: sum(e.curves[lag] * e.weight for e in est) / W for lag in est[0].curves}
    masked = {k: np.where(allowed, v, -np.inf) for k, v in curves.items()}
    lag = max(masked, key=lambda k: masked[k].max())
    j = int(np.argmax(masked[lag]))
    return float(grid[j]), lag, float(curves[lag][j]), curves


def competing_lobes(fs_grid, curve, fs_min, tol=0.1):
    """Local maxima of a coherence curve (fs >= fs_min) within tol of the best:
    more than one means the sample rate is ambiguous from this data alone."""
    R = np.where(fs_grid >= fs_min, curve, -np.inf)
    best = R.max()
    peaks = [i for i in range(1, len(R) - 1) if R[i] >= R[i - 1] and R[i] >= R[i + 1] and R[i] > best - tol]
    return [(float(fs_grid[i]), float(curve[i])) for i in peaks]


# --------------------------------------------------
# Ring-buffer diagnostics (AF10R1 UG: 1024-set ring buffer, OVERRUN = samples
# lost to buffer overflow; each capture does RESET_BUF + START ~0.1 s before
# t = 0). If the host reads slower than the sensors fill the buffer, nothing
# is lost until the buffer is full (~3-4 s); after that it overflows. What the
# file then holds depends on the firmware:
#   "overwrite"   oldest sets overwritten -> each read is 84 consecutive sets,
#                 gaps BETWEEN reads, data ~capacity/fs old
#   "drop_newest" new sets refused       -> gaps INSIDE reads, at a fixed row
# The first two tests below use no time stamps and no sample rate: only the
# mode frequency in cycles/sample, measured inside reads.
# --------------------------------------------------

def latest_overflow_onset(reads, fs=DEFAULT_SENSOR_FS_HZ, capacity=1024, block_len=84):
    """Latest read by which the ring buffer must have overflowed, assuming it
    was empty at t = 0 (START before t = 0 only makes overflow earlier).
    Returns (read index, its time stamp); (n_reads, inf) if it never must.
    From this read on, the gap-aware (between-read gap) time base holds."""
    occ_min = fs * reads.t - block_len * np.arange(reads.n)
    on = np.flatnonzero(occ_min >= capacity)
    return (int(on[0]), float(reads.t[on[0]])) if on.size else (int(reads.n), float("inf"))


def block_phases(x, batch_time, band_c, block_len=84, min_r2=0.5):
    """Mode fit in every full read. band_c: (lo, hi) cycles/sample.
    Returns dict(c, A, r2, keep, idx, X, reads)."""
    x = np.asarray(x, dtype=float)
    reads = find_reads(batch_time)
    X, idx = _full_blocks(x, reads, block_len)
    c, _, _ = in_read_frequency(X, *band_c)
    A, r2 = _block_fit(X, c)
    keep = _clean_blocks(A, r2, min_r2)
    if keep.sum() >= 5:
        c, _, _ = in_read_frequency(X[keep], *band_c, normalize=True)
        A, r2 = _block_fit(X, c)
        keep = _clean_blocks(A, r2, min_r2)
    return dict(c=c, A=A, r2=r2, keep=keep, idx=idx, X=X, reads=reads)


def boundary_contiguity(bp, block_len=84):
    """For each pair of consecutive clean reads: cos(phase step - 2*pi*c*84).
    = 1 when the second read starts right where the first ended (no samples
    lost between them); scattered when samples were lost. Plot against time
    to see when losses start. Returns (read index of 2nd read, its time stamp, score)."""
    consec = _pairs(bp["keep"], bp["idx"])
    A = bp["A"]
    dphi = np.angle(A[consec + 1] * np.conj(A[consec]))
    score = np.cos(dphi - 2 * np.pi * bp["c"] * block_len)
    b = bp["idx"][consec + 1]
    return b, bp["reads"].t[b], score


def split_contiguity(bp, ks=range(12, 73)):
    """Is each read itself one unbroken run of samples? For every split row k,
    fit the mode separately to rows [0, k) and [k, 84) of clean reads, on one
    index axis, and score cos(phase difference). ~1 for every k when reads
    are unbroken; a dip at the row where a gap sits ("drop_newest").
    Returns (ks, score per k, n blocks used)."""
    X = bp["X"][bp["keep"]]
    c = bp["c"]
    n = X.shape[1]
    out = []
    for k in ks:
        ph = []
        for rows in (np.arange(0, k), np.arange(k, n)):
            B = np.column_stack([np.cos(2 * np.pi * c * rows), np.sin(2 * np.pi * c * rows), np.ones(rows.size)])
            coef, *_ = np.linalg.lstsq(B, X[:, rows].T, rcond=None)
            ph.append(coef[0] - 1j * coef[1])
        out.append(np.mean(np.cos(np.angle(ph[1] * np.conj(ph[0])))) if len(X) else np.nan)
    return np.asarray(list(ks)), np.asarray(out), int(len(X))


def overwrite_starts(reads, fs, K, block_len=84):
    """First true sample index of each read under the "overwrite" policy:
    s_b = max(84*b, fs*T_b - K). K lumps buffer capacity, the START time
    before t = 0 and the read latency (K ~ capacity - fs*t_start_offset).
    K -> -inf is the postmortem model (gaps from the first read);
    K -> +inf is no loss at all."""
    b = np.arange(reads.n)
    return np.maximum(block_len * b, np.rint(fs * reads.t - K)).astype(np.int64)


def fit_overwrite_model(bp, fs_grid, K_grid, block_len=84):
    """Read-to-read phase coherence of the "overwrite" model over (fs, K):
    the phase step between consecutive clean reads must equal
    2*pi*c*(s_b - s_{b-1}). Returns dict with the coherence map (nfs, nK),
    best fs, K, and the two nested limits for comparison: 'postmortem'
    (gaps from the first read, best over fs) and 'no_loss'."""
    reads, A, c = bp["reads"], bp["A"], bp["c"]
    consec = _pairs(bp["keep"], bp["idx"])
    b0, b1 = bp["idx"][consec], bp["idx"][consec + 1]
    dphi = np.angle(A[consec + 1] * np.conj(A[consec]))
    t0, t1 = reads.t[b0], reads.t[b1]
    fs_grid = np.asarray(fs_grid, float)
    K_grid = np.asarray(K_grid, float)
    M = np.empty((fs_grid.size, K_grid.size))
    pm = np.empty(fs_grid.size)
    for i, fs in enumerate(fs_grid):
        s0 = np.maximum(block_len * b0[None, :], np.rint(fs * t0[None, :] - K_grid[:, None]))
        s1 = np.maximum(block_len * b1[None, :], np.rint(fs * t1[None, :] - K_grid[:, None]))
        M[i] = np.cos(dphi[None, :] - 2 * np.pi * c * (s1 - s0)).mean(axis=1) if consec.size else np.nan
        pm[i] = np.cos(dphi - 2 * np.pi * c * np.rint(fs * (t1 - t0))).mean() if consec.size else np.nan
    no_loss = float(np.cos(dphi - 2 * np.pi * c * block_len * (b1 - b0)).mean()) if consec.size else np.nan
    i, j = np.unravel_index(np.nanargmax(M), M.shape) if consec.size else (0, 0)
    return dict(map=M, fs_grid=fs_grid, K_grid=K_grid, fs=float(fs_grid[i]), K=float(K_grid[j]),
                coherence=float(M[i, j]), profile=M.max(axis=1), postmortem_profile=pm,
                postmortem=float(np.nanmax(pm)) if consec.size else np.nan, no_loss=no_loss, n_pairs=int(consec.size))
