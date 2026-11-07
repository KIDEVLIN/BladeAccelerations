"""
sim_af10r1.py

Simulates what main.collect_accel_for_duration() writes for a known signal,
under different hypotheses about how the AF10R1 hands out samples. Used by
test_accel_timing.py to check that utils/accel_timing.py can tell the
hypotheses apart and recovers the right sample rate and frequency.

Models (block = the 84 rows of one read, T_b = its CSV time stamp):
  "newest"     board keeps only the newest 84 sets at each read; block ends
               `latency` before T_b (postmortem's assumption, lag 0)
  "oldest"     board returns the 84 sets taken right after the previous read
               and drops the rest (lag 1)
  "buffered"   board buffer full: block was taken `depth` reads ago (lag = depth)
  "continuous" no samples lost: blocks are back to back at sensor rate fs,
               read stamps follow the data (the original even-spacing
               assumption is then right)
"""

import numpy as np


def read_times(n_reads, dt_ms=69.0, jitter_ms=4.0, t0=0.07, rng=None):
    rng = np.random.default_rng(rng)
    dt = np.clip(rng.normal(dt_ms, jitter_ms, n_reads), 57.0, 110.0) / 1e3
    return t0 + np.cumsum(dt) - dt[0]


def simulate(signal_fn, model="newest", fs=1493.0, duration=20.0, block=84,
             dt_ms=69.0, jitter_ms=4.0, latency=0.022, depth=12, stamp_jitter_ms=0.3, rng=0):
    """Returns (batch_time column, x rows, true sample time of each row)."""
    rng = np.random.default_rng(rng)
    if model == "continuous":
        n_reads = int(duration * fs / block)
        idx = np.arange(n_reads * block)
        t_true = idx / fs
        T = t_true[block - 1::block] + latency + rng.normal(0, stamp_jitter_ms / 1e3, n_reads)
        bt = np.repeat(T, block)
        return bt, signal_fn(t_true), t_true

    T = read_times(int(duration / (dt_ms / 1e3)), dt_ms, jitter_ms, rng=rng)
    T_meas = T + rng.normal(0, stamp_jitter_ms / 1e3, T.size)
    blocks = []
    for b in range(T.size):
        if model == "newest":
            end = np.floor((T[b] - latency) * fs)
            first = end - block + 1
        elif model == "oldest":
            prev = T[b - 1] if b else T[0] - dt_ms / 1e3
            first = np.ceil((prev - latency) * fs)
        elif model == "buffered":
            a = b - depth
            ref = T[a] if a >= 0 else T[0] - (depth - b) * dt_ms / 1e3
            end = np.floor((ref - latency) * fs)
            first = end - block + 1
        else:
            raise ValueError(model)
        blocks.append(first + np.arange(block))
    idx = np.concatenate(blocks)
    t_true = idx / fs
    bt = np.repeat(T_meas, block)
    return bt, signal_fn(t_true), t_true


def ringdown(f0=62.7, zeta=0.01, t_flick=3.5, amp=2.0, noise=0.01, rng=1):
    rng = np.random.default_rng(rng)

    def fn(t):
        tau = t - t_flick
        x = np.where(tau > 0, amp * np.exp(-zeta * 2 * np.pi * f0 * tau) * np.cos(2 * np.pi * f0 * tau + 0.3), 0.0)
        return x + noise * rng.standard_normal(t.size)
    return fn


def narrowband(f0=180.0, bw=1.0, amp=1.0, noise=0.05, rng=2):
    """Steady random vibration around f0 (like a strong production peak)."""
    rng = np.random.default_rng(rng)

    def fn(t):
        # slowly wandering phase -> peak of width ~bw
        tg = np.arange(0, t.max() + 1, 0.01)
        ph = np.cumsum(rng.standard_normal(tg.size)) * np.sqrt(2 * np.pi * bw * 0.01)
        return amp * np.cos(2 * np.pi * f0 * t + np.interp(t, tg, ph)) + noise * rng.standard_normal(t.size)
    return fn


def write_capture_csv(path, signals, model="newest", fs=1493.0, duration=20.0, rng=0, **kw):
    """Writes a CSV in main.collect_accel_for_duration()'s format.
    signals: dict column name (e.g. "s4_z") -> signal_fn; other columns get noise.
    Values are int16 counts. Returns the true row times."""
    import csv
    cols = [f"s{s}_{a}" for s in range(1, 5) for a in "xyz"]
    rng_ = np.random.default_rng(rng)
    bt, _, t_true = simulate(lambda t: np.zeros(t.size), model, fs=fs, duration=duration, rng=rng, **kw)
    data = {}
    for c in cols:
        x = rng_.normal(0, 3, t_true.size) + (16393 if c.endswith("_z") else 0)   # 1 g on z at 2g/0.061 mg
        if c in signals:
            x = x + signals[c](t_true)
        data[c] = np.clip(np.rint(x), -32768, 32767).astype(int)
    with open(path, "w", newline="") as f:
        f.write("# LIS2DS12 2g mode, 0.061 mg/LSB, raw int16, ODR=1600 Hz, t_start_perf_counter=1234.567890\n")
        w = csv.writer(f)
        w.writerow(["sample", "time"] + cols)
        for i in range(t_true.size):
            w.writerow([i, f"{bt[i]:.4f}"] + [data[c][i] for c in cols])
    return t_true


def realistic_flick(f0=62.7, zeta=0.01, t_flick=3.5, amp=2000.0, noise=8.0, rng=1,
                    transient=((290.0, 0.03, 12000.0), (160.0, 0.02, 4000.0), (650.0, 0.05, 15000.0))):
    """Ring-down as recorded: a large, fast-decaying multi-mode impact (f, zeta, amp)
    on top of the ringing first flap mode. The impact makes the first reads after the
    flick far larger than the flap mode ever is."""
    rng = np.random.default_rng(rng)
    modes = ((f0, zeta, amp),) + tuple(transient)
    phases = rng.uniform(0, 2 * np.pi, len(modes))

    def fn(t):
        tau = np.clip(t - t_flick, -1.0, None)
        x = np.zeros(t.size)
        for (f, z, a), ph in zip(modes, phases):
            x += np.where(tau > 0, a * np.exp(-z * 2 * np.pi * f * tau) * np.cos(2 * np.pi * f * tau + ph), 0.0)
        return x + noise * rng.standard_normal(t.size)
    return fn


def simulate_buffer(signal_fn, policy="overwrite", fs=1493.0, duration=10.0, block=84, capacity=1024,
                    t_start_offset=0.12, dt_ms=67.0, jitter_ms=0.8, snapshot_delay=0.015,
                    stamp_jitter_ms=0.3, rng=0):
    """The AF10R1 as its user guide describes it: a ring buffer of `capacity`
    sample-sets, filled at fs from START, emptied 84 sets per read.
    START happens t_start_offset before the capture's t = 0. Each read takes
    the oldest 84 sets present `snapshot_delay` before its time stamp.
      policy "overwrite":   when full, a new set overwrites the oldest
      policy "drop_newest": when full, new sets are discarded
    Returns (batch_time column, x rows, true sample index of each row, onset read index)."""
    rng = np.random.default_rng(rng)
    T = read_times(int(duration / (dt_ms / 1e3)), dt_ms, jitter_ms, rng=rng)
    T_meas = T + rng.normal(0, stamp_jitter_ms / 1e3, T.size)
    produced = np.floor((T - snapshot_delay + t_start_offset) * fs).astype(int)   # sets made by each read
    rows, onset = [], None
    if policy == "overwrite":
        consumed = 0
        for b, n in enumerate(produced):
            oldest = max(consumed, n - capacity)
            if oldest > consumed and onset is None:
                onset = b
            rows.append(oldest + np.arange(block))
            consumed = oldest + block
    elif policy == "drop_newest":
        stored = []                                   # true indices currently in the buffer, oldest first
        nxt = 0
        for b, n in enumerate(produced):
            while nxt < n:                            # sets produced since the last read
                if len(stored) < capacity:
                    stored.append(nxt)
                elif onset is None:
                    onset = b
                nxt += 1
            rows.append(np.array(stored[:block]))
            del stored[:block]
    else:
        raise ValueError(policy)
    idx = np.concatenate(rows)
    return np.repeat(T_meas, block), signal_fn(idx / fs), idx, onset
