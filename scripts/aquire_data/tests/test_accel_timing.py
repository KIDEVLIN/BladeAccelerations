"""
Tests for utils/accel_timing.py against simulated AF10R1 captures
(tests/sim_af10r1.py). Run from scripts/aquire_data:

    python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from utils import accel_timing as at  # noqa: E402
import sim_af10r1 as sim              # noqa: E402

FS = 1493.0
RINGDOWN = dict(dt_ms=67.0, jitter_ms=0.8, duration=10.0)      # like the 21 Oct ring-downs
PRODUCTION = dict(dt_ms=69.0, jitter_ms=4.0, duration=20.0)    # like the production captures


def peak(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(p[m])]


# ---------- reads and time bases ----------

def test_find_reads_and_stats():
    bt, x, _ = sim.simulate(sim.ringdown(), "newest", fs=FS, **RINGDOWN)
    r = at.find_reads(bt)
    assert (r.length == 84).all()
    st = at.read_stats(r, FS)
    assert st["frac_reads_full"] == 1.0
    assert 1230 < st["delivered_rate_hz"] < 1270
    assert 0.14 < st["lost_frac_at_fs"] < 0.18


def test_gap_aware_slots_match_truth():
    """With the right fs and lag, every block lands on its true samples up to
    one constant offset (the latency), and the gaps have the true length +-1."""
    bt, x, t_true = sim.simulate(sim.ringdown(), "newest", fs=FS, stamp_jitter_ms=0.2, **PRODUCTION)
    _, slots, info = at.gap_aware_times(bt, FS, lag=0)
    true_slots = np.rint(t_true * FS).astype(int)
    off = slots - true_slots
    assert np.abs(off - np.median(off)).max() <= 1
    assert info["overlaps"] == 0
    assert 0.78 < info["coverage"] < 0.84


def test_even_times_is_old_reconstruction():
    bt = np.repeat([0.1, 0.2], 4)
    t = at.even_times(at.find_reads(bt))
    assert np.allclose(t, [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])


# ---------- masked Welch ----------

def test_masked_welch_equals_scipy_without_gaps():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    f1, p1, _ = at.masked_welch(x, np.arange(x.size), 1000.0, nperseg=1024)
    f2, p2 = welch(x, fs=1000.0, nperseg=1024)
    assert np.allclose(f1, f2) and np.allclose(p1, p2)


def test_masked_welch_keeps_variance_with_gaps():
    rng = np.random.default_rng(1)
    bt, x, _ = sim.simulate(lambda t: 3.0 * rng.standard_normal(t.size), "newest", fs=FS, **PRODUCTION)
    _, slots, _ = at.gap_aware_times(bt, FS)
    f, p, _ = at.masked_welch(x, slots, FS, nperseg=2048)
    assert np.trapezoid(p, f) == pytest.approx(np.var(x), rel=0.05)


@pytest.mark.parametrize("f0", [62.7, 180.0])
def test_gap_aware_psd_finds_true_peak_even_does_not(f0):
    sig = sim.ringdown(f0=f0, zeta=0.003) if f0 < 100 else sim.narrowband(f0=f0)
    bt, x, _ = sim.simulate(sig, "newest", fs=FS, **PRODUCTION)
    _, slots, _ = at.gap_aware_times(bt, FS)
    f, p, _ = at.masked_welch(x, slots, FS, nperseg=4096)
    assert abs(peak(f, p, 20, 400) - f0) < 1.0
    # old method: evenly spaced at the delivered rate -> wrong
    r = at.find_reads(bt)
    fe, pe = welch(x - x.mean(), fs=at.read_stats(r)["delivered_rate_hz"], nperseg=4096)
    assert abs(peak(fe, pe, 20, 400) - f0) > 5.0


# ---------- timing diagnostics ----------

@pytest.mark.parametrize("model,lag", [("newest", 0), ("oldest", 1), ("buffered", 12)])
def test_estimate_timing_production_recovers_fs_and_lag(model, lag):
    bt, x, _ = sim.simulate(sim.narrowband(f0=180.0), model, fs=FS, **PRODUCTION)
    e = at.estimate_timing(x, bt, (150, 210))
    assert e.fs_hz == pytest.approx(FS, abs=3)
    assert e.lag == lag
    assert e.coherence > e.coherence_at_delivered + 0.3


@pytest.mark.parametrize("model", ["newest", "oldest"])
def test_estimate_timing_ringdown_recovers_fs(model):
    bt, x, _ = sim.simulate(sim.ringdown(), model, fs=FS, **RINGDOWN)
    e = at.estimate_timing(x, bt, (50, 75))
    assert e.fs_hz == pytest.approx(FS, abs=8)
    assert e.f0_hz == pytest.approx(62.7, abs=0.5)
    assert e.coherence_at_delivered < 0.5 < e.coherence


def test_estimate_timing_says_so_when_even_spacing_is_right():
    """If the board really were continuous (the original assumption), the
    estimate must land on the delivered rate, not 1493."""
    bt, x, _ = sim.simulate(sim.ringdown(), "continuous", fs=1220.0, **RINGDOWN)
    e = at.estimate_timing(x, bt, (50, 75))
    assert e.fs_hz == pytest.approx(1220.0, abs=8)
    assert e.coherence_at_delivered > 0.9


def test_combine_estimates_tightens_fs():
    ests = []
    for seed in range(5):
        bt, x, _ = sim.simulate(sim.ringdown(rng=seed + 10), "newest", fs=FS, rng=seed, **RINGDOWN)
        ests.append(at.estimate_timing(x, bt, (50, 75)))
    fs, lag, coh, _ = at.combine_estimates(ests)
    assert fs == pytest.approx(FS, abs=4)


def test_wrong_lag_smears_high_frequency_peaks_in_production():
    """Why the lag must be measured, not assumed: with production read jitter a
    175 Hz peak placed with the wrong anchor loses most of its height."""
    bt, x, _ = sim.simulate(sim.narrowband(f0=175.0, bw=0.2), "oldest", fs=FS, **PRODUCTION)
    r = at.find_reads(bt)
    heights = {}
    for lag in (0, 1):
        slots, _ = at.gap_aware_slots(r, FS, lag)
        f, p, _ = at.masked_welch(x, slots, FS, nperseg=4096)
        heights[lag] = p.max()
    assert heights[1] > 3 * heights[0]


def test_realistic_flick_is_not_fooled_by_the_impact():
    """A recorded flick starts with a large multi-mode impact. Only reads the
    flap mode dominates may be used, otherwise fs comes out wrong (the
    failure seen on the Oct 2026 ring-downs with the first version)."""
    ests = []
    for k in range(5):
        bt, x, _ = sim.simulate(sim.realistic_flick(rng=k), "oldest", fs=FS, rng=k, **RINGDOWN)
        ests.append(at.estimate_timing(x, bt, (55, 70)))
    assert all(e.n_pairs >= 10 for e in ests)
    fs, lag, coh, _ = at.combine_estimates(ests)
    assert fs == pytest.approx(FS, abs=4) and lag == 1
    assert coh - np.mean([e.coherence_at_delivered for e in ests]) > 0.5


@pytest.mark.parametrize("policy", ["overwrite", "drop_newest"])
def test_ring_buffer_policy_is_identified(policy):
    """AF10R1 ring buffer (UG): no loss until full, then either overwrite-oldest
    (gaps between reads) or drop-newest (gap inside each read)."""
    late, split_min = [], []
    for k in range(6):
        bt, x, idx, onset = sim.simulate_buffer(sim.realistic_flick(t_flick=1.5 + 0.6 * k, rng=k), policy, rng=k)
        bp = at.block_phases(x, bt, (55 / 1700, 70 / 1150))
        b, t, sc = at.boundary_contiguity(bp)
        late += list(sc[b > onset])
        if b.size and (b < onset).any():
            assert np.median(sc[b < onset]) > 0.9          # before the buffer fills nothing is lost
        split_min.append(at.split_contiguity(bp)[1].min())
    if policy == "overwrite":
        assert np.median(late) < 0.0 and min(split_min) > 0.85
    else:
        assert np.median(late) > 0.8 and min(split_min) < 0.7


def test_overwrite_model_recovers_fs_and_onset():
    bt, x, idx, onset = sim.simulate_buffer(sim.realistic_flick(t_flick=2.5, zeta=0.005, rng=4), "overwrite", rng=4)
    bp = at.block_phases(x, bt, (55 / 1700, 70 / 1150))
    fit = at.fit_overwrite_model(bp, np.arange(1470, 1520.1, 0.5), np.arange(300, 1200, 2.0))
    assert fit["fs"] == pytest.approx(FS, abs=4)
    assert fit["coherence"] > fit["postmortem"] and fit["coherence"] > fit["no_loss"] + 0.2
    starts = at.overwrite_starts(at.find_reads(bt), fit["fs"], fit["K"])
    assert np.abs(starts - idx[::84])[:onset].max() <= 1          # reads before the onset placed exactly
