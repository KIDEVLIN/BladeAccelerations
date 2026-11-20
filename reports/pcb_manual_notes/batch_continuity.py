# For each flap trial: fit a decaying 62.7 Hz-ish sinusoid per batch and compare
# the phase step between consecutive batches with two hypotheses:
#   contiguous: step = 2*pi*f*84/fs        (no samples lost between batches)
#   gapped:     step = 2*pi*f*T_b          (wall-clock spacing, samples lost)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd()
RUN = ROOT / "data/DTU10MW_rigid4k_ringdown"
md = json.loads((RUN / "run_metadata.json").read_text())
fl = [t for t in md["trials"] if t["label"] == "flap"]
print("onsets:", np.round(sorted(t["qc"]["onset_s"] for t in fl), 2))
F, FS = 62.7, 1500.0
res = []
for tr in fl:
    df = pd.read_csv(RUN / Path(tr["accel_csv"].replace("\\", "/")).name, comment="#")
    t = df["time"].to_numpy(); x = df["s4_z"].to_numpy(float)
    t = t - t[0]
    first = np.flatnonzero(np.r_[True, np.diff(t) > 0]); n = np.diff(np.r_[first, len(t)])
    ph, tb = [], []
    for i0, k in zip(first, n):
        if k != 84: continue
        seg = x[i0:i0+k] - x[i0:i0+k].mean()
        z = np.sum(seg * np.exp(-2j*np.pi*F*np.arange(k)/FS))
        ph.append(z); tb.append(t[i0+k-1])
    ph = np.array(ph); tb = np.array(tb)
    amp = np.abs(ph); good = amp > 0.2*amp.max()
    d = np.angle(ph[1:] * np.conj(ph[:-1]))
    ok = good[1:] & good[:-1]
    dtb = np.diff(tb)
    for td, dd, o in zip(tb[1:], d, ok):
        if o: res.append((td, dd))
    pass
res = np.array(res)
w = lambda a: np.angle(np.exp(1j*a))
pred_c = w(2*np.pi*F*84/FS)
pred_g = w(2*np.pi*F*0.0673)
print(f"predicted step: contiguous {pred_c:+.2f} rad, gapped(67.3ms) {pred_g:+.2f} rad")
for lo, hi in [(0,2),(2,4),(4,6),(6,8),(8,10)]:
    m = (res[:,0]>=lo)&(res[:,0]<hi)
    if m.sum()==0: continue
    c = np.mean(np.exp(1j*res[m,1]))
    print(f"t {lo}-{hi}s: n={m.sum():4d} mean step {np.angle(c):+.2f} rad (coherence {abs(c):.2f})")
