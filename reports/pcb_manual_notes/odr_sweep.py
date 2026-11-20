import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts/data_analysis/ringdown"))
import plot_ringdown_psd as P
RUN = ROOT / "data/DTU10MW_rigid4k_ringdown"
md = json.loads((RUN / "run_metadata.json").read_text())
FG = np.arange(0, 750, 0.25)
trials = [t for t in md["trials"] if t["label"] in ("flap", "edge")]
for odr in [1450, 1475, 1493, 1510, 1525, 1550, 1600]:
    out = {"flap": [], "edge": []}
    for tr in trials:
        csv = RUN / Path(tr["accel_csv"].replace("\\", "/")).name
        acc, rate = P.read_accel(csv, md["accel_mg_per_lsb"], odr)
        f, p = P.trial_psd(acc[["s4_z", "s4_y", "grid", "time"]], odr, tr["qc"]["onset_s"], 2048)
        ch = "s4_z" if tr["label"] == "flap" else "s4_y"
        out[tr["label"]].append(np.interp(FG, f, p[ch]))
    for lab, v in out.items():
        m = np.median(v, axis=0)
        def pk(lo, hi):
            b = (FG > lo) & (FG < hi); return FG[b][np.argmax(m[b])], m[b].max()
        if lab == "flap":
            f1, a1 = pk(40, 100)
            # comb-copy leakage: power at f1 +/- 14.9 relative to f1
            cp = [m[np.argmin(abs(FG - (f1 + k * 14.86)))] / a1 for k in (-2, -1, 1, 2)]
            print(f"odr {odr} flap: f1={f1:.2f} copies(-2,-1,+1,+2)=" + " ".join(f"{c:.3f}" for c in cp)
                  + f" | 140-200:{pk(140,200)[0]:.1f} 250-300:{pk(250,300)[0]:.1f} 380-470:{pk(380,470)[0]:.1f}")
        else:
            print(f"odr {odr} edge: 90-140:{pk(90,140)[0]:.1f} 280-340:{pk(280,340)[0]:.1f} 520-620:{pk(520,620)[0]:.1f}")
