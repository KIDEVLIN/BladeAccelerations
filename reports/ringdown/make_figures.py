#!/usr/bin/env python3
"""
make_figures.py

Figures for ringdown.tex. Run from the repo root:
    python3 reports/ringdown/make_figures.py

Reads the two tip-flick ring-down runs of 21 Oct 2026 (with and without the
mount bolt) and, for the comparison with the experiments, the production
summaries written by extract_run_summary.py (1.4+). Writes PDFs into
reports/ringdown/figs/ and prints the numbers quoted in the text.

Spectra are as in plot_ringdown_psd.py: samples on the measured 1493 Hz
grid, dropped samples masked, Welch nperseg 2048 from the flick onset,
median over the trials of one excitation. Peaks are top_peaks with the
batch-gap copies of stronger peaks dropped. Sensor 4 (nearest the tip).
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
import plot_ringdown_psd as P  # noqa: E402
from accel_spectra import MEASURED_ODR_HZ, drop_sidebands, top_peaks  # noqa: E402
import ringdown_damping as D  # noqa: E402

RUNS = {"bolted": ROOT / "data" / "DTU10MW_rigid4k_ringdown",
        "no bolt": ROOT / "data" / "DTU10MW_rigid4k_ringdown_NOBOLT"}
SUMM = ROOT / "data" / "Production_summaries"
SLOW = ["DTU10MW_390psi_300fs_v1", "DTU10MW_707psi_300fs", "DTU10MW_605psi_300fs_v1"]
# Fastest run at each of the same three pressures (605 psi only went to 700 fs).
FAST = ["DTU10MW_385psi_800fs", "DTU10MW_704psi_800fs", "DTU10MW_605psi_700fs_v1"]
OUT = Path(__file__).resolve().parent / "figs"
NPERSEG = 2048
SENSOR = "s4"
AXIS = {"y": "Edgewise ($y$)", "z": "Flapwise ($z$)"}
COLOR = {"bolted": "C0", "no bolt": "C3"}
FMAX = 450


def median_psds(run_dir):
    """{label: (f, {col: median PSD}, f_batch, n_trials)} and the rest PSD (f, psds)."""
    md = json.loads((run_dir / "run_metadata.json").read_text())
    by, rest = {}, None
    for tr in md["trials"]:
        csv = run_dir / Path(tr["accel_csv"].replace("\\", "/")).name
        acc, _ = P.read_accel(csv, md["accel_mg_per_lsb"], MEASURED_ODR_HZ)
        onset = None if tr["is_rest"] else tr["qc"].get("onset_s")
        f, psds = P.trial_psd(acc, MEASURED_ODR_HZ, onset, NPERSEG)
        if tr["is_rest"]:
            rest = (f, psds)
        else:
            by.setdefault(tr["label"], []).append((f, psds, acc.attrs["batch_hz"]))
    out = {}
    for label, trials in by.items():
        f0 = trials[0][0]
        med = {c: np.median([np.interp(f0, f, p[c]) for f, p, _ in trials], axis=0) for c in trials[0][1]}
        out[label] = (f0, med, float(np.median([b for _, _, b in trials])), len(trials))
    return md, out, rest


def modes(f, p, f_batch, n=4):
    return [int(i) for i in drop_sidebands(f, p, top_peaks(f, p), f_batch)[:n]]


def fig_spectra(data, cfgs, name):
    """Rows: flap flicks, edge flicks. Columns: edgewise y, flapwise z. One curve per configuration in cfgs."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for r, label in enumerate(("flap", "edge")):
        for c, ax_name in enumerate(("y", "z")):
            ax = axes[r, c]
            col = f"{SENSOR}_{ax_name}"
            for cfg in cfgs:
                f, med, f_batch, n = data[cfg][1][label]
                ax.semilogy(f, med[col], color=COLOR[cfg], lw=1, label=f"{cfg} ({n} flicks)")
                for k in modes(f, med[col], f_batch, 3):
                    if f[k] <= FMAX:
                        ax.plot(f[k], med[col][k], "v", color=COLOR[cfg], ms=5)
                        ax.annotate(f"{f[k]:.0f}", (f[k], med[col][k]), xytext=(3, 3),
                                    textcoords="offset points", fontsize=8, color=COLOR[cfg])
            rf, rp = data[cfgs[0]][2]
            ax.semilogy(rf, rp[col], color="0.6", ls="--", lw=0.8, label="rest (noise floor)")
            ax.set_xlim(0, FMAX)
            ax.grid(True, which="both", alpha=0.25)
            if r == 0:
                ax.set_title(AXIS[ax_name])
            if c == 0:
                ax.set_ylabel(f"{label.capitalize()} flicks\nPSD ($g^2$/Hz)")
            if r == 1:
                ax.set_xlabel("Frequency (Hz)")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def peak_psd(f, p, fm, w=1.5):
    """Largest PSD within w Hz of fm."""
    return p[np.abs(f - fm) < w].max()


# Which peak to follow in which flick and axis, for the root-motion figure.
SHAPES = {"bolted": [("62 Hz", 62, "flap", "z"), ("117 Hz", 117, "edge", "y"),
                     ("146 Hz", 146, "edge", "y"), ("162 Hz", 162, "flap", "z")],
          "no bolt": [("55 Hz", 55.4, "flap", "z"), ("85 Hz", 85, "edge", "y"),
                      ("131 Hz", 131, "flap", "z"), ("138 Hz", 138, "flap", "z")]}


def span_shapes(ax, data, cfg):
    """Amplitude of each SHAPES peak along the span, S1..S4, relative to S4."""
    d = data[cfg][1]
    for i, (name, fm, label, a) in enumerate(SHAPES[cfg]):
        f, med, _, _ = d[label]
        amp = np.sqrt([peak_psd(f, med[f"s{s}_{a}"], fm) for s in (1, 2, 3, 4)])
        amp /= amp[-1]
        print(f"  {cfg:8s} {name:9s} {fm:6.1f} Hz ({label} flicks, {a}): " + " ".join(f"{v:.2f}" for v in amp))
        ax.plot([1, 2, 3, 4], amp, "-" if a == "y" else "--", marker="o", ms=4, color=f"C{i}",
                label=f"{name} ({a})")
    ax.set_xticks([1, 2, 3, 4], ["S1\nroot", "S2", "S3", "S4\ntip"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Amplitude relative to S4")
    ax.set_title(cfg, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)


def fig_root(data):
    """root_motion.pdf: bolted edge-flick edgewise PSD at the root (S1) and tip (S4), and the bolted span
    shapes. span_compare.pdf: span shapes with and without the bolt."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    f, med, _, _ = data["bolted"][1]["edge"]
    for s, c in (("s1", "C1"), ("s4", "C0")):
        ax.semilogy(f, med[f"{s}_y"], color=c, lw=1, label=f"{s.upper()} ({'root' if s == 's1' else 'tip'})")
    for fm in (117, 146, 162):
        ax.axvline(fm, color="0.5", ls=":", lw=0.8)
        ax.text(fm, 1.5e-4, f" {fm}", fontsize=8, color="0.3")
    ax.set_xlim(40, 200)
    ax.set_ylim(1e-7, 3e-4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD ($g^2$/Hz)")
    ax.set_title("Bolted, edge flicks, edgewise ($y$)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    print("\n== root motion: sqrt(PSD) of S1..S4 relative to S4")
    span_shapes(axes[1], data, "bolted")
    fig.savefig(OUT / "root_motion.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, constrained_layout=True)
    for ax, cfg in zip(axes, ("bolted", "no bolt")):
        span_shapes(ax, data, cfg)
    fig.savefig(OUT / "span_compare.pdf")
    plt.close(fig)


def quiet_psds(runs):
    """Median S4 y/z PSD over the quietest third of angles of each production run."""
    out = {}
    for run in runs:
        ang = pd.read_csv(SUMM / run / "angle_summary.csv")
        npz = np.load(SUMM / run / "spectra.npz")
        ang = ang[[f"psd_accel_axes_{int(i)}" in npz.files for i in ang["index"]]]
        q = ang.nsmallest(max(3, len(ang) // 3), "nvar_s4")
        f = npz[f"f_accel_{int(q['index'].iloc[0])}"]
        P_ = {a: np.median([npz[f"psd_accel_axes_{int(i)}"][3, j] for i in q["index"]], axis=0)
              for j, a in ((1, "y"), (2, "z"))}
        info = json.loads((SUMM / run / "run_info.json").read_text())
        out[run] = (f, P_, len(q), info)
    return out


def fig_production(slow, table, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, constrained_layout=True)
    for i, (run, (f, P_, nq, info)) in enumerate(slow.items()):
        for ax, a in zip(axes, ("y", "z")):
            ax.semilogy(f, P_[a], color=("C1", "C2", "C4")[i], lw=1,
                        label=f"{run.replace('DTU10MW_', '')} ({info['density_kg_m3']:.0f} kg/m$^3$, "
                              f"{info['velocity_m_s']:.1f} m/s)")
    for ax, a in zip(axes, ("y", "z")):
        for cfg, ls in (("bolted", ":"), ("no bolt", "--")):
            for fm in table[cfg]:
                if fm <= 350:
                    ax.axvline(fm, color=COLOR[cfg], ls=ls, lw=0.9)
        ax.set_xlim(20, 350)
        ax.set_ylim(3e-6, 3e-2)          # same scale for the slow and fast figures
        ax.set_title(AXIS[a])
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("PSD ($g^2$/Hz)")
    axes[1].plot([], [], color=COLOR["no bolt"], ls="--", label="no-bolt ring-down modes")
    axes[1].plot([], [], color=COLOR["bolted"], ls=":", label="bolted ring-down modes")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


class DampingArgs:
    snr, min_reads = 5.0, 5                     # ringdown_damping.py defaults


def damping(data_runs):
    """First flap and edge mode damping per run (ringdown_damping.py), and a synthetic check of the method."""
    out = {}
    for cfg, run in data_runs.items():
        df, ex = D.analyse(run, DampingArgs)
        out[cfg] = (df, ex)
        print(f"\n== damping, {cfg}")
        for mode, t in df.groupby("mode"):
            ok = t.dropna(subset=["zeta"])
            z = ok.zeta * 100
            print(f"  {mode}: f {t.f_hz.iloc[0]:.1f} Hz, {len(ok)}/{len(t)} flicks, zeta median {z.median():.2f}% "
                  f"(IQR {z.quantile(.25):.2f}-{z.quantile(.75):.2f}, range {z.min():.2f}-{z.max():.2f}), "
                  f"reads {ok.n_reads.median():.0f}, R2 {ok.r2.median():.3f}, "
                  f"early {ok.zeta_early.median() * 100:.2f}% / late {ok.zeta_late.median() * 100:.2f}%, "
                  f"fit window {ok.t_start.median():.2f}-{ok.t_end.median():.2f} s after onset, "
                  f"amplitude {ok.peak_amp.median():.2f} g down to {DampingArgs.snr * ok.noise.median():.3f} g")
    # synthetic check: known zeta on the real read/gap timing of a bolted flap capture
    run = RUNS["bolted"]
    md = json.loads((run / "run_metadata.json").read_text())
    tr = next(t for t in md["trials"] if t["label"] == "flap")
    acc, raw = D.read_trial(run, md, tr)
    t, on = acc["time"].to_numpy(), tr["qc"]["onset_s"]
    rng = np.random.default_rng(1)
    print("\n== damping method, synthetic check (20 realizations each; real read/gap timing)")
    for zeta, f in ((0.005, 62.0), (0.013, 62.0), (0.02, 55.4)):
        est = []
        for _ in range(20):
            dt = np.clip(t - on, 0, None)
            x = (5 * np.exp(-zeta * 2 * np.pi * f * dt) * np.cos(2 * np.pi * f * t + rng.uniform(0, 6.3))
                 + 1.5 * np.exp(-0.01 * 2 * np.pi * 117 * dt) * np.cos(2 * np.pi * 117 * t + 1)
                 + 2.0 * np.exp(-0.012 * 2 * np.pi * 162 * dt) * np.cos(2 * np.pi * 162 * t)) * (t >= on) \
                + 0.03 * rng.standard_normal(len(t))
            a2, r2 = acc.copy(), raw.copy()
            a2[f"{SENSOR}_z"], r2[f"{SENSOR}_z"] = x, 0
            env = D.read_amplitudes(a2, r2, f"{SENSOR}_z", [f, 117.0, 162.0])
            est.append(D.fit_decay(env, f, on, 0.03 * np.sqrt(2 / 84), 5.0, 5).get("zeta", np.nan))
        est = np.array(est) * 100
        print(f"  true {zeta * 100:.2f}% at {f:g} Hz (with 117 and 162 Hz modes, noise): recovered median "
              f"{np.nanmedian(est):.3f}%, range {np.nanmin(est):.3f}-{np.nanmax(est):.3f}%")
    return out


def damping_example(ax, res, cfg):
    """Mode amplitude per read after one flap flick, and the exponential fit."""
    df, ex = res[cfg]
    trial, env, r, onset, noise = ex["1st flap"]
    tt_, a = env[:, 0] - onset, env[:, 1]
    ax.semilogy(tt_, a, ".", color="0.6", label="each read")
    ax.semilogy(tt_[r["use"]], a[r["use"]], "o", color=COLOR[cfg], ms=4, label="used in fit")
    tt, pfit = r["fit"]
    ax.semilogy(tt - onset, np.exp(np.polyval(pfit, tt)), color="k", lw=1.2,
                label=rf"fit, $\zeta$ = {r['zeta'] * 100:.2f}%")
    ax.axhline(noise * DampingArgs.snr, color="0.4", ls=":", lw=1, label="5 x noise floor")
    ax.set_xlim(-0.3, 1.6)
    ax.set_xlabel("Time after flick onset (s)")
    ax.set_ylabel("Mode amplitude in each read (g)")
    ax.set_title(f"{cfg}: 1st flap mode, flick {trial}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


def damping_scatter(ax, res, cfgs):
    """Damping ratio of every flap flick, one column per configuration."""
    for i, cfg in enumerate(cfgs):
        z = res[cfg][0].query("mode == '1st flap'").dropna(subset=["zeta"]).zeta * 100
        ax.scatter(np.full(len(z), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(z)), z, s=14,
                   color=COLOR[cfg], alpha=0.8)
        ax.hlines(z.median(), i - 0.25, i + 0.25, color="k", lw=2)
    ax.set_xticks(range(len(cfgs)), cfgs)
    ax.set_xlim(-0.6, len(cfgs) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"1st flap damping ratio $\zeta$ (%)")
    ax.set_title("Every flap flick (bar: median)", fontsize=10)


def fig_damping(res, cfgs, name):
    """Example fit for the last configuration in cfgs, and zeta of every flap flick of each."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 0.6 + 0.2 * len(cfgs)]))
    damping_example(axes[0], res, cfgs[-1])
    damping_scatter(axes[1], res, cfgs)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data, table = {}, {}
    for cfg, run in RUNS.items():
        md, d, rest = median_psds(run)
        data[cfg] = (md, d, rest)
        print(f"\n== {cfg}: {run.name}  {md['start_time']} .. {md.get('end_time')}  "
              f"mounting {md['mounting_angle_deg']} deg, sweep {md['sweep_angle_deg']} deg")
        found = set()
        for label, ax_name in (("edge", "y"), ("flap", "z")):
            f, med, f_batch, n = d[label]
            col = f"{SENSOR}_{ax_name}"
            ks = modes(f, med[col], f_batch)
            print(f"  {label} flicks: {n} trials, read cycle {f_batch:.2f} Hz, df {f[1] - f[0]:.2f} Hz; "
                  f"{col} peaks: " + ", ".join(f"{f[k]:.1f}" for k in ks))
            # The same mode found in both flick types lands a bin or a few apart (62/63, 82/85): keep one.
            for k in ks:
                if all(abs(f[k] - g) > 4 for g in found):
                    found.add(round(f[k]))
        # Which axis each peak lives in, and which flick excites it: S4 PSD at the peak, y and z, both flicks.
        print("  peak   flap-flick y / z      edge-flick y / z   (g^2/Hz)")
        for fm in sorted(found):
            vals = []
            for label in ("flap", "edge"):
                f, med, _, _ = d[label]
                k = int(np.argmin(np.abs(f - fm)))
                w = slice(max(k - 2, 0), k + 3)
                vals.append(f"{med[f'{SENSOR}_y'][w].max():.1e} / {med[f'{SENSOR}_z'][w].max():.1e}")
            print(f"  {fm:4d}   {vals[0]:20s}  {vals[1]}")
        table[cfg] = sorted(found)
    fig_spectra(data, ["bolted"], "ringdown_bolted")
    fig_spectra(data, ["bolted", "no bolt"], "ringdown_compare")
    fig_root(data)
    for name, runs in (("production_slow", SLOW), ("production_fast", FAST)):
        psds = quiet_psds(runs)
        print(f"\n== {name}: quietest third of angles, S4 peaks (no copy removal needed for the two largest):")
        for run, (f, P_, nq, info) in psds.items():
            pk = {a: sorted(round(f[i]) for i in top_peaks(f, P_[a], 4, fmin=40, min_prom=0.15) if f[i] < 350)
                  for a in "yz"}
            amp = {a: {fm: P_[a][np.abs(f - fm) < 4].max() for fm in (85, 110, 131)} for a in "yz"}
            print(f"  {run}: {nq} angles, rho {info['density_kg_m3']:.1f}, U {info['velocity_m_s']:.2f}  "
                  f"y {pk['y']}  z {pk['z']}  |  PSD at 85/110/131 Hz  y "
                  + " ".join(f"{v:.1e}" for v in amp["y"].values()) + "  z "
                  + " ".join(f"{v:.1e}" for v in amp["z"].values()))
        fig_production(psds, table, name)
    res = damping(RUNS)
    fig_damping(res, ["bolted"], "damping_bolted")
    fig_damping(res, ["bolted", "no bolt"], "damping_compare")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
