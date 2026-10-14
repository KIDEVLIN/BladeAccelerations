#!/usr/bin/env python3
"""
plot_psd.py

Accelerometer spectra from the per-angle Welch PSDs stored by extract_run_summary.py (version 1.2+:
<summaries>/<run>/spectra.npz).  No raw data is read.  Three modes:

  map      One dataset.  Top: normalized variance vs motor angle.  Bottom: PSD as colour on a
           (motor angle, frequency) plane, sharing the angle axis, so every variance peak can be read
           against the frequencies that make it up (are all peaks at the same frequency? which
           frequency does a peak sit at, and does it move with angle?).
               python plot_psd.py Production_summaries --mode map --run 704/800 --sensor S4 --component z

  spectra  One dataset.  PSD curves at chosen angles overlaid: peak angles vs a quiet angle.
               python plot_psd.py Production_summaries --mode spectra --run 704/800 --component z
                         (auto: the 3 strongest well-separated variance peaks + one typical quiet angle)
               python plot_psd.py Production_summaries --mode spectra --run 704/800 --angles 120 170 200 -50

  runs     Several datasets.  The PSD at each run's own variance peak (or at a fixed angle), coloured by
           Re, Ca_e or U, with a lower panel of the dominant peak frequency against the same quantity
           (flat in Hz -> fixed-frequency/structural; flat in St -> scales with the flow).
               python plot_psd.py Production_summaries --mode runs --run 704 --colorby re --xaxis st
               python plot_psd.py Production_summaries --mode runs --run all --at 170 --colorby ca

First argument: the Production_summaries folder (or the master_angles.csv inside it).
--run takes folder names, short keys PSI/FAN[/VERSION], substrings, or all (same rules as plot_angle_sweep.py).

Options
    --sensor S4            accelerometer (default S4)
    --component {mag,x,y,z}   |a| (default) or one sensor axis (axes need extract_run_summary.py 1.2+)
    --xaxis {freq,st}      frequency in Hz (default) or Strouhal number St = f * c_avg / U, with c_avg =
                           --chord (default 0.0138 m) and U the measured freestream velocity of that run.
                           With st the PSD is converted to a density per unit St (S(St) = S(f) * U / c_avg),
                           so areas under curves still equal variances.
    --norm {ac,raw}        ac (default): PSD / a_c^2, so runs at different conditions are comparable
                           (same normalization as the variance plots); raw: g^2/Hz (g^2 per unit St)
    --fmin / --fmax        frequency range shown and searched for peaks (Hz; default 5 .. 500)
    --flog                 log frequency axis (map: vertical axis; spectra/runs: x axis is always log)
    --smooth N             moving average over N frequency bins before plotting/peak picking in the
                           spectra and runs modes (default 5, about 3 Hz at the default resolution)
    --angles A [A ...]     spectra mode: motor angles to show (nearest measured), or "auto" (default)
    --npeaks N, --sep DEG  auto selection: number of peaks and minimum separation (default 3, 15 deg)
    --at {peak,ANGLE}      runs mode: each run's variance-peak angle (default) or one fixed angle
    --colorby {re,ca,u}    runs mode colouring (default re)
    --track                map mode: mark the dominant-peak frequency at angles whose variance is >= 3x the median
    --vmin / --vmax        map colour limits (default 1st and 99.9th percentile of the PSD)
    --chord M              average chord for St (default 0.0138)
    --out DIR, --dpi N

Outputs (default <summaries>/figures): a PNG and CSVs with the plotted numbers (peak tables; spectra values).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LogNorm, Normalize                  # noqa: E402
from matplotlib.cm import ScalarMappable                          # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.ticker import MultipleLocator                     # noqa: E402
from scipy.ndimage import uniform_filter1d                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_angle_sweep import (INK, MUTED, SURFACE, CAT, resolve, short_name, slug, parse_sensors,  # noqa: E402
                              style_axes, conditions_line, save, clip_matrix)

AXES = ("x", "y", "z")
CMAP = "turbo"
MAX_GAP_DEG = 20.0          # holes in the angle sampling larger than this are left blank in the map


# --------------------------------------------------
# Data
# --------------------------------------------------

def var_col(sensor, comp):
    return f"nvar_s{sensor}" if comp == "mag" else f"nvar_s{sensor}_{comp}"


def load_run(summ, run, sensor, comp):
    d = summ / run
    ang = pd.read_csv(d / "angle_summary.csv")
    col = var_col(sensor, comp)
    if col not in ang.columns:
        raise SystemExit(f"{run}: column {col} missing from angle_summary.csv; rerun extract_run_summary.py "
                         "(version 1.2+ writes per-axis columns and spectra).")
    npz = np.load(d / "spectra.npz")
    keys = set(npz.files)
    clip = clip_matrix(ang, None if comp == "mag" else comp)[:, sensor - 1]
    th, nv, P, cl, f = [], [], [], [], None
    for pos, r in ang.iterrows():
        i = int(r["index"]) if "index" in ang.columns else pos
        if f"f_accel_{i}" not in keys or not np.isfinite(r["motor_angle_deg"]):
            continue
        if comp == "mag":
            p = npz[f"psd_accel_{i}"][sensor - 1]
        else:
            if f"psd_accel_axes_{i}" not in keys:
                raise SystemExit(f"{run}: no per-axis spectra in spectra.npz; rerun extract_run_summary.py (1.2+).")
            p = npz[f"psd_accel_axes_{i}"][sensor - 1, AXES.index(comp)]
        f = npz[f"f_accel_{i}"]
        th.append(r["motor_angle_deg"]); nv.append(r[col]); P.append(np.asarray(p, float)); cl.append(clip[pos])
    if not th:
        raise SystemExit(f"{run}: no spectra found in spectra.npz")
    o = np.argsort(th)
    return dict(th=np.array(th)[o], nv=np.array(nv)[o], P=np.array(P)[o], f=np.asarray(f, float),
                clip=np.array(cl)[o])


def transform(D, info, args, smooth=1):
    """Return x axis (Hz or St) and PSD matrix (n_angles, nf) in the requested normalization."""
    U, ac = info["velocity_m_s"], info["a_c_g"]
    P = uniform_filter1d(D["P"], smooth, axis=-1, mode="nearest") if smooth > 1 else D["P"].copy()
    if args.xaxis == "st":
        x, S = D["f"] * args.chord / U, P * U / args.chord
    else:
        x, S = D["f"], P
    if args.norm == "ac":
        S = S / ac ** 2
    return x, S


def band(D, args):
    return (D["f"] >= args.fmin) & (D["f"] <= args.fmax)


def peak_of(D, S, args):
    """Index of the dominant spectral peak (within the band) for each PSD row."""
    m = band(D, args)
    idx = np.where(m)[0]
    k = idx[np.argmax(S[:, m], axis=1)]
    return k


def comp_tag(args):
    return f"S{args.sensor_id}" + ("" if args.component == "mag" else f"_{args.component}")


def sym_psd(args):
    sub = "|a|" if args.component == "mag" else f"a_{args.component}"
    var = "St" if args.xaxis == "st" else "f"
    num = rf"S_{{{sub}}}({var})"
    if args.norm == "ac":
        unit = "per unit St" if args.xaxis == "st" else "1/Hz"
        return rf"${num}\,/\,a_c^2$   ({unit})"
    unit = "g$^2$ per unit St" if args.xaxis == "st" else "g$^2$/Hz"
    return rf"${num}$   ({unit})"


def xlabel(args):
    return r"Strouhal number  $St = f\,\bar{c}/U$" if args.xaxis == "st" else "Frequency (Hz)"


def band_label(args):
    return "|a|" if args.component == "mag" else f"{args.component}-axis"


def fmt_x(v, args):
    return f"St {v:.3g}" if args.xaxis == "st" else f"{v:.3g} Hz"


# --------------------------------------------------
# Mode: map
# --------------------------------------------------

def fig_map(run, D, info, args, out):
    x, S = transform(D, info, args)
    m = band(D, args)
    th, nv = D["th"], D["nv"]
    # blank the middle of angle holes (targeted sweeps) instead of smearing colour across them
    th2, S2 = [th[0]], [S[0]]
    for k in range(1, len(th)):
        if th[k] - th[k - 1] > MAX_GAP_DEG:
            th2.append(0.5 * (th[k] + th[k - 1])); S2.append(np.full(S.shape[1], np.nan))
        th2.append(th[k]); S2.append(S[k])
    th2, S2 = np.array(th2), np.array(S2)
    vals = S[:, m][np.isfinite(S[:, m]) & (S[:, m] > 0)]
    vmin = args.vmin if args.vmin else np.percentile(vals, 1)
    vmax = args.vmax if args.vmax else np.percentile(vals, 99.9)

    fig = plt.figure(figsize=(11.5, 7.8), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.3], width_ratios=[1, 0.025])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])

    ax1.plot(th, nv, color=CAT[0], lw=1.6, marker="o", ms=4.2, mec=SURFACE, mew=0.6, zorder=3)
    if D["clip"].any():
        ax1.scatter(th[D["clip"]], nv[D["clip"]], s=95, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=6)
    pos = nv[np.isfinite(nv) & (nv > 0)]
    ax1.set_yscale("log"); ax1.set_ylim(pos.min() / 1.6, pos.max() * 1.7)
    ax1.set_ylabel(r"Normalized variance" + "\n" + (r"$\mathrm{Var}(|a|)/a_c^2$" if args.component == "mag"
                                                  else rf"$\mathrm{{Var}}(a_{args.component})/a_c^2$"), color=INK)
    style_axes(ax1)
    ax1.tick_params(labelbottom=False)
    ax1.set_title(conditions_line(run, info) + f"   ·   S{args.sensor_id} {band_label(args)}", loc="left",
                  fontsize=11, color=INK)

    pm = ax2.pcolormesh(th2, x[m], S2[:, m].T, shading="nearest", cmap=CMAP, norm=LogNorm(vmin, vmax),
                        rasterized=True)
    if args.track:
        k = peak_of(D, S, args)
        on = nv >= 3 * np.nanmedian(nv)            # only where the variance is clearly above the quiet level
        ax2.scatter(th[on], x[k][on], s=14, facecolor="white", edgecolor=INK, linewidth=0.7, zorder=5)
    ax2.set_ylim(x[m].min(), x[m].max())
    if args.flog:
        ax2.set_yscale("log")
    ax2.set_ylabel(xlabel(args), color=INK)
    ax2.set_xlabel("Motor angle (deg)", color=INK)
    ax2.set_xlim(th.min() - 6, th.max() + 6)
    ax2.xaxis.set_major_locator(MultipleLocator(30))
    ax2.tick_params(colors=MUTED, which="both")
    for s in ax2.spines.values():
        s.set_color(MUTED)
    cb = fig.colorbar(pm, cax=cax)
    cb.set_label(sym_psd(args), color=INK)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_edgecolor(MUTED)
    save(fig, out, args.dpi)

    k = peak_of(D, S, args)
    pd.DataFrame(dict(motor_angle_deg=th, nvar=nv, peak_x=x[k], peak_hz=D["f"][k],
                      peak_psd=S[np.arange(len(k)), k])).to_csv(out.with_suffix(".csv"), index=False)


# --------------------------------------------------
# Mode: spectra
# --------------------------------------------------

def pick_angles(D, args):
    th, nv = D["th"], D["nv"]
    if args.angles and [a.lower() for a in args.angles] != ["auto"]:
        out = []
        for a in args.angles:
            j = int(np.argmin(np.abs(th - float(a))))
            if j not in [o[0] for o in out]:
                out.append((j, "angle"))
        return out
    chosen = []
    for j in np.argsort(-np.where(np.isfinite(nv), nv, -np.inf)):
        if all(abs(th[j] - th[c]) >= args.sep for c in chosen):
            chosen.append(int(j))
        if len(chosen) >= args.npeaks:
            break
    out = [(j, "peak") for j in sorted(chosen, key=lambda j: th[j])]
    q = int(np.argmin(np.abs(nv - np.nanmedian(nv))))
    out.append((q, "quiet"))
    return out


def fig_spectra(run, D, info, args, out):
    x, S = transform(D, info, args, args.smooth)
    m = band(D, args)
    sel = pick_angles(D, args)
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=SURFACE, constrained_layout=True)
    rows, tab = [], pd.DataFrame({"f_hz": D["f"], "St": D["f"] * args.chord / info["velocity_m_s"]})
    ci = 0
    for j, kind in sel:
        if kind == "quiet":
            col, ls = MUTED, (0, (4, 3))
        else:
            col, ls = CAT[ci % len(CAT)], "-"
            ci += 1
        s = S[j]
        k = np.where(m)[0][np.argmax(s[m])]
        label = (f"{D['th'][j]:.0f}°" + ("  quiet reference" if kind == "quiet" else "")
                 + f"  ·  nvar {D['nv'][j]:.2g}  ·  peak {fmt_x(x[k], args)}"
                 + ("" if args.xaxis == "freq" else f" ({D['f'][k]:.3g} Hz)"))
        ax.plot(x[m], s[m], color=col, lw=1.5, ls=ls, label=label, zorder=3)
        ax.scatter([x[k]], [s[k]], s=34, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        rows.append(dict(run_name=run, motor_angle_deg=D["th"][j], kind=kind, nvar=D["nv"][j],
                         peak_hz=D["f"][k], peak_St=D["f"][k] * args.chord / info["velocity_m_s"],
                         peak_psd=s[k], clipped=bool(D["clip"][j])))
        tab[f"psd_{D['th'][j]:.0f}deg"] = D["P"][j]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x[m].min(), x[m].max())
    ax.set_xlabel(xlabel(args), color=INK)
    ax.set_ylabel(sym_psd(args), color=INK)
    style_axes(ax)
    ax.grid(True, which="minor", color="#efeeea", linewidth=0.5, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(conditions_line(run, info) + f"   ·   S{args.sensor_id} {band_label(args)}", loc="left",
                 fontsize=11, color=INK)
    save(fig, out, args.dpi)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    tab.to_csv(out.with_name(out.stem + "_values.csv"), index=False)


# --------------------------------------------------
# Mode: runs
# --------------------------------------------------

COLORBY = {"re": ("reynolds_number", "Reynolds number  Re", True),
           "ca": ("cauchy_edgewise", r"$Ca_e$", True),
           "u": ("velocity_m_s", "Freestream velocity U (m/s)", False)}


def fig_runs(runs, data, infos, args, out):
    key, clabel, logc = COLORBY[args.colorby]
    vals = np.array([infos[r][key] for r in runs], float)
    order = np.argsort(vals)
    runs, vals = [runs[i] for i in order], vals[order]
    norm = (LogNorm if logc else Normalize)(vals.min(), vals.max() if vals.max() > vals.min() else vals.min() * 1.01)
    cmap = plt.get_cmap(CMAP)
    fig = plt.figure(figsize=(11, 8.2), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1, 0.025])
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])
    rows = []
    for r, v in zip(runs, vals):
        D, info = data[r], infos[r]
        x, S = transform(D, info, args, args.smooth)
        m = band(D, args)
        j = (int(np.argmax(np.where(np.isfinite(D["nv"]), D["nv"], -np.inf))) if args.at == "peak"
             else int(np.argmin(np.abs(D["th"] - float(args.at)))))
        s = S[j]
        k = np.where(m)[0][np.argmax(s[m])]
        col = cmap(norm(v))
        ax.plot(x[m], s[m], color=col, lw=1.4, zorder=3)
        ax.scatter([x[k]], [s[k]], s=26, color=col, edgecolor=INK, linewidth=0.6, zorder=5)
        axp.scatter([v], [x[k]], s=46, color=col, edgecolor=INK, linewidth=0.7, zorder=4)
        rows.append(dict(run_name=r, colorby=args.colorby, value=v, motor_angle_deg=D["th"][j], nvar=D["nv"][j],
                         peak_hz=D["f"][k], peak_St=D["f"][k] * args.chord / info["velocity_m_s"], peak_psd=s[k],
                         clipped=bool(D["clip"][j])))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel(args), color=INK)
    ax.set_ylabel(sym_psd(args), color=INK)
    style_axes(ax)
    ax.grid(True, which="minor", color="#efeeea", linewidth=0.5, zorder=0)
    where = "at each run's variance peak" if args.at == "peak" else f"at motor angle ≈ {float(args.at):g}°"
    ax.set_title(f"S{args.sensor_id} {band_label(args)}   ·   PSD {where}   ·   {len(runs)} runs", loc="left",
                 fontsize=11, color=INK)
    if logc:
        axp.set_xscale("log")
    axp.set_xlabel(clabel, color=INK)
    axp.set_ylabel("Peak St" if args.xaxis == "st" else "Peak frequency (Hz)", color=INK)
    style_axes(axp)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(clabel, color=INK)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_edgecolor(MUTED)
    save(fig, out, args.dpi)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", help="Production_summaries folder (or the master_angles.csv inside it)")
    ap.add_argument("--mode", choices=["map", "spectra", "runs"], required=True)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--sensor", default="S4")
    ap.add_argument("--component", choices=["mag", "x", "y", "z"], default="mag")
    ap.add_argument("--xaxis", choices=["freq", "st"], default="freq")
    ap.add_argument("--norm", choices=["ac", "raw"], default="ac")
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--flog", action="store_true")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--angles", nargs="+")
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--sep", type=float, default=15.0)
    ap.add_argument("--at", default="peak")
    ap.add_argument("--colorby", choices=list(COLORBY), default="re")
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--chord", type=float, default=0.0138)
    ap.add_argument("--out")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    summ = Path(args.summaries)
    summ = summ.parent if summ.is_file() else summ
    runs_csv = summ / "master_runs.csv"
    if not runs_csv.exists():
        raise SystemExit(f"{runs_csv} not found; give the Production_summaries folder")
    rdf = pd.read_csv(runs_csv).set_index("run_name")
    sensors = parse_sensors([args.sensor])
    if len(sensors) != 1:
        raise SystemExit("--sensor takes one accelerometer (S1..S4)")
    args.sensor_id = sensors[0]
    out = Path(args.out) if args.out else summ / "figures"
    targets, batch = resolve(sorted(rdf.index), args.run)
    for r in targets:
        for k in ("velocity_m_s", "a_c_g"):
            if not np.isfinite(rdf.loc[r, k]):
                raise SystemExit(f"{r}: no measured {k} in master_runs.csv (needed for normalization / St)")
    infos = {r: rdf.loc[r].to_dict() for r in targets}
    tag = comp_tag(args) + ("_st" if args.xaxis == "st" else "") + ("_raw" if args.norm == "raw" else "")

    if args.mode == "runs":
        if len(targets) < 2:
            raise SystemExit("--mode runs needs several datasets (e.g. --run 704psi or --run all)")
        data = {r: load_run(summ, r, args.sensor_id, args.component) for r in targets}
        name = f"psd_runs_{args.colorby}_{'all' if batch else '_'.join(slug(r) for r in targets)}_{tag}"
        if len(name) > 150:
            name = f"psd_runs_{args.colorby}_{len(targets)}runs_{tag}"
        png = out / f"{name}.png"
        fig_runs(targets, data, infos, args, png)
        print(f"Wrote {png}  (+ .csv table)")
        return
    for r in targets:
        D = load_run(summ, r, args.sensor_id, args.component)
        png = out / f"psd_{args.mode}_{r}_{tag}.png"
        (fig_map if args.mode == "map" else fig_spectra)(r, D, infos[r], args, png)
        print(f"Wrote {png}  (+ .csv table)")


if __name__ == "__main__":
    main()