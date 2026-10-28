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
           Peak regions (flags from angle_metrics.py, the same ones plot_design_space.py --flag shows):
               python plot_psd.py Production_summaries --mode runs --run all --region 170 --colorby flag
               python plot_psd.py Production_summaries --mode runs --run all --region 170 --flag-false grey
               python plot_psd.py Production_summaries --mode runs --run all --region 170 --flag-false hide --colorby ca

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
    --premult              spectra and runs modes: plot f*S(f) (St*S(St) with --xaxis st) on a LINEAR y axis against the
                           log frequency axis. Because  integral S df = integral f*S d(ln f), equal areas under the curve
                           now mean equal variance (energy), so the peaks can be compared by eye; on the log-log plot
                           they cannot. The quantity is per unit ln f (normalized: dimensionless). Peak markers and the
                           peak frequencies still come from the plain PSD S(f), not from f*S
    --yscale {log,linear}  y axis of the spectra and runs modes (default log, or linear with --premult)
    --fmin / --fmax        frequency range shown and searched for peaks (Hz; default 5 .. 500)
    --flog                 log frequency axis (map: vertical axis; spectra/runs: x axis is always log)
    --smooth N             moving average over N frequency bins before plotting/peak picking in the
                           spectra and runs modes (default 5, about 3 Hz at the default resolution)
    --angles A [A ...]     spectra mode: motor angles to show (nearest measured), or "auto" (default)
    --npeaks N, --sep DEG  auto selection: number of peaks and minimum separation (default 3, 15 deg)
    --at {peak,ANGLE}      runs mode: each run's variance-peak angle (default) or one fixed angle. The angle
                           (or the range of angles, for peak) is in the title and in the file name
                           (..._atpeak_... / ..._at170deg_...), so different angles never overwrite each other
    --no-peak              runs mode: leave out the peak-frequency markers and the lower peak-frequency panel
    --colorby {re,ca,u,flag}   runs mode colouring (default re). flag: orange = the run has a peak in --region,
                           grey = no peak, dashed = region not measured (needs --region); the lower panel then
                           uses --peakx for its x axis
    --peakx {re,ca,u}      runs mode with --colorby flag: quantity on the x axis of the peak-frequency panel (default re)
    --region R             runs mode: peak region from angle_metrics.py (170 or r170). With --at peak each run's PSD is
                           taken at its variance peak INSIDE that window, not at its overall maximum
    --flag-false {show,grey,hide}   runs mode, needs --region. Runs whose flag is False (measured, no peak in the
                           region) are drawn as usual (show, default), greyed out behind the others (grey), or left
                           out (hide). Runs whose region was not measured (blank flag) are never greyed or hidden.
                           Greyed runs get no peak marker and no point in the peak-frequency panel (their
                           "peak" would only be the noise floor).
    --metrics PATH         master_metrics_S<sensor>.csv holding the flags (default: next to master_angles.csv; run
                           angle_metrics.py --sensor S4 first). Flags come from the |a| variance of that sensor, also
                           when --component is x, y or z
    --track                map mode: mark the dominant-peak frequency at angles whose variance is >= 3x the median
    --vmin / --vmax        map colour limits (default 1st and 99.9th percentile of the PSD)
    --chord M              average chord for St (default 0.0138)
    --transparent / --no-transparent   PNG background: transparent (default) or the off-white of earlier versions.
                           The text and lines are dark, so a transparent PNG belongs on a light slide background
    --box / --no-box       frame all four sides of the plot (default: box)
    --fontsize N           base font size in points (default 10). Axis labels and ticks use it; titles, legends and
                           annotations scale in proportion, so 14 makes everything 1.4x larger
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
                              style_axes, conditions_line, save, clip_matrix, add_style_args, apply_style, frame, fs)

from plot_design_space import FLAG_TRUE, as_flag, pick_regions, region_window    # noqa: E402

AXES = ("x", "y", "z")
CMAP = "turbo"
GREY_LINE = "#b5b4ae"       # curves of runs without a peak in the chosen region (greyed or flag-coloured)
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
    if args.premult:
        num = rf"{var}\," + num
        per = "ln St" if args.xaxis == "st" else "ln f"
        if args.norm == "ac":
            return rf"${num}\,/\,a_c^2$   (variance per {per})"
        return rf"${num}$   (g$^2$ per {per})"
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


def y_of(args, x, s):
    """What is drawn: S, or x*S with --premult (x = f or St, so the plot is per unit ln x)."""
    return s * x if args.premult else s


def set_yaxis(ax, args):
    if args.ylin:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
    else:
        ax.set_yscale("log")
    if args.premult:
        ax.text(0.995, 0.015, "equal area under the curve = equal variance", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=fs(8), color=MUTED)


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
                  fontsize=fs(11), color=INK)

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
    frame(ax2)
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
        sp = y_of(args, x, s)
        ax.plot(x[m], sp[m], color=col, lw=1.5, ls=ls, label=label, zorder=3)
        ax.scatter([x[k]], [sp[k]], s=34, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        rows.append(dict(run_name=run, motor_angle_deg=D["th"][j], kind=kind, nvar=D["nv"][j],
                         peak_hz=D["f"][k], peak_St=D["f"][k] * args.chord / info["velocity_m_s"],
                         peak_psd=s[k], clipped=bool(D["clip"][j])))
        tab[f"psd_{D['th'][j]:.0f}deg"] = D["P"][j]
    ax.set_xscale("log")
    set_yaxis(ax, args)
    ax.set_xlim(x[m].min(), x[m].max())
    ax.set_xlabel(xlabel(args), color=INK)
    ax.set_ylabel(sym_psd(args), color=INK)
    style_axes(ax)
    ax.grid(True, which="minor", color="#efeeea", linewidth=0.5, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=fs(9), labelcolor=INK)
    ax.set_title(conditions_line(run, info) + f"   ·   S{args.sensor_id} {band_label(args)}", loc="left",
                 fontsize=fs(11), color=INK)
    save(fig, out, args.dpi)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    tab.to_csv(out.with_name(out.stem + "_values.csv"), index=False)


# --------------------------------------------------
# Mode: runs
# --------------------------------------------------

COLORBY = {"re": ("reynolds_number", "Reynolds number  Re", True),
           "ca": ("cauchy_edgewise", r"$Ca_e$", True),
           "u": ("velocity_m_s", "Freestream velocity U (m/s)", False)}


def load_region_flags(summ, args, targets):
    """(region name, {run: True/False/None}, (lo, hi) or None) from master_metrics_S<sensor>.csv.
    None = the region was not measured in that run (blank flag), which is not the same as False."""
    mpath = Path(args.metrics) if args.metrics else summ / f"master_metrics_S{args.sensor_id}.csv"
    if not mpath.exists():
        raise SystemExit(f"{mpath} not found; run:  python angle_metrics.py {summ / 'master_angles.csv'} "
                         f"--sensor S{args.sensor_id}  (or pass --metrics)")
    m = pd.read_csv(mpath)
    name = pick_regions(m, [args.region], mpath)[0]
    m[f"flag_{name}"] = as_flag(m[f"flag_{name}"])
    f = m.set_index("run_name")[f"flag_{name}"]
    flags = {r: (None if r not in f.index or pd.isna(f[r]) else bool(f[r])) for r in targets}
    missing = [r for r in targets if r not in f.index]
    if missing:
        print(f"WARNING: no flags in {mpath.name} for: {', '.join(missing)} (treated as not measured)")
    if "baseline_reliable" in m.columns:
        unrel = [r for r in targets if r in f.index and m.set_index("run_name").loc[r, "baseline_reliable"] == False]  # noqa: E712
        if unrel:
            print("WARNING: own-median baseline unreliable (partial sweep), so these flags are not trustworthy "
                  "until angle_metrics.py gets --baseline-file:\n  " + "\n  ".join(unrel))
    return name, flags, region_window(m, name)


def run_style(args, fl, v, cmap, norm):
    """(colour, linewidth, linestyle, zorder, greyed) of one run's curve."""
    if args.colorby == "flag":
        if fl is True:
            return FLAG_TRUE, 1.6, "-", 4, False
        if fl is False:
            return GREY_LINE, 1.1, "-", 2, True
        return MUTED, 1.2, (0, (4, 3)), 3, False
    if args.flag_false == "grey" and fl is False:
        return GREY_LINE, 1.1, "-", 2, True
    return cmap(norm(v)), 1.4, "-", 3, False


def fig_runs(runs, data, infos, args, out, flags=None, window=None):
    by_flag = args.colorby == "flag"
    key, clabel, logc = COLORBY[args.peakx if by_flag else args.colorby]
    vals = np.array([infos[r][key] for r in runs], float)
    order = np.argsort(vals)
    runs, vals = [runs[i] for i in order], vals[order]
    norm = (LogNorm if logc else Normalize)(vals.min(), vals.max() if vals.max() > vals.min() else vals.min() * 1.01)
    cmap = plt.get_cmap(CMAP)
    show_pk = not args.no_peak
    wr = [1] if by_flag else [1, 0.025]
    cax = None
    if show_pk:
        fig = plt.figure(figsize=(11, 8.2), facecolor=SURFACE, constrained_layout=True)
        gs = fig.add_gridspec(2, len(wr), height_ratios=[2.2, 1], width_ratios=wr)
        ax = fig.add_subplot(gs[0, 0])
        axp = fig.add_subplot(gs[1, 0])
        if not by_flag:
            cax = fig.add_subplot(gs[:, 1])
    else:
        fig = plt.figure(figsize=(11, 5.8), facecolor=SURFACE, constrained_layout=True)
        gs = fig.add_gridspec(1, len(wr), width_ratios=wr)
        ax = fig.add_subplot(gs[0, 0])
        if not by_flag:
            cax = fig.add_subplot(gs[0, 1])
    rows, angs, used = [], [], []
    for r, v in zip(runs, vals):
        D, info = data[r], infos[r]
        fl = flags.get(r) if flags else None
        x, S = transform(D, info, args, args.smooth)
        m = band(D, args)
        if args.at == "peak" and window:
            inw = (D["th"] >= window[0]) & (D["th"] <= window[1]) & np.isfinite(D["nv"])
            if not inw.any():
                print(f"NOTE: {r}: no measured angle in the {window[0]:g}-{window[1]:g} deg window; left out")
                continue
            j = int(np.flatnonzero(inw)[np.argmax(D["nv"][inw])])
        elif args.at == "peak":
            j = int(np.argmax(np.where(np.isfinite(D["nv"]), D["nv"], -np.inf)))
        else:
            j = int(np.argmin(np.abs(D["th"] - float(args.at))))
        s = S[j]
        k = np.where(m)[0][np.argmax(s[m])]
        col, lw, ls, z, grey = run_style(args, fl, v, cmap, norm)
        sp = y_of(args, x, s)
        ax.plot(x[m], sp[m], color=col, lw=lw, ls=ls, zorder=z)
        angs.append(D["th"][j])
        used.append((fl, grey))
        if show_pk and not grey:      # a run with no peak in the region has no peak frequency (it would be the noise-floor edge)
            ax.scatter([x[k]], [sp[k]], s=26, color=col, edgecolor=INK, linewidth=0.6, zorder=z + 2)
            axp.scatter([v], [x[k]], s=46, color=col, edgecolor=INK, linewidth=0.7, zorder=z + 1)
        rows.append(dict(run_name=r, colorby=args.colorby, value=v, motor_angle_deg=D["th"][j], nvar=D["nv"][j],
                         peak_hz=D["f"][k], peak_St=D["f"][k] * args.chord / info["velocity_m_s"], peak_psd=s[k],
                         clipped=bool(D["clip"][j]), region=args.region_name if flags else "",
                         flag="" if fl is None else fl))
    if not used:
        raise SystemExit("no runs left to plot")
    ax.set_xscale("log")
    set_yaxis(ax, args)
    ax.set_xlabel(xlabel(args), color=INK)
    ax.set_ylabel(sym_psd(args), color=INK)
    style_axes(ax)
    ax.grid(True, which="minor", color="#efeeea", linewidth=0.5, zorder=0)
    lo, hi = min(angs), max(angs)
    span = f"{lo:g}°" if np.isclose(lo, hi, atol=0.05) else f"{lo:g}° to {hi:g}°"
    if args.at == "peak":
        inside = f" inside {window[0]:g}–{window[1]:g}°" if window else ""
        where = f"at each run's variance-peak angle{inside} ({span})"
    else:
        where = f"at motor angle {float(args.at):g}°" + ("" if span == f"{float(args.at):g}°"
                                                         else f" (nearest measured: {span})")
    note = ""
    if flags:
        nt = sum(f is True for f, _ in used)
        note = f"   ·   region {args.region_name}: {nt} of {len(used)} runs with a peak"
    ax.set_title(f"S{args.sensor_id} {band_label(args)}   ·   PSD {where}\n{len(used)} runs" + note,
                 loc="left", fontsize=fs(11), color=INK)
    handles = []
    if by_flag:
        n_t = sum(f is True for f, _ in used)
        n_f = sum(f is False for f, _ in used)
        handles = [Line2D([], [], color=FLAG_TRUE, lw=2.2, label=f"peak in region ({n_t})"),
                   Line2D([], [], color=GREY_LINE, lw=2.2, label=f"no peak ({n_f})")]
        if any(f is None for f, _ in used):
            handles.append(Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 3)),
                                  label=f"region not measured ({sum(f is None for f, _ in used)})"))
    elif any(g for _, g in used):
        handles = [Line2D([], [], color=GREY_LINE, lw=2.2, label=f"no peak in region ({sum(g for _, g in used)})")]
    if handles:
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=fs(9), labelcolor=INK)
    if show_pk:
        if logc:
            axp.set_xscale("log")
        axp.set_xlabel(clabel, color=INK)
        axp.set_ylabel("Peak St" if args.xaxis == "st" else "Peak frequency (Hz)", color=INK)
        style_axes(axp)
    if cax is not None:
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
    ap.add_argument("--premult", action="store_true")
    ap.add_argument("--yscale", choices=["log", "linear"])
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--flog", action="store_true")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--angles", nargs="+")
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--sep", type=float, default=15.0)
    ap.add_argument("--at", default="peak")
    ap.add_argument("--colorby", choices=list(COLORBY) + ["flag"], default="re")
    ap.add_argument("--peakx", choices=list(COLORBY), default="re")
    ap.add_argument("--region", help="peak region from angle_metrics.py, e.g. 170 or r170 (runs mode)")
    ap.add_argument("--flag-false", choices=["show", "grey", "hide"], default="show")
    ap.add_argument("--metrics", help="master_metrics_S<sensor>.csv with the region flags")
    ap.add_argument("--no-peak", action="store_true",
                    help="runs mode: do not mark or plot the dominant peak frequency (PSD curves only)")
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--chord", type=float, default=0.0138)
    ap.add_argument("--out")
    add_style_args(ap)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    apply_style(args)

    args.ylin = (args.yscale or ("linear" if args.premult else "log")) == "linear"
    if args.mode == "map" and (args.premult or args.yscale):
        raise SystemExit("--premult and --yscale apply to --mode spectra and runs (map shows the PSD as colour)")
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
    tag += "_premult" if args.premult else ""
    if args.ylin != args.premult:
        tag += "_lin" if args.ylin else "_logy"

    flags, window = None, None
    if args.colorby == "flag" or args.flag_false != "show":
        if not args.region:
            raise SystemExit("--colorby flag and --flag-false need --region (e.g. --region 170)")
    if args.region:
        if args.mode != "runs":
            raise SystemExit("--region, --colorby flag and --flag-false apply to --mode runs")
        args.region_name, flags, window = load_region_flags(summ, args, targets)
        if args.flag_false == "hide":
            dropped = [r for r in targets if flags[r] is False]
            targets = [r for r in targets if flags[r] is not False]
            print(f"--flag-false hide: left out {len(dropped)} run(s) with no peak in {args.region_name}: "
                  + (", ".join(dropped) if dropped else "none"))

    if args.mode == "runs":
        if len(targets) < 2:
            raise SystemExit("--mode runs needs several datasets (e.g. --run 704psi or --run all)")
        data = {r: load_run(summ, r, args.sensor_id, args.component) for r in targets}
        at = "atpeak" if args.at == "peak" else f"at{float(args.at):g}deg"
        extra = (f"_{args.region_name}" + {"grey": "_grey", "hide": "_hide"}.get(args.flag_false, "")
                 if flags is not None else "") + ("_nopeak" if args.no_peak else "")
        name = f"psd_runs_{args.colorby}_{'all' if batch else '_'.join(slug(r) for r in targets)}_{at}_{tag}{extra}"
        if len(name) > 150:
            name = f"psd_runs_{args.colorby}_{len(targets)}runs_{at}_{tag}{extra}"
        png = out / f"{name}.png"
        fig_runs(targets, data, infos, args, png, flags, window)
        print(f"Wrote {png}  (+ .csv table)")
        return
    for r in targets:
        D = load_run(summ, r, args.sensor_id, args.component)
        png = out / f"psd_{args.mode}_{r}_{tag}.png"
        (fig_map if args.mode == "map" else fig_spectra)(r, D, infos[r], args, png)
        print(f"Wrote {png}  (+ .csv table)")


if __name__ == "__main__":
    main()