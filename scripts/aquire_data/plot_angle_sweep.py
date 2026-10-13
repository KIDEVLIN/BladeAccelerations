#!/usr/bin/env python3
"""
plot_angle_sweep.py

Normalized acceleration variance vs motor angle, built from master_angles.csv (written by
extract_run_summary.py); no raw data is read. Three uses:

  1. One dataset, all four accelerometers (or a chosen subset)
         python plot_angle_sweep.py Production_summaries/master_angles.csv --run 704/800
         python plot_angle_sweep.py Production_summaries/master_angles.csv --run 704/800 --sensor S1 S4

  2. Several datasets compared on the chosen accelerometer(s), overlaid on one axis
         python plot_angle_sweep.py Production_summaries/master_angles.csv --run 385/800 704/600 --sensor S4
         python plot_angle_sweep.py Production_summaries/master_angles.csv --run 385/700 704/400 --sensor S4

  3. The same comparison as one row per dataset (shared axes), for more than a few runs
         python plot_angle_sweep.py Production_summaries/master_angles.csv --run 704/400 704/500 704/600 704/700 --sensor S4 --layout stacked

  --run all makes one single-dataset figure per run (in <out>/angle_sweeps/).

Put the master_angles.csv path first; --run and --sensor take several values.

--run accepts the full folder name, a short key PSI/FAN[/VERSION] (version defaults to 1, so 385/700 is
the unversioned run and 385/700/2 is _v2), a unique substring, or "all". A substring that matches
several runs (e.g. 704psi) compares them all.

Options
    --sensor S4 [S1 ...]   accelerometers to plot (default all four)
    --layout {overlay,stacked}   comparison layout (default overlay; >8 runs switches to a colour ramp)
    --yscale {log,linear}  variance axis (default log: the quiet floor and the peaks are both visible)
    --bands                shade the vibrating angle ranges (needs master_metrics.csv from angle_metrics.py).
                           Single dataset and stacked: also draws the baseline and threshold. The ranges
                           refer to the worst-sensor curve, not to the sensors you selected.
    --no-title             drop the title line
    --no-frame-panel       drop the inclination / angle-of-attack panel
    --runs PATH, --metrics PATH   master_runs.csv / master_metrics.csv (default: next to master_angles.csv)
    --out DIR, --dpi N

Outputs (in --out, default <master_angles folder>/figures): a PNG and a CSV with the plotted numbers.

Encoding
    single dataset / stacked rows: color + marker shape = accelerometer (S1 circle, S2 square, S3 triangle, S4 diamond)
    overlay: color = dataset (legend gives Re and Ca_e), marker shape = accelerometer
    ring around a point = that sensor had >=1% of samples at full scale at that angle (clipped, the
                          variance there is a LOWER BOUND)

All datasets are normalized by their own a_c, so heights are comparable between runs.

Requires: numpy pandas matplotlib
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.ticker import MultipleLocator                    # noqa: E402

INK = "#1a1a19"
MUTED = "#6b6b66"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
# reference categorical palette, slots 1-8 in order (adjacent pairs validated for lines)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SENSOR_COLORS = CAT[:4]
SENSOR_MARKERS = ["o", "s", "^", "D"]
NAME_RE = re.compile(r"^(?P<blade>.+?)_(?P<psi>\d+)psi_(?P<fs>\d+)fs(?:_v(?P<v>\d+))?$")
KEY_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)(?:\s*/\s*v?(\d+))?\s*$")
NS = 4


# --------------------------------------------------
# Selecting runs and sensors
# --------------------------------------------------

def short_name(run):
    m = NAME_RE.match(run)
    if not m:
        return run
    return f"{m['psi']}/{m['fs']}" + (f" v{int(m['v'])}" if m["v"] else "")


def slug(run):
    return short_name(run).replace("/", "-").replace(" ", "")


def resolve_one(names, key):
    if key in names:
        return [key]
    m = KEY_RE.match(key)
    if m:
        psi, fs, v = int(m[1]), int(m[2]), int(m[3] or 1)
        hit = [n for n in names if NAME_RE.match(n)
               and int(NAME_RE.match(n)["psi"]) == psi and int(NAME_RE.match(n)["fs"]) == fs
               and int(NAME_RE.match(n)["v"] or 1) == v]
        if hit:
            return hit
    hit = [n for n in names if key in n]
    if hit:
        return hit
    raise SystemExit(f"--run {key!r} matched no run. Available:\n  " + "\n  ".join(names))


def resolve(names, keys):
    if len(keys) == 1 and keys[0].lower() == "all":
        return list(names), True
    out = []
    for k in keys:
        for n in resolve_one(names, k):
            if n not in out:
                out.append(n)
    return out, False


def parse_sensors(vals):
    if not vals:
        return list(range(1, NS + 1))
    out = []
    for v in vals:
        m = re.fullmatch(r"[sS]?(\d)", v.strip())
        if not m or not 1 <= int(m[1]) <= NS:
            raise SystemExit(f"--sensor {v!r}: use S1 to S{NS}")
        if int(m[1]) not in out:
            out.append(int(m[1]))
    return sorted(out)


# --------------------------------------------------
# Data helpers
# --------------------------------------------------

def clipped_sensors(s):
    if not isinstance(s, str) or not s:
        return set()
    return {int(x) for x in re.findall(r"s(\d)_", s)}


def prepare(g):
    g = g.sort_values("motor_angle_deg")
    nv = np.column_stack([g[f"nvar_s{i}"].to_numpy(float) for i in range(1, NS + 1)])
    clip = np.zeros_like(nv, dtype=bool)
    comp = g["accel_clip_components"] if "accel_clip_components" in g else pd.Series([None] * len(g), index=g.index)
    for r, s in enumerate(comp):
        for sid in clipped_sensors(s):
            if 1 <= sid <= NS:
                clip[r, sid - 1] = True
    return dict(
        th=g["motor_angle_deg"].to_numpy(float), nv=nv, clip=clip,
        touched=g["accel_touched_max"].astype(bool).to_numpy() if "accel_touched_max" in g else np.zeros(len(g), bool),
        incl=g["inclination_deg"].to_numpy(float) if "inclination_deg" in g else None,
        aoa=g["aoa_deg"].to_numpy(float) if "aoa_deg" in g else None,
    )


def break_jumps(x, y, jump=180.0):
    """Insert NaN where a wrapped angle jumps so the line does not sweep across the plot."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.where(np.abs(np.diff(y)) > jump)[0]
    if len(idx):
        x = np.insert(x, idx + 1, np.nan)
        y = np.insert(y, idx + 1, np.nan)
    return x, y


def parse_bands(s):
    out = []
    if isinstance(s, str):
        for part in s.split(";"):
            m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*[–-]\s*(-?\d+(?:\.\d+)?)\s*$", part)
            if m:
                out.append((float(m[1]), float(m[2])))
    return out


def run_label(run, info, conditions=True):
    s = short_name(run)
    if conditions and info:
        bits = []
        if np.isfinite(info.get("reynolds_number", np.nan)):
            bits.append(f"Re {info['reynolds_number']:.3g}")
        if np.isfinite(info.get("cauchy_edgewise", np.nan)):
            bits.append(f"Ca$_e$ {info['cauchy_edgewise']:.2g}")
        if bits:
            s += "  (" + ", ".join(bits) + ")"
    return s


def conditions_line(run, info):
    bits = [short_name(run)]
    if info:
        for key, fmt in (("velocity_m_s", "U = {:.2f} m/s"), ("density_kg_m3", "ρ = {:.1f} kg/m³"),
                         ("reynolds_number", "Re = {:.3g}"), ("cauchy_edgewise", "Ca$_e$ = {:.2g}")):
            v = info.get(key)
            if v is not None and np.isfinite(v):
                bits.append(fmt.format(v))
        if isinstance(info.get("accel_fs"), str) and info["accel_fs"]:
            bits.append(f"accel range ±{info['accel_fs'].replace('g', '')} g")
    return "   ·   ".join(bits)


# --------------------------------------------------
# Drawing helpers
# --------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both")


def set_variance_limits(ax, arrays, yscale):
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if yscale == "log":
        pos = vals[vals > 0]
        ax.set_yscale("log")
        ax.set_ylim(pos.min() / 1.6, pos.max() * 1.7)
    else:
        ax.set_ylim(0, vals.max() * 1.08)


def draw_sensors(ax, d, sensors):
    for s in sensors:
        i = s - 1
        ax.plot(d["th"], d["nv"][:, i], color=SENSOR_COLORS[i], lw=1.6, marker=SENSOR_MARKERS[i], ms=4.6,
                mec=SURFACE, mew=0.6, zorder=3 + i * 0.1)
        m = d["clip"][:, i]
        if m.any():
            ax.scatter(d["th"][m], d["nv"][m, i], s=95, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=6)


def draw_bands(ax, met, color=MUTED, alpha=0.13, lines=True):
    if met is None:
        return
    for s, e in parse_bands(met.get("bands")):
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0, zorder=1)
    if lines:
        for yv, txt in ((met.get("baseline"), "baseline"), (met.get("threshold"), "threshold")):
            if yv is not None and np.isfinite(yv):
                ax.axhline(yv, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
                ax.annotate(txt, (1.0, yv), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=8, color=MUTED)


def draw_frame_panel(ax2, th, incl, aoa):
    for y0, label, color, ls in ((incl, "Inclination", INK, "-"), (aoa, "Angle of attack", MUTED, (0, (5, 2)))):
        if y0 is None:
            continue
        x, y = break_jumps(th, y0)
        ax2.plot(x, y, color=color, lw=1.7, ls=ls, zorder=3)
        fin = np.isfinite(y)
        if fin.any():
            j = np.where(fin)[0][-1]
            ax2.annotate(label, (x[j], y[j]), xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                         color=color, va="center", annotation_clip=False)
    ax2.set_ylabel("Blade frame (deg)", color=INK)
    style_axes(ax2)


def frame_available(d):
    return d["incl"] is not None and d["aoa"] is not None and (np.isfinite(d["incl"]).any() or np.isfinite(d["aoa"]).any())


def frames_agree(runs, data):
    ref = pd.DataFrame({"th": np.round(data[runs[0]]["th"], 1), "i0": data[runs[0]]["incl"], "a0": data[runs[0]]["aoa"]})
    for r in runs[1:]:
        d = data[r]
        cur = pd.DataFrame({"th": np.round(d["th"], 1), "i1": d["incl"], "a1": d["aoa"]})
        m = ref.merge(cur, on="th")
        if len(m) == 0:
            return False
        if np.nanmax(np.abs(m["i0"] - m["i1"])) > 1.0 or np.nanmax(np.abs(m["a0"] - m["a1"])) > 1.0:
            return False
    return True


def finish_x(ax_last, th_all):
    ax_last.set_xlabel("Motor angle (deg)", color=INK)
    ax_last.set_xlim(np.nanmin(th_all) - 6, np.nanmax(th_all) + 6)
    ax_last.xaxis.set_major_locator(MultipleLocator(30))


YLABEL = r"Normalized variance   $\mathrm{Var}(|a|)/a_c^2$"


def sensor_handles(sensors, any_clip, with_color=True):
    h = []
    for s in sensors:
        h.append(Line2D([], [], color=SENSOR_COLORS[s - 1] if with_color else MUTED, marker=SENSOR_MARKERS[s - 1],
                        ms=6, lw=1.6 if with_color else 0, mec=SURFACE, mew=0.6, label=f"S{s}"))
    if any_clip:
        h.append(Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, mew=1.2, label="clipped (lower bound)"))
    return h


def save(fig, out_png, dpi):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def long_table(runs, data, sensors):
    rows = []
    for r in runs:
        d = data[r]
        for s in sensors:
            for k in range(len(d["th"])):
                rows.append(dict(run_name=r, sensor=f"S{s}", motor_angle_deg=d["th"][k],
                                 nvar=d["nv"][k, s - 1], clipped=bool(d["clip"][k, s - 1])))
    return pd.DataFrame(rows)


# --------------------------------------------------
# Figures
# --------------------------------------------------

def fig_single(run, d, info, met, sensors, args, out_png):
    has_frame = not args.no_frame_panel and frame_available(d)
    if has_frame:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, facecolor=SURFACE,
                                      gridspec_kw=dict(height_ratios=[3, 1.05]), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE, constrained_layout=True)
        ax2 = None
    draw_sensors(ax, d, sensors)
    set_variance_limits(ax, [d["nv"][:, s - 1] for s in sensors], args.yscale)
    if args.bands:
        draw_bands(ax, met)
    ax.set_ylabel(YLABEL, color=INK)
    handles = sensor_handles(sensors, d["clip"][:, [s - 1 for s in sensors]].any())
    if args.bands and met is not None and parse_bands(met.get("bands")):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.2, lw=0, label="vibrating range"))
    ax.legend(handles=handles, loc="upper left", ncol=min(len(handles), 4), frameon=False, fontsize=9,
              labelcolor=INK, columnspacing=1.4, handletextpad=0.5)
    style_axes(ax)
    if not args.no_title:
        ax.set_title(conditions_line(run, info), loc="left", fontsize=11, color=INK)
    last = ax
    if ax2 is not None:
        draw_frame_panel(ax2, d["th"], d["incl"], d["aoa"])
        ax.tick_params(labelbottom=False)
        last = ax2
    finish_x(last, d["th"])
    save(fig, out_png, args.dpi)

    tab = pd.DataFrame({"motor_angle_deg": d["th"]})
    for s in sensors:
        tab[f"nvar_s{s}"] = d["nv"][:, s - 1]
        tab[f"s{s}_clipped"] = d["clip"][:, s - 1]
    tab["touched_full_scale"] = d["touched"]
    if d["incl"] is not None:
        tab["inclination_deg"], tab["aoa_deg"] = d["incl"], d["aoa"]
    tab.to_csv(out_png.with_suffix(".csv"), index=False)


def fig_overlay(runs, data, infos, mets, sensors, args, out_png):
    if len(runs) <= len(CAT):
        cols = CAT[:len(runs)]
    else:
        # more runs than distinct categorical hues: ramp (turbo) in the order the runs were given,
        # so order them by Re or Ca_e on the command line and colour then encodes that order
        cmap = plt.get_cmap("turbo")
        cols = [cmap(0.05 + 0.9 * k / (len(runs) - 1)) for k in range(len(runs))]
        print(f"NOTE: {len(runs)} runs > {len(CAT)}: colours are a turbo ramp in the order given "
              "(low->high), so runs are told apart by order, not by distinct hue.")
    ref = data[runs[0]]
    show_frame = (not args.no_frame_panel and frame_available(ref))
    if show_frame and not frames_agree(runs, data):
        print("NOTE: these runs have different inclination/angle-of-attack maps (different sweep or mounting "
              "angle); the frame panel is omitted.")
        show_frame = False
    if show_frame:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, facecolor=SURFACE,
                                      gridspec_kw=dict(height_ratios=[3, 1.05]), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE, constrained_layout=True)
        ax2 = None
    arrays, any_clip = [], False
    for k, run in enumerate(runs):
        d, col = data[run], cols[k]
        for s in sensors:
            i = s - 1
            ax.plot(d["th"], d["nv"][:, i], color=col, lw=1.6, marker=SENSOR_MARKERS[i], ms=4.8,
                    mec=SURFACE, mew=0.6, zorder=3 + k * 0.1)
            arrays.append(d["nv"][:, i])
            m = d["clip"][:, i]
            if m.any():
                any_clip = True
                ax.scatter(d["th"][m], d["nv"][m, i], s=95, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=6)
        if args.bands:
            draw_bands(ax, mets.get(run), color=col, alpha=0.09, lines=False)
    set_variance_limits(ax, arrays, args.yscale)
    ax.set_ylabel(YLABEL, color=INK)

    handles = [Line2D([], [], color=cols[k], lw=2.2, label=run_label(r, infos.get(r))) for k, r in enumerate(runs)]
    if len(sensors) > 1:
        handles += sensor_handles(sensors, False, with_color=False)
    if any_clip:
        handles.append(Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, mew=1.2, label="clipped (lower bound)"))
    ax.legend(handles=handles, loc="upper left", ncol=1 if len(handles) <= 4 else (2 if len(handles) <= 12 else 3), frameon=False,
              fontsize=9 if len(handles) <= 12 else 8,
              labelcolor=INK)
    style_axes(ax)
    if not args.no_title:
        ax.set_title("Accelerometer " + ", ".join(f"S{s}" for s in sensors), loc="left", fontsize=11, color=INK)
    last = ax
    if ax2 is not None:
        draw_frame_panel(ax2, ref["th"], ref["incl"], ref["aoa"])
        ax.tick_params(labelbottom=False)
        last = ax2
    finish_x(last, np.concatenate([data[r]["th"] for r in runs]))
    save(fig, out_png, args.dpi)
    long_table(runs, data, sensors).to_csv(out_png.with_suffix(".csv"), index=False)


def fig_stacked(runs, data, infos, mets, sensors, args, out_png):
    n = len(runs)
    ref = data[runs[0]]
    show_frame = (not args.no_frame_panel and frame_available(ref))
    if show_frame and not frames_agree(runs, data):
        print("NOTE: these runs have different inclination/angle-of-attack maps; the frame panel is omitted.")
        show_frame = False
    rows = n + (1 if show_frame else 0)
    fig = plt.figure(figsize=(11, 2.55 * n + (1.9 if show_frame else 0.4) + 0.5), facecolor=SURFACE,
                     constrained_layout=True)
    gs = fig.add_gridspec(rows, 1, height_ratios=[1.0] * n + ([0.75] if show_frame else []))
    axes = []
    for k in range(n):
        axes.append(fig.add_subplot(gs[k], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None))
    arrays = [data[r]["nv"][:, s - 1] for r in runs for s in sensors]
    any_clip = False
    for ax, run in zip(axes, runs):
        d = data[run]
        draw_sensors(ax, d, sensors)
        any_clip = any_clip or d["clip"][:, [s - 1 for s in sensors]].any()
        if args.bands:
            draw_bands(ax, mets.get(run))
        style_axes(ax)
        ax.set_title(conditions_line(run, infos.get(run)), loc="left", fontsize=10, color=INK)
        ax.tick_params(labelbottom=False)
    set_variance_limits(axes[0], arrays, args.yscale)
    axes[0].legend(handles=sensor_handles(sensors, any_clip), loc="upper left", ncol=5, frameon=False,
                   fontsize=9, labelcolor=INK, columnspacing=1.4, handletextpad=0.5)
    fig.supylabel(YLABEL, color=INK, fontsize=10)
    last = axes[-1]
    if show_frame:
        ax2 = fig.add_subplot(gs[n], sharex=axes[0])
        draw_frame_panel(ax2, ref["th"], ref["incl"], ref["aoa"])
        last = ax2
    finish_x(last, np.concatenate([data[r]["th"] for r in runs]))
    last.tick_params(labelbottom=True)
    save(fig, out_png, args.dpi)
    long_table(runs, data, sensors).to_csv(out_png.with_suffix(".csv"), index=False)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master_angles", help="path to master_angles.csv")
    ap.add_argument("--run", nargs="+", required=True, help="one or more runs (see above), or all")
    ap.add_argument("--sensor", nargs="+", help="accelerometers to plot, e.g. S4 or S1 S4 (default all four)")
    ap.add_argument("--layout", choices=["overlay", "stacked"], default="overlay",
                    help="layout when several runs are given (default overlay)")
    ap.add_argument("--yscale", choices=["log", "linear"], default="log")
    ap.add_argument("--bands", action="store_true", help="shade vibrating ranges (needs master_metrics.csv)")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--no-frame-panel", action="store_true")
    ap.add_argument("--runs", help="master_runs.csv (default: next to master_angles.csv)")
    ap.add_argument("--metrics", help="master_metrics.csv (default: next to master_angles.csv)")
    ap.add_argument("--out", help="output folder (default: <master_angles folder>/figures)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    src = Path(args.master_angles)
    df = pd.read_csv(src)
    need = {"run_name", "motor_angle_deg"} | {f"nvar_s{i}" for i in range(1, NS + 1)}
    if not need <= set(df.columns):
        raise SystemExit(f"{src} is missing columns: {sorted(need - set(df.columns))}")
    runs_path = Path(args.runs) if args.runs else src.parent / "master_runs.csv"
    runs_df = pd.read_csv(runs_path).set_index("run_name") if runs_path.exists() else None
    met_path = Path(args.metrics) if args.metrics else src.parent / "master_metrics.csv"
    met_df = pd.read_csv(met_path).set_index("run_name") if met_path.exists() else None
    if args.bands and met_df is None:
        raise SystemExit(f"--bands needs {met_path}; run angle_metrics.py first")

    sensors = parse_sensors(args.sensor)
    stag = "" if len(sensors) == NS else "_" + "".join(f"S{s}" for s in sensors)
    targets, batch = resolve(sorted(df["run_name"].unique()), args.run)
    out = Path(args.out) if args.out else src.parent / "figures"

    data = {r: prepare(df[df["run_name"] == r]) for r in targets}
    infos = {r: runs_df.loc[r].to_dict() for r in targets if runs_df is not None and r in runs_df.index}
    mets = {r: met_df.loc[r].to_dict() for r in targets if met_df is not None and r in met_df.index}

    if batch or len(targets) == 1:
        folder = out / "angle_sweeps" if batch else out
        for r in targets:
            png = folder / f"angle_sweep_{r}{stag}.png"
            fig_single(r, data[r], infos.get(r), mets.get(r), sensors, args, png)
            print(f"Wrote {png}  (+ .csv table)")
    else:
        name = "angle_sweep_compare_" + "_".join(slug(r) for r in targets) + (stag or "_allsensors")
        if args.layout == "stacked":
            name += "_stacked"
        png = out / f"{name}.png"
        (fig_stacked if args.layout == "stacked" else fig_overlay)(targets, data, infos, mets, sensors, args, png)
        print(f"Wrote {png}  (+ .csv table)")


if __name__ == "__main__":
    main()