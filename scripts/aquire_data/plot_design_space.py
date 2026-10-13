#!/usr/bin/env python3
"""
plot_design_space.py

Goal 1: experimental design space. One point per dataset on a Reynolds number vs average
edgewise Cauchy number plot, colored by a vibration metric.

Inputs (written by the earlier pipeline stages, no raw data is read):
    master_runs.csv      from extract_run_summary.py
    master_angles.csv    from extract_run_summary.py (only needed for --sensor)
    master_metrics.csv   from angle_metrics.py (needed for the excess, width and weighted-p90 options;
                         with --sensor S4 it is master_metrics_S4.csv from angle_metrics.py --sensor S4)

Usage (from scripts/aquire_data):
    python plot_design_space.py Production_summaries/master_runs.csv
    python plot_design_space.py Production_summaries/master_runs.csv --panels p90
    python plot_design_space.py Production_summaries/master_runs.csv --panels p90 --p90 weighted
    python plot_design_space.py Production_summaries/master_runs.csv --panels excess
    python plot_design_space.py Production_summaries/master_runs.csv --panels p90,excess,width --no-labels
    python plot_design_space.py Production_summaries/master_runs.csv --sensor S4              # S4 only
    python plot_design_space.py Production_summaries/master_runs.csv --sensor S4 --panels p90,excess

Options
    --panels LIST       comma-separated metrics, one panel each (default max,p90; "both" = max,p90)
                          max     maximum normalized variance over sensors and angles
                          p90     90th percentile over angles (see --p90)
                          excess  angle-averaged excess variance over the run's quiet baseline,
                                  averaged over the full 360 deg (exact for full sweeps, a lower
                                  bound for partial ones): the recommended cross-run metric
                          width   vibrating width in degrees, i.e. angle range where the variance
                                  exceeds k x baseline (k is set in angle_metrics.py)
    --p90 {count,weighted}
                        count (default): plain percentile, every measured angle counts equally
                        weighted: each angle counts by the span of motor angle it represents
                        (taken from master_metrics.csv)
    --cbar {log,linear} colorbar scale for the variance-type metrics (default log; linear starts
                        at 0). The width panel is always linear.
    --separate-cbars    give max and p90 their own colorbars instead of one shared colorbar
    --full-only         plot only runs with a full sweep (coverage of at least 95% of 360 deg);
                        partial sweeps are left out of the figure and the table
    --sensor S1..S4     use one accelerometer instead of the worst of the four. The maximum and the
                        90th percentile (count) are recomputed from that sensor's per-angle values in
                        master_angles.csv, and the clipped (square) marker means THAT sensor clipped.
                        The weighted 90th percentile, excess and width panels need
                        master_metrics_S4.csv: run  angle_metrics.py master_angles.csv --sensor S4  first.
    --angles PATH       master_angles.csv (default: next to master_runs.csv)
    --metrics PATH      master_metrics.csv (default: next to master_runs.csv; master_metrics_S4.csv with --sensor S4)
    --no-labels         hide the dataset names (e.g. 390/300)
    --out DIR, --dpi N

Outputs (in --out, default <master_runs folder>/figures):
    design_space[_<panels>][_weighted][_S4][_fullonly][_linear].png
    design_space_table[_S4].csv    the numbers behind the plot

Encoding
    color  = the metric of each panel
    shape  = square if at least one angle of that run had accelerometer clipping (the metric is
             then a LOWER BOUND), circle otherwise
    dashed ring (excess and width panels) = partial sweep, the metric is a LOWER BOUND
    repeats (same pressure/fan setting, v1, v2, ...) share one label

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
from matplotlib.colors import LogNorm, Normalize                 # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

INK = "#1a1a19"
MUTED = "#6b6b66"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
CMAP = plt.get_cmap("turbo")      # identical to MATLAB's turbo

VAR_LABEL = r"Normalized acceleration variance   $\mathrm{Var}(|a|)/a_c^2$"
METRICS = {
    "max": dict(col="nvar_max", kind="var", title="Maximum over angles", cbar=VAR_LABEL),
    "p90": dict(col="nvar_p90", kind="var", title="90th percentile over angles", cbar=VAR_LABEL),
    "excess": dict(col="excess_mean_full", kind="excess", title="Angle-averaged excess variance",
                   cbar="Excess normalized variance over baseline,\naveraged over 360°"),
    "width": dict(col="vibrating_width_deg", kind="width", title="Vibrating width",
                  cbar="Angle range with variance above threshold (deg)"),
}

NEEDED = ["run_name", "nominal_psi", "fan_setting", "version", "reynolds_number",
          "cauchy_edgewise", "nvar_max", "nvar_p90", "n_accel_clipped"]
FROM_METRICS = ["excess_mean_full", "vibrating_width_deg", "nvar_p90_weighted", "full_sweep",
                "baseline", "baseline_reliable", "threshold"]


# --------------------------------------------------
# Data
# --------------------------------------------------

def load(path):
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    df = df.dropna(subset=["reynolds_number", "cauchy_edgewise", "nvar_max"]).copy()
    df["clipped"] = df["n_accel_clipped"].fillna(0) > 0
    # one label per test condition (pressure/fan setting); repeats are listed together
    df["cond"] = df["nominal_psi"].astype(int).astype(str) + "/" + \
        df["fan_setting"].astype(int).astype(str)
    return df


def sensor_id(text):
    m = re.fullmatch(r"[sS]?([1-4])", str(text).strip())
    if not m:
        raise SystemExit(f"--sensor {text!r}: use S1 to S4")
    return int(m[1])


def apply_sensor(df, angles_path, sid):
    """Replace the all-sensor maximum / 90th percentile / clip count with those of one accelerometer."""
    a = pd.read_csv(angles_path)
    col = f"nvar_s{sid}"
    if not {"run_name", "motor_angle_deg", col} <= set(a.columns):
        raise SystemExit(f"{angles_path} needs run_name, motor_angle_deg and {col}")
    rows = []
    for run, g in a.dropna(subset=["motor_angle_deg", col]).groupby("run_name"):
        v = g[col].to_numpy(float)
        comp = g["accel_clip_components"] if "accel_clip_components" in g.columns else []
        rows.append(dict(
            run_name=run, nvar_max=float(v.max()),
            nvar_max_angle_deg=float(g["motor_angle_deg"].to_numpy(float)[int(np.argmax(v))]),
            nvar_p90=float(np.quantile(v, 0.9)),
            n_accel_clipped=int(sum(sid in {int(x) for x in re.findall(r"s(\d)_", c)}
                                    for c in comp if isinstance(c, str)))))
    s = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in ("nvar_max", "nvar_max_angle_deg", "nvar_p90", "n_accel_clipped",
                                      "nvar_max_unclipped") if c in df.columns]).merge(s, on="run_name", how="left")
    df = df.dropna(subset=["nvar_max"]).copy()
    df["clipped"] = df["n_accel_clipped"].fillna(0) > 0
    return df


def merge_metrics(df, path):
    m = pd.read_csv(path)
    keep = ["run_name"] + [c for c in FROM_METRICS if c in m.columns]
    return df.merge(m[keep], on="run_name", how="left")


def condition_labels(df):
    out = []
    for cond, g in df.groupby("cond", sort=False):
        vers = sorted(g["version"].astype(int))
        if len(vers) == 1:
            tag = ""
        elif vers == list(range(vers[0], vers[-1] + 1)):
            tag = f" (v{vers[0]}–v{vers[-1]})"
        else:
            tag = " (v" + ",v".join(map(str, vers)) + ")"
        out.append((np.exp(np.log(g["reynolds_number"]).mean()),
                    np.exp(np.log(g["cauchy_edgewise"]).mean()), cond + tag))
    return out


# --------------------------------------------------
# Drawing
# --------------------------------------------------

def draw(ax, df, metric, title, norm, labels=True, ylabel=True, partial_ring=False):
    for clipped, marker, size in ((False, "o", 150), (True, "s", 150)):
        d = df[(df["clipped"] == clipped)].dropna(subset=[metric])
        if len(d):
            ax.scatter(d["reynolds_number"], d["cauchy_edgewise"], c=d[metric], cmap=CMAP,
                       norm=norm, marker=marker, s=size, edgecolor=SURFACE, linewidth=1.6,
                       zorder=3)
            # thin dark outline under the surface-colored ring keeps pale points visible
            ax.scatter(d["reynolds_number"], d["cauchy_edgewise"], facecolor="none",
                       marker=marker, s=size + 55, edgecolor=INK, linewidth=0.7, zorder=2)
    if partial_ring and "full_sweep" in df.columns:
        d = df[df["full_sweep"] == False].dropna(subset=[metric])          # noqa: E712
        if len(d):
            ax.scatter(d["reynolds_number"], d["cauchy_edgewise"], facecolor="none", marker="o",
                       s=420, edgecolor=MUTED, linewidth=1.0, linestyle=(0, (2.5, 2)), zorder=2)
    ax._label_specs = condition_labels(df) if labels else []
    ax._points = df[["reynolds_number", "cauchy_edgewise"]].to_numpy()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(df["reynolds_number"].min() / 1.25, df["reynolds_number"].max() * 1.9)
    ax.set_ylim(df["cauchy_edgewise"].min() / 1.5, df["cauchy_edgewise"].max() * 1.7)
    ax.set_xlabel(r"$Re=\rho U \bar{c}/\mu$", color=INK)
    if ylabel:
        ax.set_ylabel(r"$Ca_e=\rho U^2 b^4/B_e$", color=INK)
    ticks = [6e4, 1e5, 1.5e5, 2e5, 3e5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,.0f}" for t in ticks])
    ax.minorticks_off()
    ax.tick_params(labelleft=True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both")


CANDIDATES = [(10, 6, "left", "bottom"), (10, -6, "left", "top"), (-10, 6, "right", "bottom"),
              (-10, -6, "right", "top"), (0, 14, "center", "bottom"), (0, -14, "center", "top"),
              (22, 0, "left", "center"), (-22, 0, "right", "center"),
              (20, 16, "left", "bottom"), (-20, 16, "right", "bottom"),
              (20, -16, "left", "top"), (-20, -16, "right", "top")]


def place_labels(fig, ax):
    """Put each label at the first candidate offset that hits no other label or marker."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(ax._points)
    axbox = ax.get_window_extent(renderer)
    placed = []

    def hits(bb):
        if any(bb.expanded(1.05, 1.2).overlaps(o) for o in placed):
            return True
        for px, py in pts:                       # keep clear of every marker (~12 px radius)
            dx = max(bb.x0 - px, 0, px - bb.x1)
            dy = max(bb.y0 - py, 0, py - bb.y1)
            if dx * dx + dy * dy < 13 ** 2:
                return True
        return not (axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1)

    for x, y, text in ax._label_specs:
        best = None
        for dx, dy, ha, va in CANDIDATES:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            va=va, fontsize=8, color=INK, zorder=4)
            bb = t.get_window_extent(renderer)
            if not hits(bb):
                best = bb
                break
            t.remove()
        if best is None:                          # nothing free: fall back to the first candidate
            dx, dy, ha, va = CANDIDATES[0]
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            va=va, fontsize=8, color=INK, zorder=4)
            best = t.get_window_extent(renderer)
        placed.append(best)


def make_norm(values, kind, scale):
    v = pd.concat(values).dropna()
    if kind == "width" or scale == "linear":
        return Normalize(vmin=0.0, vmax=float(v.max()))
    return LogNorm(vmin=float(v[v > 0].min()), vmax=float(v.max()))


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master_runs", help="path to master_runs.csv")
    ap.add_argument("--sensor", help="plot one accelerometer (S1..S4) instead of the worst of the four")
    ap.add_argument("--angles", help="path to master_angles.csv (default: next to master_runs.csv)")
    ap.add_argument("--metrics", help="path to master_metrics.csv (default: next to master_runs.csv)")
    ap.add_argument("--out", help="output folder (default: <master_runs folder>/figures)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--panels", default="max,p90",
                    help="comma-separated: max, p90, excess, width (default max,p90)")
    ap.add_argument("--p90", choices=["count", "weighted"], default="count")
    ap.add_argument("--cbar", choices=["log", "linear"], default="log")
    ap.add_argument("--separate-cbars", action="store_true")
    ap.add_argument("--full-only", action="store_true", help="plot only full-sweep runs")
    ap.add_argument("--no-labels", action="store_true", help="hide dataset names")
    args = ap.parse_args()

    panels = ["max", "p90"] if args.panels == "both" else [p.strip() for p in args.panels.split(",") if p.strip()]
    bad = [p for p in panels if p not in METRICS]
    if bad or not panels:
        raise SystemExit(f"--panels must be a comma-separated list of {list(METRICS)}; got {args.panels!r}")

    src = Path(args.master_runs)
    out = Path(args.out) if args.out else src.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = load(src)

    sid = sensor_id(args.sensor) if args.sensor and args.sensor.lower() != "all" else None
    if sid:
        apath = Path(args.angles) if args.angles else src.parent / "master_angles.csv"
        if not apath.exists():
            raise SystemExit(f"--sensor needs {apath} (or pass --angles)")
        df = apply_sensor(df, apath, sid)

    # metrics that depend on the chosen sensor (weighted p90, excess, width) vs the sweep coverage only
    needs_sensor_metrics = (args.p90 == "weighted" and "p90" in panels
                            or any(p in ("excess", "width") for p in panels))
    needs_metrics = needs_sensor_metrics or args.full_only
    mname = f"master_metrics_S{sid}.csv" if sid else "master_metrics.csv"
    mpath = Path(args.metrics) if args.metrics else src.parent / mname
    if needs_metrics:
        if not mpath.exists() and sid and args.full_only and not needs_sensor_metrics \
                and (src.parent / "master_metrics.csv").exists():
            mpath = src.parent / "master_metrics.csv"      # coverage does not depend on the sensor
        if not mpath.exists():
            hint = f"python angle_metrics.py {src.parent / 'master_angles.csv'}" + (f" --sensor S{sid}" if sid else "")
            raise SystemExit(f"{mpath} not found; run:  {hint}  (or pass --metrics)")
        df = merge_metrics(df, mpath)

    if args.full_only:
        keep = df["full_sweep"] == True                                              # noqa: E712
        dropped = df.loc[~keep, "run_name"].tolist()
        df = df[keep].copy()
        print(f"--full-only: plotting {len(df)} runs; left out {len(dropped)} partial or unmatched: "
              + (", ".join(dropped) if dropped else "none"))
        if df.empty:
            raise SystemExit("no full-sweep runs to plot")

    df["nvar_p90_count"] = df["nvar_p90"]
    titles = {p: METRICS[p]["title"] + (f" (S{sid})" if sid else "") for p in panels}
    if args.p90 == "weighted" and "p90" in panels:
        if "nvar_p90_weighted" not in df.columns:
            raise SystemExit(f"{mpath} has no nvar_p90_weighted column")
        df["nvar_p90"] = df["nvar_p90_weighted"]
        titles["p90"] = titles["p90"].replace(" (S", " (weighted by angular span, S", 1) if sid \
            else titles["p90"] + " (weighted by angular span)"
    if any(p in ("excess", "width") for p in panels) and "baseline_reliable" in df.columns:
        unrel = df.loc[df["baseline_reliable"] == False, "run_name"].tolist()        # noqa: E712
        if unrel:
            print("WARNING: these runs have an unreliable own-median baseline (partial sweep); "
                  "excess/width are not trustworthy until you pass --baseline-file to angle_metrics.py:\n  "
                  + "\n  ".join(unrel))
    if "width" in panels and {"threshold", "baseline"} <= set(df.columns):
        k = (df["threshold"] / df["baseline"]).median()
        titles["width"] = f"Vibrating width (variance > {k:g} × baseline)" + (f" (S{sid})" if sid else "")

    # panels of the same kind (max/p90) share one norm and colorbar unless --separate-cbars
    groups = {}
    for p in panels:
        key = p if (args.separate_cbars or METRICS[p]["kind"] != "var") else "var"
        groups.setdefault(key, []).append(p)
    norms = {key: make_norm([df[METRICS[p]["col"]] for p in ps], METRICS[ps[0]]["kind"], args.cbar)
             for key, ps in groups.items()}
    norm_of = {p: norms[key] for key, ps in groups.items() for p in ps}

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n + 1.8 * len(groups), 6.4), sharex=True, sharey=True,
                             facecolor=SURFACE, constrained_layout=True, squeeze=False)
    axes = list(axes[0])
    for i, (ax, p) in enumerate(zip(axes, panels)):
        draw(ax, df, METRICS[p]["col"], titles[p], norm_of[p], labels=(not args.no_labels and i == 0),
             ylabel=(i == 0), partial_ring=METRICS[p]["kind"] in ("excess", "width"))

    for key, ps in groups.items():
        axs = [axes[panels.index(p)] for p in ps]
        sm = plt.cm.ScalarMappable(norm=norms[key], cmap=CMAP)
        cb = fig.colorbar(sm, ax=axs, shrink=0.85, pad=0.015)
        cb.set_label(METRICS[ps[0]]["cbar"], color=INK)
        cb.outline.set_edgecolor(MUTED)
        cb.ax.tick_params(colors=MUTED, which="both")

    handles = [
        Line2D([], [], marker="o", ls="", ms=9, mfc="#bdbdb8", mec=INK, mew=0.7, label="no clipped angles"),
        Line2D([], [], marker="s", ls="", ms=9, mfc="#bdbdb8", mec=INK, mew=0.7,
               label=(f"≥1 angle with S{sid} clipped" if sid else "≥1 clipped angle") + " (value is a lower bound)"),
    ]
    if any(METRICS[p]["kind"] in ("excess", "width") for p in panels) and "full_sweep" in df.columns \
            and (df["full_sweep"] == False).any():                                   # noqa: E712
        handles.append(Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=MUTED, mew=1.0,
                              label="partial sweep (value is a lower bound)"))
    axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)

    if not args.no_labels:
        place_labels(fig, axes[0])

    name = "design_space"
    if panels != ["max", "p90"]:
        name += "_" + "_".join(panels)
    if args.p90 == "weighted" and "p90" in panels:
        name += "_weighted"
    if sid:
        name += f"_S{sid}"
    if args.full_only:
        name += "_fullonly"
    if args.cbar == "linear":
        name += "_linear"
    png = out / f"{name}.png"
    fig.savefig(png, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)

    cols = ["run_name", "reynolds_number", "cauchy_edgewise", "nvar_max", "nvar_max_angle_deg",
            "nvar_p90", "nvar_p90_count", "excess_mean_full", "vibrating_width_deg", "full_sweep",
            "nvar_max_unclipped", "n_accel_clipped", "accel_fs"]
    cols = [c for c in cols if c in df.columns]
    table = df.sort_values("nvar_max", ascending=False)[cols]
    tpath = out / (f"design_space_table_S{sid}.csv" if sid else "design_space_table.csv")
    table.to_csv(tpath, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nWrote {png}\n      {tpath}")


if __name__ == "__main__":
    main()