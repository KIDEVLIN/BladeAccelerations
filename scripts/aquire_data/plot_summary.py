#!/usr/bin/env python3
"""
plot_summary.py

Plots a run from its summary files alone -- coordinate_frame_angles.csv,
run_metadata.json and (if present) tunnel_conditions_summary.csv -- for
when the raw per-angle angle_*.csv / _load.csv captures aren't
available (so postprocessing_pipeline.py / plot_variance_sweep.py have
nothing to load).

Usage:
    python3 plot_summary.py <run_dir> [--out analysis_summary]

Produces, in <run_dir>/<out>/:
    accel_variance_map.png   -- inclination vs angle of attack, colored
    load_variance_map.png       by variance
    accel_variance_sweep.png -- variance vs motor angle
    load_variance_sweep.png
                                One untitled, LaTeX-typeset figure each
                                (presentation use). Accel variance here is
                                the logged one (variance of the 4-sensor
                                MEAN magnitude, see
                                live_variance_plot.accel_magnitude_variance),
                                so there is no per-sensor breakdown.
    summary_diagnostics.png  -- acquisition QC: settled motor angle error
                                (actual - requested) and accel/load sync
                                offset, vs requested motor angle.

Run/tunnel conditions (velocity, Re, density) are printed to the
console, and added as a footer line on the diagnostics figure. Only depends on numpy/pandas/matplotlib -- no hardware
libraries -- so it runs on any analysis machine.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Accel variance is logged in mg^2 (sensor units); plots show g^2.
MG2_TO_G2 = 1e-6


def load_summary(run_dir):
    """Returns (angles_df, metadata_dict, tunnel_series_or_None)."""
    run_dir = Path(run_dir)
    angles_path = run_dir / "coordinate_frame_angles.csv"
    if not angles_path.exists():
        raise FileNotFoundError(f"{angles_path} not found -- nothing to plot")
    angles = pd.read_csv(angles_path)

    meta_path = run_dir / "run_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    tunnel_path = run_dir / "tunnel_conditions_summary.csv"
    tunnel = pd.read_csv(tunnel_path).iloc[0] if tunnel_path.exists() else None
    if tunnel is None and "tunnel_conditions" in metadata:
        tunnel = pd.Series(metadata["tunnel_conditions"])

    return angles, metadata, tunnel


def _conditions_text(run_name, metadata, tunnel):
    parts = [run_name]
    if "sweep_angle_deg" in metadata:
        parts.append(f"sweep {metadata['sweep_angle_deg']:g} deg")
    if "mounting_angle_deg" in metadata:
        parts.append(f"mounting {metadata['mounting_angle_deg']:g} deg")
    if "sample_duration_s" in metadata:
        parts.append(f"{metadata['sample_duration_s']:g} s/angle")
    if tunnel is not None:
        if "velocity_freestream_m_s" in tunnel:
            parts.append(f"U = {tunnel['velocity_freestream_m_s']:.2f} m/s")
        if "reynolds_number" in tunnel:
            parts.append(f"Re = {tunnel['reynolds_number']:.3g}")
        if "density_kg_m3" in tunnel:
            parts.append(f"rho = {tunnel['density_kg_m3']:.2f} kg/m^3")
    return "   |   ".join(parts)


def _latex_rc():
    """LaTeX-typeset text: real LaTeX (text.usetex) when a `latex` binary
    is on PATH, otherwise matplotlib's built-in Computer Modern mathtext,
    which looks the same without needing a TeX install."""
    if shutil.which("latex"):
        return {"text.usetex": True, "font.family": "serif", "font.size": 12}
    return {"mathtext.fontset": "cm", "font.family": "serif",
            "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True,
            "font.size": 12}


ACCEL_VAR_LABEL = r"$\mathrm{Var}(|\mathbf{a}|)\ (g^2)$"
LOAD_VAR_LABEL = r"$\mathrm{Var}(|\mathbf{F}|)\ (\mathrm{N}^2)$"


def _plot_map(incl, aoa, values, cmap, cbar_label, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(incl, aoa, c=values, cmap=cmap)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-180, 180)
    ax.set_xlabel(r"Inclination angle $(^\circ)$")
    ax.set_ylabel(r"Angle of attack $(^\circ)$")
    fig.colorbar(sc, ax=ax, label=cbar_label)
    _finish(fig, out_path)


def _plot_sweep(motor, values, fmt, color, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(motor, values, fmt, color=color)
    ax.set_xlabel(r"Motor angle $(^\circ)$")
    ax.set_ylabel(ylabel)
    _finish(fig, out_path)


def plot_summary_variance(angles, out_dir):
    """One untitled, LaTeX-typeset figure per panel: accel/load variance
    maps (inclination vs angle of attack) and sweeps (vs motor angle)."""
    df = angles.sort_values("motor_angle_actual_deg")
    motor = df["motor_angle_actual_deg"].to_numpy()
    incl = df["inclination_angle_deg_shifted"].to_numpy()
    aoa = df["angle_of_attack_deg_shifted"].to_numpy()
    accel_var = df["accel_variance"].to_numpy() * MG2_TO_G2
    has_load = "load_variance" in df and df["load_variance"].notna().any()

    with plt.rc_context(_latex_rc()):
        _plot_map(incl, aoa, accel_var, "viridis", ACCEL_VAR_LABEL,
                  out_dir / "accel_variance_map.png")
        _plot_sweep(motor, accel_var, "o-", "tab:blue", ACCEL_VAR_LABEL,
                    out_dir / "accel_variance_sweep.png")
        if has_load:
            load_var = df["load_variance"].to_numpy(dtype=float)
            _plot_map(incl, aoa, load_var, "plasma", LOAD_VAR_LABEL,
                      out_dir / "load_variance_map.png")
            _plot_sweep(motor, load_var, "s-", "tab:red", LOAD_VAR_LABEL,
                        out_dir / "load_variance_sweep.png")


def plot_summary_diagnostics(angles, out_path, footer=None):
    df = angles.sort_values("motor_angle_requested_deg")
    requested = df["motor_angle_requested_deg"].to_numpy()

    panels = [("Motor angle error (deg)", "Settled motor angle - requested",
               (df["motor_angle_actual_deg"] - df["motor_angle_requested_deg"]).to_numpy())]
    if "sync_offset_ms" in df:
        panels.append(("Sync offset (ms)", "Load-cell start - accel start",
                       df["sync_offset_ms"].to_numpy(dtype=float)))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)
    for ax, (ylabel, title, values) in zip(axes[0], panels):
        ax.plot(requested, values, "o-", color="tab:gray")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Requested motor angle (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    _finish(fig, out_path, footer)


def _finish(fig, out_path, footer=None):
    if footer:
        fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default="analysis_summary",
                        help="Output folder, relative to run_dir (default: analysis_summary)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    angles, metadata, tunnel = load_summary(run_dir)
    print(f"Loaded {len(angles)} angles from {run_dir}")
    footer = _conditions_text(run_dir.name, metadata, tunnel)

    print(f"  {footer}")

    plot_summary_variance(angles, out_dir)
    plot_summary_diagnostics(angles, out_dir / "summary_diagnostics.png", footer)


if __name__ == "__main__":
    main()
