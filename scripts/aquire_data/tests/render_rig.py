"""
render_rig.py

Renders the wind-tunnel rig at a given motor angle -- vertical shaft,
horizontal arm, swept blade hanging from the arm tip -- and computes the
blade's inflow angle of attack and inclination from explicit vectors,
independently of utils/coordinate_transforms.py (whose outputs are printed
for comparison).

Run from the repo root:
    poetry run python scripts/aquire_data/tests/render_rig.py [--motor 120] [--sweep 30] [--mounting 84.105]

Geometry, from the experimentalists' sketch (WhatsApp image, 29 Sep 2026),
seen from above with the flow along +x (left to right on the page):
  - motor 0:   the arm points across the flow (+y, up the page); the chord is
               drawn roughly along the flow, leading edge upstream;
               inclination 0, "mounting angle = aoa".
  - motor +90: arm downstream, inclination +30.
  - motor -90: arm upstream,  inclination -30.
  So the motor turns the arm clockwise seen from above.
Blade: hangs below the arm (root at the arm tip, above the blade tip; the
  swept frame of coordinate_transforms has z "up the blade from tip to
  root"), leaning by `sweep` within the vertical plane of the arm. The sign
  of the lean is fixed by the sketch: +30 inclination with the arm
  downstream means the flow runs tip -> root there, so the tip leans back
  toward the shaft.
Chord: perpendicular to the span, at `mounting` to the arm (the mounting
  angle of the run metadata, as defined in utils/coordinate_transforms.py).
  90 = exactly perpendicular to the arm; the production runs record 84.105,
  i.e. the chord 5.9 deg off perpendicular and at 5.9 deg to the flow at
  motor 0.
Angle of attack: angle of the flow, projected into the section plane, from
  the chord (LE -> TE), positive as the flow turns toward the arm-tip side;
  so it grows with motor angle. Inclination: asin(-U.s), positive when the
  flow runs along the span from tip to root.
Lengths are illustrative except the blade (0.28 m span, 13.8 mm mean chord);
the arm length is assumed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_parked_turbine import airfoil, cylinder, box, load_transform, unit  # noqa: E402

FIG_DIR = HERE / "figs"
SPAN = 0.28
CHORD = 0.0138
ARM = 0.20          # assumed
SHAFT_UP = 0.18     # shaft length above the arm (to the motor)
BLADE = "#e0702b"
WIND = "#1b9e77"
CHORD_VIS = 3.0     # chord drawn 3x for visibility in the 3D/side views


def rig_vectors(motor_deg, sweep_deg, aoa0_deg):
    th = np.radians(motor_deg)
    y_m = np.array([np.sin(th), np.cos(th), 0.0])      # arm direction, shaft -> arm tip
    z = np.array([0.0, 0.0, 1.0])
    x_m = np.cross(y_m, z)                              # perpendicular to arm; = +x at motor 0
    sw = np.radians(abs(sweep_deg))
    s = -np.cos(sw) * z - np.sin(sw) * y_m              # root -> tip: down, tip leaning toward the shaft
    e1 = x_m                                            # in the section plane (already normal to s)
    e2 = unit(y_m - (y_m @ s) * s)                      # in the section plane, toward the arm tip
    phi = np.radians(aoa0_deg)
    c = np.cos(phi) * e1 - np.sin(phi) * e2             # LE -> TE
    n = np.sin(phi) * e1 + np.cos(phi) * e2             # c turned 90 deg toward e2
    return dict(y_m=y_m, x_m=x_m, z=z, s=s, c=c, n=n)


def angles(U, v):
    U = unit(U)
    aoa = np.degrees(np.arctan2(U @ v["n"], U @ v["c"]))
    incl = np.degrees(np.arcsin(np.clip(-(U @ v["s"]), -1, 1)))
    return aoa, incl


def blade_quads(root, s, c, scale=1.0):
    rs = np.linspace(0, SPAN, 16)
    ch = lambda r: CHORD * scale * (1.25 - 0.5 * r / SPAN)
    le = [root + r * s - 0.25 * ch(r) * c for r in rs]
    te = [root + r * s + 0.75 * ch(r) * c for r in rs]
    return [[le[i], le[i + 1], te[i + 1], te[i]] for i in range(len(rs) - 1)], np.array(le), np.array(te)


def draw_3d(ax, v, motor):
    O = np.zeros(3)
    tip_arm = O + ARM * v["y_m"]
    polys, cols = [], []
    polys += cylinder(O - 0.01 * v["z"], O + SHAFT_UP * v["z"], 0.012)
    cols += ["0.55"] * 24
    polys += box(O + (SHAFT_UP + 0.04) * v["z"], v["x_m"], v["y_m"], v["z"], 0.08, 0.08, 0.08)
    cols += ["0.35"] * 6
    polys += cylinder(O, tip_arm, 0.009)
    cols += ["0.60"] * 24
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"))
    quads, le, te = blade_quads(tip_arm, v["s"], v["c"], CHORD_VIS)
    ax.add_collection3d(Poly3DCollection(quads, facecolors=BLADE, edgecolors="none"))
    ax.plot(*le.T, color="k", lw=1.0)
    ax.text(*(tip_arm + 0.03 * v["z"]), "root", fontsize=7)
    ax.text(*(tip_arm + SPAN * v["s"] - 0.03 * v["z"]), "tip", fontsize=7)
    for dy in (-0.2, 0.1):
        for dz in (-0.05, -0.22):
            ax.quiver(-0.42, dy, dz, 0.16, 0, 0, color=WIND, lw=1.3, arrow_length_ratio=0.3)
    ax.text(-0.42, -0.2, 0.0, "flow", color=WIND, fontsize=9)
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(-0.35, 0.3)
    ax.set_box_aspect((1, 1, 0.93))
    ax.view_init(elev=16, azim=-60)
    ax.set_axis_off()
    ax.set_title(f"(a) Rig at motor angle {motor:g}°", loc="left", fontsize=10)


def draw_top(ax, motor, sweep, aoa0, highlight=False):
    """From above in the sketch's style: flow left to right, shaft circle,
    arm, and the blade's root section drawn true to shape (symmetric
    section, enlarged), chord at 90 - aoa0 to the arm, leading edge
    upstream at motor 0."""
    v = rig_vectors(motor, sweep, aoa0)
    aoa, incl = angles([1.0, 0, 0], v)
    th = np.radians(motor)
    y_m = np.array([np.sin(th), np.cos(th)])            # arm direction (2D)
    x_m = np.array([np.cos(th), -np.sin(th)])           # perpendicular; = +x at motor 0
    phi = np.radians(aoa0)
    c = np.cos(phi) * x_m - np.sin(phi) * y_m            # LE -> TE (horizontal part)
    nn = np.sin(phi) * x_m + np.cos(phi) * y_m
    L = 1.0
    end = L * y_m
    ax.add_patch(plt.Circle((0, 0), 0.22, fill=False, color="0.25", lw=1.2))
    ax.plot(*np.c_[0.22 * y_m, end].reshape(2, 2), color="0.45", lw=4, solid_capstyle="butt")
    xa, ya = airfoil(m=0.0, th=0.16)
    k = 0.9
    P = np.array([end + k * ((x - 0.4) * c + y * nn) for x, y in zip(xa, ya)])
    ax.fill(P[:, 0], P[:, 1], color=BLADE, zorder=3)
    le = end - 0.4 * k * c
    ax.plot(*le, "ko", ms=3, zorder=4)
    ax.text(*(le - 0.28 * c), "LE", fontsize=7, ha="center", va="center")
    ax.annotate("", xy=(-1.1, -1.55), xytext=(-1.75, -1.55),
                arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.4))
    ax.text(-1.75, -1.35, "U", color=WIND, fontsize=9)
    txt = f"AoA {aoa:+.1f}°\nincl {incl:+.1f}°"
    ax.text(1.75, -1.75, txt, fontsize=8, ha="right", va="bottom",
            fontweight="bold" if highlight else "normal")
    ax.set_title(f"motor {motor:+g}°", loc="left", fontsize=9, fontweight="bold" if highlight else "normal")
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    if highlight:
        for sp in ax.spines.values():
            sp.set_linewidth(1.6)


def draw_side(ax, v, incl):
    """Looking along x_m (perpendicular to the arm): the arm and the blade's
    lean lie in the page; the flow's component along the arm is drawn."""
    ym, z = v["y_m"], v["z"]
    P = lambda p: np.array([p @ ym, p @ z])
    O = np.zeros(3)
    tip_arm = ARM * ym
    ax.plot(*np.c_[P(O), P(O + SHAFT_UP * z)], color="0.55", lw=4)
    ax.add_patch(plt.Rectangle((-0.04, SHAFT_UP), 0.08, 0.08, color="0.35"))
    ax.text(0, SHAFT_UP + 0.1, "motor", ha="center", fontsize=7)
    ax.plot(*np.c_[P(O), P(tip_arm)], color="0.6", lw=4)
    bt = tip_arm + SPAN * v["s"]
    ax.plot(*np.c_[P(tip_arm), P(bt)], color=BLADE, lw=5, solid_capstyle="round")
    ax.text(*(P(tip_arm) + [0.015, 0.02]), "root", fontsize=7)
    ax.text(*(P(bt) + [0, -0.03]), "tip", fontsize=7, ha="center", va="top")
    ax.plot(*np.c_[P(tip_arm), P(tip_arm - SPAN * z)], color="0.6", lw=0.6, ls=":")
    U = np.array([1.0, 0, 0])
    ua = U @ ym
    mid = P(tip_arm + 0.5 * SPAN * v["s"])
    ax.annotate("", xy=mid + np.array([0.15 * ua, 0.0]), xytext=mid,
                arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.6))
    ax.text(mid[0] + 0.17 * max(ua, 0) + 0.01, mid[1] - 0.01, f"flow along arm\n{ua:+.2f} U", color=WIND, fontsize=8, va="center")
    lab = "tip → root" if incl > 0 else "root → tip"
    ax.text(0.03, -0.42, f"inclination {incl:+.1f}°: flow runs {lab}", fontsize=8.5, ha="center")
    ax.text(0.03, -0.355, "← shaft side   |   arm-tip side →", fontsize=7, color="0.4", ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 0.4)
    ax.set_ylim(-0.46, 0.32)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Side view, normal to the arm", loc="left", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--motor", type=float, default=120.0)
    ap.add_argument("--sweep", type=float, default=30.0, help="lean of the blade from vertical (deg)")
    ap.add_argument("--mounting", type=float, default=84.105,
                    help="angle between chord and arm (deg); 90 = perpendicular")
    args = ap.parse_args()

    args.aoa0 = 90.0 - args.mounting                  # chord angle to the flow at motor 0
    v = rig_vectors(args.motor, args.sweep, args.aoa0)
    aoa, incl = angles([1.0, 0, 0], v)
    print(f"Rig at motor {args.motor:g} deg, sweep {args.sweep:g} deg, AoA at motor 0 = {args.aoa0:g} deg (first principles)")
    print(f"  angle of attack {aoa:+.2f} deg (= {aoa % 360:.2f}),  inclination {incl:+.2f} deg")
    # sanity: the sketch's three cases
    for m in (0, 90, -90):
        a_, i_ = angles([1.0, 0, 0], rig_vectors(m, args.sweep, args.aoa0))
        print(f"  check motor {m:+4d}: AoA {a_:+7.2f}  incl {i_:+6.2f}")
    print(f"Repository transforms (sweep {-args.sweep:g}, mounting {args.mounting:g}):")
    for src in ("master", "branch"):
        ct = load_transform(src)
        r = ct.compute_frame_angles(args.motor, -args.sweep, args.mounting)
        print(f"  {src:6s}: AoA {r['angle_of_attack_deg_shifted']:+8.2f}  incl {r['inclination_angle_deg_shifted']:+7.2f}")

    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.08)
    draw_3d(fig.add_subplot(gs[0], projection="3d"), v, args.motor)
    sub = gs[1].subgridspec(2, 2, wspace=0.06, hspace=0.18)
    for i, m in enumerate((0.0, 90.0, -90.0, args.motor)):
        draw_top(fig.add_subplot(sub[i // 2, i % 2]), m, args.sweep, args.aoa0, highlight=(i == 3))
    fig.text(0.37 + 0.005, 0.93, "(b) From above, as in the sketch", fontsize=10)
    draw_side(fig.add_subplot(gs[2]), v, incl)
    fig.text(0.5, 0.0, f"Sweep {args.sweep:g}° (tip leans toward the shaft), chord at {args.mounting:g}° to the arm.  "
             "Black line / dot: leading edge.  Chord drawn enlarged; arm length assumed.",
             ha="center", fontsize=8, color="0.3")
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"rig_motor{args.motor:g}_mounting{args.mounting:g}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
