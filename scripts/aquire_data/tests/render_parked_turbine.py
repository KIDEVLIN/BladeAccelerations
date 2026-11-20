"""
render_parked_turbine.py

Renders a parked three-bladed turbine in the "bunny ears" position (two
blades up at +-60 deg azimuth, one straight down) at a given yaw error, and
computes the inflow angle of attack and inclination of each blade from
first principles (explicit span / chord / normal vectors), independently of
utils/coordinate_transforms.py. The transform's outputs for the equivalent
test conditions are printed alongside for comparison.

Run from the repo root:
    poetry run python scripts/aquire_data/tests/render_parked_turbine.py [--yaw 30] [--pitch 90]

Conventions (stated here, used everywhere below)
------------------------------------------------
Global frame: x downwind (the wind is U = +x), z up, y = z cross x.
Yaw: the rotor axis a (pointing downwind, hub -> nacelle) is turned by `yaw`
    counter-clockwise seen from above, so a = (cos yaw, sin yaw, 0). With
    yaw > 0 the wind has a component toward the right of an observer who
    stands upwind and looks down the rotor axis.
Front view: seen from upwind along a. "right" = a x z.
Azimuth psi: 0 = blade straight up, positive clockwise in the front view
    (the usual rotation sense). Bunny ears: psi = -60 (left), +60 (right),
    180 (down).
Span s: root -> tip unit vector. Rotation direction t = ds/dpsi.
Pitch: 0 = chord in the rotor plane with the leading edge pointing in the
    rotation direction (chord c = LE->TE = -t, pressure side facing upwind).
    Pitch 90 = feathered: leading edge turned into the wind (c = +a).
Normal n: from the pressure side to the suction side (at pitch 0, n = +a).
Angle of attack: atan2(U.n, U.c) -- the angle of the wind, projected into
    the blade section plane, from the chord line; positive when the wind
    strikes the pressure side. 0 = flow from LE to TE, +-180 = from the TE.
Inclination: asin(-U.s / |U|) -- positive when the wind runs along the
    span from tip to root (the sign convention of the experimentalists'
    sketch and coordinate_transforms).
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIG_DIR = HERE / "figs"

HUB_H = 1.0         # hub height
R = 0.72            # rotor radius
R_ROOT = 0.06       # blade root radius (hub)
OVERHANG = 0.10     # hub centre upwind of tower axis
BLADES = {"left": -60.0, "right": 60.0, "down": 180.0}
COLOR = {"left": "#2a6fdb", "right": "#e0702b", "down": "0.30"}
WIND = "#1b9e77"


# --------------------------------------------------
# Geometry and inflow angles
# --------------------------------------------------

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def frame(yaw_deg):
    g = np.radians(yaw_deg)
    a = np.array([np.cos(g), np.sin(g), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    right = np.cross(a, z)
    return a, right, z


def blade_axes(psi_deg, pitch_deg, a, right, z):
    """Span s, rotation direction t, chord c (LE->TE) and normal n."""
    p = np.radians(psi_deg)
    s = np.cos(p) * z + np.sin(p) * right
    t = -np.sin(p) * z + np.cos(p) * right
    c0, n0 = -t, a                       # pitch 0
    th = np.radians(pitch_deg)           # feathering turns the LE upwind: c -> +a at 90 deg
    c = np.cos(th) * c0 + np.sin(th) * n0
    n = -np.sin(th) * c0 + np.cos(th) * n0
    return s, t, c, n


def inflow_angles(U, s, c, n):
    U = unit(U)
    aoa = np.degrees(np.arctan2(U @ n, U @ c))
    incl = np.degrees(np.arcsin(np.clip(-(U @ s), -1, 1)))
    return aoa, incl


# --------------------------------------------------
# Comparison with the repository transforms
# --------------------------------------------------

def load_transform(source):
    """coordinate_transforms from the working tree ('branch') or from origin/master."""
    if source == "branch":
        path = REPO / "scripts" / "aquire_data" / "utils" / "coordinate_transforms.py"
        code = path.read_text()
    else:
        code = subprocess.run(["git", "show", "origin/master:scripts/aquire_data/utils/coordinate_transforms.py"],
                              cwd=REPO, capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader(f"ct_{source}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(code, mod.__dict__)
    return mod


# --------------------------------------------------
# Drawing helpers
# --------------------------------------------------

def chord_len(r):
    """Tapered planform: max chord near the root, thin tip."""
    x = (r - R_ROOT) / (R - R_ROOT)
    return 0.11 * (1 - 0.72 * x) * (0.75 + 0.25 * np.clip(x / 0.15, 0, 1))


def blade_surface(hub, s, c, n=None):
    """Quads of a flat tapered blade (quarter chord on the span axis)."""
    rs = np.linspace(R_ROOT, R, 18)
    le = [hub + r * s - 0.25 * chord_len(r) * c for r in rs]
    te = [hub + r * s + 0.75 * chord_len(r) * c for r in rs]
    quads = [[le[i], le[i + 1], te[i + 1], te[i]] for i in range(len(rs) - 1)]
    return quads, np.array(le), np.array(te)


def cylinder(p0, p1, rad, k=24):
    ax_ = unit(p1 - p0)
    tmp = np.array([1.0, 0, 0]) if abs(ax_[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(ax_, tmp))
    e2 = np.cross(ax_, e1)
    th = np.linspace(0, 2 * np.pi, k + 1)
    ring0 = [p0 + rad * (np.cos(q) * e1 + np.sin(q) * e2) for q in th]
    ring1 = [p1 + rad * 0.75 * (np.cos(q) * e1 + np.sin(q) * e2) for q in th]
    return [[ring0[i], ring0[i + 1], ring1[i + 1], ring1[i]] for i in range(k)]


def box(center, a, right, z, L, W, H):
    corners = {}
    for i, sa in enumerate((-1, 1)):
        for j, sr in enumerate((-1, 1)):
            for k, sz in enumerate((-1, 1)):
                corners[i, j, k] = center + sa * L / 2 * a + sr * W / 2 * right + sz * H / 2 * z
    faces = []
    for fixed in range(3):
        for v in (0, 1):
            idx = [(0, 0), (0, 1), (1, 1), (1, 0)]
            f = []
            for p, q in idx:
                key = [p, q]
                key.insert(fixed, v)
                f.append(corners[tuple(key)])
            faces.append(f)
    return faces


def sphere(center, rad, k=14):
    u = np.linspace(0, 2 * np.pi, k + 1)
    v = np.linspace(0, np.pi, k // 2 + 1)
    P = lambda uu, vv: center + rad * np.array([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)])
    return [[P(u[i], v[j]), P(u[i + 1], v[j]), P(u[i + 1], v[j + 1]), P(u[i], v[j + 1])]
            for i in range(k) for j in range(k // 2)]


# --------------------------------------------------
# Figure
# --------------------------------------------------

def draw_3d(ax, yaw, pitch, geo):
    a, right, z, hub = geo["a"], geo["right"], geo["z"], geo["hub"]
    polys, cols = [], []
    polys += cylinder(np.array([0, 0, 0.0]), np.array([0, 0, HUB_H - 0.05]), 0.035)
    cols += ["0.80"] * 24
    nac_c = hub + (OVERHANG + 0.08) * a
    polys += box(nac_c, a, right, z, 0.26, 0.09, 0.09)
    cols += ["0.70"] * 6
    polys += sphere(hub, 0.055)
    cols += ["0.62"] * (14 * 7)
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors="none", alpha=1.0)
    ax.add_collection3d(pc)
    for name, psi in BLADES.items():
        s, t, c, n = geo["axes"][name]
        quads, le, te = blade_surface(hub, s, c)
        ax.add_collection3d(Poly3DCollection(quads, facecolors=COLOR[name], edgecolors="none", alpha=0.95))
        ax.plot(*le.T, color="k", lw=1.2)                 # leading edge
    # wind arrows upstream
    for dy in (-0.35, 0.0, 0.35):
        for dz in (0.65, 1.25):
            p = np.array([-1.05, dy, dz])
            ax.quiver(*p, 0.45, 0, 0, color=WIND, lw=1.4, arrow_length_ratio=0.25)
    ax.text(-1.05, -0.35, 1.42, "wind", color=WIND, fontsize=9)
    # ground disc
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(0.9 * np.cos(th), 0.9 * np.sin(th), 0 * th, color="0.85", lw=0.8)
    ax.set_xlim(-1.1, 0.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0, 2.0)
    ax.set_box_aspect((2.0, 2.0, 2.0))
    ax.view_init(elev=14, azim=-72)
    ax.set_axis_off()
    ax.set_title(f"(a) Parked, bunny ears, yaw {yaw:g}°, pitch {pitch:g}°", loc="left", fontsize=10)


def draw_top(ax, yaw, geo):
    a, right, hub = geo["a"], geo["right"], geo["hub"]
    ax.add_patch(plt.Circle((0, 0), 0.035, color="0.75"))
    nac = [hub + OVERHANG * a + d for d in (-0.05 * a, 0.21 * a)]
    ax.plot([nac[0][0], nac[1][0]], [nac[0][1], nac[1][1]], color="0.6", lw=7, solid_capstyle="butt")
    for name in ("left", "right", "down"):
        s, t, c, n = geo["axes"][name]
        quads, le, te = blade_surface(hub, s, c)
        poly = np.r_[le[:, :2], te[::-1, :2]]
        ax.fill(poly[:, 0], poly[:, 1], color=COLOR[name], alpha=0.85, lw=0, zorder=3 if name != "down" else 2)
    # rotor axis and wind
    ax.annotate("", xy=hub[:2] + 0.65 * a[:2], xytext=hub[:2] - 0.55 * a[:2],
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8, ls="--"))
    ax.text(*(hub[:2] + 0.68 * a[:2]), "rotor axis", fontsize=8, va="center")
    ax.annotate("", xy=(-0.35, -0.85), xytext=(-0.95, -0.85), arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.6))
    ax.text(-0.95, -0.78, "wind", color=WIND, fontsize=9)
    arc = Arc(hub[:2], 0.7, 0.7, theta1=0, theta2=yaw, color="k", lw=0.8)
    ax.add_patch(arc)
    ax.plot([hub[0], hub[0] + 0.45], [hub[1], hub[1]], color="0.4", lw=0.6, ls=":")
    ax.text(hub[0] + 0.38, hub[1] + 0.06, f"yaw {yaw:g}°", fontsize=8)
    ax.text(hub[0] + 0.0, hub[1] + 0.68, "left blade (seen from upwind)", fontsize=7, color=COLOR["left"], ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) From above", loc="left", fontsize=10)


def draw_front(ax, yaw, geo):
    """Looking downwind along the rotor axis from upwind: right = a x z."""
    a, right, z, hub = geo["a"], geo["right"], geo["z"], geo["hub"]
    P = lambda p: np.array([(p - hub) @ right, (p - hub) @ z + HUB_H])
    ax.plot([0, 0], [0, HUB_H - 0.05], color="0.85", lw=6, solid_capstyle="butt", zorder=1)
    ax.add_patch(plt.Circle((0, HUB_H), R, fill=False, ls=":", color="0.7"))
    for name in ("down", "left", "right"):
        s, t, c, n = geo["axes"][name]
        quads, le, te = blade_surface(hub, s, c)
        # feathered blades are edge-on from upwind: draw the span line with a
        # width standing in for the blade thickness, plus the projected planform
        poly = np.array([P(p) for p in np.r_[le, te[::-1]]])
        ax.fill(poly[:, 0], poly[:, 1], color=COLOR[name], alpha=0.9, lw=0, zorder=3)
        root, tip = P(hub + R_ROOT * s), P(hub + R * s)
        ax.plot([root[0], tip[0]], [root[1], tip[1]], color=COLOR[name], lw=4, solid_capstyle="round", zorder=3)
        lab = {"left": "left\n$\\psi=-60°$", "right": "right\n$\\psi=+60°$", "down": "down $\\psi=180°$"}[name]
        if name == "down":
            ax.text(tip[0] + 0.08, tip[1] + 0.05, lab, color=COLOR[name], ha="left", va="bottom", fontsize=8)
        else:
            ax.text(tip[0] * 1.12, tip[1] + 0.08, lab, color=COLOR[name], ha="center", va="bottom", fontsize=8)
    ax.add_patch(plt.Circle((0, HUB_H), 0.05, color="0.6", zorder=4))
    # in-plane component of the wind (what moves along the blades)
    U = np.array([1.0, 0, 0])
    lat = U @ right
    ax.annotate("", xy=(-0.85 + 0.6 * lat, 1.85), xytext=(-0.85, 1.85),
                arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.6))
    ax.text(-0.85 + 0.6 * lat + 0.05, 1.85, f"in-plane wind\nU sin(yaw) = {lat:.2f} U", color=WIND, fontsize=8, va="center")
    ax.annotate("", xy=(0.30, 1.70), xytext=(-0.05, 1.77),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=0.8, connectionstyle="arc3,rad=-0.2"))
    ax.text(0.12, 1.80, "rotation", fontsize=7, color="0.4", ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) From upwind, along the rotor axis", loc="left", fontsize=10)


def airfoil(m=0.04, p=0.4, th=0.15, k=80):
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    yt = 5 * th * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
    return np.r_[x, x[::-1]], np.r_[yc + yt, (yc - yt)[::-1]]


def draw_section(ax, name, aoa, color):
    """Section view: chord to the right (LE at left), suction side up. The
    projected wind arrives at angle `aoa` below the chord when positive (it
    strikes the lower, pressure side)."""
    x, y = airfoil()
    ax.fill(x - 0.25, y, color=color, alpha=0.9, lw=0)
    ax.plot([-0.25, 0.75], [0, 0], color="k", lw=0.6, ls="--")
    ax.text(-0.3, 0.0, "LE", ha="right", va="center", fontsize=7)
    ax.text(0.8, 0.0, "TE", ha="left", va="center", fontsize=7)
    ax.text(0.27, -0.12, "pressure side", fontsize=6.5, ha="center", color="0.35")
    ang = np.radians(aoa)
    d = np.array([np.cos(ang), np.sin(ang)])
    # positive AoA: wind has +n component (pressure -> suction = up), so it comes from below
    start = -0.95 * d
    ax.annotate("", xy=-0.32 * d, xytext=start, arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.6))
    ax.add_patch(Arc((0, 0), 1.0, 1.0, theta1=min(180, 180 + aoa), theta2=max(180, 180 + aoa), color="k", lw=0.8))
    ax.text(-0.62, 0.05 if aoa < 0 else -0.08, f"{aoa:+.1f}°", fontsize=9, ha="right",
            va="bottom" if aoa < 0 else "top")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 0.95)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.set_title(f"{name} blade: angle of attack", loc="left", fontsize=9, color=color)


def draw_incl(ax, name, incl, color):
    """Side view of the blade, span vertical (tip up). The wind is drawn at
    its angle to the plane normal to the span: tipward component -> pointing
    up the page (root->tip, negative inclination)."""
    ax.fill([-0.04, 0.04, 0.025, -0.025], [-0.5, -0.5, 0.55, 0.55], color=color, alpha=0.9, lw=0)
    ax.text(0, 0.6, "tip", ha="center", fontsize=7)
    ax.text(0, -0.58, "root", ha="center", va="top", fontsize=7)
    ax.plot([-0.75, 0.75], [0, 0], color="k", lw=0.6, ls="--")
    ang = np.radians(-incl)                 # positive inclination = tip->root = pointing down the page
    d = np.array([np.cos(ang), np.sin(ang)])
    ax.annotate("", xy=-0.06 * d, xytext=-0.78 * d, arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.6))
    ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=min(180, 180 - incl), theta2=max(180, 180 - incl), color="k", lw=0.8))
    ax.text(-0.5, 0.05 if incl > 0 else -0.06, f"{incl:+.1f}°", fontsize=9, ha="right",
            va="bottom" if incl > 0 else "top")
    lab = "tip → root" if incl > 0 else "root → tip"
    ax.text(0.0, -0.78, f"wind runs {lab}", fontsize=7.5, ha="center", color="0.3")
    ax.set_aspect("equal")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.9, 0.7)
    ax.axis("off")
    ax.set_title(f"{name} blade: inclination", loc="left", fontsize=9, color=color)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--yaw", type=float, default=30.0, help="yaw error in degrees")
    ap.add_argument("--pitch", type=float, default=90.0, help="blade pitch in degrees (90 = feathered)")
    args = ap.parse_args()

    a, right, z = frame(args.yaw)
    hub = np.array([0.0, 0.0, HUB_H]) - OVERHANG * a
    U = np.array([1.0, 0.0, 0.0])
    geo = dict(a=a, right=right, z=z, hub=hub,
               axes={k: blade_axes(psi, args.pitch, a, right, z) for k, psi in BLADES.items()})
    res = {k: inflow_angles(U, *[geo["axes"][k][i] for i in (0, 2, 3)]) for k in BLADES}

    print(f"Parked, bunny ears, yaw {args.yaw:g} deg, pitch {args.pitch:g} deg (first principles)")
    print(f"{'blade':8s}{'azimuth':>9s}{'AoA':>9s}{'incl':>9s}")
    for k, psi in BLADES.items():
        print(f"{k:8s}{psi:9.0f}{res[k][0]:9.2f}{res[k][1]:9.2f}")

    # Same conditions through the repository transforms: motor = yaw, sweep = -azimuth
    # (as in plot_test_matrix_vs_turbine.py); mounting = AoA at zero yaw (branch) or pitch (master).
    print("\nRepository transforms, motor = yaw, sweep = -azimuth:")
    for src in ("branch", "master"):
        try:
            ct = load_transform(src)
        except Exception as e:                         # noqa: BLE001
            print(f"  {src}: could not load ({e})")
            continue
        for k, psi in BLADES.items():
            if k == "down":
                continue
            if src == "branch":
                mount = 90.0 - args.pitch                   # AoA at zero yaw for this pitch
            else:
                mount = args.pitch
            r = ct.compute_frame_angles(args.yaw, -psi, mount)
            print(f"  {src:6s} {k:6s}: AoA {r['angle_of_attack_deg_shifted']:8.2f}  incl {r['inclination_angle_deg_shifted']:7.2f}")

    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 0.9], hspace=0.05, wspace=0.08)
    ax3 = fig.add_subplot(gs[0, :2], projection="3d")
    draw_3d(ax3, args.yaw, args.pitch, geo)
    draw_top(fig.add_subplot(gs[0, 2]), args.yaw, geo)
    draw_front(fig.add_subplot(gs[0, 3]), args.yaw, geo)
    for i, k in enumerate(("left", "right")):
        draw_section(fig.add_subplot(gs[1, 2 * i]), k.capitalize(), res[k][0], COLOR[k])
        draw_incl(fig.add_subplot(gs[1, 2 * i + 1]), k.capitalize(), res[k][1], COLOR[k])
    fig.text(0.5, 0.035,
             "Black line on each blade: leading edge.  Left/right as seen from upwind.  "
             "AoA > 0: wind strikes the pressure side.  Inclination > 0: wind runs along the span from tip to root.",
             ha="center", fontsize=8, color="0.3")
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"parked_bunny_ears_yaw{args.yaw:g}_pitch{args.pitch:g}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
