"""
plot_test_matrix.py

Scatter plot of angle of attack vs inclination angle across the test
matrix: mounting angle 0 deg, sweep angle 30 deg, and motor angles from
-180 to 180 deg, colored by motor angle.

Run from the repo root with:
    python3 scripts/aquire_data/tests/plot_test_matrix.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import coordinate_transforms as ct  # noqa: E402


# --------------------------------------------------
# Test matrix
# --------------------------------------------------

MOUNTING_ANGLE_DEG = 0.0
SWEEP_ANGLE_DEG = 30.0
MOTOR_ANGLES_DEG = np.arange(-180, 181, 10)

FIG_DIR = Path(__file__).resolve().parent / "figs"


def main():
    fig, ax = plt.subplots(figsize=(7, 5))

    inclination_deg = []
    aoa_deg = []
    for motor_angle_deg in MOTOR_ANGLES_DEG:
        angles = ct.compute_frame_angles(
            motor_angle_deg, SWEEP_ANGLE_DEG, MOUNTING_ANGLE_DEG
        )
        inclination_deg.append(angles["inclination_angle_deg_shifted"])
        aoa_deg.append(angles["angle_of_attack_deg_shifted"])

    # Cyclic colormap since motor angle -180 and 180 are the same position
    sc = ax.scatter(inclination_deg, aoa_deg, c=MOTOR_ANGLES_DEG, s=25,
                    cmap="twilight_shifted", vmin=-180, vmax=180)
    cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(-180, 181, 45))
    cbar.set_label("Motor angle (deg)")

    ax.set_xlabel("Inclination angle (deg)")
    ax.set_ylabel("Angle of attack (deg)")
    ax.set_title(f"Test matrix (mounting angle = {MOUNTING_ANGLE_DEG:g}°, "
                 f"sweep angle = {SWEEP_ANGLE_DEG:g}°)")
    ax.set_yticks(np.arange(-180, 181, 45))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out_path = FIG_DIR / "test_matrix_aoa_vs_inclination.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
