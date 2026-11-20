"""
test_coordinate_transforms.py

Checks utils/coordinate_transforms.py against
  1. the experimentalists' sketch of the rig (29 Sep 2026),
  2. an independent construction from explicit arm / span / chord vectors,
  3. the collaborator's original implementation (origin/master, frozen
     below), which agrees for horizontal tunnel flow,
  4. a hand calculation of the production case at motor 120 deg, and
  5. geometric properties that must hold for any rig.

Run from the repo root with:
    poetry run python -m unittest discover -s scripts/aquire_data/tests -v
"""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import coordinate_transforms as ct  # noqa: E402

DEG = np.pi / 180

# Geometry of all 32 production runs (run_metadata.json)
PROD_SWEEP = -30.0
PROD_MOUNTING = 84.105

MOTOR_GRID = np.arange(-180.0, 180.0, 7.5)
SWEEP_GRID = [-45.0, -30.0, -10.0, 0.0, 15.0, 30.0]
MOUNTING_GRID = [60.0, 84.105, 90.0, 110.0]


def angdiff(a, b):
    """Smallest signed difference between two angles in degrees."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


# --------------------------------------------------
# Independent references
# --------------------------------------------------

def rig_reference(motor_deg, sweep_deg, mounting_deg, U=(1.0, 0.0, 0.0)):
    """Inclination and angle of attack (deg) built directly from vectors in
    the global frame, without the module's rotation matrices.

    arm a: horizontal, across the flow at motor 0 (+y), clockwise from above
    p:     horizontal, perpendicular to the arm, = +x at motor 0 (p x a = z)
    r:     tip -> root along the span; sweep tilts it from vertical toward
           -sin(sweep) * a (negative sweep: root outward, tip toward shaft)
    chord: in the section plane (normal to r), at `mounting` to the arm,
           = p when mounting = 90 (leading edge upstream at motor 0)
    """
    th, sw, mt = np.radians([motor_deg, sweep_deg, mounting_deg])
    U = np.asarray(U, dtype=float)
    U = U / np.linalg.norm(U)
    z = np.array([0.0, 0.0, 1.0])
    a = np.array([np.sin(th), np.cos(th), 0.0])
    p = np.array([np.cos(th), -np.sin(th), 0.0])
    r = np.cos(sw) * z - np.sin(sw) * a
    a_s = a - (a @ r) * r
    a_s /= np.linalg.norm(a_s)                 # arm direction within the section plane
    c = np.sin(mt) * p - np.cos(mt) * a_s      # chord, LE -> TE
    n = np.cos(mt) * p + np.sin(mt) * a_s      # chord turned 90 deg toward the arm tip side
    incl = np.degrees(np.arcsin(np.clip(U @ r, -1, 1)))
    aoa = np.degrees(np.arctan2(U @ n, U @ c))
    return incl, aoa


def master_reference(motor_deg, sweep_deg, mounting_deg, U=(1.0, 0.0, 0.0)):
    """The collaborator's original transform (origin/master, unchanged since
    2 Sep 2026), frozen here: turbine-style frames, blade z root -> tip,
    mounting angle used as a pitch. Returns (inclination, AoA) in degrees."""
    yaw, az, pitch = np.radians([motor_deg, sweep_deg, mounting_deg])
    R1 = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, np.cos(az), -np.sin(az)], [0, np.sin(az), np.cos(az)]])
    u, v, w = R2 @ (R1 @ np.asarray(U, dtype=float))
    incl = np.arctan2(-w, np.sqrt(u ** 2 + v ** 2))
    aoa = np.arctan2(u, -v) - pitch
    return np.degrees(incl), np.degrees(aoa)


def module(motor_deg, sweep_deg, mounting_deg, U=(1.0, 0.0, 0.0)):
    r = ct.compute_frame_angles(motor_deg, sweep_deg, mounting_deg, U_global=U)
    return r["inclination_angle_deg_shifted"], r["angle_of_attack_deg_shifted"]


# --------------------------------------------------
# 1. The experimentalists' sketch
# --------------------------------------------------

class TestSketch(unittest.TestCase):
    """Seen from above with the flow left to right: motor 0 -> arm across
    the flow, inclination 0, chord along the flow; motor +90 -> arm
    downstream, inclination +30; motor -90 -> arm upstream, inclination
    -30. These hold for the production sweep (-30 deg)."""

    def test_inclination_cases(self):
        for motor, expected in ((0.0, 0.0), (90.0, 30.0), (-90.0, -30.0)):
            for mounting in MOUNTING_GRID:
                with self.subTest(motor=motor, mounting=mounting):
                    incl, _ = module(motor, PROD_SWEEP, mounting)
                    self.assertAlmostEqual(incl, expected, places=9)

    def test_motor_turns_arm_clockwise_from_above(self):
        # At +90 the arm points downstream: the flow lies along mount +y.
        np.testing.assert_allclose(ct.global_to_mount([1.0, 0, 0], 90 * DEG), [0, 1, 0], atol=1e-12)
        # At -90 it points upstream: the flow runs along mount -y.
        np.testing.assert_allclose(ct.global_to_mount([1.0, 0, 0], -90 * DEG), [0, -1, 0], atol=1e-12)

    def test_chord_along_flow_at_motor_0(self):
        # Chord perpendicular to the arm -> AoA 0 at motor 0 for any sweep.
        for sweep in SWEEP_GRID:
            with self.subTest(sweep=sweep):
                _, aoa = module(0.0, sweep, 90.0)
                self.assertAlmostEqual(aoa, 0.0, places=9)


# --------------------------------------------------
# 2. Independent vector construction (any flow direction)
# --------------------------------------------------

class TestAgainstVectorReference(unittest.TestCase):

    def check(self, U):
        worst_i = worst_a = 0.0
        for motor, sweep, mounting in itertools.product(MOTOR_GRID, SWEEP_GRID, MOUNTING_GRID):
            i_m, a_m = module(motor, sweep, mounting, U)
            i_r, a_r = rig_reference(motor, sweep, mounting, U)
            worst_i = max(worst_i, abs(i_m - i_r))
            if abs(i_r) < 89.9:                    # AoA undefined for flow along the span
                worst_a = max(worst_a, abs(angdiff(a_m, a_r)))
        self.assertLess(worst_i, 1e-9)
        self.assertLess(worst_a, 1e-9)

    def test_horizontal_flow(self):
        self.check((1.0, 0.0, 0.0))

    def test_flow_with_vertical_and_lateral_components(self):
        # Exercises every component of both rotations.
        self.check((1.0, 0.3, -0.4))
        self.check((0.2, -0.7, 0.5))


# --------------------------------------------------
# 3. The collaborator's original implementation
# --------------------------------------------------

class TestAgainstMaster(unittest.TestCase):
    """origin/master is built in turbine-style frames (blade z root -> tip,
    pitch instead of the chord-to-arm angle) but describes the same rig;
    for horizontal flow the angles must agree exactly. (With a vertical flow
    component they differ: master's blade frame stands the blade up.)"""

    def test_horizontal_flow_grid(self):
        for motor, sweep, mounting in itertools.product(MOTOR_GRID, SWEEP_GRID, MOUNTING_GRID):
            with self.subTest(motor=motor, sweep=sweep, mounting=mounting):
                i_m, a_m = module(motor, sweep, mounting)
                i_k, a_k = master_reference(motor, sweep, mounting)
                self.assertAlmostEqual(i_m, i_k, places=9)
                self.assertAlmostEqual(angdiff(a_m, a_k), 0.0, places=9)


# --------------------------------------------------
# 4. Hand calculation: production geometry, motor 120 deg
# --------------------------------------------------

class TestProductionCase120(unittest.TestCase):
    """Flow in mount components at motor 120: (cos 120, sin 120, 0) =
    (-0.5, 0.866, 0). The swept z axis (tip -> root) is (0, 0.5, 0.866) in
    mount components for sweep -30, so w_s = 0.866 * 0.5 = 0.433 and
    inclination = asin(0.433) = +25.659 deg (tip -> root). The swept y axis
    is (0, 0.866, -0.5), so v_s = 0.75, u_s = -0.5, and the in-plane flow
    angle is atan2(0.75, -0.5) = 123.690 deg; adding 90 - 84.105 = 5.895
    gives AoA = 129.585 deg."""

    def test_values(self):
        incl, aoa = module(120.0, PROD_SWEEP, PROD_MOUNTING)
        self.assertAlmostEqual(incl, np.degrees(np.arcsin(0.5 * np.sqrt(3) / 2)), places=9)
        self.assertAlmostEqual(incl, 25.659, places=3)
        self.assertAlmostEqual(aoa, np.degrees(np.arctan2(0.75, -0.5)) + 90 - PROD_MOUNTING, places=9)
        self.assertAlmostEqual(aoa, 129.585, places=3)

    def test_not_the_1_oct_values(self):
        # The 1 Oct 2026 version gave incl -25.659, AoA -152.205 here.
        incl, aoa = module(120.0, PROD_SWEEP, PROD_MOUNTING)
        self.assertGreater(incl, 0)
        self.assertGreater(abs(angdiff(aoa, -152.205)), 70)


# --------------------------------------------------
# 5. Geometric properties
# --------------------------------------------------

class TestProperties(unittest.TestCase):

    def test_no_sweep_no_inclination(self):
        for motor in MOTOR_GRID:
            self.assertAlmostEqual(module(motor, 0.0, PROD_MOUNTING)[0], 0.0, places=9)

    def test_inclination_bounded_by_sweep_and_peaks_with_arm_along_flow(self):
        for sweep in SWEEP_GRID:
            incl = np.array([module(m, sweep, PROD_MOUNTING)[0] for m in MOTOR_GRID])
            self.assertLessEqual(np.abs(incl).max(), abs(sweep) + 1e-9)
            self.assertAlmostEqual(abs(module(90.0, sweep, PROD_MOUNTING)[0]), abs(sweep), places=9)

    def test_arm_across_flow_no_inclination(self):
        for sweep in SWEEP_GRID:
            for motor in (0.0, 180.0, -180.0):
                self.assertAlmostEqual(module(motor, sweep, PROD_MOUNTING)[0], 0.0, places=9)

    def test_aoa_at_motor_0_is_90_minus_mounting(self):
        for sweep, mounting in itertools.product(SWEEP_GRID, MOUNTING_GRID):
            self.assertAlmostEqual(module(0.0, sweep, mounting)[1], 90.0 - mounting, places=9)

    def test_flow_along_arm_is_aoa_90_for_perpendicular_chord(self):
        for sweep in SWEEP_GRID:
            self.assertAlmostEqual(module(90.0, sweep, 90.0)[1], 90.0, places=9)
            self.assertAlmostEqual(module(-90.0, sweep, 90.0)[1], -90.0, places=9)

    def test_aoa_grows_with_motor_angle(self):
        for sweep in SWEEP_GRID:
            aoa = np.unwrap(np.radians([module(m, sweep, PROD_MOUNTING)[1] for m in MOTOR_GRID]))
            self.assertTrue(np.all(np.diff(aoa) > 0))

    def test_mounting_angle_only_shifts_aoa(self):
        for motor, sweep in itertools.product(MOTOR_GRID, SWEEP_GRID):
            i1, a1 = module(motor, sweep, 90.0)
            i2, a2 = module(motor, sweep, 84.105)
            self.assertAlmostEqual(i1, i2, places=9)
            self.assertAlmostEqual(angdiff(a2, a1), 5.895, places=9)

    def test_mirror_sweep_flips_inclination_only(self):
        for motor in MOTOR_GRID:
            i1, a1 = module(motor, -30.0, PROD_MOUNTING)
            i2, a2 = module(motor, 30.0, PROD_MOUNTING)
            self.assertAlmostEqual(i1, -i2, places=9)
            self.assertAlmostEqual(angdiff(a1, a2), 0.0, places=9)


# --------------------------------------------------
# Low-level functions
# --------------------------------------------------

class TestLowLevel(unittest.TestCase):

    def test_identity_rotations(self):
        U = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(ct.global_to_mount(U, 0.0), U)
        np.testing.assert_allclose(ct.mount_to_swept(U, 0.0), U)

    def test_swept_z_is_tip_to_root(self):
        # For sweep -30 the tip -> root axis is (0, sin 30, cos 30) in mount
        # components (root outward of the tip); it must map onto swept +z.
        r_mount = [0.0, np.sin(30 * DEG), np.cos(30 * DEG)]
        np.testing.assert_allclose(ct.mount_to_swept(r_mount, -30 * DEG), [0, 0, 1], atol=1e-12)

    def test_inclination_sign(self):
        self.assertAlmostEqual(ct.inclination_from_swept([1.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(ct.inclination_from_swept([0.0, 0.0, 1.0]), 90 * DEG)    # tip -> root
        self.assertAlmostEqual(ct.inclination_from_swept([0.0, 0.0, -1.0]), -90 * DEG)  # root -> tip
        self.assertAlmostEqual(ct.inclination_from_swept([1.0, 0.0, 1.0]), 45 * DEG)

    def test_aoa_low_level(self):
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, 0.0, 0.0], 90 * DEG), 0.0)
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, 1.0, 0.0], 90 * DEG), 45 * DEG)
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, 1.0, 5.0], 90 * DEG),
                               ct.angle_of_attack_from_swept([1.0, 1.0, 0.0], 90 * DEG))

    def test_wrap(self):
        self.assertAlmostEqual(ct._wrap_deg(190.0), -170.0)
        self.assertAlmostEqual(ct._wrap_deg(-190.0), 170.0)
        self.assertAlmostEqual(ct._wrap_deg(180.0), -180.0)


if __name__ == "__main__":
    unittest.main()
