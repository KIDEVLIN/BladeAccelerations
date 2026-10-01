"""
test_coordinate_transforms.py

Lightweight sanity checks for utils/coordinate_transforms.py using cases
that are easy to work out by hand.

Run from the repo root with:
    python3 -m unittest discover -s scripts/aquire_data/tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import coordinate_transforms as ct  # noqa: E402


DEG = np.pi / 180


# --------------------------------------------------
# Hand-checked test conditions (mounting angle 0, sweep angle 30 deg)
# --------------------------------------------------

SWEEP_ANGLE_DEG = 30.0
MOUNTING_ANGLE_DEG = 0.0

# motor_angle_deg -> (expected inclination_deg, expected angle_of_attack_deg)
EXPECTED_FRAME_ANGLES = {
    0.0: (0, 0),
    90.0: (30, 90),
    -90.0: (-30, -90),
}


class TestComputeFrameAngles(unittest.TestCase):

    def test_expected_cases(self):
        for motor_angle_deg, (inclination_deg, aoa_deg) in EXPECTED_FRAME_ANGLES.items():
            with self.subTest(motor_angle_deg=motor_angle_deg):
                if inclination_deg is None or aoa_deg is None:
                    self.skipTest(f"expected values not set for motor angle {motor_angle_deg}")
                result = ct.compute_frame_angles(motor_angle_deg, SWEEP_ANGLE_DEG,
                                                 MOUNTING_ANGLE_DEG)
                self.assertAlmostEqual(result["inclination_angle_deg_shifted"],
                                       inclination_deg, places=6)
                self.assertAlmostEqual(result["angle_of_attack_deg_shifted"],
                                       aoa_deg, places=6)



# --------------------------------------------------
# Frame transforms
# --------------------------------------------------


class TestGlobalToMount(unittest.TestCase):

    def test_zero_motor_angle_is_identity(self):
        U = np.array([1.0, 0, 0])
        np.testing.assert_allclose(ct.global_to_mount(U, 0.0), U)

    def test_90_deg_rotation(self):
        np.testing.assert_allclose(ct.global_to_mount([1.0, 0.0, 0.0], 90 * DEG),
                                   [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(ct.global_to_mount([1.0, 0.0, 0.0], -90 * DEG),
                                   [0.0, -1.0, 0.0], atol=1e-12)



class TestMountToSwept(unittest.TestCase):

    def test_zero_sweep_is_identity(self):
        U = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(ct.mount_to_swept(U, 0.0), U)

    def test_90_deg_rotation(self):
        np.testing.assert_allclose(ct.mount_to_swept([0.0, 1.0, 0.0], 90 * DEG),
                                   [0.0, 0.0, 1.0], atol=1e-12)


# --------------------------------------------------
# Angle extraction
# --------------------------------------------------


class TestInclinationFromSwept(unittest.TestCase):

    def test_flow_normal_to_blade(self):
        self.assertAlmostEqual(ct.inclination_from_swept([1.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(ct.inclination_from_swept([0.0, 1.0, 0.0]), 0.0)

    def test_flow_along_blade(self):
        # Tip-to-root (negative z) is positive inclination.
        self.assertAlmostEqual(ct.inclination_from_swept([0.0, 0.0, 1.0]), 90 * DEG)
        self.assertAlmostEqual(ct.inclination_from_swept([0.0, 0.0, -1.0]), -90 * DEG)

    def test_45_deg(self):
        self.assertAlmostEqual(ct.inclination_from_swept([1.0, 0.0, 1.0]), 45 * DEG)
        self.assertAlmostEqual(ct.inclination_from_swept([0.0, -1.0, -1.0]), -45 * DEG)


class TestAngleOfAttackFromSwept(unittest.TestCase):

    def test_flow_along_x(self):
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, 0.0, 0.0], 0.0), 0.0)

    def test_45_deg(self):
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, 1.0, 0.0], 0.0), 45 * DEG)
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, -1.0, 0.0], 0.0), -45 * DEG)

    def test_mounting_angle_offset(self):
        U = [1.0, 0.5, 0.0]
        base = ct.angle_of_attack_from_swept(U, 0.0)
        self.assertAlmostEqual(ct.angle_of_attack_from_swept(U, 5.895 * DEG),
                               base + 5.895 * DEG)

    def test_independent_of_vertical_component(self):
        self.assertAlmostEqual(ct.angle_of_attack_from_swept([1.0, 1.0, 5.0], 0.0),
                               ct.angle_of_attack_from_swept([1.0, 1.0, 0.0], 0.0))


if __name__ == "__main__":
    unittest.main()
