"""
test_angle_chain.py

End-to-end checks of the chain that turns a motor position into the flow
angles stored with the data:

    main.py constants (sweep, chord offset -> mounting angle)
    motor.py / run_loader.py  encoder counts <-> motor angle (sign, scale)
    coordinate_transforms     motor angle -> inclination, angle of attack
    coordinate_frame_angles.csv   written at acquisition (lab PC)
    angle_summary.csv / master_angles.csv   written by extract_run_summary.py

test_coordinate_transforms.py checks the transform itself; this file checks
that every link around it uses it the same way. The data tests run only
where data/Production_10-8-26 and data/Production_summaries exist (the data
are not in git).

Run from the repo root with:
    poetry run python -m unittest discover -s scripts/aquire_data/tests -v
"""

import ast
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

AQ = Path(__file__).resolve().parents[1]
REPO = AQ.parents[1]
sys.path.insert(0, str(AQ))
from utils import coordinate_transforms as ct  # noqa: E402
from utils.run_loader import load_encoder_log, windowed_average_angle  # noqa: E402

RAW = REPO / "data" / "Production_10-8-26"
SUMM = REPO / "data" / "Production_summaries"
HAVE_DATA = RAW.is_dir() and SUMM.is_dir()


def angdiff(a, b):
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def module_constants(path, names):
    """Top-level constants of a script that cannot be imported here (main.py
    and motor.py need the lab hardware libraries): evaluates simple
    assignments in order, so later ones may use earlier ones."""
    env = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(path), "eval"),
                                               {"__builtins__": {}}, dict(env))
            except Exception:                                   # noqa: BLE001 -- non-constant, skip
                pass
    return {n: env[n] for n in names}


def motor_conversions():
    """Motor.degrees_to_counts / counts_to_degrees, lifted out of motor.py."""
    tree = ast.parse((AQ / "motor.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Motor")
    funcs = [f for f in cls.body if isinstance(f, ast.FunctionDef)
             and f.name in ("degrees_to_counts", "counts_to_degrees")]
    stub = ast.Module(body=[ast.ClassDef(name="M", bases=[], keywords=[], body=funcs, decorator_list=[])],
                      type_ignores=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(stub), "motor.py", "exec"), ns)
    return ns["M"]


def production_runs():
    return sorted(d for d in RAW.iterdir() if (d / "run_metadata.json").exists()) if HAVE_DATA else []


# --------------------------------------------------
# Acquisition constants
# --------------------------------------------------

class TestAcquisitionConstants(unittest.TestCase):

    def setUp(self):
        self.c = module_constants(AQ / "main.py", ["SWEEP_ANGLE_DEG", "CHORD_OFFSET_DEG", "MOUNTING_ANGLE_DEG"])

    def test_values(self):
        self.assertEqual(self.c["SWEEP_ANGLE_DEG"], -30.0)
        self.assertEqual(self.c["CHORD_OFFSET_DEG"], 5.895)
        self.assertEqual(self.c["MOUNTING_ANGLE_DEG"], 90.0 - self.c["CHORD_OFFSET_DEG"])
        # bit-identical to the value the production runs stored (-5.895 + 90)
        self.assertEqual(self.c["MOUNTING_ANGLE_DEG"], -5.895 + 90)

    def test_chord_offset_is_aoa_at_motor_0(self):
        r = ct.compute_frame_angles(0.0, self.c["SWEEP_ANGLE_DEG"], self.c["MOUNTING_ANGLE_DEG"])
        self.assertAlmostEqual(r["angle_of_attack_deg_shifted"], self.c["CHORD_OFFSET_DEG"], places=9)
        self.assertAlmostEqual(r["inclination_angle_deg_shifted"], 0.0, places=9)

    def test_sketch_with_acquisition_constants(self):
        for motor, incl in ((90.0, 30.0), (-90.0, -30.0)):
            r = ct.compute_frame_angles(motor, self.c["SWEEP_ANGLE_DEG"], self.c["MOUNTING_ANGLE_DEG"])
            self.assertAlmostEqual(r["inclination_angle_deg_shifted"], incl, places=9)


# --------------------------------------------------
# Encoder counts <-> motor angle
# --------------------------------------------------

class TestMotorAngleSign(unittest.TestCase):

    def setUp(self):
        self.M = motor_conversions()
        self.m = self.M()
        self.m.counts_per_deg = 694.44

    def test_round_trip(self):
        for deg in (-180.0, -90.0, -12.3, 0.0, 45.0, 120.0, 270.0):
            self.assertAlmostEqual(self.m.counts_to_degrees(self.m.degrees_to_counts(deg)), deg,
                                   delta=0.5 / 694.44)

    def test_run_loader_uses_the_motor_sign(self):
        # An encoder log held at the counts for 120 deg must average to +120 deg.
        for deg in (120.0, -75.0):
            counts = self.m.degrees_to_counts(deg)
            log = pd.DataFrame({"time_sec": np.arange(0, 70, 0.01), "encoder_counts": counts})
            mean, n, sd = windowed_average_angle(log, abs_start=100.0, window_start=105.0,
                                                 duration_s=60.0, counts_per_deg=694.44)
            self.assertGreater(n, 0)
            self.assertAlmostEqual(mean, self.m.counts_to_degrees(counts), places=9)
            self.assertAlmostEqual(mean, deg, delta=0.5 / 694.44)


# --------------------------------------------------
# The recorded data
# --------------------------------------------------

@unittest.skipUnless(HAVE_DATA, "production data not present")
class TestRecordedData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = module_constants(AQ / "main.py", ["SWEEP_ANGLE_DEG", "MOUNTING_ANGLE_DEG"])
        cls.runs = production_runs()

    def test_all_runs_found(self):
        self.assertEqual(len(self.runs), 32)

    def test_metadata_matches_main(self):
        for d in self.runs:
            md = json.loads((d / "run_metadata.json").read_text())
            with self.subTest(run=d.name):
                self.assertEqual(md["sweep_angle_deg"], self.c["SWEEP_ANGLE_DEG"])
                self.assertEqual(md["mounting_angle_deg"], self.c["MOUNTING_ANGLE_DEG"])

    def test_acquisition_angles_match_fixed_transform(self):
        """coordinate_frame_angles.csv was written on the lab PC by master's
        transform during each run. Recomputing every row from its measured
        motor angle with the fixed transform must give the same angles."""
        n = 0
        for d in self.runs:
            f = pd.read_csv(d / "coordinate_frame_angles.csv")
            for _, r in f.iterrows():
                a = ct.compute_frame_angles(r.motor_angle_actual_deg, r.sweep_angle_deg, r.mounting_angle_deg)
                with self.subTest(run=d.name, motor=r.motor_angle_requested_deg):
                    self.assertAlmostEqual(a["inclination_angle_deg_shifted"], r.inclination_angle_deg_shifted,
                                           places=9)
                    self.assertAlmostEqual(float(angdiff(a["angle_of_attack_deg"], r.angle_of_attack_deg)), 0.0,
                                           places=9)
                n += 1
        self.assertGreater(n, 1000)

    def test_motor_reached_requested_angle(self):
        for d in self.runs:
            f = pd.read_csv(d / "coordinate_frame_angles.csv")
            with self.subTest(run=d.name):
                self.assertLess(np.abs(f.motor_angle_actual_deg - f.motor_angle_requested_deg).max(), 0.2)

    def test_encoder_log_reproduces_recorded_motor_angle(self):
        """run_loader's windowed encoder average (the post-processing route)
        against the motor angle main.py recorded, on the real encoder logs."""
        checked = 0
        for d in self.runs:
            if not (d / "encoder_log.txt").exists():
                continue
            md = json.loads((d / "run_metadata.json").read_text())
            enc_df, enc_start = load_encoder_log(d / "encoder_log.txt")
            if enc_start is None:
                continue
            f = pd.read_csv(d / "coordinate_frame_angles.csv")
            for _, r in f.iterrows():
                mean, n, _ = windowed_average_angle(enc_df, enc_start, r.accel_t_start,
                                                    md["sample_duration_s"], md["counts_per_deg"])
                if mean is None:
                    continue
                with self.subTest(run=d.name, motor=r.motor_angle_requested_deg):
                    self.assertAlmostEqual(mean, r.motor_angle_actual_deg, delta=0.05)
                checked += 1
        self.assertGreater(checked, 100)

    def test_summaries_use_fixed_transform(self):
        n = 0
        for d in self.runs:
            s = pd.read_csv(SUMM / d.name / "angle_summary.csv")
            for _, r in s.iterrows():
                a = ct.compute_frame_angles(r.motor_angle_deg, self.c["SWEEP_ANGLE_DEG"],
                                            self.c["MOUNTING_ANGLE_DEG"])
                with self.subTest(run=d.name, motor=r.requested_angle_deg):
                    self.assertAlmostEqual(r.inclination_deg, a["inclination_angle_deg_shifted"], places=9)
                    self.assertAlmostEqual(r.aoa_deg, a["angle_of_attack_deg_shifted"], places=9)
                n += 1
        self.assertGreater(n, 1000)

    def test_summary_motor_angle_is_the_recorded_one(self):
        """Each summary row's motor angle is the one main.py recorded for the
        capture its accel file holds. Some runs repeat an angle; the second
        capture overwrote the file, so the row is matched on the accel
        file's start time, not on the requested angle."""
        for d in self.runs:
            s = pd.read_csv(SUMM / d.name / "angle_summary.csv")
            f = pd.read_csv(d / "coordinate_frame_angles.csv")
            for _, r in s.iterrows():
                with open(d / r.accel_file) as fh:
                    t0 = float(fh.readline().split("t_start_perf_counter=")[1].split(",")[0])
                rec = f[np.isclose(f.accel_t_start, t0, atol=1e-6)]
                with self.subTest(run=d.name, motor=r.requested_angle_deg):
                    self.assertEqual(len(rec), 1)
                    self.assertEqual(rec.motor_angle_requested_deg.iloc[0], r.requested_angle_deg)
                    self.assertAlmostEqual(r.motor_angle_deg, rec.motor_angle_actual_deg.iloc[0], places=9)

    def test_master_table_matches_run_summaries(self):
        m = pd.read_csv(SUMM / "master_angles.csv")
        for d in self.runs:
            s = pd.read_csv(SUMM / d.name / "angle_summary.csv")
            mm = m[m.run_name == d.name].sort_values("index")
            with self.subTest(run=d.name):
                np.testing.assert_allclose(mm.inclination_deg.to_numpy(), s.inclination_deg.to_numpy(), atol=1e-12)
                np.testing.assert_allclose(mm.aoa_deg.to_numpy(), s.aoa_deg.to_numpy(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
