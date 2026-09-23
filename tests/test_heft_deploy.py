from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from project_mujoco_visualizer.errors import MotionFormatError
from project_mujoco_visualizer.heft_deploy import (
    HEFT_FIELDS, HEFT_JOINT_NAMES, check_reference, convert_csv, main,
    resample_motion, validate_deploy,
)
from project_mujoco_visualizer.motion_loader import load_motion_csv


class HeftDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "motion.csv"
        # Reversed joints and xyzw column ordering exercise semantic loading.
        with self.source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "timestamp", "root_translateX", "root_translateY", "root_translateZ",
                "root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w",
                *(name + "_dof" for name in reversed(HEFT_JOINT_NAMES)),
            ])
            for i in range(3):
                angle = i * np.pi / 3
                sign = -1 if i == 1 else 1
                writer.writerow([10 + i * .04, i, 0, 1, 0, 0,
                                 sign * np.sin(angle / 2), sign * np.cos(angle / 2),
                                 *(j + i * .08 for j in reversed(range(21)))])

    def test_reference_schema_mapping_slerp_and_velocity(self):
        reference = Path(__file__).parent / "fixtures/heft_deploy_reference.csv"
        check_reference(reference)
        output = self.root / "motion_deploy.csv"
        self.assertEqual(convert_csv(self.source, output), 5)
        with reference.open(encoding="utf-8-sig", newline="") as stream:
            expected_header = next(csv.reader(stream, delimiter="\t"))
        with output.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            self.assertEqual(next(reader), expected_header)
            values = np.asarray(list(reader), dtype=float)
        np.testing.assert_allclose(values[:, 0], [0, .5, 1, 1.5, 2])
        np.testing.assert_allclose(values[:, 7:28], np.arange(21)[None, :] + np.arange(5)[:, None] * .04)
        np.testing.assert_allclose(values[:, 28:], 2)
        np.testing.assert_allclose(values[1, 3:7], [np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(values[:, 3:7], axis=1), 1)
        self.assertTrue(np.all(np.sum(values[:-1, 3:7] * values[1:, 3:7], axis=1) > 0))
        with self.assertRaisesRegex(ValueError, "new file"):
            convert_csv(self.source, output)

    def test_invalid_joints_and_timing(self):
        motion = load_motion_csv(self.source)
        for names, message in [
            (motion.joint_names[:-1], "missing"),
            (motion.joint_names[:-1] + (motion.joint_names[0],), "duplicate"),
            (motion.joint_names[:-1] + ("unknown",), "unexpected"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(MotionFormatError, message):
                resample_motion(replace(motion, joint_names=names))
        with self.assertRaisesRegex(MotionFormatError, "25 Hz"):
            resample_motion(replace(motion, timestamps=np.array([0, .04, .09])))
        with self.assertRaisesRegex(MotionFormatError, "at least two"):
            resample_motion(replace(motion, timestamps=np.array([0.])))

    def test_output_validation(self):
        times, values = resample_motion(load_motion_csv(self.source))
        for broken, message in [(values[:, :-1], "count"), (values * np.nan, "NaN/Inf")]:
            with self.assertRaisesRegex(MotionFormatError, message):
                validate_deploy(HEFT_FIELDS, broken, times)
        with self.assertRaisesRegex(MotionFormatError, "order"):
            validate_deploy(tuple(reversed(HEFT_FIELDS)), values, times)
        with self.assertRaisesRegex(MotionFormatError, "50 Hz"):
            validate_deploy(HEFT_FIELDS, values, times * 2)
        values[:, 3:7] *= 2
        with self.assertRaisesRegex(MotionFormatError, "normalized"):
            validate_deploy(HEFT_FIELDS, values, times)

    def test_recursive_batch_continues_after_bad_input(self):
        sub = self.root / "nested"
        sub.mkdir()
        (sub / "other.csv").write_bytes(self.source.read_bytes())
        (self.root / "bad.csv").write_text("bad\n1\n", encoding="utf-8")
        self.assertEqual(main([str(self.root), "--recursive"]), 1)
        self.assertTrue((self.root / "deploy/motion_deploy.csv").is_file())
        self.assertTrue((self.root / "deploy/nested/other_deploy.csv").is_file())
        self.assertFalse((self.root / "deploy/bad_deploy.csv").exists())


if __name__ == "__main__":
    unittest.main()
