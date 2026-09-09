from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import shutil

from project_mujoco_visualizer.errors import MotionFormatError
from project_mujoco_visualizer.motion_loader import load_motion_csv


class MotionLoaderTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "motion.csv"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        return path

    def test_discovers_named_fields_and_sampling(self) -> None:
        path = self._write(
            "timestamp,root_translateX,root_translateY,root_translateZ,"
            "root_quat_w,root_quat_x,root_quat_y,root_quat_z,"
            "leg_b_joint_dof,leg_a_joint_dof\n"
            "0,1,2,3,1,0,0,0,0.2,0.1\n"
            "0.04,2,3,4,1,0,0,0,0.3,0.2\n"
        )
        motion = load_motion_csv(path)
        self.assertEqual(motion.frame_count, 2)
        self.assertAlmostEqual(motion.sample_rate_hz, 25.0)
        self.assertEqual(motion.joint_names, ("leg_b_joint", "leg_a_joint"))
        self.assertEqual(tuple(motion.root_quaternion[0]), (1.0, 0.0, 0.0, 0.0))

    def test_rejects_duplicate_joint_semantics(self) -> None:
        path = self._write(
            "timestamp,root_translateX,root_translateY,root_translateZ,"
            "root_quat_w,root_quat_x,root_quat_y,root_quat_z,joint_dof,joint_pos\n"
            "0,0,0,1,1,0,0,0,0,0\n"
            "0.1,0,0,1,1,0,0,0,0,0\n"
        )
        with self.assertRaisesRegex(MotionFormatError, "Duplicate joint fields"):
            load_motion_csv(path)

    def test_requires_explicit_rate_without_timestamp(self) -> None:
        path = self._write(
            "root_translateX,root_translateY,root_translateZ,root_quat_w,"
            "root_quat_x,root_quat_y,root_quat_z,joint_dof\n"
            "0,0,1,1,0,0,0,0\n"
        )
        with self.assertRaisesRegex(MotionFormatError, "--sample-rate"):
            load_motion_csv(path)

    def test_explicit_xyzw_conversion_is_wxyz(self) -> None:
        path = self._write(
            "timestamp,root_translateX,root_translateY,root_translateZ,"
            "root_quat_0,root_quat_1,root_quat_2,root_quat_3,joint_dof\n"
            "0,0,0,1,0,0,0,1,0\n"
            "0.1,0,0,1,0,0,0,1,0\n"
        )
        motion = load_motion_csv(path, quat_order="xyzw")
        self.assertEqual(tuple(motion.root_quaternion[0]), (1.0, 0.0, 0.0, 0.0))
        self.assertIn("xyzw -> wxyz", motion.quaternion_input_order)


if __name__ == "__main__":
    unittest.main()
