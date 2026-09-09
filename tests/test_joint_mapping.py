from __future__ import annotations

import unittest
from types import SimpleNamespace
import shutil

import numpy as np

from project_mujoco_visualizer.errors import JointMappingError
from project_mujoco_visualizer.joint_mapping import build_qpos, create_joint_mapping
from project_mujoco_visualizer.model_loader import JointInfo, ModelDescription
from project_mujoco_visualizer.motion_loader import load_motion_csv


class JointMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.motion_path = self._temporary_motion()
        self.motion = load_motion_csv(self.motion_path)
        self.root = JointInfo(0, "root_free", "free", 0, 7, 0)
        self.joint_a = JointInfo(1, "joint_a", "hinge", 7, 1, 0)
        self.joint_b = JointInfo(2, "joint_b", "hinge", 8, 1, 1)
        fake_model = SimpleNamespace(nq=9, qpos0=np.zeros(9))
        self.model = ModelDescription(
            path=self.motion_path,
            model=fake_model,
            root_joint=self.root,
            joints=(self.joint_a, self.joint_b),
        )

    def _temporary_motion(self):
        import tempfile
        from pathlib import Path

        directory = Path(tempfile.mkdtemp())
        path = directory / "motion.csv"
        path.write_text(
            "timestamp,root_translateX,root_translateY,root_translateZ,"
            "root_quat_w,root_quat_x,root_quat_y,root_quat_z,joint_b_dof,joint_a_dof\n"
            "0,1,2,3,1,0,0,0,20,10\n"
            "0.1,4,5,6,1,0,0,0,21,11\n",
            encoding="utf-8",
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        return path

    def test_maps_by_name_and_writes_correct_qpos_addresses(self) -> None:
        mapping = create_joint_mapping(self.model, self.motion)
        qpos = build_qpos(self.model, self.motion, mapping, 0)
        self.assertTrue(np.array_equal(qpos[:3], [1, 2, 3]))
        self.assertTrue(np.array_equal(qpos[3:7], [1, 0, 0, 0]))
        self.assertEqual(qpos[7], 10)
        self.assertEqual(qpos[8], 20)

    def test_reports_missing_and_extra_names(self) -> None:
        self.model = ModelDescription(
            path=self.model.path,
            model=self.model.model,
            root_joint=self.root,
            joints=(self.joint_a, JointInfo(2, "joint_c", "hinge", 8, 1, 1)),
        )
        with self.assertRaisesRegex(JointMappingError, "missing model joints.*joint_c"):
            create_joint_mapping(self.model, self.motion)


if __name__ == "__main__":
    unittest.main()
