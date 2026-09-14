from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_mujoco_visualizer.errors import MotionFormatError
from project_mujoco_visualizer.motion_loader import load_motion
from project_mujoco_visualizer.cli import _select_motion
from project_mujoco_visualizer.viewer import MotionPlaylist


class PklLoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "motion.pkl"
        self.payload = dict(
            fps=30, root_pos=np.zeros((2, 3)),
            root_rot=np.array([[0., 0., .6, .8], [0., 0., 0., 1.]]),
            dof_pos=np.array([[2., 1.], [4., 3.]]),
            joint_names=["joint_b", "joint_a"],
            link_body_list=["base_link"], local_body_pos=np.zeros((2, 1, 3)),
        )

    def write(self, payload):
        with self.path.open("wb") as stream:
            pickle.dump(payload, stream)

    def test_load_and_discovery(self):
        self.write(self.payload)
        motion = load_motion(self.path)
        np.testing.assert_allclose(motion.root_quaternion[0], [.8, 0., 0., .6])
        np.testing.assert_allclose(motion.timestamps, [0., 1 / 30])
        np.testing.assert_allclose(motion.joint_values, self.payload["dof_pos"])
        self.assertEqual(motion.joint_names, ("joint_b", "joint_a"))
        self.assertEqual(motion.path, self.path.resolve())
        self.assertEqual(motion.delimiter, "pkl")
        self.assertEqual(_select_motion(None, self.path.parent), self.path.resolve())
        self.assertIn(self.path.resolve(), MotionPlaylist(self.path, lambda p: None).paths)

    def test_invalid_payloads(self):
        for payload, message in (
            ([], "dictionary"),
            (dict(self.payload, fps=0), "positive and finite"),
            (dict(self.payload, root_rot=np.zeros((2, 4))), "near-zero"),
            (dict(self.payload, dof_pos=np.zeros((3, 2))), "frame dimensions"),
            (dict(self.payload, joint_names=["a", "a"]), "duplicates"),
            (dict(self.payload, local_body_pos=np.zeros((2, 2, 3))), "local_body_pos shape"),
        ):
            with self.subTest(message=message):
                self.write(payload)
                with self.assertRaisesRegex(MotionFormatError, message):
                    load_motion(self.path)

    def test_corrupt_pickle(self):
        self.path.write_bytes(b"not a pickle")
        with self.assertRaisesRegex(MotionFormatError, "Could not read PKL"):
            load_motion(self.path)
