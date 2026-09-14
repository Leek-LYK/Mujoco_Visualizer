from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_mujoco_visualizer.errors import MotionFormatError
from project_mujoco_visualizer.motion_loader import load_motion, load_motion_npz


class NpzLoaderTests(unittest.TestCase):
    def _write_npz(
        self,
        *,
        include_duplicate: bool = False,
        fps: np.ndarray | None = None,
    ) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        names = np.asarray(["joint_b", "joint_a"] if not include_duplicate else ["joint_a", "joint_a"])
        path = directory / "motion.npz"
        np.savez(
            path,
            fps=np.asarray(50, dtype=np.int32) if fps is None else fps,
            root_pos=np.asarray([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], dtype=np.float32),
            # NPZ convention is x,y,z,w; this is identity in both input and output.
            root_rot=np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            dof_pos=np.asarray([[20.0, 10.0], [21.0, 11.0]], dtype=np.float32),
            joint_names=names,
            body_names=np.asarray(["base_link"]),
            local_body_pos=np.zeros((2, 1, 3), dtype=np.float32),
            local_body_rot=np.asarray([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        )
        return path

    def test_loads_npz_and_converts_xyzw_to_wxyz(self) -> None:
        path = self._write_npz()
        motion = load_motion_npz(path)
        self.assertEqual(motion.frame_count, 2)
        self.assertAlmostEqual(motion.sample_rate_hz, 50.0)
        self.assertEqual(motion.joint_names, ("joint_b", "joint_a"))
        self.assertEqual(tuple(motion.root_quaternion[0]), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(motion.quaternion_input_order, "xyzw -> wxyz (explicit NPZ conversion)")
        self.assertEqual(load_motion(path).frame_count, 2)

    def test_accepts_one_element_fps_array(self) -> None:
        path = self._write_npz(fps=np.asarray([50], dtype=np.int32))
        motion = load_motion_npz(path)
        self.assertAlmostEqual(motion.sample_rate_hz, 50.0)

    def test_loads_whole_body_npz_without_names(self) -> None:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        path = directory / "whole_body.npz"
        np.savez(
            path,
            fps=np.asarray([50], dtype=np.int32),
            joint_pos=np.zeros((2, 21), dtype=np.float32),
            body_pos_w=np.concatenate(
                (
                    np.asarray([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], dtype=np.float32),
                    np.zeros((2, 27, 3), dtype=np.float32),
                ),
                axis=1,
            ),
            body_quat_w=np.concatenate(
                (
                    np.asarray(
                        [[[0.9238795, 0.0, 0.3826834, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
                        dtype=np.float32,
                    ),
                    np.tile(
                        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        (2, 27, 1),
                    ),
                ),
                axis=1,
            ),
        )

        motion = load_motion_npz(path)

        self.assertEqual(motion.frame_count, 2)
        self.assertEqual(motion.joint_count, 21)
        self.assertAlmostEqual(motion.sample_rate_hz, 50.0)
        np.testing.assert_allclose(motion.root_position[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(motion.root_quaternion[0], [0.9238795, 0.0, 0.3826834, 0.0])
        self.assertEqual(motion.quaternion_input_order, "wxyz (body_quat_w root body)")

    def test_rejects_multi_value_fps_array(self) -> None:
        path = self._write_npz(fps=np.asarray([50, 60], dtype=np.int32))
        with self.assertRaisesRegex(MotionFormatError, "exactly one value"):
            load_motion_npz(path)

    def test_rejects_duplicate_npz_joint_names(self) -> None:
        path = self._write_npz(include_duplicate=True)
        with self.assertRaisesRegex(MotionFormatError, "contains duplicates"):
            load_motion_npz(path)


if __name__ == "__main__":
    unittest.main()
