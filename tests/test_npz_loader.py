from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from project_mujoco_visualizer.errors import MotionFormatError
from project_mujoco_visualizer.joint_mapping import build_qpos_sequence, create_joint_mapping
from project_mujoco_visualizer.model_loader import JointInfo, ModelDescription
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


class _UnexpectedPickleGlobal:
    def __reduce__(self):
        return eval, ("1 / 0",)


class QposNpzLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "umr.npz"
        self.qpos = np.asarray(
            [[1, 2, 3, 0.8, 0, 0, 0.6, 20, 10], [4, 5, 6, 1, 0, 0, 0, 21, 11]],
            dtype=np.float32,
        )
        self.payload = dict(
            qpos=self.qpos,
            fps=np.asarray([30], dtype=np.float32),
            robot_joint_names=np.asarray(["joint_b", "joint_a"], dtype=object),
            robot_xml=np.asarray("/original/machine/unavailable.xml"),
        )

    def write(self, **overrides) -> Path:
        np.savez_compressed(self.path, **dict(self.payload, **overrides))
        return self.path

    def test_loads_umr_object_names_and_preserves_wxyz_and_fps(self) -> None:
        motion = load_motion(self.write())
        self.assertEqual(motion.joint_names, ("joint_b", "joint_a"))
        self.assertEqual(motion.quaternion_input_order, "wxyz (MuJoCo qpos root)")
        np.testing.assert_array_equal(motion.root_position, self.qpos[:, :3])
        np.testing.assert_array_equal(motion.root_quaternion, self.qpos[:, 3:7])
        np.testing.assert_array_equal(motion.joint_values, self.qpos[:, 7:])
        np.testing.assert_allclose(motion.timestamps, [0, 1 / 30])
        np.testing.assert_allclose(motion.frame_durations, [1 / 30, 1 / 30])

    def test_maps_qpos_joint_columns_by_name(self) -> None:
        motion = load_motion(self.write())
        model = ModelDescription(
            path=self.path,
            model=SimpleNamespace(nq=9, qpos0=np.zeros(9)),
            root_joint=JointInfo(0, "root_free", "free", 0, 7, 0),
            joints=(JointInfo(1, "joint_a", "hinge", 7, 1, 6),
                    JointInfo(2, "joint_b", "hinge", 8, 1, 7)),
        )
        sequence = build_qpos_sequence(model, motion, create_joint_mapping(model, motion))
        np.testing.assert_array_equal(sequence[:, :7], self.qpos[:, :7])
        np.testing.assert_array_equal(sequence[:, 7:], self.qpos[:, [8, 7]])

    def test_accepts_unicode_and_bytes_names(self) -> None:
        for dtype in (str, bytes):
            with self.subTest(dtype=dtype):
                motion = load_motion(self.write(robot_joint_names=np.asarray(["joint_b", "joint_a"], dtype=dtype)))
                self.assertEqual(motion.joint_names, ("joint_b", "joint_a"))

    def test_rejects_missing_names(self) -> None:
        np.savez(self.path, qpos=self.qpos, fps=30)
        with self.assertRaisesRegex(MotionFormatError, "missing required field 'robot_joint_names'"):
            load_motion(self.path)

    def test_rejects_invalid_data(self) -> None:
        for overrides, message in (
            (dict(qpos=np.zeros(9)), "must be 2D"),
            (dict(qpos=np.zeros((2, 6))), "7 columns"),
            (dict(qpos=np.empty((0, 9))), "zero frames"),
            (dict(qpos=np.full((2, 9), np.nan)), "non-finite"),
            (dict(qpos=np.zeros((2, 9))), "near-zero norm"),
            (dict(qpos=self.qpos * 2), "norm differs"),
            (dict(robot_joint_names=np.asarray(["a"])), "has 1 names"),
            (dict(robot_joint_names=np.asarray(["a"], dtype=object)), "must have shape"),
            (dict(robot_joint_names=np.asarray(["a", "a"], dtype=object)), "duplicates"),
            (dict(robot_joint_names=np.asarray([123, None], dtype=object)), "must be a string"),
            (dict(fps=0), "positive and finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(MotionFormatError, message):
                    load_motion(self.write(**overrides))

    def test_rejects_arbitrary_pickle_globals_in_object_names(self) -> None:
        path = self.write(robot_joint_names=np.asarray([_UnexpectedPickleGlobal(), "joint_a"], dtype=object))
        with self.assertRaisesRegex(MotionFormatError, "Unsupported global.*builtins.eval"):
            load_motion(path)

    def test_rejects_conflicting_rate_and_quaternion_override(self) -> None:
        path = self.write()
        with self.assertRaisesRegex(MotionFormatError, "disagrees with NPZ fps"):
            load_motion(path, sample_rate_hz=50)
        with self.assertRaisesRegex(MotionFormatError, "fixed by its schema"):
            load_motion(path, quat_order="xyzw")


if __name__ == "__main__":
    unittest.main()
