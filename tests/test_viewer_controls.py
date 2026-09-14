from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np

from project_mujoco_visualizer.playback import PlaybackController
from project_mujoco_visualizer.viewer import MotionPlaylist, _handle_keycode, _is_quit_keycode
from project_mujoco_visualizer.errors import MotionFormatError


class ViewerControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.glfw = SimpleNamespace(
            KEY_SPACE=32,
            KEY_RIGHT=262,
            KEY_LEFT=263,
            KEY_EQUAL=61,
            KEY_MINUS=45,
            KEY_F10=299,
            KEY_F8=297,
            KEY_BACKSPACE=259,
        )
        self.controller = PlaybackController(
            np.arange(4, dtype=float).reshape(4, 1),
            np.asarray([0.0, 0.1, 0.2, 0.3]),
            np.full(4, 0.1),
            loop=False,
        )

    def test_single_keycode_dispatch_preserves_all_controls(self) -> None:
        self.assertTrue(_handle_keycode(self.glfw.KEY_SPACE, self.controller, self.glfw))
        self.assertFalse(self.controller.playing)

        self.assertTrue(_handle_keycode(self.glfw.KEY_RIGHT, self.controller, self.glfw))
        self.assertEqual(self.controller.current_frame, 1)
        self.assertTrue(_handle_keycode(self.glfw.KEY_LEFT, self.controller, self.glfw))
        self.assertEqual(self.controller.current_frame, 0)

        self.assertTrue(_handle_keycode(self.glfw.KEY_EQUAL, self.controller, self.glfw))
        self.assertEqual(self.controller.speed, 2.0)
        self.assertTrue(_handle_keycode(self.glfw.KEY_MINUS, self.controller, self.glfw))
        self.assertEqual(self.controller.speed, 1.0)

        self.assertTrue(_handle_keycode(ord("L"), self.controller, self.glfw))
        self.assertTrue(self.controller.loop)
        self.assertTrue(_handle_keycode(ord("R"), self.controller, self.glfw))
        self.assertEqual(self.controller.current_frame, 0)
        self.assertTrue(_is_quit_keycode(self.glfw.KEY_F10, self.glfw))
        self.assertFalse(_is_quit_keycode(ord("Q"), self.glfw))
        self.assertFalse(_handle_keycode(ord("x"), self.controller, self.glfw))

    def test_playlist_order_wrap_and_playback_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("c.NPZ", "a.csv", "b.npz", "ignored.txt"):
                (root / name).touch()
            (root / "nested.csv").mkdir()
            def load(path):
                return PlaybackController(np.ones((2, 1)), np.array([0., .2]), np.full(2, .2))
            playlist = MotionPlaylist(root / "b.npz", load)
            self.controller.step(2)
            self.controller.set_speed(2)
            current = playlist.switch(1, self.controller)
            self.assertEqual(playlist.path.name, "c.NPZ")
            self.assertEqual(current.current_frame, 0)
            self.assertEqual(current.frame_count, 2)
            self.assertFalse(current.playing)
            self.assertFalse(current.loop)
            self.assertEqual(current.speed, 2)
            current = playlist.switch(1, current)
            self.assertEqual(playlist.path.name, "a.csv")
            playlist.switch(-1, current)
            self.assertEqual(playlist.path.name, "c.NPZ")

    def test_failed_switch_preserves_selection_and_single_file_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.csv").touch()
            loader = Mock(side_effect=MotionFormatError("invalid motion"))
            playlist = MotionPlaylist(root / "a.csv", loader)
            self.assertIs(playlist.switch(1, self.controller), self.controller)
            loader.assert_not_called()
            (root / "b.npz").touch()
            playlist = MotionPlaylist(root / "a.csv", loader)
            self.controller.step(2)
            with self.assertRaises(MotionFormatError):
                playlist.switch(1, self.controller)
            self.assertEqual(playlist.path.name, "a.csv")
            self.assertEqual(self.controller.current_frame, 2)


if __name__ == "__main__":
    unittest.main()
