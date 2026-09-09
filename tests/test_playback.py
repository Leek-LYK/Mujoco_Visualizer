from __future__ import annotations

import unittest

import numpy as np

from project_mujoco_visualizer.playback import PlaybackController


class PlaybackTests(unittest.TestCase):
    def _controller(self, *, loop: bool = False) -> PlaybackController:
        qpos = np.arange(4, dtype=float).reshape(4, 1)
        timestamps = np.asarray([0.0, 0.1, 0.2, 0.3])
        durations = np.full(4, 0.1)
        return PlaybackController(qpos, timestamps, durations, loop=loop)

    def test_play_pause_and_single_frame_step(self) -> None:
        controller = self._controller()
        controller.set_playing(False)
        self.assertFalse(controller.advance(1.0))
        self.assertEqual(controller.current_frame, 0)
        self.assertEqual(controller.step(1), 1)
        self.assertEqual(controller.current_time, 0.1)
        self.assertFalse(controller.playing)
        self.assertTrue(controller.toggle_play_pause())
        self.assertTrue(controller.advance(0.1))
        self.assertEqual(controller.current_frame, 2)

    def test_loop_and_speed(self) -> None:
        controller = self._controller(loop=True)
        controller.set_speed(2.0)
        self.assertEqual(controller.speed, 2.0)
        controller.advance(0.2)
        self.assertEqual(controller.current_frame, 0)
        self.assertTrue(controller.loop)


if __name__ == "__main__":
    unittest.main()
