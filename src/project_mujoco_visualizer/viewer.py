"""Minimal official MuJoCo passive viewer frontend."""

from __future__ import annotations

import importlib
import sys
import time
from typing import Any

from .errors import ModelLoadError
from .motion_loader import MotionData
from .playback import PlaybackController


def _load_mujoco() -> Any:
    try:
        return importlib.import_module("mujoco")
    except Exception as exc:
        raise ModelLoadError(
            "Could not import mujoco for the viewer "
            f"({type(exc).__name__}: {exc})."
        ) from exc


def _print_status(controller: PlaybackController, *, newline: bool = False) -> None:
    status = "playing" if controller.playing else "paused"
    loop = "on" if controller.loop else "off"
    message = (
        f"frame {controller.current_frame + 1}/{controller.frame_count} | "
        f"time {controller.current_time:.3f} s | {status} | "
        f"speed {controller.speed:g}x | loop {loop}"
    )
    if newline:
        print(message, flush=True)
    else:
        sys.stdout.write("\r" + message + " " * 8)
        sys.stdout.flush()


def run_viewer(
    model: Any,
    motion: MotionData,
    controller: PlaybackController,
) -> None:
    """Open ``mujoco.viewer.launch_passive`` and run keyboard-controlled playback."""

    mujoco = _load_mujoco()
    try:
        from mujoco import viewer as mujoco_viewer
        from mujoco.glfw import glfw
    except Exception as exc:
        raise ModelLoadError(
            "The official MuJoCo viewer backend could not be imported "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    def key_callback(key: int, _scancode: int, action: int, _mods: int) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_SPACE:
            controller.toggle_play_pause()
        elif key == glfw.KEY_RIGHT:
            controller.step(1)
        elif key == glfw.KEY_LEFT:
            controller.step(-1)
        elif key in (glfw.KEY_EQUAL, ord("+")):
            controller.scale_speed(2.0)
        elif key in (glfw.KEY_MINUS, ord("-")):
            controller.scale_speed(0.5)
        elif key == ord("L") or key == ord("l"):
            controller.toggle_loop()
        elif key == ord("R") or key == ord("r"):
            controller.reset()
        _print_status(controller, newline=True)

    data = mujoco.MjData(model)
    with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        data.qpos[:] = controller.current_qpos()
        mujoco.mj_forward(model, data)
        viewer.sync()
        print(
            "Controls: Space play/pause | Left/Right single frame | +/- speed | "
            "L loop | R reset | close the viewer to exit",
            flush=True,
        )
        _print_status(controller, newline=True)
        last_time = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = max(0.0, min(now - last_time, 0.25))
            last_time = now
            changed = controller.advance(elapsed)
            if changed or not controller.playing:
                data.qpos[:] = controller.current_qpos()
                mujoco.mj_forward(model, data)
            viewer.sync()
            _print_status(controller)
            time.sleep(0.001)
    print()
