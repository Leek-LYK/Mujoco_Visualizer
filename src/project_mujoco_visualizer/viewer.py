"""Minimal official MuJoCo passive viewer frontend."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Callable
from typing import Any

from .errors import ModelLoadError, VisualizerError
from .motion_loader import MotionData
from .playback import PlaybackController


class MotionPlaylist:
    """Load adjacent motion files before committing a playback switch."""

    def __init__(self, path: Path, loader: Callable[[Path], PlaybackController]) -> None:
        self.path = path.resolve()
        self.paths = sorted(
            (p for p in self.path.parent.iterdir()
             if p.is_file() and p.suffix.lower() in {".csv", ".npz"}),
            key=lambda p: (p.name.casefold(), p.name),
        )
        self.loader = loader

    def switch(self, delta: int, current: PlaybackController) -> PlaybackController:
        index = (self.paths.index(self.path) + delta) % len(self.paths)
        target = self.paths[index]
        if target == self.path:
            return current
        replacement = self.loader(target)
        replacement.set_speed(current.speed)
        replacement.set_loop(current.loop)
        replacement.set_playing(current.playing)
        self.path = target
        return replacement


def _is_quit_keycode(keycode: int, glfw: Any) -> bool:
    """Return whether the application quit key was pressed."""

    return keycode == glfw.KEY_F10


def _handle_keycode(keycode: int, controller: PlaybackController, glfw: Any) -> bool:
    """Apply one MuJoCo passive-viewer keycode.

    MuJoCo 3.12.0 calls ``key_callback`` with one GLFW keycode argument.  It
    does not pass GLFW's ``scancode``, ``action`` or ``mods`` values, so key
    handling must not filter on a separate press action here.
    """

    if keycode == glfw.KEY_SPACE:
        controller.toggle_play_pause()
    elif keycode == glfw.KEY_RIGHT:
        controller.step(1)
    elif keycode == glfw.KEY_LEFT:
        controller.step(-1)
    elif keycode in (glfw.KEY_EQUAL, ord("+")):
        controller.scale_speed(2.0)
    elif keycode in (glfw.KEY_MINUS, ord("-")):
        controller.scale_speed(0.5)
    elif keycode in (ord("L"), ord("l")):
        controller.toggle_loop()
    elif keycode in (ord("R"), ord("r")):
        controller.reset()
    else:
        return False
    return True


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


def _configure_initial_camera(mujoco: Any, model: Any, data: Any, viewer: Any) -> tuple[float, float, float] | None:
    """Start a tracking camera centered on the compiled model's ``base_link``."""

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
    if body_id < 0:
        print("Camera: body 'base_link' not found; keeping MuJoCo default view.", flush=True)
        return None

    base_position = tuple(float(value) for value in data.xpos[body_id])
    extent = max(1.0, float(model.stat.extent))
    distance = max(2.5, 1.6 * extent)
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body_id
        viewer.cam.lookat[:] = base_position
        viewer.cam.distance = distance
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
    return base_position[0], base_position[1], base_position[2]


def run_viewer(
    model: Any,
    motion: MotionData,
    controller: PlaybackController,
    *,
    motion_loader: Callable[[Path], PlaybackController] | None = None,
) -> None:
    """Open ``mujoco.viewer.launch_passive`` and run keyboard-controlled playback."""

    mujoco = _load_mujoco()
    try:
        from mujoco import viewer as mujoco_viewer
        import glfw
    except Exception as exc:
        raise ModelLoadError(
            "The official MuJoCo viewer backend could not be imported "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    exit_requested = threading.Event()
    pending_keys: SimpleQueue[int] = SimpleQueue()
    playlist = MotionPlaylist(motion.path, motion_loader) if motion_loader else None

    def key_callback(keycode: int) -> None:
        if _is_quit_keycode(keycode, glfw):
            exit_requested.set()
            print("F10 pressed; closing viewer...", flush=True)
            return
        pending_keys.put(keycode)

    data = mujoco.MjData(model)
    try:
        with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            data.qpos[:] = controller.current_qpos()
            mujoco.mj_forward(model, data)
            camera_target = _configure_initial_camera(mujoco, model, data, viewer)
            viewer.sync()
            if camera_target is not None:
                print(
                    "Camera: lookat base_link "
                    f"({camera_target[0]:.3f}, {camera_target[1]:.3f}, {camera_target[2]:.3f}), "
                    f"distance={max(2.5, 1.6 * max(1.0, float(model.stat.extent))):.3f}, "
                    "mode=tracking, "
                    "elevation=-20 deg",
                    flush=True,
                )
            print(
                "Controls: Space play/pause | Left/Right single frame | Up/Down previous/next motion | +/- speed | "
                "L loop | R reset | F10 quit (native Ctrl+Q also works)",
                flush=True,
            )
            _print_status(controller, newline=True)
            print(f"Motion: {motion.path}", flush=True)
            last_time = time.perf_counter()
            while viewer.is_running() and not exit_requested.is_set():
                switched = False
                while True:
                    try:
                        keycode = pending_keys.get_nowait()
                    except Empty:
                        break
                    if playlist is not None and keycode in (glfw.KEY_UP, glfw.KEY_DOWN):
                        try:
                            replacement = playlist.switch(-1 if keycode == glfw.KEY_UP else 1, controller)
                        except (VisualizerError, OSError, ValueError) as exc:
                            print(f"\nCould not switch motion: {exc}", flush=True)
                        else:
                            switched = switched or replacement is not controller
                            controller = replacement
                            print(f"\nMotion: {playlist.path}", flush=True)
                        last_time = time.perf_counter()
                    elif _handle_keycode(keycode, controller, glfw):
                        _print_status(controller, newline=True)
                now = time.perf_counter()
                elapsed = max(0.0, min(now - last_time, 0.25))
                last_time = now
                changed = controller.advance(0.0 if switched else elapsed)
                if switched or changed or not controller.playing:
                    data.qpos[:] = controller.current_qpos()
                    mujoco.mj_forward(model, data)
                viewer.sync()
                _print_status(controller)
                time.sleep(0.001)
            if exit_requested.is_set():
                viewer.close()
    except KeyboardInterrupt:
        print("\nCtrl+C received; viewer closed.", flush=True)
    print()
