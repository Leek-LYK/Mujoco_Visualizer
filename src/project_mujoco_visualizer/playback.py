"""Frame and wall-clock playback state, independent of the GUI."""

from __future__ import annotations

import threading

import numpy as np


class PlaybackController:
    """Control a pre-built qpos sequence with timestamp-aware timing."""

    def __init__(
        self,
        qpos_sequence: np.ndarray,
        timestamps: np.ndarray,
        frame_durations: np.ndarray,
        *,
        loop: bool = True,
        speed: float = 1.0,
    ) -> None:
        sequence = np.asarray(qpos_sequence, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)
        durations = np.asarray(frame_durations, dtype=np.float64)
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            raise ValueError("qpos_sequence must be a non-empty 2D array.")
        if times.shape != (sequence.shape[0],):
            raise ValueError("timestamps must contain one value per qpos frame.")
        if durations.shape != (sequence.shape[0],) or np.any(durations <= 0):
            raise ValueError("frame_durations must contain one positive value per frame.")
        if not np.isfinite(sequence).all() or not np.isfinite(times).all():
            raise ValueError("Playback arrays must be finite.")
        self._qpos = sequence
        self._timestamps = times
        self._durations = durations
        self._loop = bool(loop)
        self._speed = 1.0
        self._playing = True
        self._frame_index = 0
        self._accumulator = 0.0
        self._lock = threading.RLock()
        self.set_speed(speed)

    @property
    def frame_count(self) -> int:
        return int(self._qpos.shape[0])

    @property
    def current_frame(self) -> int:
        with self._lock:
            return self._frame_index

    @property
    def current_time(self) -> float:
        with self._lock:
            return float(self._timestamps[self._frame_index])

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def loop(self) -> bool:
        with self._lock:
            return self._loop

    def current_qpos(self) -> np.ndarray:
        with self._lock:
            return self._qpos[self._frame_index].copy()

    def toggle_play_pause(self) -> bool:
        with self._lock:
            self._playing = not self._playing
            return self._playing

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = bool(playing)

    def set_loop(self, enabled: bool) -> bool:
        with self._lock:
            self._loop = bool(enabled)
            return self._loop

    def toggle_loop(self) -> bool:
        with self._lock:
            self._loop = not self._loop
            return self._loop

    def set_speed(self, speed: float) -> float:
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError(f"Playback speed must be positive and finite, got {speed!r}.")
        with self._lock:
            self._speed = float(np.clip(speed, 0.0625, 16.0))
            return self._speed

    def scale_speed(self, factor: float) -> float:
        return self.set_speed(self.speed * factor)

    def step(self, delta: int) -> int:
        """Move one or more frames and pause, returning the new frame index."""

        if not isinstance(delta, int):
            raise TypeError("delta must be an integer frame count")
        with self._lock:
            target = self._frame_index + delta
            if self._loop:
                target %= self.frame_count
            else:
                target = max(0, min(self.frame_count - 1, target))
            self._frame_index = target
            self._accumulator = 0.0
            self._playing = False
            return self._frame_index

    def reset(self) -> None:
        with self._lock:
            self._frame_index = 0
            self._accumulator = 0.0

    def advance(self, elapsed_seconds: float) -> bool:
        """Advance according to wall time; return whether the frame changed."""

        if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        with self._lock:
            if not self._playing or self.frame_count == 1:
                return False
            self._accumulator += float(elapsed_seconds) * self._speed
            changed = False
            while self._accumulator >= self._durations[self._frame_index]:
                self._accumulator -= self._durations[self._frame_index]
                if self._frame_index == self.frame_count - 1:
                    if not self._loop:
                        self._accumulator = 0.0
                        self._playing = False
                        break
                    self._frame_index = 0
                else:
                    self._frame_index += 1
                changed = True
            return changed
