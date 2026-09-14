"""Command-line entry point for the motion visualizer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from .errors import VisualizerError
from .joint_mapping import build_qpos_sequence, create_joint_mapping, mapping_report_lines
from .model_loader import load_model
from .motion_loader import load_motion
from .playback import PlaybackController
from .viewer import run_viewer


def _default_asset_dir() -> Path:
    return Path("robot_asset") / "roban_s22_handball"


def _select_motion(path: str | None, data_dir: str | Path) -> Path:
    if path is not None:
        requested = Path(path).expanduser().resolve()
        if not requested.is_file():
            raise VisualizerError(f"Motion file does not exist: {requested}")
        return requested
    directory = Path(data_dir).expanduser().resolve()
    candidates = tuple(sorted(directory.glob("*.csv")))
    if not candidates:
        candidates = tuple(sorted(directory.glob("*.npz")))
    if not candidates:
        candidates = tuple(sorted(directory.glob("*.pkl")))
    if not candidates:
        raise VisualizerError(f"No .csv, .npz or .pkl motion files found under {directory}")
    print(f"--motion not supplied; selected first motion file: {candidates[0]}")
    return candidates[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project_mujoco_visualizer",
        description="Play named CSV, NPZ or PKL robot motion data in a MuJoCo MJCF model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="MJCF/XML file or directory; omitted means auto-discover below --asset-dir.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=_default_asset_dir(),
        help="Robot asset directory used for automatic XML discovery.",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        help="CSV, NPZ or PKL motion file; omitted means select the first supported file below --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory scanned for CSV/NPZ/PKL when --motion is omitted.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        help="Explicit sample rate in Hz, required only when CSV has no time column.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("auto", "wxyz", "xyzw"),
        default="auto",
        help="Order for generic root_quat_0..3 columns; named w/x/y/z fields are semantic.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed multiplier.")
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start with loop playback enabled (default: on; use --no-loop to disable).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Load, map, and construct qpos frames without opening a viewer.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        motion_path = _select_motion(str(args.motion) if args.motion else None, args.data_dir)
        motion = load_motion(
            motion_path,
            sample_rate_hz=args.sample_rate,
            quat_order=args.quat_order,
        )
        print("Motion analysis:")
        print("\n".join(motion.report_lines()))

        model = load_model(args.model, asset_dir=args.asset_dir)
        if model.candidates:
            print("MJCF/XML discovery:")
            for index, candidate in enumerate(model.candidates):
                selected = " [selected]" if candidate == model.path else ""
                print(f"  {index + 1}. {candidate}{selected}")
        mapping = create_joint_mapping(model, motion)
        print("Joint mapping:")
        print("\n".join(mapping_report_lines(model, motion, mapping)))
        qpos_sequence = build_qpos_sequence(model, motion, mapping)
        print(
            f"qpos validation: {qpos_sequence.shape[0]} frames x {qpos_sequence.shape[1]} values; "
            "root quaternion preserved in MuJoCo wxyz order"
        )
        if args.analyze_only:
            return 0
        if not math.isfinite(args.speed) or args.speed <= 0:
            raise VisualizerError(f"--speed must be positive and finite, got {args.speed!r}.")
        controller = PlaybackController(
            qpos_sequence,
            motion.timestamps,
            motion.frame_durations,
            loop=args.loop,
            speed=args.speed,
        )
        def load_playback(path: Path) -> PlaybackController:
            next_motion = load_motion(
                path, sample_rate_hz=args.sample_rate, quat_order=args.quat_order,
            )
            next_mapping = create_joint_mapping(model, next_motion)
            next_qpos = build_qpos_sequence(model, next_motion, next_mapping)
            return PlaybackController(
                next_qpos, next_motion.timestamps, next_motion.frame_durations,
            )

        run_viewer(model.model, motion, controller, motion_loader=load_playback)
    except VisualizerError as exc:
        parser.error(str(exc))
    return 0
