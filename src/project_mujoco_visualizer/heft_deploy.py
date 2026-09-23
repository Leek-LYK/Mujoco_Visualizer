"""Convert named 25 Hz Roban S17 motions to the 50 Hz HEFT CSV contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .errors import MotionFormatError, VisualizerError
from .motion_loader import MotionData, load_motion_csv


# Deployment order, deliberately independent of CSV and MuJoCo qpos order.
HEFT_JOINT_NAMES = (
    "waist_yaw_joint",
    *(f"leg_{side}{i}_joint" for side in ("l", "r") for i in range(1, 7)),
    *(f"zarm_{side}{i}_joint" for side in ("l", "r") for i in range(1, 5)),
)
HEFT_FIELDS = (
    "body_pos_x", "body_pos_y", "body_pos_z",
    "body_quat_w", "body_quat_x", "body_quat_y", "body_quat_z",
    *(f"joint_pos{i:02d}" for i in range(21)),
    *(f"joint_vel_{i:02d}" for i in range(21)),
)
HEFT_DELIMITER = "\t"  # Verified against main_220728_walk_forward_stop_002__A044_deploy.csv.


def check_reference(path: Path) -> None:
    """Check an independently supplied HEFT example's exact field order."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = tuple(next(csv.reader(stream, delimiter=HEFT_DELIMITER), ()))
    if header != HEFT_FIELDS:
        raise MotionFormatError(
            f"HEFT reference field order mismatch: {path}\n"
            f"Expected: {HEFT_FIELDS}\nActual: {header}"
        )


def resample_motion(motion: MotionData) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first/last sample times: N input frames become 2*N-1 frames.

    Positions are linear, rotations use shortest-arc SLERP. Velocities use
    central differences internally and one-sided differences at endpoints.
    No extrapolation, angle wrapping, coordinate or unit conversion is applied.
    """
    names = motion.joint_names
    duplicates = sorted({name for name in names if names.count(name) > 1})
    missing = sorted(set(HEFT_JOINT_NAMES) - set(names))
    extra = sorted(set(names) - set(HEFT_JOINT_NAMES))
    if duplicates or missing or extra or motion.joint_count != 21:
        raise MotionFormatError(
            f"Expected 21 Roban S17 joints; missing={missing}, "
            f"duplicate={duplicates}, unexpected={extra}"
        )
    if motion.timestamp_column is None or motion.frame_count < 2:
        raise MotionFormatError("25 Hz input requires timestamps and at least two frames.")
    times = motion.timestamps - motion.timestamps[0]
    if not np.allclose(np.diff(times), 1 / 25, rtol=0, atol=1e-6):
        raise MotionFormatError("Input timestamps must be uniformly sampled at 25 Hz (dt=0.04 s).")
    target = np.arange(2 * motion.frame_count - 1, dtype=float) / 50
    if not np.isclose(times[-1], target[-1], rtol=0, atol=1e-6):
        raise MotionFormatError("Input duration does not align with the 50 Hz grid.")
    # Remove only timestamp rounding noise at the endpoints for scipy's bounds check.
    times[-1] = target[-1]
    joints = motion.joint_values[:, [names.index(name) for name in HEFT_JOINT_NAMES]]
    positions = np.column_stack((motion.root_position, joints))
    interpolated = np.column_stack([
        np.interp(target, times, column) for column in positions.T
    ])
    rotations = Rotation.from_quat(motion.root_quaternion, scalar_first=True)
    quaternion = Slerp(times, rotations)(target).as_quat(scalar_first=True)
    # Equivalent q/-q inputs must not introduce sign jumps in the output.
    for i in range(1, len(quaternion)):
        if np.dot(quaternion[i - 1], quaternion[i]) < 0:
            quaternion[i] *= -1
    velocity = np.gradient(interpolated[:, 3:], target, axis=0, edge_order=1)
    values = np.column_stack((interpolated[:, :3], quaternion, interpolated[:, 3:], velocity))
    validate_deploy(HEFT_FIELDS, values, target)
    return target, values


def validate_deploy(fields: tuple[str, ...], values: np.ndarray, times: np.ndarray) -> None:
    """Validate the 49-column payload and its implicit 50 Hz sampling grid."""
    if fields != HEFT_FIELDS:
        raise MotionFormatError("Invalid HEFT output field order (expected 7 root + 21 position + 21 velocity).")
    if values.shape != (len(times), 49) or len(times) < 2:
        raise MotionFormatError(f"Invalid HEFT frame/column count: {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(times).all():
        raise MotionFormatError("HEFT output contains NaN/Inf.")
    if not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-9):
        raise MotionFormatError("HEFT output timing is not 50 Hz.")
    if not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, rtol=0, atol=1e-9):
        raise MotionFormatError("HEFT output quaternion is not normalized.")


def convert_csv(source: Path, output: Path) -> int:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if source == output or output.exists():
        raise ValueError(f"Output must be a new file: {output}")
    motion = load_motion_csv(source, sample_rate_hz=25)
    times, values = resample_motion(motion)
    # Validate serialized data before publishing an output file.
    with tempfile.TemporaryDirectory(prefix="heft_deploy_") as directory:
        candidate = Path(directory) / "deploy.csv"
        with candidate.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=HEFT_DELIMITER)
            writer.writerow(HEFT_FIELDS)
            writer.writerows(values)
        with candidate.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter=HEFT_DELIMITER)
            fields = tuple(next(reader))
            restored = np.asarray(list(reader), dtype=float)
        validate_deploy(fields, restored, times)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(candidate.read_text(encoding="utf-8"))
    return len(times)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One CSV or a directory of CSV files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="Output filename (single input only).")
    group.add_argument("--output-dir", type=Path, help="Default: <input directory>/deploy.")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories; preserve relative paths.")
    parser.add_argument("--reference", type=Path, help="Validate exact header against an existing HEFT CSV.")
    args = parser.parse_args(argv)
    try:
        source = args.input.expanduser().resolve()
        if args.reference:
            check_reference(args.reference)
        if source.is_file():
            sources = [source]
            base = source.parent
        elif source.is_dir():
            if args.output:
                raise ValueError("--output requires a single input file; use --output-dir for batches.")
            base = source
            sources = sorted(p for p in (base.rglob("*") if args.recursive else base.iterdir())
                             if p.is_file() and p.suffix.lower() == ".csv"
                             and not p.stem.lower().endswith("_deploy"))
        else:
            raise ValueError(f"Input does not exist: {source}")
        destination = (args.output_dir or base / "deploy").expanduser().resolve()
        if source.is_dir():
            sources = [p for p in sources if not p.is_relative_to(destination)]
        if not sources:
            raise ValueError("No input CSV files found (existing *_deploy.csv files are excluded).")
        failed = 0
        for path in sources:
            output = args.output or destination / path.relative_to(base).with_name(path.stem + "_deploy.csv")
            try:
                frames = convert_csv(path, output)
                print(f"OK: {path} -> {output} ({frames} frames, 50 Hz, 49 columns)")
            except (OSError, ValueError, csv.Error, VisualizerError) as exc:
                failed += 1
                print(f"ERROR: {path}: {exc}")
        print(f"Finished: {len(sources) - failed} succeeded, {failed} failed.")
        return 1 if failed else 0
    except (OSError, ValueError, csv.Error, VisualizerError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
