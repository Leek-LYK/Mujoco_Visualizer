"""Add zero-pose quintic transitions before and after Roban motion CSVs.

Run with no arguments to process data/roban/大脑生成/roban_out/roban_out into its smoothed
subdirectory. Original motion values are preserved; timestamps are shifted.
Root position/orientation are held at the respective endpoint during padding.
The transition has zero endpoint velocity/acceleration; the untouched motion
itself need not have zero endpoint derivatives, so global C2 is not guaranteed.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data/roban/大脑生成/roban_out/roban_out"


def smooth_csv(source: Path, output: Path, intro: float = 2.0,
               outro: float = 2.0) -> tuple[int, int, float, float]:
    """Pad joint columns ending in _dof; keep source samples exactly once.

    Durations round up to whole frame intervals. Input must have uniform timing
    (allowing microsecond timestamp rounding). All default joint angles are 0.
    Existing output files are never overwritten.
    """
    if any(not math.isfinite(t) or t <= 0 for t in (intro, outro)):
        raise ValueError("Transition durations must be finite and positive.")
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError(f"Output must be a new file: {output}")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        rows = [row for row in reader if row]
    if len(header) != len(set(header)) or "timestamp" not in header:
        raise ValueError(f"Expected unique columns and timestamp: {source}")
    joints = [i for i, name in enumerate(header) if name.endswith("_dof")]
    root = {"root_translateX", "root_translateY", "root_translateZ",
            "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"}
    if not joints or set(header) != root | {"timestamp"} | {header[i] for i in joints}:
        raise ValueError(f"Expected timestamp, seven root fields and *_dof joints: {source}")
    if len(rows) < 2 or any(len(row) != len(header) for row in rows):
        raise ValueError(f"Expected at least two complete motion frames: {source}")
    values = [[float(cell) for cell in row] for row in rows]
    if any(not math.isfinite(x) for row in values for x in row):
        raise ValueError(f"Non-finite motion values: {source}")
    ti = header.index("timestamp")
    times = [row[ti] for row in values]
    steps = [b - a for a, b in zip(times, times[1:])]
    dt = statistics.median(steps)
    if dt <= 0 or any(step <= 0 or not math.isclose(step, dt, rel_tol=1e-4, abs_tol=2e-6)
                      for step in steps):
        raise ValueError(f"Expected increasing, uniformly sampled timestamps: {source}")
    n_intro = max(1, math.ceil(intro / dt - 1e-9))
    n_outro = max(1, math.ceil(outro / dt - 1e-9))
    shift = n_intro * dt

    def transition(endpoint: list[str], numeric: list[float], u: float,
                   timestamp: float, entering: bool) -> list[str]:
        weight = u ** 3 * (10 + u * (-15 + 6 * u))
        if not entering:
            weight = 1 - weight
        row = endpoint.copy()
        row[ti] = format(timestamp, ".12g")
        for j in joints:
            row[j] = format(weight * numeric[j], ".12g")
        return row

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        # The original first frame supplies u=1; do not duplicate it.
        for k in range(n_intro):
            writer.writerow(transition(rows[0], values[0], k / n_intro,
                                        times[0] + k * dt, True))
        for original, timestamp in zip(rows, times):
            row = original.copy()
            row[ti] = format(timestamp + shift, ".12g")
            writer.writerow(row)
        # The original last frame supplies u=0; do not duplicate it.
        for k in range(1, n_outro + 1):
            writer.writerow(transition(rows[-1], values[-1], k / n_outro,
                                        times[-1] + shift + k * dt, False))
    return len(rows), len(rows) + n_intro + n_outro, shift, n_outro * dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="One CSV or a directory of CSVs (nonrecursive).")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: smoothed subdirectory beside input CSVs.")
    parser.add_argument("--intro-seconds", type=float, default=2.0)
    parser.add_argument("--outro-seconds", type=float, default=2.0)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    sources = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    output_dir = args.output_dir or (source if source.is_dir() else source.parent) / "smoothed"
    try:
        if not sources:
            raise ValueError(f"No CSV files found: {source}")
        # Check all destinations before writing the batch.
        for path in sources:
            if (output_dir / path.name).exists():
                raise ValueError(f"Output already exists: {output_dir / path.name}")
        for path in sources:
            before, after, intro, outro = smooth_csv(
                path, output_dir / path.name, args.intro_seconds, args.outro_seconds)
            print(f"{path.name}: {before} -> {after} frames; +{intro:g}s / +{outro:g}s")
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Saved {len(sources)} files to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
