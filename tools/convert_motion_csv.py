"""Convert a commented retargeting CSV to the visualizer's named CSV format."""

from __future__ import annotations

import argparse
import csv
import io
import tempfile
from pathlib import Path

from project_mujoco_visualizer.errors import VisualizerError
from project_mujoco_visualizer.motion_loader import load_motion_csv


def convert_csv(source: Path, output: Path) -> tuple[int, float]:
    """Preserve numeric text and timing, and validate before creating output."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("Input and output must be different files.")
    if output.exists():
        raise ValueError(f"Output already exists; choose another --output: {output}")

    lines = source.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    content = "".join(line for line in lines if not line.lstrip().startswith("#"))
    reader = csv.reader(io.StringIO(content, newline=""))
    header = next((row for row in reader if any(cell.strip() for cell in row)), None)
    if header is None:
        raise ValueError("CSV contains no header or motion data.")
    aliases = {
        "root_qx": "root_quat_x", "root_qy": "root_quat_y",
        "root_qz": "root_quat_z", "root_qw": "root_quat_w",
    }
    names = [aliases.get(name.strip(), name.strip().removeprefix("dof_")) for name in header]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(names)
    writer.writerows(reader)
    converted = buffer.getvalue()

    # Reuse the visualizer's numeric, quaternion, joint and timestamp checks.
    with tempfile.TemporaryDirectory(prefix="motion_csv_") as directory:
        candidate = Path(directory) / "converted.csv"
        candidate.write_text(converted, encoding="utf-8", newline="")
        motion = load_motion_csv(candidate)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(converted)
    return motion.frame_count, motion.sample_rate_hz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source CSV with optional # metadata lines.")
    parser.add_argument("--output", type=Path, help="Default: <input directory>/converted/<input filename>.")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output or source.parent / "converted" / source.name
    try:
        frames, rate = convert_csv(source, output)
    except (OSError, ValueError, csv.Error, VisualizerError) as exc:
        parser.error(str(exc))
    print(f"Converted: {output.resolve()}")
    print(f"Validated: {frames} frames, {rate:g} Hz; original timestamps and values preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
