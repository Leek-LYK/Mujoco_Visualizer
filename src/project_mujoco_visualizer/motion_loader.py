"""CSV motion loading and schema discovery.

The loader intentionally does not assume that the CSV columns are in MuJoCo
order.  It first identifies semantic root/time fields, then exposes every
remaining field as a named joint candidate.  The model-aware validation is
performed by :mod:`joint_mapping`.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .errors import MotionFormatError


_TIME_ALIASES = {
    "timestamp",
    "time",
    "times",
    "t",
    "frames",
    "frametime",
    "timesec",
    "timeseconds",
}

_POSITION_ALIASES = {
    "x": ("roottranslatex", "roottranslationx", "rootpositionx", "rootposx", "rootx"),
    "y": ("roottranslatey", "roottranslationy", "rootpositiony", "rootposy", "rooty"),
    "z": ("roottranslatez", "roottranslationz", "rootpositionz", "rootposz", "rootz"),
}

_QUATERNION_ALIASES = {
    "w": ("rootquatw", "rootquaternionw", "rootorientationw", "rootrotw"),
    "x": ("rootquatx", "rootquaternionx", "rootorientationx", "rootrotx"),
    "y": ("rootquaty", "rootquaterniony", "rootorientationy", "rootroty"),
    "z": ("rootquatz", "rootquaternionz", "rootorientationz", "rootrotz"),
}

_JOINT_SUFFIX_RE = re.compile(
    r"(?:[_\s]*(?:dof|position|pos|angle|jointangle|qpos|q))$", re.IGNORECASE
)


def _normalise_header(value: str) -> str:
    """Return a comparison key without separators or a UTF-8 BOM."""

    return re.sub(r"[^a-z0-9]+", "", value.strip().lstrip("\ufeff").lower())


def _format_names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names) or "<none>"


def _find_unique_alias(header: Sequence[str], aliases: Sequence[str], field: str) -> str:
    normalised = {_normalise_header(name): name for name in header}
    matches = [normalised[alias] for alias in aliases if alias in normalised]
    if len(matches) > 1:
        raise MotionFormatError(
            f"Ambiguous {field} field: {_format_names(matches)}. "
            "Keep exactly one semantic column for each component."
        )
    if not matches:
        raise MotionFormatError(
            f"Missing {field} field. Tried aliases: {_format_names(aliases)}. "
            f"Available columns: {_format_names(header)}"
        )
    return matches[0]


def _parse_float(value: str, *, path: Path, row: int, column: str) -> float:
    text = value.strip()
    if not text:
        raise MotionFormatError(f"Empty numeric value at CSV row {row}, column {column!r}.")
    try:
        result = float(text)
    except ValueError as exc:
        raise MotionFormatError(
            f"Invalid numeric value {value!r} at {path} row {row}, column {column!r}."
        ) from exc
    if not math.isfinite(result):
        raise MotionFormatError(
            f"Non-finite numeric value {value!r} at {path} row {row}, column {column!r}."
        )
    return result


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return ","
    return dialect.delimiter


def _strip_joint_suffix(header: str) -> str:
    """Convert e.g. ``zarm_l1_joint_dof`` to ``zarm_l1_joint``."""

    stripped = header.strip().lstrip("\ufeff")
    candidate = _JOINT_SUFFIX_RE.sub("", stripped)
    return candidate.strip(" _-")


def _find_generic_quaternion_columns(
    header: Sequence[str], quat_order: str
) -> tuple[tuple[str, ...], str] | None:
    """Find a four-column root quaternion when component names are absent.

    Generic names such as ``root_quat_0`` are inherently ambiguous.  They are
    accepted only when the caller explicitly supplies ``wxyz`` or ``xyzw``.
    """

    candidates: list[tuple[int, str]] = []
    for index, name in enumerate(header):
        key = _normalise_header(name)
        if key in {"rootquat0", "rootquat1", "rootquat2", "rootquat3"}:
            candidates.append((int(key[-1]), name))
    if not candidates:
        return None
    if len(candidates) != 4 or {number for number, _ in candidates} != {0, 1, 2, 3}:
        raise MotionFormatError(
            "Generic root quaternion fields must contain exactly root_quat_0 through "
            "root_quat_3."
        )
    if quat_order == "auto":
        raise MotionFormatError(
            "Generic root_quat_0..3 fields need an explicit --quat-order wxyz or xyzw."
        )
    by_number = dict(candidates)
    ordered = tuple(by_number[number] for number in range(4))
    if quat_order == "wxyz":
        return ordered, "wxyz (explicit generic-column order)"
    return (ordered[3], ordered[0], ordered[1], ordered[2]), "xyzw -> wxyz (explicit conversion)"


@dataclass(frozen=True)
class MotionData:
    """A validated, named motion sequence."""

    path: Path
    field_names: tuple[str, ...]
    timestamps: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_names: tuple[str, ...]
    joint_values: np.ndarray
    timestamp_column: str | None
    root_position_columns: tuple[str, str, str]
    root_quaternion_columns: tuple[str, str, str, str]
    quaternion_input_order: str
    delimiter: str
    sample_rate_hz: float
    frame_durations: np.ndarray
    uniform_timing: bool

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def nominal_frame_dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def joint_values_by_name(self) -> dict[str, np.ndarray]:
        return {
            name: self.joint_values[:, index]
            for index, name in enumerate(self.joint_names)
        }

    def frame_joint_values(self, frame_index: int) -> dict[str, float]:
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"Frame index {frame_index} is outside [0, {self.frame_count}).")
        return {
            name: float(self.joint_values[frame_index, index])
            for index, name in enumerate(self.joint_names)
        }

    def report_lines(self) -> list[str]:
        """Return a concise analysis report for the CLI."""

        start = float(self.timestamps[0])
        end = float(self.timestamps[-1])
        timing = "uniform" if self.uniform_timing else "irregular"
        time_description = (
            f"{self.timestamp_column!r} [s]"
            if self.timestamp_column is not None
            else "generated from --sample-rate [s]"
        )
        return [
            f"CSV: {self.path}",
            f"  fields: {len(self.field_names)} ({self.delimiter!r} delimiter)",
            f"  frames: {self.frame_count}, time: {time_description}, "
            f"range={start:.6g}..{end:.6g} s",
            f"  sampling: {self.sample_rate_hz:.6g} Hz, dt={self.nominal_frame_dt:.6g} s ({timing})",
            "  root position: " + ", ".join(
                f"{axis} <- {column!r}"
                for axis, column in zip(("x", "y", "z"), self.root_position_columns)
            ),
            "  root orientation: "
            + ", ".join(self.root_quaternion_columns)
            + f"; input={self.quaternion_input_order}, MuJoCo qpos order=wxyz",
            f"  joint candidates: {self.joint_count} ({_format_names(self.joint_names)})",
        ]


def load_motion_csv(
    path: str | Path,
    *,
    sample_rate_hz: float | None = None,
    quat_order: str = "auto",
    quaternion_norm_tolerance: float = 1e-3,
) -> MotionData:
    """Read and semantically analyse a motion CSV.

    ``quat_order`` applies only to generic numbered quaternion columns.  When
    fields are named ``root_quat_w``/``x``/``y``/``z``, their component names
    determine the input order and the returned array is assembled as MuJoCo's
    ``w, x, y, z`` order.
    """

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise MotionFormatError(f"Motion CSV does not exist: {csv_path}")
    if quat_order not in {"auto", "wxyz", "xyzw"}:
        raise MotionFormatError("quat_order must be one of: auto, wxyz, xyzw")
    if sample_rate_hz is not None and (
        not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0
    ):
        raise MotionFormatError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz!r}.")

    try:
        raw_text = csv_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MotionFormatError(f"CSV is not valid UTF-8: {csv_path}") from exc
    delimiter = _detect_delimiter(raw_text[:8192])

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise MotionFormatError(f"CSV is empty: {csv_path}") from exc
        header = [cell.strip().lstrip("\ufeff") for cell in raw_header]
        if not header or any(not name for name in header):
            raise MotionFormatError("CSV header contains an empty field name.")
        normalised_header = [_normalise_header(name) for name in header]
        duplicates = sorted(
            {
                name
                for name in normalised_header
                if normalised_header.count(name) > 1
            }
        )
        if duplicates:
            duplicate_display = [
                "/".join(header[index] for index, key in enumerate(normalised_header) if key == duplicate)
                for duplicate in duplicates
            ]
            raise MotionFormatError(
                "Duplicate CSV header names are not allowed: "
                + "; ".join(duplicate_display)
            )

        rows: list[list[str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise MotionFormatError(
                    f"CSV row {row_number} has {len(row)} fields, expected {len(header)}."
                )
            rows.append(
                [
                    _parse_float(value, path=csv_path, row=row_number, column=header[index])
                    for index, value in enumerate(row)
                ]
            )

    if not rows:
        raise MotionFormatError(f"CSV contains a header but no data rows: {csv_path}")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():  # Defensive check after per-cell validation.
        raise MotionFormatError(f"CSV contains non-finite values: {csv_path}")
    column_index = {name: index for index, name in enumerate(header)}

    time_matches = [
        name for name in header if _normalise_header(name) in _TIME_ALIASES
    ]
    if len(time_matches) > 1:
        raise MotionFormatError(f"Ambiguous timestamp columns: {_format_names(time_matches)}")
    timestamp_column = time_matches[0] if time_matches else None
    if timestamp_column is not None:
        timestamps = values[:, column_index[timestamp_column]].copy()
        if timestamps.shape[0] > 1:
            differences = np.diff(timestamps)
            if np.any(differences <= 0):
                raise MotionFormatError(
                    f"Timestamp column {timestamp_column!r} must be strictly increasing."
                )
            nominal_dt = float(np.median(differences))
            if not math.isfinite(nominal_dt) or nominal_dt <= 0:
                raise MotionFormatError("Could not derive a positive frame interval from timestamps.")
            if sample_rate_hz is not None and not math.isclose(
                sample_rate_hz, 1.0 / nominal_dt, rel_tol=0.02, abs_tol=1e-9
            ):
                raise MotionFormatError(
                    f"Explicit sample rate {sample_rate_hz:g} Hz disagrees with timestamp-derived "
                    f"rate {1.0 / nominal_dt:g} Hz. Remove --sample-rate or correct it."
                )
            frame_durations = np.concatenate((differences, np.asarray([nominal_dt])))
            effective_rate = 1.0 / nominal_dt
            uniform_timing = bool(np.allclose(differences, nominal_dt, rtol=1e-5, atol=1e-8))
        else:
            if sample_rate_hz is None:
                raise MotionFormatError(
                    "A one-frame CSV needs --sample-rate because no frame interval can be inferred."
                )
            effective_rate = float(sample_rate_hz)
            frame_durations = np.asarray([1.0 / effective_rate])
            uniform_timing = True
    else:
        if sample_rate_hz is None:
            raise MotionFormatError(
                "CSV has no timestamp/time column. Supply --sample-rate so frame timing is explicit."
            )
        effective_rate = float(sample_rate_hz)
        timestamps = np.arange(values.shape[0], dtype=np.float64) / effective_rate
        frame_durations = np.full(values.shape[0], 1.0 / effective_rate, dtype=np.float64)
        uniform_timing = True

    position_columns = tuple(
        _find_unique_alias(header, _POSITION_ALIASES[axis], f"root position {axis}")
        for axis in ("x", "y", "z")
    )

    named_quaternion_columns: tuple[str, str, str, str] | None = None
    if all(
        any(_normalise_header(name) in aliases for name in header)
        for aliases in _QUATERNION_ALIASES.values()
    ):
        named_quaternion_columns = tuple(
            _find_unique_alias(header, _QUATERNION_ALIASES[component], f"root quaternion {component}")
            for component in ("w", "x", "y", "z")
        )  # type: ignore[assignment]
    generic_quaternion = _find_generic_quaternion_columns(header, quat_order)
    if named_quaternion_columns is not None and generic_quaternion is not None:
        raise MotionFormatError(
            "Both named and generic root quaternion columns were found; keep one representation."
        )
    if named_quaternion_columns is not None:
        quaternion_columns = named_quaternion_columns
        quaternion_input_order = "named components assembled as wxyz"
    elif generic_quaternion is not None:
        quaternion_columns, quaternion_input_order = generic_quaternion
    else:
        raise MotionFormatError(
            "Missing root orientation quaternion. Expected root_quat_w/x/y/z or "
            "root_quat_0..3 with an explicit order."
        )

    position = values[:, [column_index[column] for column in position_columns]].copy()
    quaternion = values[:, [column_index[column] for column in quaternion_columns]].copy()
    quaternion_norms = np.linalg.norm(quaternion, axis=1)
    if np.any(quaternion_norms <= 1e-12):
        bad = int(np.flatnonzero(quaternion_norms <= 1e-12)[0])
        raise MotionFormatError(f"Root quaternion at frame {bad} has near-zero norm.")
    if np.max(np.abs(quaternion_norms - 1.0)) > quaternion_norm_tolerance:
        maximum = float(np.max(np.abs(quaternion_norms - 1.0)))
        raise MotionFormatError(
            f"Root quaternion norm differs from 1 by up to {maximum:.6g}; "
            f"allowed tolerance is {quaternion_norm_tolerance:.6g}."
        )

    excluded_columns = {
        timestamp_column,
        *position_columns,
        *quaternion_columns,
    }
    joint_columns = [name for name in header if name not in excluded_columns]
    joint_names: list[str] = []
    joint_indices: list[int] = []
    seen_joints: dict[str, str] = {}
    for column in joint_columns:
        joint_name = _strip_joint_suffix(column)
        if not joint_name:
            raise MotionFormatError(f"Could not derive a joint name from column {column!r}.")
        if joint_name in seen_joints:
            raise MotionFormatError(
                f"Duplicate joint fields {seen_joints[joint_name]!r} and {column!r} "
                f"both map to joint {joint_name!r}."
            )
        seen_joints[joint_name] = column
        joint_names.append(joint_name)
        joint_indices.append(column_index[column])

    joint_values = values[:, joint_indices].copy() if joint_indices else np.empty((len(rows), 0))
    return MotionData(
        path=csv_path,
        field_names=tuple(header),
        timestamps=timestamps,
        root_position=position,
        root_quaternion=quaternion,
        joint_names=tuple(joint_names),
        joint_values=joint_values,
        timestamp_column=timestamp_column,
        root_position_columns=position_columns,  # type: ignore[arg-type]
        root_quaternion_columns=quaternion_columns,  # type: ignore[arg-type]
        quaternion_input_order=quaternion_input_order,
        delimiter=delimiter,
        sample_rate_hz=effective_rate,
        frame_durations=frame_durations,
        uniform_timing=uniform_timing,
    )
