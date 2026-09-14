"""Name-based CSV-to-MuJoCo joint mapping and qpos construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import JointMappingError
from .model_loader import JointInfo, ModelDescription
from .motion_loader import MotionData


@dataclass(frozen=True)
class JointBinding:
    joint_name: str
    input_field: str
    csv_index: int
    qpos_adr: int
    joint_type: str


@dataclass(frozen=True)
class JointMapping:
    """A complete mapping from semantic motion fields to model qpos slots."""

    root_joint: JointInfo
    root_position_qpos_adr: int
    root_quaternion_qpos_adr: int
    bindings: tuple[JointBinding, ...]

    @property
    def joint_count(self) -> int:
        return len(self.bindings)


def create_joint_mapping(model: ModelDescription, motion: MotionData) -> JointMapping:
    """Create and validate a strict name-based mapping.

    The order of ``motion.joint_names`` is deliberately irrelevant.  Every
    scalar MuJoCo joint must be present exactly once, and every CSV candidate
    must resolve to a scalar model joint.
    """

    if model.root_joint is None:
        raise JointMappingError(
            "The model has no free root joint, but the CSV contains root position and "
            "quaternion fields. A free root joint is required for this visualizer."
        )
    if model.unsupported_joints:
        unsupported = ", ".join(
            f"{joint.name} ({joint.type_name}, qpos size {joint.qpos_size})"
            for joint in model.unsupported_joints
        )
        raise JointMappingError(
            "The model contains non-scalar non-root joints that this CSV format cannot "
            f"map safely: {unsupported}"
        )

    model_by_name = {joint.name: joint for joint in model.scalar_joints}
    csv_by_name: dict[str, int] = {}
    duplicate_names: list[str] = []
    for index, name in enumerate(motion.joint_names):
        if name in csv_by_name:
            duplicate_names.append(name)
        csv_by_name[name] = index
    if duplicate_names:
        raise JointMappingError(
            "Duplicate CSV joint names: " + ", ".join(sorted(set(duplicate_names)))
        )

    missing = sorted(set(model_by_name) - set(csv_by_name))
    extra = sorted(set(csv_by_name) - set(model_by_name))
    if missing or extra:
        problems: list[str] = []
        if missing:
            problems.append("missing model joints: " + ", ".join(missing))
        if extra:
            problems.append("CSV joints not present in model: " + ", ".join(extra))
        raise JointMappingError("Joint name mapping failed; " + "; ".join(problems))

    bindings = tuple(
        JointBinding(
            joint_name=joint.name,
            input_field=_input_field_for_joint(motion, joint.name),
            csv_index=csv_by_name[joint.name],
            qpos_adr=joint.qpos_adr,
            joint_type=joint.type_name,
        )
        for joint in model.scalar_joints
    )
    return JointMapping(
        root_joint=model.root_joint,
        root_position_qpos_adr=model.root_joint.qpos_adr,
        root_quaternion_qpos_adr=model.root_joint.qpos_adr + 3,
        bindings=bindings,
    )


def _input_field_for_joint(motion: MotionData, joint_name: str) -> str:
    for index, name in enumerate(motion.joint_names):
        if name == joint_name:
            # CSV fields use the semantic name with a recognized suffix. NPZ
            # fields carry the names in the joint_names array, so fall back to
            # the semantic name when no per-joint field name exists.
            for column in motion.field_names:
                if column == joint_name or column.lower().startswith(joint_name.lower() + "_"):
                    return column
            return name
    raise JointMappingError(f"Internal error: no CSV column for joint {joint_name!r}.")


def build_qpos(
    model: ModelDescription,
    motion: MotionData,
    mapping: JointMapping,
    frame_index: int,
) -> np.ndarray:
    """Construct one complete MuJoCo qpos vector from a CSV frame."""

    if not 0 <= frame_index < motion.frame_count:
        raise IndexError(f"Frame index {frame_index} is outside [0, {motion.frame_count}).")
    if motion.joint_count != mapping.joint_count:
        raise JointMappingError(
            f"Motion has {motion.joint_count} joint values but mapping has "
            f"{mapping.joint_count} bindings. Rebuild the mapping."
        )
    qpos0 = np.asarray(getattr(model.model, "qpos0", np.zeros(int(model.model.nq))), dtype=np.float64)
    if qpos0.ndim != 1 or qpos0.shape[0] != int(model.model.nq):
        raise JointMappingError(
            f"Model qpos0 has shape {qpos0.shape}, expected ({int(model.model.nq)},)."
        )
    qpos = qpos0.copy()
    root_position_adr = mapping.root_position_qpos_adr
    root_quaternion_adr = mapping.root_quaternion_qpos_adr
    qpos[root_position_adr : root_position_adr + 3] = motion.root_position[frame_index]
    qpos[root_quaternion_adr : root_quaternion_adr + 4] = motion.root_quaternion[frame_index]
    for binding in mapping.bindings:
        qpos[binding.qpos_adr] = motion.joint_values[frame_index, binding.csv_index]
    if not np.isfinite(qpos).all():
        raise JointMappingError(f"Constructed qpos contains non-finite values at frame {frame_index}.")
    return qpos


def build_qpos_sequence(
    model: ModelDescription,
    motion: MotionData,
    mapping: JointMapping,
) -> np.ndarray:
    """Construct all qpos frames once, making playback deterministic."""

    sequence = np.vstack(
        [build_qpos(model, motion, mapping, frame_index) for frame_index in range(motion.frame_count)]
    )
    return sequence


def mapping_report_lines(model: ModelDescription, motion: MotionData, mapping: JointMapping) -> list[str]:
    lines = [
        f"Model: {model.path}",
        f"  discovered candidates: {len(model.candidates) or 1}",
        f"  selected model joints: {len(model.scalar_joints)} scalar + "
        f"root {mapping.root_joint.name!r}",
        f"  root qpos: position[{mapping.root_position_qpos_adr}:{mapping.root_position_qpos_adr + 3}], "
        f"quaternion[{mapping.root_quaternion_qpos_adr}:{mapping.root_quaternion_qpos_adr + 4}] "
        "(MuJoCo order wxyz)",
        f"  name-based mappings: {mapping.joint_count}",
    ]
    for binding in mapping.bindings:
        lines.append(
            f"    {binding.joint_name}: input field {binding.input_field!r} "
            f"-> qpos[{binding.qpos_adr}] ({binding.joint_type})"
        )
    if model.candidate_errors:
        lines.append("  skipped invalid candidates:")
        lines.extend(f"    {error}" for error in model.candidate_errors)
    return lines
