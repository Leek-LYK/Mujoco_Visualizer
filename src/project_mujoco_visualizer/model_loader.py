"""MJCF/XML discovery and MuJoCo model introspection."""

from __future__ import annotations

import importlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelLoadError


@dataclass(frozen=True)
class JointInfo:
    """The MuJoCo addresses needed to write a joint into ``qpos``."""

    id: int
    name: str
    type_name: str
    qpos_adr: int
    qpos_size: int
    dof_adr: int


@dataclass(frozen=True)
class ModelDescription:
    """A compiled model plus its named root/scalar joint inventory."""

    path: Path
    model: Any
    root_joint: JointInfo | None
    joints: tuple[JointInfo, ...]
    candidates: tuple[Path, ...] = ()
    candidate_errors: tuple[str, ...] = ()

    @property
    def scalar_joints(self) -> tuple[JointInfo, ...]:
        return tuple(joint for joint in self.joints if joint.qpos_size == 1)

    @property
    def unsupported_joints(self) -> tuple[JointInfo, ...]:
        return tuple(joint for joint in self.joints if joint.qpos_size != 1)


def _mujoco() -> Any:
    try:
        return importlib.import_module("mujoco")
    except Exception as exc:  # ImportError and Windows DLL loading errors.
        raise ModelLoadError(
            "Could not import the official mujoco Python package. "
            "Check the current uv environment and whether mujoco.dll can be loaded "
            f"in this process ({type(exc).__name__}: {exc})."
        ) from exc


def _candidate_score(path: Path) -> tuple[int, str]:
    """Prefer an explicit scene/include entry while remaining deterministic."""

    score = 0
    lowered = path.name.lower()
    if lowered in {"scene.xml", "scene.mjcf"}:
        score += 100
    if "scene" in lowered:
        score += 20
    try:
        root = ET.parse(path).getroot()
        if root.tag.lower() == "mujoco":
            score += 5
        if root.find("include") is not None:
            score += 30
    except (ET.ParseError, OSError):
        score -= 100
    return score, str(path).lower()


def find_xml_candidates(asset_dir: str | Path) -> tuple[Path, ...]:
    """Find all MJCF/XML files below an asset directory in selection order."""

    directory = Path(asset_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ModelLoadError(f"Robot asset directory does not exist: {directory}")
    candidates = sorted(
        {
            path.resolve()
            for pattern in ("*.xml", "*.mjcf")
            for path in directory.rglob(pattern)
            if path.is_file() and _is_mjcf_root(path)
        },
        key=lambda path: (-_candidate_score(path)[0], _candidate_score(path)[1]),
    )
    if not candidates:
        raise ModelLoadError(f"No MJCF/XML entry file found under {directory}")
    return tuple(candidates)


def _is_mjcf_root(path: Path) -> bool:
    """Exclude package/metadata XML files from model entry discovery."""

    try:
        return ET.parse(path).getroot().tag.lower() == "mujoco"
    except (ET.ParseError, OSError):
        return False


def _joint_type_details(mujoco: Any, joint_type: int) -> tuple[str, int]:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return "free", 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return "ball", 4
    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return "slide", 1
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return "hinge", 1
    return f"unknown({joint_type})", 0


def describe_compiled_model(
    model: Any,
    path: str | Path,
    *,
    candidates: tuple[Path, ...] = (),
    candidate_errors: tuple[str, ...] = (),
) -> ModelDescription:
    """Extract names and qpos addresses from a compiled ``MjModel``."""

    mujoco = _mujoco()
    joints: list[JointInfo] = []
    roots: list[JointInfo] = []
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            name = f"joint_{joint_id}"
        type_name, qpos_size = _joint_type_details(mujoco, int(model.jnt_type[joint_id]))
        info = JointInfo(
            id=joint_id,
            name=name,
            type_name=type_name,
            qpos_adr=int(model.jnt_qposadr[joint_id]),
            qpos_size=qpos_size,
            dof_adr=int(model.jnt_dofadr[joint_id]),
        )
        if type_name == "free":
            roots.append(info)
        else:
            joints.append(info)
    if len(roots) > 1:
        raise ModelLoadError(
            f"Model {Path(path)} has {len(roots)} free root joints; exactly one is supported."
        )
    return ModelDescription(
        path=Path(path).resolve(),
        model=model,
        root_joint=roots[0] if roots else None,
        joints=tuple(joints),
        candidates=candidates,
        candidate_errors=candidate_errors,
    )


def load_model(
    model_path: str | Path | None = None,
    *,
    asset_dir: str | Path | None = None,
) -> ModelDescription:
    """Compile an explicitly supplied model or discover one below ``asset_dir``.

    When discovery is used, candidates are attempted in a deterministic order;
    the preferred ``scene.xml`` can therefore include the robot and floor while
    a standalone robot XML remains a fallback if the scene is invalid.
    """

    if model_path is not None:
        requested = Path(model_path).expanduser().resolve()
        if requested.is_dir():
            candidates = find_xml_candidates(requested)
        else:
            candidates = (requested,)
    else:
        if asset_dir is None:
            raise ModelLoadError("Supply --model or --asset-dir so a model can be located.")
        candidates = find_xml_candidates(asset_dir)

    errors: list[str] = []
    mujoco: Any | None = None
    for candidate in candidates:
        if not candidate.is_file():
            errors.append(f"{candidate}: file does not exist")
            continue
        if mujoco is None:
            mujoco = _mujoco()
        try:
            model = mujoco.MjModel.from_xml_path(str(candidate))
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        return describe_compiled_model(
            model,
            candidate,
            candidates=candidates,
            candidate_errors=tuple(errors),
        )

    details = "\n".join(f"  - {error}" for error in errors) or "  - no candidate was attempted"
    raise ModelLoadError("Could not compile any MJCF/XML candidate:\n" + details)
