#!/usr/bin/env python3
"""Generate geometry-free trajectory-quality sidecars from NavSim CF poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_cowa_world_model.training.cf_trajectory_quality import (  # noqa: E402  # isort: skip
    CF_QUALITY_SCHEMA,
    compute_counterfactual_trajectory_quality,
)
from app.vjepa_cowa_world_model.training.counterfactual_quality_sidecar_contract import (  # noqa: E402  # isort: skip
    CF_QUALITY_GENERATOR_VERSION,
    CF_QUALITY_POSE_OVERLAY_COORD_FRAME,
    FORMAL_V2_CF_QUALITY_FINGERPRINT_SCOPES,
    FORMAL_V2_CF_QUALITY_MAX_PROGRESS_M,
    FORMAL_V2_CF_QUALITY_TIMESTEP_SEC,
    FORMAL_V2_CF_QUALITY_WEIGHTS,
    formal_v2_cf_quality_timeline_contract,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (  # noqa: E402  # isort: skip
    FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST,
    formal_v2_navsim_cf_root_name_for_annotation_path,
    validate_formal_v2_navsim_cf_annotation_source,
)
from app.vjepa_cowa_world_model.training.navsim_data import (  # noqa: E402  # isort: skip
    load_counterfactual_annotations,
)
from app.vjepa_cowa_world_model.training.pose_overlay import (  # noqa: E402  # isort: skip
    PoseOverlayReader,
)

GENERATOR_VERSION = CF_QUALITY_GENERATOR_VERSION
_CF_SCENE_PATTERN = re.compile(r"^.+_cf_\d{6}_\d{6}$")
_QUALITY_WEIGHTS = dict(FORMAL_V2_CF_QUALITY_WEIGHTS)
_PKL_FINGERPRINT_SCOPES = frozenset({"content_sha256", "relative_path_identity"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_files(root: Path, paths: Sequence[Path]) -> str:
    """Hash relative names and contents of a deterministic source file set."""

    digest = hashlib.sha256()
    for path in sorted({candidate.resolve() for candidate in paths}):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _relative_path_identity(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix().encode("utf-8")
    return hashlib.sha256(relative).hexdigest()


def _sha256_relative_path_identities(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({candidate.resolve() for candidate in paths}):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
    return digest.hexdigest()


def _require_source_root(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {resolved}")
    return resolved


def _discover_cf_scenes(pkl_root: Path, *, scene_names: Optional[Sequence[str]] = None) -> list[Path]:
    if scene_names is None:
        paths = sorted(path.resolve() for path in pkl_root.glob("*.pkl") if not path.name.startswith("."))
    else:
        if isinstance(scene_names, (str, bytes)) or not isinstance(scene_names, Sequence):
            raise TypeError("scene_names must be a sequence of counterfactual scene names")
        names = list(scene_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("scene_names must contain non-empty strings")
        if names != sorted(set(names)):
            raise ValueError("scene_names must be sorted and unique")
        paths = [(pkl_root / f"{name}.pkl").resolve() for name in names]
        missing = [path.stem for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"selected counterfactual NavSim scenes do not exist: {missing[:8]}")
    if not paths:
        raise FileNotFoundError(f"no counterfactual NavSim .pkl scenes found under {pkl_root}")
    invalid_names = [path.stem for path in paths if _CF_SCENE_PATTERN.fullmatch(path.stem) is None]
    if invalid_names:
        raise ValueError(f"NavSim PKL root contains non-counterfactual scene names: {invalid_names[:8]}")
    return paths


def _formal_v2_selected_scene_names(annotations_path: Path, *, camera_name: str) -> list[str]:
    """Resolve the exact runtime-selected Formal-v2 hazard cohort."""

    if type(camera_name) is not str or not camera_name.strip():
        raise ValueError("camera_name must be a non-empty string")
    root_name = formal_v2_navsim_cf_root_name_for_annotation_path(annotations_path)
    annotation_contract = validate_formal_v2_navsim_cf_annotation_source(
        annotations_path,
        root_name=root_name,
    )
    annotations = load_counterfactual_annotations(
        str(annotations_path),
        camera_name,
        require_trajectory_match=True,
    )
    selected = sorted(
        scene_name
        for scene_name, annotation in annotations.items()
        if annotation["distortion"] is False
        and annotation["trajectory_match"] is True
        and annotation["accident"] is True
        and annotation["accident_type"] in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
    )
    if not selected:
        raise ValueError("Formal-v2 annotations select no matched hazard scenes")
    expected_selected_count = annotation_contract["selected_record_count"]
    if len(selected) != expected_selected_count:
        raise ValueError(
            f"Formal-v2 NavSim {root_name} selected_record_count mismatch: "
            f"expected={expected_selected_count}, actual={len(selected)}"
        )
    return selected


def _full_local_trajectory(
    reader: PoseOverlayReader, scene_name: str, *, timestep_sec: float
) -> tuple[np.ndarray, Path]:
    """Load every validated pose and express it in the first-pose ego frame."""

    table = reader._load_scene(scene_name)
    if table.frame_indices.shape[0] < 2:
        raise ValueError(f"counterfactual pose scene={scene_name!r} requires at least 2 poses")
    if np.any(np.diff(table.frame_indices) <= 0):
        raise ValueError(f"counterfactual pose scene={scene_name!r} frame indices must be strictly increasing")

    batch = reader.build_states_actions_ego_dynamics(
        scene_name,
        table.frame_indices.tolist(),
        action_dim=3,
        dt=timestep_sec,
    )
    xy_yaw = np.asarray(batch.states[:, [0, 1, 5]], dtype=np.float64)
    if not np.isfinite(xy_yaw).all():
        raise ValueError(f"counterfactual pose scene={scene_name!r} contains non-finite trajectory values")

    delta_xy = xy_yaw[:, :2] - xy_yaw[0, :2]
    anchor_yaw = float(xy_yaw[0, 2])
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    local = np.empty_like(xy_yaw)
    local[:, 0] = cos_yaw * delta_xy[:, 0] + sin_yaw * delta_xy[:, 1]
    local[:, 1] = -sin_yaw * delta_xy[:, 0] + cos_yaw * delta_xy[:, 1]
    yaw_delta = xy_yaw[:, 2] - anchor_yaw
    local[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))

    pose_path = reader._resolve_scene_file(scene_name)
    if pose_path is None:
        raise RuntimeError(f"resolved pose disappeared for counterfactual scene={scene_name!r}")
    return local, pose_path.resolve()


def _formal_v2_future_local_trajectory(
    reader: PoseOverlayReader,
    scene_name: str,
    *,
    timestep_sec: float,
) -> tuple[np.ndarray, Path]:
    """Load the unique CF start window and express its eight targets at frame-3."""

    timeline = formal_v2_cf_quality_timeline_contract()
    requested_indices = list(range(timeline["num_total_frames"]))
    try:
        batch = reader.build_states_actions_ego_dynamics(
            scene_name,
            requested_indices,
            action_dim=3,
            dt=timestep_sec,
        )
    except KeyError as exc:
        raise ValueError(
            f"Formal-v2 counterfactual pose scene={scene_name!r} must cover the exact 12-frame "
            "dataset window frame indices 0..11"
        ) from exc
    xy_yaw = np.asarray(batch.states[:, [0, 1, 5]], dtype=np.float64)
    if xy_yaw.shape != (timeline["num_total_frames"], 3) or not np.isfinite(xy_yaw).all():
        raise ValueError(f"Formal-v2 counterfactual pose scene={scene_name!r} must yield finite [12, 3] ego poses")

    anchor_index = timeline["anchor_frame_index"]
    target_indices = np.asarray(timeline["scored_frame_indices"], dtype=np.int64)
    delta_xy = xy_yaw[target_indices, :2] - xy_yaw[anchor_index, :2]
    anchor_yaw = float(xy_yaw[anchor_index, 2])
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    future_local = np.empty((timeline["num_target_frames"], 3), dtype=np.float64)
    future_local[:, 0] = cos_yaw * delta_xy[:, 0] + sin_yaw * delta_xy[:, 1]
    future_local[:, 1] = -sin_yaw * delta_xy[:, 0] + cos_yaw * delta_xy[:, 1]
    yaw_delta = xy_yaw[target_indices, 2] - anchor_yaw
    future_local[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))

    pose_path = reader._resolve_scene_file(scene_name)
    if pose_path is None:
        raise RuntimeError(f"resolved pose disappeared for counterfactual scene={scene_name!r}")
    return future_local, pose_path.resolve()


def _quality_entry(
    future_trajectory: np.ndarray,
    *,
    pose_count: int,
    timestep_sec: float,
    max_progress_m: float,
) -> dict:
    future = torch.from_numpy(future_trajectory.astype(np.float32, copy=False)).reshape(
        1,
        1,
        future_trajectory.shape[0],
        3,
    )
    quality = compute_counterfactual_trajectory_quality(
        future,
        torch.ones((1, 1), dtype=future.dtype),
        dataset_domains=["counterfactual"],
        timestep_sec=timestep_sec,
        max_progress_m=max_progress_m,
    )
    metrics = {
        "progress_score": float(quality.progress_score.item()),
        "reverse_risk": float(quality.reverse_risk.item()),
        "comfort_score": float(quality.comfort_score.item()),
        "path_efficiency": float(quality.path_efficiency.item()),
    }
    quality_score = (
        _QUALITY_WEIGHTS["progress"] * metrics["progress_score"]
        + _QUALITY_WEIGHTS["non_reverse"] * (1.0 - metrics["reverse_risk"])
        + _QUALITY_WEIGHTS["comfort"] * metrics["comfort_score"]
        + _QUALITY_WEIGHTS["path_efficiency"] * metrics["path_efficiency"]
    )
    return {
        "status": "passed",
        "pose_count": int(pose_count),
        "scored_pose_count": int(future_trajectory.shape[0]),
        "metrics": metrics,
        "quality_score": quality_score,
    }


def generate_sidecar(
    *,
    pkl_root: str | Path,
    pose_overlay_root: str | Path,
    output_path: str | Path,
    pose_overlay_coord_frame: str,
    pose_overlay_txt_start_seconds: float,
    timestep_sec: float = 0.5,
    max_progress_m: float = 20.0,
    scene_names: Optional[Sequence[str]] = None,
    pkl_fingerprint_scope: str = "content_sha256",
    formal_v2_timeline: bool = False,
) -> dict:
    """Generate and exclusively create one loader-compatible CF quality JSON."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    timestep_sec = float(timestep_sec)
    pose_overlay_txt_start_seconds = float(pose_overlay_txt_start_seconds)
    max_progress_m = float(max_progress_m)
    if pkl_fingerprint_scope not in _PKL_FINGERPRINT_SCOPES:
        raise ValueError(
            f"pkl_fingerprint_scope must be one of {sorted(_PKL_FINGERPRINT_SCOPES)}, "
            f"got {pkl_fingerprint_scope!r}"
        )
    if type(formal_v2_timeline) is not bool:
        raise TypeError("formal_v2_timeline must be an exact boolean")
    if pose_overlay_coord_frame != CF_QUALITY_POSE_OVERLAY_COORD_FRAME:
        raise ValueError(
            "pose_overlay_coord_frame must be exactly "
            f"{CF_QUALITY_POSE_OVERLAY_COORD_FRAME!r}, got {pose_overlay_coord_frame!r}"
        )
    if not math.isfinite(timestep_sec) or timestep_sec <= 0.0:
        raise ValueError(f"timestep_sec must be finite and positive, got {timestep_sec}")
    if not math.isfinite(max_progress_m) or max_progress_m <= 0.0:
        raise ValueError(f"max_progress_m must be finite and positive, got {max_progress_m}")
    if not math.isfinite(pose_overlay_txt_start_seconds) or pose_overlay_txt_start_seconds < 0.0:
        raise ValueError(
            "pose_overlay_txt_start_seconds must be finite and non-negative, " f"got {pose_overlay_txt_start_seconds}"
        )
    if formal_v2_timeline:
        if timestep_sec != FORMAL_V2_CF_QUALITY_TIMESTEP_SEC:
            raise ValueError(f"Formal-v2 quality timestep_sec must be exactly {FORMAL_V2_CF_QUALITY_TIMESTEP_SEC}")
        if max_progress_m != FORMAL_V2_CF_QUALITY_MAX_PROGRESS_M:
            raise ValueError(f"Formal-v2 quality max_progress_m must be exactly {FORMAL_V2_CF_QUALITY_MAX_PROGRESS_M}")
        if pose_overlay_txt_start_seconds != 0.0:
            raise ValueError("Formal-v2 quality pose_overlay_txt_start_seconds must be exactly 0.0")
        if pkl_fingerprint_scope != FORMAL_V2_CF_QUALITY_FINGERPRINT_SCOPES["navsim_pkl"]:
            raise ValueError(
                "Formal-v2 quality pkl_fingerprint_scope must be exactly "
                f"{FORMAL_V2_CF_QUALITY_FINGERPRINT_SCOPES['navsim_pkl']!r}"
            )

    resolved_pkl_root = _require_source_root(Path(pkl_root), name="NavSim PKL root")
    resolved_pose_root = _require_source_root(Path(pose_overlay_root), name="pose overlay root")
    pkl_paths = _discover_cf_scenes(resolved_pkl_root, scene_names=scene_names)
    reader = PoseOverlayReader(
        resolved_pose_root,
        coord_frame=pose_overlay_coord_frame,
        required=True,
        txt_start_seconds=pose_overlay_txt_start_seconds,
    )

    scenes: dict[str, dict] = {}
    pose_paths: list[Path] = []
    for pkl_path in pkl_paths:
        scene_name = pkl_path.stem
        if formal_v2_timeline:
            future_trajectory, pose_path = _formal_v2_future_local_trajectory(
                reader,
                scene_name,
                timestep_sec=timestep_sec,
            )
            pose_count = formal_v2_cf_quality_timeline_contract()["num_total_frames"]
        else:
            local_trajectory, pose_path = _full_local_trajectory(reader, scene_name, timestep_sec=timestep_sec)
            future_trajectory = local_trajectory[1:]
            pose_count = int(local_trajectory.shape[0])
        entry = _quality_entry(
            future_trajectory,
            pose_count=pose_count,
            timestep_sec=timestep_sec,
            max_progress_m=max_progress_m,
        )
        entry["source_fingerprints"] = {
            "navsim_pkl": (
                _sha256_file(pkl_path)
                if pkl_fingerprint_scope == "content_sha256"
                else _relative_path_identity(resolved_pkl_root, pkl_path)
            ),
            "pose_overlay": _sha256_file(pose_path),
        }
        scenes[scene_name] = entry
        pose_paths.append(pose_path)

    payload = {
        "schema": CF_QUALITY_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "fingerprint_algorithm": "sha256",
        "timestep_sec": timestep_sec,
        "pose_overlay_coord_frame": pose_overlay_coord_frame,
        "pose_overlay_txt_start_seconds": pose_overlay_txt_start_seconds,
        "max_progress_m": max_progress_m,
        "weights": dict(_QUALITY_WEIGHTS),
        "source_fingerprint_scopes": {
            "navsim_pkl": pkl_fingerprint_scope,
            "pose_overlay": "content_sha256",
        },
        **({"timeline_contract": formal_v2_cf_quality_timeline_contract()} if formal_v2_timeline else {}),
        "scene_count": len(scenes),
        "source_fingerprints": {
            "navsim_pkl_root": {
                "fingerprint": (
                    _sha256_files(resolved_pkl_root, pkl_paths)
                    if pkl_fingerprint_scope == "content_sha256"
                    else _sha256_relative_path_identities(resolved_pkl_root, pkl_paths)
                ),
                "file_count": len(pkl_paths),
            },
            "pose_overlay_root": {
                "fingerprint": _sha256_files(resolved_pose_root, pose_paths),
                "file_count": len(set(pose_paths)),
            },
        },
        "scenes": scenes,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl-root", required=True, type=Path, help="Counterfactual NavSim PKL directory.")
    parser.add_argument("--pose-overlay-root", required=True, type=Path, help="PoseOverlayReader source directory.")
    parser.add_argument("--output", required=True, type=Path, help="New output JSON; existing files are rejected.")
    parser.add_argument("--timestep-sec", type=float, default=0.5)
    parser.add_argument(
        "--pose-overlay-coord-frame",
        choices=(CF_QUALITY_POSE_OVERLAY_COORD_FRAME,),
        required=True,
    )
    parser.add_argument("--pose-overlay-txt-start-seconds", type=float, required=True)
    parser.add_argument("--max-progress-m", type=float, default=20.0)
    parser.add_argument(
        "--pkl-fingerprint-scope",
        choices=sorted(_PKL_FINGERPRINT_SCOPES),
        default="content_sha256",
    )
    parser.add_argument("--formal-v2-timeline", action="store_true")
    parser.add_argument(
        "--formal-v2-annotations",
        type=Path,
        help="Annotation JSON used to select the exact Formal-v2 matched-hazard cohort.",
    )
    parser.add_argument("--camera-name", help="Camera name used to normalize annotation scene IDs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if (args.formal_v2_annotations is None) != (args.camera_name is None):
        raise ValueError("--formal-v2-annotations and --camera-name must be provided together")
    if args.formal_v2_annotations is not None and not args.formal_v2_timeline:
        raise ValueError("--formal-v2-annotations requires --formal-v2-timeline")
    scene_names = (
        None
        if args.formal_v2_annotations is None
        else _formal_v2_selected_scene_names(
            args.formal_v2_annotations,
            camera_name=args.camera_name,
        )
    )
    generate_sidecar(
        pkl_root=args.pkl_root,
        pose_overlay_root=args.pose_overlay_root,
        output_path=args.output,
        pose_overlay_coord_frame=args.pose_overlay_coord_frame,
        pose_overlay_txt_start_seconds=args.pose_overlay_txt_start_seconds,
        timestep_sec=args.timestep_sec,
        max_progress_m=args.max_progress_m,
        scene_names=scene_names,
        pkl_fingerprint_scope=args.pkl_fingerprint_scope,
        formal_v2_timeline=args.formal_v2_timeline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
