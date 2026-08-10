"""事故标注格式规范化与严格校验工具。"""

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

ANNO_FIELDS = (
    "distortion",
    "accident",
    "accident_type",
    "accident_frame_idx",
    "trajectory_match",
    "reverse",
    "static",
    "run_red_light",
    "suggested_action",
)

KNOWN_ACCIDENT_TYPES = {
    "自车行为引起",
    "非自车行为引起",
    "有事故但与自车无关",
}
ACTION_ORDER = ("停车", "左避让", "右避让", "保持直行")


def _validate_video_frame_count(video_frame_count: int) -> None:
    if isinstance(video_frame_count, bool) or not isinstance(video_frame_count, int) or video_frame_count <= 0:
        raise ValueError(f"video_frame_count must be a positive integer, got {video_frame_count!r}")


def _normalize_optional_bool(value: Any, *, scene: str, field: str) -> Any:
    if value == "":
        return None
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"Annotation {scene!r} has non-bool {field}={value!r}")


def _normalize_actions(value: Any, *, scene: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Annotation {scene!r} has non-list suggested_action={value!r}")
    unknown = [action for action in value if action not in ACTION_ORDER]
    if unknown:
        raise ValueError(f"Annotation {scene!r} has unknown suggested_action={unknown[0]!r}")
    selected = set(value)
    return [action for action in ACTION_ORDER if action in selected]


def _normalize_frame_index(
    value: Any,
    *,
    scene: str,
    accident: bool,
    video_frame_count: int,
    change_counts: Counter,
    review_records: List[Dict[str, Any]],
) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Annotation {scene!r} has non-integer accident_frame_idx={value!r}")
    if value == -1:
        return None
    if not accident:
        change_counts["non_accident_frame_cleared"] += 1
        return None
    if not 1 <= value <= video_frame_count:
        change_counts["out_of_range_frame_cleared"] += 1
        review_records.append(
            {
                "scene": scene,
                "field": "accident_frame_idx",
                "source_value": value,
                "reason": f"outside 1-based video frame range [1, {video_frame_count}]",
            }
        )
        return None
    return value


def normalize_annotation_records(
    records: List[Dict[str, Any]], *, video_frame_count: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """将事故标注记录规范化为固定九字段格式。"""
    _validate_video_frame_count(video_frame_count)
    if not isinstance(records, list) or not records:
        raise ValueError("Accident annotations must be a non-empty list")

    normalized: List[Dict[str, Any]] = []
    changed_record_count = 0
    change_counts: Counter = Counter()
    review_records: List[Dict[str, Any]] = []
    seen_scenes = set()
    for source_record in records:
        if not isinstance(source_record, dict):
            raise ValueError(f"Annotation entry must be a mapping, got {source_record!r}")
        scene = source_record.get("scene")
        source_annos = source_record.get("annos")
        if not isinstance(scene, str) or not scene:
            raise ValueError(f"Annotation entry has invalid scene={scene!r}")
        if scene in seen_scenes:
            raise ValueError(f"Duplicate annotation scene {scene!r}")
        seen_scenes.add(scene)
        if not isinstance(source_annos, dict):
            raise ValueError(f"Annotation {scene!r} has non-mapping annos={source_annos!r}")
        if "accident_frame_idx" not in source_annos:
            change_counts["missing_accident_frame_idx_filled"] += 1
        if "trajectory_match" not in source_annos:
            change_counts["missing_trajectory_match_filled"] += 1
        source_frame_idx = source_annos.get("accident_frame_idx")
        if isinstance(source_frame_idx, int) and not isinstance(source_frame_idx, bool) and source_frame_idx == -1:
            change_counts["frame_sentinel_to_null"] += 1

        distortion = source_annos.get("distortion")
        if not isinstance(distortion, bool):
            raise ValueError(f"Annotation {scene!r} has non-bool distortion={distortion!r}")
        if distortion:
            replacement = {field: None for field in ANNO_FIELDS}
            replacement["distortion"] = True
            change_counts["distortion_records_cleared"] += 1
        else:
            accident = source_annos.get("accident")
            if not isinstance(accident, bool):
                raise ValueError(f"Annotation {scene!r} has non-bool accident={accident!r}")
            accident_type = source_annos.get("accident_type")
            if accident:
                if accident_type not in KNOWN_ACCIDENT_TYPES:
                    raise ValueError(f"Annotation {scene!r} has unknown accident_type={accident_type!r}")
            else:
                if accident_type not in {"", "正常", None}:
                    raise ValueError(f"Annotation {scene!r} has invalid normal accident_type={accident_type!r}")
                if accident_type != "正常":
                    change_counts["normal_accident_type_repaired"] += 1
                accident_type = "正常"

            accident_frame_idx = _normalize_frame_index(
                source_frame_idx,
                scene=scene,
                accident=accident,
                video_frame_count=video_frame_count,
                change_counts=change_counts,
                review_records=review_records,
            )
            trajectory_match = _normalize_optional_bool(
                source_annos.get("trajectory_match"), scene=scene, field="trajectory_match"
            )
            reverse = _normalize_optional_bool(source_annos.get("reverse"), scene=scene, field="reverse")
            static = _normalize_optional_bool(source_annos.get("static"), scene=scene, field="static")
            run_red_light = _normalize_optional_bool(
                source_annos.get("run_red_light"), scene=scene, field="run_red_light"
            )
            suggested_action = _normalize_actions(source_annos.get("suggested_action"), scene=scene)
            if not accident:
                if suggested_action not in (None, []):
                    raise ValueError(f"Annotation {scene!r} is normal but has suggested_action={suggested_action!r}")
                suggested_action = []
            elif suggested_action != source_annos.get("suggested_action"):
                change_counts["suggested_action_canonicalized"] += 1

            replacement = {
                "distortion": False,
                "accident": accident,
                "accident_type": accident_type,
                "accident_frame_idx": accident_frame_idx,
                "trajectory_match": trajectory_match,
                "reverse": reverse,
                "static": static,
                "run_red_light": run_red_light,
                "suggested_action": suggested_action,
            }

        if source_annos != replacement:
            changed_record_count += 1
        normalized.append({"scene": scene, "annos": deepcopy(replacement)})

    return normalized, {
        "record_count": len(normalized),
        "changed_record_count": changed_record_count,
        "change_counts": dict(sorted(change_counts.items())),
        "review_records": review_records,
    }


def accident_frame_idx_to_tensor_index(value: Any, *, video_frame_count: int) -> Any:
    """将 1-based 人工标注帧号转换为 0-based 张量下标。"""
    _validate_video_frame_count(video_frame_count)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= video_frame_count:
        raise ValueError(f"accident_frame_idx={value!r} is outside 1-based video frame range [1, {video_frame_count}]")
    return value - 1


def _validate_clip(*, scene: str, frame_idx: Any, clip_dir: Path, video_frame_count: int) -> Tuple[int, int]:
    clip_path = clip_dir / f"{scene}.clip"
    if not clip_path.is_file():
        if frame_idx is not None:
            raise ValueError(f"Annotation {scene!r} with accident_frame_idx={frame_idx} requires a clip file")
        return 0, 1
    try:
        with clip_path.open("r", encoding="utf-8") as f:
            clip = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read annotation clip {clip_path}: {exc}") from exc
    sequence = clip.get("sequence") if isinstance(clip, dict) else None
    expected_length = video_frame_count + 1
    if not isinstance(sequence, list) or len(sequence) != expected_length:
        raise ValueError(
            f"Annotation clip {clip_path} must contain {expected_length} sequence items "
            f"({video_frame_count} video frames plus one trajectory overlay)"
        )
    return 1, 0


def validate_annotation_records(
    records: List[Dict[str, Any]], *, video_frame_count: int, clip_dir: Any = None
) -> Dict[str, int]:
    """严格校验统一后的事故标注记录。"""
    _validate_video_frame_count(video_frame_count)
    if not isinstance(records, list) or not records:
        raise ValueError("Accident annotations must be a non-empty list")
    resolved_clip_dir = Path(clip_dir) if clip_dir is not None else None
    if resolved_clip_dir is not None and not resolved_clip_dir.is_dir():
        raise ValueError(f"clip_dir does not exist or is not a directory: {resolved_clip_dir}")

    seen_scenes = set()
    clip_count = 0
    missing_clip_count = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"scene", "annos"}:
            raise ValueError(f"Annotation entry must contain exactly 'scene' and 'annos', got {record!r}")
        scene = record["scene"]
        annos = record["annos"]
        if not isinstance(scene, str) or not scene:
            raise ValueError(f"Annotation entry has invalid scene={scene!r}")
        if scene in seen_scenes:
            raise ValueError(f"Duplicate annotation scene {scene!r}")
        seen_scenes.add(scene)
        if not isinstance(annos, dict) or tuple(annos) != ANNO_FIELDS:
            actual_fields = tuple(annos) if isinstance(annos, dict) else type(annos).__name__
            raise ValueError(
                f"Annotation {scene!r} must contain canonical annotation fields {ANNO_FIELDS}, got {actual_fields}"
            )

        distortion = annos["distortion"]
        if not isinstance(distortion, bool):
            raise ValueError(f"Annotation {scene!r} has non-bool distortion={distortion!r}")
        if distortion:
            non_null = [field for field in ANNO_FIELDS[1:] if annos[field] is not None]
            if non_null:
                raise ValueError(f"Distorted annotation {scene!r} has non-null fields {non_null}")
        else:
            accident = annos["accident"]
            if not isinstance(accident, bool):
                raise ValueError(f"Annotation {scene!r} has non-bool accident={accident!r}")
            accident_type = annos["accident_type"]
            if accident and accident_type not in KNOWN_ACCIDENT_TYPES:
                raise ValueError(f"Annotation {scene!r} has unknown accident_type={accident_type!r}")
            if not accident and accident_type != "正常":
                raise ValueError(f"Normal annotation {scene!r} must have accident_type='正常'")

            frame_idx = annos["accident_frame_idx"]
            if not accident and frame_idx is not None:
                raise ValueError(f"Normal annotation {scene!r} must have null accident_frame_idx")
            accident_frame_idx_to_tensor_index(frame_idx, video_frame_count=video_frame_count)
            for field in ("trajectory_match", "reverse", "static", "run_red_light"):
                value = annos[field]
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"Annotation {scene!r} has non-bool {field}={value!r}")

            actions = annos["suggested_action"]
            if actions is not None:
                canonical_actions = _normalize_actions(actions, scene=scene)
                if actions != canonical_actions:
                    raise ValueError(
                        f"Annotation {scene!r} must use canonical suggested_action order {canonical_actions!r}"
                    )
            if not accident and actions != []:
                raise ValueError(f"Normal annotation {scene!r} must have suggested_action=[]")

        if resolved_clip_dir is not None:
            found, missing = _validate_clip(
                scene=scene,
                frame_idx=annos["accident_frame_idx"],
                clip_dir=resolved_clip_dir,
                video_frame_count=video_frame_count,
            )
            clip_count += found
            missing_clip_count += missing

    return {
        "record_count": len(records),
        "clip_count": clip_count,
        "missing_clip_count": missing_clip_count,
    }
