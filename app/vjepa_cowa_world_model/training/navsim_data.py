"""NavSim dataloader utilities for world model training."""

import hashlib
import json
import os
import pickle
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

if TYPE_CHECKING:
    from torch.utils.data import Sampler

from app.vjepa_cowa_world_model.training import collate
from app.vjepa_cowa_world_model.training.configs.data import (
    NAVSIM_DEFAULT_MAX_AGENTS,
    require_positive_navsim_max_agents,
    resolve_navsim_root_max_agents,
)
from app.vjepa_cowa_world_model.training.counterfactual_quality_sidecar_contract import (
    CF_QUALITY_SCHEMA,
    FORMAL_V2_CF_QUALITY_TIMESTEP_SEC,
    formal_v2_cf_quality_timeline_contract,
    validate_counterfactual_quality_sidecar_metadata,
)
from app.vjepa_cowa_world_model.training.counterfactual_supervision import KNOWN_HAZARD_TYPES
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST,
    FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION,
    FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA,
    validate_formal_v2_navsim_effective_root,
)
from app.vjepa_cowa_world_model.training.navsim_scene_filter_contract import load_navsim_scene_filter_contract
from app.vjepa_cowa_world_model.training.pose_overlay import PoseOverlayReader
from app.vjepa_cowa_world_model.utils.metrics import BEV_SIZE, _rasterize_agents_to_bev
from src.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_REQUIRE_ALL_FRAMES = "all_frames"
IMAGE_REQUIRE_OBSERVED_ONLY = "observed_only"
VALID_IMAGE_REQUIRE_POLICIES = {IMAGE_REQUIRE_ALL_FRAMES, IMAGE_REQUIRE_OBSERVED_ONLY}
WINDOW_START_SLIDING = "sliding"
WINDOW_START_COUNTERFACTUAL_SCENE_START = "counterfactual_scene_start"
VALID_WINDOW_START_POLICIES = {WINDOW_START_SLIDING, WINDOW_START_COUNTERFACTUAL_SCENE_START}
TIMESTAMP_POLICY_ROOT_CONTIGUOUS = "root_contiguous_v1"
TIMESTAMP_POLICY_ELIGIBLE_WINDOW_BOUNDARY = "eligible_window_boundary_v1"
VALID_TIMESTAMP_POLICIES = {
    TIMESTAMP_POLICY_ROOT_CONTIGUOUS,
    TIMESTAMP_POLICY_ELIGIBLE_WINDOW_BOUNDARY,
}
REAL_AGENT_GEOMETRY_SOURCE = "logged_nuscenes_gt"
REAL_AGENT_COORDINATE_FRAME = "per_frame_ego"
_CF_QUALITY_WEIGHTS = {
    "progress": 0.4,
    "non_reverse": 0.2,
    "comfort": 0.2,
    "path_efficiency": 0.2,
}


class CameraMetadataError(Exception):
    """相机内参/外参缺失等不可重试的致命数据错误（point 11/12）。

    __getitem__ 对这类异常不做随机换窗口重试，直接向上抛出（fail-loud），
    避免偶发缺失被重试静默掩盖。
    """


class AgentGeometryOverflowError(ValueError):
    """Real agent geometry exceeds the configured fixed batch capacity."""


@dataclass
class SceneRecord:
    scene_name: str
    pkl_path: str
    camera_dir: str
    valid_frame_indices: List[int]
    camera_dirs: Optional[Dict[str, str]] = None
    frame_count: Optional[int] = None
    # Per-raw-frame capture timestamps (microseconds). Used to reject windows
    # whose GT span crosses a real recording gap (missing timesteps), which the
    # raw-index continuity check could not detect. None => timestamps were
    # unavailable in the PKL and the time-gap check is skipped for this scene.
    frame_timestamps: Optional[List[int]] = None
    # Per-raw-frame NavSim tokens. None => at least one frame lacks a token
    # (e.g. converted nuScenes exports); official token anchoring raises on
    # such scenes while stride mode ignores this field.
    frame_tokens: Optional[List[str]] = None


@dataclass
class WindowRecord:
    """A single sliding-window sample inside a scene (log).

    Attributes
    ----------
    scene_idx : int
        Index into the parent ``scenes`` list (for PKL loading / caching).
    start_pos : int
        Raw frame start index inside the scene.
    """

    scene_idx: int
    start_pos: int


class BalancedRootConcatDataset(Dataset):
    """Balanced virtual concat for mixing multiple NavSim roots under DistributedSampler.

    The virtual index alternates roots and cycles shorter roots up to the longest root length:
    for two roots with lengths 2 and 3, the root/local indices are
    ``(0,0), (1,0), (0,1), (1,1), (0,0), (1,2)``.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        root_repeats: Optional[Sequence[int]] = None,
    ):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("BalancedRootConcatDataset requires at least one dataset")
        self.lengths = [len(dataset) for dataset in self.datasets]
        empty = [idx for idx, length in enumerate(self.lengths) if length <= 0]
        if empty:
            raise ValueError(f"BalancedRootConcatDataset cannot include empty datasets: indices={empty}")
        self.max_length = max(self.lengths)
        if root_repeats is None:
            repeats = [1 for _ in self.datasets]
        else:
            repeats = [int(repeat) for repeat in root_repeats]
            if len(repeats) != len(self.datasets):
                raise ValueError(
                    f"root_repeats length {len(repeats)} does not match datasets length {len(self.datasets)}"
                )
            if any(repeat <= 0 for repeat in repeats):
                raise ValueError(f"root_repeats must be positive integers, got {repeats}")
        self.root_repeats = repeats
        self.root_schedule = [
            dataset_idx for dataset_idx, repeat in enumerate(self.root_repeats) for _ in range(repeat)
        ]

    def __len__(self) -> int:
        return self.max_length * len(self.root_schedule)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        schedule_pos = int(index) % len(self.root_schedule)
        dataset_idx = self.root_schedule[schedule_pos]
        local_idx = (int(index) // len(self.root_schedule)) % self.lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


_COUNTERFACTUAL_SCENE_RE = re.compile(r"^(?P<base>.+)_cf_(?P<start>\d{6})_(?P<end>\d{6})$")


class RootTaggedDataset(Dataset):
    """Copy and stamp one root's samples before any root-composition policy."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        domain: str,
        dataset_root_name: str,
        dataset_root_index: int,
        future_agent_geometry_valid: bool,
        load_agent_annotations: bool = True,
        annotation_selection: str = "all_valid",
    ) -> None:
        self.dataset = dataset
        self.domain = str(domain)
        if self.domain not in {"real", "counterfactual"}:
            raise ValueError(f"domain must be 'real' or 'counterfactual', got {domain!r}")
        self.dataset_root_name = str(dataset_root_name)
        if not self.dataset_root_name:
            raise ValueError("dataset_root_name must be non-empty")
        self.dataset_root_index = int(dataset_root_index)
        self.load_agent_annotations = bool(load_agent_annotations)
        self.geometry_present = self.domain == "real" and self.load_agent_annotations
        requested_future_geometry = bool(future_agent_geometry_valid)
        if self.domain == "counterfactual" and requested_future_geometry:
            raise ValueError("counterfactual roots must never set future_agent_geometry_valid=true")
        self.future_agent_geometry_valid = requested_future_geometry
        self.annotation_selection = str(annotation_selection)
        if self.annotation_selection not in {"all_valid", "safe_only", FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION}:
            raise ValueError(
                "annotation_selection must be 'all_valid', 'safe_only', or "
                f"{FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}, got {annotation_selection!r}"
            )
        if self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION and self.domain != "counterfactual":
            raise ValueError(f"{FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION} is counterfactual-only")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        source = self.dataset[index]
        if not isinstance(source, dict):
            raise TypeError(f"RootTaggedDataset expects dict samples, got {type(source).__name__}")
        item = dict(source)
        scene_name = str(item.get("scene_name", ""))
        if not scene_name:
            raise ValueError("RootTaggedDataset requires every sample to have a non-empty scene_name")
        if self.domain == "counterfactual":
            match = _COUNTERFACTUAL_SCENE_RE.fullmatch(scene_name)
            if match is None:
                raise ValueError(
                    "counterfactual base_scene_id requires scene_name matching "
                    f"'<base>_cf_<6-digit-start>_<6-digit-end>', got {scene_name!r}"
                )
            base_scene_id = match.group("base")
            window_start_pos = int(match.group("start"))
        else:
            base_scene_id = scene_name
            window_start_pos = int(item.get("window_start_pos", index))
        sample_token = item.get("sample_token")
        if sample_token is None:
            stable_sample_id = collate.make_stable_sample_id(
                "navsim",
                self.dataset_root_name,
                scene_name,
                window_start_pos,
            )
        else:
            if not isinstance(sample_token, str) or not sample_token:
                raise ValueError(f"NavSim sample_token must be a non-empty string, got {sample_token!r}")
            stable_sample_id = collate.make_stable_sample_id(
                "navsim",
                self.dataset_root_name,
                "token",
                sample_token,
            )

        annotation_valid = bool(item.get("cf_annotation_valid", False))
        is_hazard = bool(item.get("cf_is_hazard", False))
        hazard_type = str(item.get("cf_hazard_type", ""))
        if self.domain == "real" and (annotation_valid or is_hazard or hazard_type):
            raise ValueError("real samples cannot carry counterfactual hazard annotations")
        if self.domain == "counterfactual" and not annotation_valid:
            raise ValueError("counterfactual samples require cf_annotation_valid=true")
        if self.annotation_selection == "safe_only" and is_hazard:
            raise ValueError("safe_only counterfactual root leaked a hazard sample at runtime")
        if self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION and (
            not is_hazard or hazard_type not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
        ):
            raise ValueError(
                "Formal-v2 counterfactual sample must remain a hazard with accident_type in the exact allowlist"
            )
        sample_future_geometry_valid = bool(item.get("future_agent_geometry_valid", False))
        if self.domain == "counterfactual" and sample_future_geometry_valid:
            raise ValueError("counterfactual samples must never set future_agent_geometry_valid=true")

        quality_present = bool(item.get("cf_quality_present", False))
        quality_field_names = (
            "cf_quality",
            "cf_progress_score",
            "cf_reverse_risk",
            "cf_comfort_score",
            "cf_path_efficiency",
        )
        quality_values = {field_name: float(item.get(field_name, float("nan"))) for field_name in quality_field_names}
        if self.domain == "real":
            if quality_present or any(np.isfinite(value) for value in quality_values.values()):
                raise ValueError("real samples cannot carry counterfactual trajectory quality")
            quality_present = False
        elif quality_present:
            invalid_quality = [
                field_name
                for field_name, value in quality_values.items()
                if not np.isfinite(value) or not 0.0 <= value <= 1.0
            ]
            if invalid_quality:
                raise ValueError(
                    "counterfactual trajectory quality fields must be finite in [0, 1]; " f"invalid={invalid_quality}"
                )
            if item.get("cf_quality_schema") != CF_QUALITY_SCHEMA:
                raise ValueError(f"counterfactual trajectory quality requires schema={CF_QUALITY_SCHEMA}")
            if item.get("cf_quality_source") != "trajectory_quality_sidecar":
                raise ValueError("counterfactual trajectory quality requires trajectory_quality_sidecar source")
        elif any(np.isfinite(value) for value in quality_values.values()):
            raise ValueError("cf_quality_present=false cannot carry finite counterfactual quality fields")

        raw_agent_count = item.get("raw_agent_count")
        if self.geometry_present:
            raw_agent_count = np.asarray(raw_agent_count)
            if raw_agent_count.ndim != 1 or not np.issubdtype(raw_agent_count.dtype, np.integer):
                raise ValueError("real NavSim samples require integer raw_agent_count metadata with shape [T]")
            if np.any(raw_agent_count < 0):
                raise ValueError(f"raw_agent_count must be non-negative, got {raw_agent_count.tolist()}")
            if "agent_boxes" not in item or raw_agent_count.shape[0] != np.asarray(item["agent_boxes"]).shape[0]:
                raise ValueError("raw_agent_count length must match the agent_boxes time dimension")
            raw_agent_count = raw_agent_count.astype(np.int64, copy=True)
        else:
            raw_agent_count = None
            if not self.load_agent_annotations:
                transport_fields = ("agent_boxes", "agent_mask", "bev_segmentation")
                missing_transport = [field_name for field_name in transport_fields if field_name not in item]
                if missing_transport:
                    raise ValueError(
                        "load_agent_annotations=false requires zero geometry transport tensors; "
                        f"missing {missing_transport}"
                    )
                nonzero_transport = [
                    field_name for field_name in transport_fields if np.asarray(item[field_name]).any()
                ]
                if nonzero_transport:
                    raise ValueError(
                        "load_agent_annotations=false requires zero geometry transport tensors; "
                        f"nonzero {nonzero_transport}"
                    )

        item.update(
            {
                "dataset_domain": self.domain,
                "dataset_root_name": self.dataset_root_name,
                "dataset_root_index": self.dataset_root_index,
                "base_scene_id": base_scene_id,
                "window_start_pos": window_start_pos,
                "future_agent_geometry_valid": self.future_agent_geometry_valid and sample_future_geometry_valid,
                "geometry_present": self.geometry_present,
                "geometry_source": REAL_AGENT_GEOMETRY_SOURCE if self.geometry_present else None,
                "geometry_coordinate_frame": REAL_AGENT_COORDINATE_FRAME if self.geometry_present else None,
                "coordinate_frame": REAL_AGENT_COORDINATE_FRAME if self.geometry_present else None,
                "agent_geometry_truncated": False if self.geometry_present else None,
                "raw_agent_count": raw_agent_count,
                "stable_sample_id": stable_sample_id,
                "sample_id": stable_sample_id,
                "cf_annotation_valid": annotation_valid,
                "cf_is_hazard": is_hazard,
                "cf_hazard_type": hazard_type,
                "cf_quality_present": quality_present,
                **quality_values,
                "cf_quality_schema": CF_QUALITY_SCHEMA if quality_present else None,
                "cf_quality_source": "trajectory_quality_sidecar" if quality_present else None,
            }
        )
        return item

    def cvoi_pair_key(self, index: int) -> tuple[str, int]:
        """Return the factual-pair identity without decoding the sample."""

        provider = getattr(self.dataset, "cvoi_pair_key", None)
        if not callable(provider):
            raise ValueError(
                f"NavSim root {self.dataset_root_name!r} does not expose cvoi_pair_key; "
                "atomic Formal-v2 real/CF pairing cannot be verified"
            )
        raw_key = provider(index)
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ValueError(f"NavSim root {self.dataset_root_name!r} returned invalid CVoI pair key {raw_key!r}")
        base_scene_id, window_start_pos = raw_key
        if type(base_scene_id) is not str or not base_scene_id or type(window_start_pos) is not int:
            raise ValueError(f"NavSim root {self.dataset_root_name!r} returned invalid CVoI pair key {raw_key!r}")
        return base_scene_id, window_start_pos

    def cvoi_hazard_label(self, index: int) -> tuple[bool, str]:
        """Return the indexed CF hazard label without decoding video frames."""

        provider = getattr(self.dataset, "cvoi_hazard_label", None)
        if not callable(provider):
            raise ValueError(
                f"NavSim root {self.dataset_root_name!r} does not expose cvoi_hazard_label; "
                "Formal-v2 CF cohort cannot be verified"
            )
        raw_label = provider(index)
        if not isinstance(raw_label, tuple) or len(raw_label) != 2:
            raise ValueError(f"NavSim root {self.dataset_root_name!r} returned invalid CVoI hazard label")
        is_hazard, hazard_type = raw_label
        if type(is_hazard) is not bool or type(hazard_type) is not str:
            raise ValueError(f"NavSim root {self.dataset_root_name!r} returned invalid CVoI hazard label")
        return is_hazard, hazard_type


@dataclass(frozen=True)
class MatchedRealCounterfactualPair:
    """One atomic factual/counterfactual unit assigned to exactly one rank."""

    pair_key: tuple[str, int]
    real: Dict[str, Any]
    counterfactual: Dict[str, Any]


class MatchedRealCounterfactualPairDataset(Dataset):
    """Pair every retained Formal-v2 CF hazard with its unique factual window.

    The dataset index is a *pair* index, so an ordinary ``DistributedSampler``
    may shuffle and shard it without ever assigning the two rows to different
    ranks. The collate function expands pair units only after rank-local batch
    sampling.
    """

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        roots = list(datasets)
        if len(roots) != 2 or any(not isinstance(root, RootTaggedDataset) for root in roots):
            raise ValueError("Formal-v2 matched pairing requires exactly two RootTaggedDataset roots")
        by_domain = {root.domain: root for root in roots}
        if set(by_domain) != {"real", "counterfactual"}:
            raise ValueError("Formal-v2 matched pairing requires exactly one real and one counterfactual root")
        self.real_dataset = by_domain["real"]
        self.counterfactual_dataset = by_domain["counterfactual"]
        if self.counterfactual_dataset.annotation_selection != FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
            raise ValueError(
                "Formal-v2 matched pairing requires counterfactual annotation_selection="
                f"{FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}"
            )

        factual_indices: dict[tuple[str, int], int] = {}
        for index in range(len(self.real_dataset)):
            key = self.real_dataset.cvoi_pair_key(index)
            if key in factual_indices:
                raise ValueError(f"duplicate factual CVoI pair key {key!r}")
            factual_indices[key] = index

        seen_counterfactual: set[tuple[str, int]] = set()
        pairs: list[tuple[tuple[str, int], int, int]] = []
        for counterfactual_index in range(len(self.counterfactual_dataset)):
            key = self.counterfactual_dataset.cvoi_pair_key(counterfactual_index)
            if key in seen_counterfactual:
                raise ValueError(f"duplicate counterfactual CVoI pair key {key!r}")
            seen_counterfactual.add(key)
            factual_index = factual_indices.get(key)
            if factual_index is None:
                raise ValueError(f"missing factual CVoI pair for key {key!r}")
            is_hazard, hazard_type = self.counterfactual_dataset.cvoi_hazard_label(counterfactual_index)
            if not is_hazard or hazard_type not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST:
                raise ValueError(
                    f"counterfactual CVoI pair {key!r} must remain a hazard in the exact accident_type allowlist"
                )
            pairs.append((key, factual_index, counterfactual_index))
        if not pairs:
            raise ValueError("Formal-v2 matched pairing requires at least one retained counterfactual sample")
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> MatchedRealCounterfactualPair:
        key, factual_index, counterfactual_index = self.pairs[index]
        real = self.real_dataset[factual_index]
        counterfactual = self.counterfactual_dataset[counterfactual_index]
        for name, sample, expected_domain in (
            ("factual", real, "real"),
            ("counterfactual", counterfactual, "counterfactual"),
        ):
            actual_key = (sample.get("base_scene_id"), sample.get("window_start_pos"))
            if sample.get("dataset_domain") != expected_domain or actual_key != key:
                raise ValueError(
                    f"{name} CVoI pair identity drift: expected domain/key={expected_domain!r}/{key!r}, "
                    f"got {sample.get('dataset_domain')!r}/{actual_key!r}"
                )
        if counterfactual.get("cf_is_hazard") is not True:
            raise ValueError(f"counterfactual CVoI pair {key!r} must remain a hazard")
        hazard_type = counterfactual.get("cf_hazard_type")
        if hazard_type not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST:
            raise ValueError(
                f"counterfactual CVoI pair {key!r} hazard type must be in the exact accident_type allowlist"
            )
        return MatchedRealCounterfactualPair(pair_key=key, real=real, counterfactual=counterfactual)


def load_navsim_scene_filter(path: str) -> Tuple[set, List[str]]:
    """Parse an official NavSim scene-filter yaml (navtrain/navtest).

    Returns ``(log_names, tokens)`` where ``log_names`` is the set of log
    stems and ``tokens`` is the ordered list of current-frame tokens defining
    the official samples. Raises on missing file, missing/empty sections, or
    duplicate tokens.
    """
    contract = load_navsim_scene_filter_contract(path)
    return set(contract.log_names), list(contract.tokens)


def load_counterfactual_annotations(
    path: str,
    camera_name: str,
    *,
    require_trajectory_match: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load counterfactual clip annotations (train_annos.json / val_annos.json).

    Legacy annotation scenes such as ``scene-0001_CAM_F0_000010_000019_gen``
    are re-mapped to the pseudo-pkl stem ``scene-0001_cf_000010_000019``.
    Converted annotation files may already use that pseudo-pkl stem; both forms
    normalize to the same lookup key used by ``SceneRecord.scene_name``.

    标注方案（生成事故视频打标）的字段语义：
    - ``distortion``   严重失真 → 其余字段为 null，整条不可用；
    - ``trajectory_match``  轨迹图是否与视频匹配；true 可用于 planner 训练；
    - ``accident``     bool；注意"自车行为引起"包含未碰撞的异常驾驶（逆行/闯红灯等），
                       语义是 hazard 而非严格 collision；
    - ``accident_type``  自车行为引起 / 非自车行为引起 / 有事故但与自车无关 / 正常；
    - ``reverse``/``static``/``run_red_light``  自车行为标签（倒车/全程静止/闯红灯），
                       不是生成质量标签，不得当作过滤依据。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Counterfactual annotations json does not exist: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Counterfactual annotations must be a non-empty list: {path}")
    camera_token = f"_{camera_name}_"
    raw_scene_pattern = re.compile(rf"^(?P<base>.+)_{re.escape(camera_name)}_(?P<start>\d{{6}})_(?P<end>\d{{6}})_gen$")
    annotations: Dict[str, Dict[str, Any]] = {}
    for entry in payload:
        scene = entry.get("scene")
        annos = entry.get("annos")
        if not isinstance(scene, str) or not isinstance(annos, dict):
            raise ValueError(f"Annotation entry must have str 'scene' and dict 'annos', got {entry!r}: {path}")
        raw_match = raw_scene_pattern.fullmatch(scene)
        if raw_match is not None:
            stem = f"{raw_match.group('base')}_cf_{raw_match.group('start')}_{raw_match.group('end')}"
        elif _COUNTERFACTUAL_SCENE_RE.fullmatch(scene) is not None:
            stem = scene
        else:
            raise ValueError(
                f"Annotation scene {scene!r} does not match the expected "
                f"'<scene>{camera_token}<start>_<end>_gen' or '<scene>_cf_<start>_<end>' pattern: {path}"
            )
        if stem in annotations:
            raise ValueError(f"Duplicate annotation scene {scene!r} (stem {stem!r}): {path}")
        distortion = annos.get("distortion")
        if not isinstance(distortion, bool):
            raise ValueError(f"Annotation {scene!r} has non-bool distortion={distortion!r}: {path}")
        if require_trajectory_match and "trajectory_match" not in annos:
            raise ValueError(f"Annotation {scene!r} is missing required trajectory_match: {path}")
        trajectory_match = annos.get("trajectory_match")
        if require_trajectory_match and trajectory_match is not None and not isinstance(trajectory_match, bool):
            raise ValueError(f"Annotation {scene!r} has non-bool trajectory_match={trajectory_match!r}: {path}")
        accident = annos.get("accident")
        accident_type = annos.get("accident_type")
        if distortion:
            # 导出约定：失真条目其余字段为空串，内容无意义（标注方案里失真条目的
            # 事故栏是占位值），只允许整体剔除，不允许读取。
            accident = False
            accident_type = ""
        else:
            if not isinstance(accident, bool):
                raise ValueError(
                    f"Annotation {scene!r} is not distorted but has non-bool accident={accident!r}; "
                    f"tri-state '' is only valid on distorted entries: {path}"
                )
            if accident and (not isinstance(accident_type, str) or not accident_type):
                raise ValueError(f"Annotation {scene!r} has accident=true but empty accident_type: {path}")
            accident_type = str(accident_type)
            if accident and accident_type not in KNOWN_HAZARD_TYPES:
                raise ValueError(
                    f"Annotation {scene!r} has unknown hazard type {accident_type!r}; "
                    f"known types are {sorted(KNOWN_HAZARD_TYPES)}: {path}"
                )
        annotations[stem] = {
            "scene": scene,
            "distortion": bool(distortion),
            "accident": bool(accident),
            "accident_type": accident_type,
            "trajectory_match": trajectory_match,
        }
    return annotations


_CF_QUALITY_COMPONENT_FIELDS = (
    "progress_score",
    "reverse_risk",
    "comfort_score",
    "path_efficiency",
)
_CF_QUALITY_FORBIDDEN_KEY_PARTS = (
    "agent_box",
    "collision",
    "future_agent",
    "off_road",
    "offroad",
    "safety",
    "ttc",
)


def _reject_cf_quality_forbidden_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if any(token in normalized_key for token in _CF_QUALITY_FORBIDDEN_KEY_PARTS):
                raise ValueError(f"counterfactual trajectory quality contains forbidden field {path}.{key}")
            _reject_cf_quality_forbidden_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_cf_quality_forbidden_fields(nested, path=f"{path}[{index}]")


def _unit_interval_quality_value(value: Any, *, field_name: str, scene_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"trajectory quality scene {scene_name!r} field {field_name!r} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"trajectory quality scene {scene_name!r} field {field_name!r} must be finite in [0, 1]")
    return normalized


def load_counterfactual_trajectory_quality(
    path: str,
    *,
    expected_pose_overlay_txt_start_seconds: float | None = None,
    require_formal_v2_contract: bool = False,
    pose_overlay_reader: PoseOverlayReader | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load canonical geometry-free quality components keyed by CF scene.

    The aggregate score is derived from the recorded components using the
    immutable CVoI quality weights. A sidecar-provided aggregate is accepted
    only as a consistency check; it is never trusted as an independent label.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Counterfactual trajectory quality json does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pose_sha256_by_scene = validate_counterfactual_quality_sidecar_metadata(
        payload,
        source=path,
        expected_pose_overlay_txt_start_seconds=expected_pose_overlay_txt_start_seconds,
        require_formal_v2_contract=require_formal_v2_contract,
    )
    if pose_overlay_reader is not None and pose_sha256_by_scene:
        pose_overlay_reader.bind_expected_scene_sha256(pose_sha256_by_scene)
    scenes = payload.get("scenes")
    if not isinstance(scenes, Mapping) or not scenes:
        raise ValueError(f"Counterfactual trajectory quality requires a non-empty scenes mapping: {path}")
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_scene_name, raw_entry in scenes.items():
        scene_name = str(raw_scene_name)
        if _COUNTERFACTUAL_SCENE_RE.fullmatch(scene_name) is None:
            raise ValueError(f"trajectory quality scene name is not a counterfactual scene: {scene_name!r}")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"trajectory quality scene {scene_name!r} must be a mapping")
        _reject_cf_quality_forbidden_fields(raw_entry, path=f"scenes.{scene_name}")
        status = raw_entry.get("status")
        if status is not None and status != "passed":
            raise ValueError(f"trajectory quality scene {scene_name!r} must have status='passed', got {status!r}")
        raw_metrics = raw_entry.get("metrics", raw_entry)
        if not isinstance(raw_metrics, Mapping):
            raise ValueError(f"trajectory quality scene {scene_name!r} metrics must be a mapping")
        components = {
            name: _unit_interval_quality_value(raw_metrics.get(name), field_name=name, scene_name=scene_name)
            for name in _CF_QUALITY_COMPONENT_FIELDS
        }
        quality_score = (
            _CF_QUALITY_WEIGHTS["progress"] * components["progress_score"]
            + _CF_QUALITY_WEIGHTS["non_reverse"] * (1.0 - components["reverse_risk"])
            + _CF_QUALITY_WEIGHTS["comfort"] * components["comfort_score"]
            + _CF_QUALITY_WEIGHTS["path_efficiency"] * components["path_efficiency"]
        )
        declared_score = raw_entry.get("quality_score", raw_metrics.get("quality_score"))
        if declared_score is not None:
            declared_score = _unit_interval_quality_value(
                declared_score,
                field_name="quality_score",
                scene_name=scene_name,
            )
            if not np.isclose(declared_score, quality_score, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"trajectory quality scene {scene_name!r} quality_score does not match canonical components"
                )
        normalized[scene_name] = {
            "cf_quality_present": True,
            "cf_quality": quality_score,
            "cf_progress_score": components["progress_score"],
            "cf_reverse_risk": components["reverse_risk"],
            "cf_comfort_score": components["comfort_score"],
            "cf_path_efficiency": components["path_efficiency"],
            "cf_quality_schema": CF_QUALITY_SCHEMA,
            "cf_quality_source": "trajectory_quality_sidecar",
        }
    return normalized


class NavSimWorldModelDataset(Dataset):
    """Dataset that adapts NavSim logs to the world-model batch contract."""

    # 碰撞检测需要过滤的动态 agent 类型
    _DYNAMIC_AGENT_TYPES = {"vehicle", "pedestrian", "bicycle"}

    def __init__(
        self,
        data_path: str,
        sensor_blobs_path: str,
        camera_name: str = "CAM_F0",
        camera_names: Optional[Sequence[str]] = None,
        frames_per_clip: int = 20,
        fps: int = 2,
        base_fps: float = 2.0,
        tubelet_size: int = 2,
        transform: Any = None,
        proposal_transform: Any = None,
        max_scenes: Optional[int] = None,
        action_dim: int = 3,
        cache_size: int = 8,
        index_cache: bool = True,
        window_stride: int = 1,
        max_frame_gap: int = 3,
        max_agents: int = NAVSIM_DEFAULT_MAX_AGENTS,
        load_agent_annotations: bool = True,
        image_require_policy: str = IMAGE_REQUIRE_ALL_FRAMES,
        num_observed_frames: Optional[int] = None,
        scene_filter_yaml: Optional[str] = None,
        pose_overlay_path: Optional[str] = None,
        pose_overlay_coord_frame: str = "opencv_first_frame",
        pose_overlay_txt_start_seconds: float = 1.5,
        pose_overlay_required: bool = False,
        tail_seconds: Optional[float] = None,
        window_start_policy: str = WINDOW_START_SLIDING,
        timestamp_policy: Optional[str] = None,
        annotations_path: Optional[str] = None,
        annotations_drop_distorted: Optional[bool] = None,
        annotations_require_trajectory_match: bool = False,
        annotations_accident_type_allowlist: Optional[Sequence[str]] = None,
        trajectory_quality_path: Optional[str] = None,
        annotation_selection: str = "all_valid",
        is_validation: bool = False,
    ):
        self.data_path = data_path
        self.sensor_blobs_path = sensor_blobs_path
        self.camera_names = list(camera_names) if camera_names is not None else [camera_name]
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera")
        self.camera_name = self.camera_names[0] if camera_names is not None else camera_name
        self.multiview_enabled = len(self.camera_names) > 1
        self.frames_per_clip = int(frames_per_clip)
        self.fps = int(max(1, fps))
        if isinstance(base_fps, bool) or not isinstance(base_fps, (int, float)) or not np.isfinite(base_fps):
            raise ValueError(f"base_fps must be a finite positive number, got {base_fps!r}")
        self.base_fps = float(base_fps)
        if self.base_fps <= 0.0:
            raise ValueError(f"base_fps must be a finite positive number, got {base_fps!r}")
        self.tubelet_size = int(max(1, tubelet_size))
        self.transform = transform
        self.proposal_transform = proposal_transform
        self.is_validation = bool(is_validation)
        self.max_scenes = max_scenes
        self.action_dim = action_dim
        self.cache_size = max(1, int(cache_size))
        self.index_cache = index_cache
        self.window_stride = max(1, int(window_stride))
        self.max_frame_gap = max(1, int(max_frame_gap))
        self.max_agents = require_positive_navsim_max_agents(max_agents)
        self.load_agent_annotations = bool(load_agent_annotations)
        self.image_require_policy = str(image_require_policy).lower()
        if self.image_require_policy not in VALID_IMAGE_REQUIRE_POLICIES:
            raise ValueError(
                f"image_require_policy must be one of {sorted(VALID_IMAGE_REQUIRE_POLICIES)}, "
                f"got {image_require_policy!r}"
            )
        self.num_observed_frames = (
            int(num_observed_frames) if num_observed_frames is not None else self.frames_per_clip
        )
        if self.num_observed_frames < 1 or self.num_observed_frames > self.frames_per_clip:
            raise ValueError(
                "num_observed_frames must be in [1, frames_per_clip], "
                f"got {self.num_observed_frames} for frames_per_clip={self.frames_per_clip}"
            )
        self.required_image_frames = (
            self.frames_per_clip if self.image_require_policy == IMAGE_REQUIRE_ALL_FRAMES else self.num_observed_frames
        )

        # Official token anchoring: windows are anchored on the scene-filter
        # token list instead of the stride grid. The yaml path doubles as the
        # mode switch.
        self.scene_filter_yaml = scene_filter_yaml
        self.token_mode = scene_filter_yaml is not None
        if pose_overlay_required and not pose_overlay_path:
            raise ValueError("pose_overlay_required=true requires a non-empty pose_overlay_path")
        self.pose_overlay_reader = (
            PoseOverlayReader(
                pose_overlay_path,
                coord_frame=pose_overlay_coord_frame,
                required=bool(pose_overlay_required),
                txt_start_seconds=pose_overlay_txt_start_seconds,
            )
            if pose_overlay_path
            else None
        )
        if self.token_mode:
            if self.max_scenes is not None:
                raise ValueError(
                    "max_scenes cannot be combined with scene_filter_yaml; "
                    "official token anchoring requires full log coverage "
                    "(use a truncated scene-filter yaml for smoke runs)"
                )
            self._filter_log_names, self._filter_tokens = load_navsim_scene_filter(scene_filter_yaml)
            filter_fingerprint = json.dumps(
                {"log_names": sorted(self._filter_log_names), "tokens": self._filter_tokens},
                sort_keys=True,
            )
            self._scene_filter_hash = hashlib.sha256(filter_fingerprint.encode()).hexdigest()[:16]
            logger.info(
                "NavSim official token anchoring enabled: yaml=%s, log_names=%d, tokens=%d "
                "(window_stride is ignored in this mode)",
                scene_filter_yaml,
                len(self._filter_log_names),
                len(self._filter_tokens),
            )
        else:
            self._filter_log_names = None
            self._filter_tokens = None
            self._scene_filter_hash = None

        self.timestamp_policy_was_explicit = timestamp_policy is not None
        self.timestamp_policy = (
            str(timestamp_policy)
            if timestamp_policy is not None
            else (TIMESTAMP_POLICY_ELIGIBLE_WINDOW_BOUNDARY if self.token_mode else TIMESTAMP_POLICY_ROOT_CONTIGUOUS)
        )
        if self.timestamp_policy not in VALID_TIMESTAMP_POLICIES:
            raise ValueError(
                f"timestamp_policy must be one of {sorted(VALID_TIMESTAMP_POLICIES)}, " f"got {timestamp_policy!r}"
            )
        if self.token_mode != (self.timestamp_policy == TIMESTAMP_POLICY_ELIGIBLE_WINDOW_BOUNDARY):
            raise ValueError(
                "eligible_window_boundary_v1 requires scene_filter_yaml and token anchoring; "
                "root_contiguous_v1 forbids scene_filter_yaml"
            )

        self.sample_step = max(1, int(round(self.base_fps / float(self.fps))))
        self.min_valid_frames = 1 + (self.frames_per_clip - 1) * self.sample_step
        self.tail_seconds = None if tail_seconds is None else float(tail_seconds)
        if self.tail_seconds is not None and self.tail_seconds <= 0.0:
            raise ValueError(f"tail_seconds must be positive when set, got {tail_seconds}")
        self.window_start_policy = str(window_start_policy)
        if self.window_start_policy not in VALID_WINDOW_START_POLICIES:
            raise ValueError(
                f"window_start_policy must be one of {sorted(VALID_WINDOW_START_POLICIES)}, "
                f"got {window_start_policy!r}"
            )
        if self.window_start_policy == WINDOW_START_COUNTERFACTUAL_SCENE_START:
            if self.token_mode:
                raise ValueError("counterfactual_scene_start cannot be combined with scene_filter_yaml")
            if self.tail_seconds is not None:
                raise ValueError("counterfactual_scene_start requires tail_seconds=None")

        # Counterfactual clip annotations (sample-level accident/distortion labels).
        self.annotation_selection = str(annotation_selection)
        if self.annotation_selection not in {"all_valid", "safe_only", FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION}:
            raise ValueError(
                "annotation_selection must be 'all_valid', 'safe_only', or "
                f"{FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}, got {annotation_selection!r}"
            )
        self.annotations_accident_type_allowlist = annotations_accident_type_allowlist
        if self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
            expected_allowlist = list(FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST)
            if (
                type(self.annotations_accident_type_allowlist) is not list
                or self.annotations_accident_type_allowlist != expected_allowlist
            ):
                raise ValueError(
                    "annotations_accident_type_allowlist must be exactly "
                    f"{expected_allowlist!r} for {FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION}"
                )
            if (
                annotations_path is None
                or annotations_drop_distorted is not True
                or annotations_require_trajectory_match is not True
            ):
                raise ValueError(
                    f"{FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION} requires annotations_path, "
                    "annotations_drop_distorted=true, and annotations_require_trajectory_match=true"
                )
        elif self.annotations_accident_type_allowlist is not None:
            raise ValueError(
                "annotations_accident_type_allowlist is valid only with annotation_selection="
                f"{FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}"
            )
        self.annotations_path = annotations_path or None
        if not isinstance(annotations_require_trajectory_match, bool):
            raise ValueError(
                "annotations_require_trajectory_match must be boolean, "
                f"got {annotations_require_trajectory_match!r}"
            )
        self.annotations_require_trajectory_match = annotations_require_trajectory_match
        if self.annotations_require_trajectory_match and self.annotations_path is None:
            raise ValueError("annotations_require_trajectory_match=true requires annotations_path")
        if self.annotation_selection == "safe_only" and (
            self.annotations_path is None or annotations_drop_distorted is not True
        ):
            raise ValueError(
                "safe_only annotation_selection requires annotations_path and " "annotations_drop_distorted=true"
            )
        if self.annotations_path is not None:
            if annotations_drop_distorted is None:
                raise ValueError(
                    "annotations_path is set but annotations_drop_distorted is not; "
                    "set it explicitly (true/false) — no implicit default for a data-dropping switch"
                )
            if self.token_mode:
                raise ValueError(
                    "annotations_path cannot be combined with scene_filter_yaml (official token anchoring): "
                    "annotation-based scene dropping would violate full log coverage"
                )
            self.annotations_drop_distorted = bool(annotations_drop_distorted)
            self._scene_annotations: Optional[Dict[str, Dict[str, Any]]] = load_counterfactual_annotations(
                self.annotations_path,
                camera_name=self.camera_name,
                require_trajectory_match=self.annotations_require_trajectory_match,
            )
        else:
            if annotations_drop_distorted is not None:
                raise ValueError(
                    "annotations_drop_distorted is set but annotations_path is not; "
                    "did you forget to configure annotations_path for this root?"
                )
            self.annotations_drop_distorted = False
            self._scene_annotations = None
        self.trajectory_quality_path = trajectory_quality_path or None
        if self.trajectory_quality_path is not None:
            if self.annotations_path is None:
                raise ValueError(
                    "trajectory_quality_path requires annotations_path for counterfactual scene alignment"
                )
            if self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
                expected_timeline = formal_v2_cf_quality_timeline_contract()
                runtime_timeline = {
                    "num_total_frames": self.frames_per_clip,
                    "num_observed_frames": self.num_observed_frames,
                    "num_target_frames": self.frames_per_clip - self.num_observed_frames,
                    "window_start_policy": self.window_start_policy,
                    "sample_step": self.sample_step,
                    "timestep_sec": 1.0 / self.base_fps,
                }
                expected_runtime = {
                    "num_total_frames": expected_timeline["num_total_frames"],
                    "num_observed_frames": expected_timeline["num_observed_frames"],
                    "num_target_frames": expected_timeline["num_target_frames"],
                    "window_start_policy": expected_timeline["window_start_policy"],
                    "sample_step": expected_timeline["sample_step"],
                    "timestep_sec": FORMAL_V2_CF_QUALITY_TIMESTEP_SEC,
                }
                if runtime_timeline != expected_runtime:
                    actual_fields = ", ".join(f"{name}={value}" for name, value in runtime_timeline.items())
                    raise ValueError(
                        "Formal-v2 CF quality dataset timeline must be exactly 12 total / 4 observed / "
                        f"8 target frames from scene start: {actual_fields}"
                    )
                if self.pose_overlay_reader is None:
                    raise ValueError("Formal-v2 CF quality requires pose_overlay_path for live SHA256 verification")
                if self.pose_overlay_reader.txt_start_seconds != 0.0:
                    raise ValueError("Formal-v2 CF quality pose overlay timeline must start at exactly 0.0 seconds")
            self._scene_trajectory_quality: Optional[Dict[str, Dict[str, Any]]] = (
                load_counterfactual_trajectory_quality(
                    self.trajectory_quality_path,
                    expected_pose_overlay_txt_start_seconds=pose_overlay_txt_start_seconds,
                    require_formal_v2_contract=(self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION),
                    pose_overlay_reader=self.pose_overlay_reader,
                )
            )
        else:
            self._scene_trajectory_quality = None

        # Real-time continuity threshold for GT trajectory frames. Adjacent raw
        # frames nominally sit 1/base_fps apart; a recording gap (missing
        # timesteps) shows up as a larger inter-frame timestamp delta and would
        # corrupt the GT trajectory. A window is rejected if any consecutive
        # pair in its raw span exceeds `max_frame_gap` nominal steps (+ half a
        # step of jitter tolerance). With max_frame_gap=1 @ 2Hz this is 0.75s,
        # so normal 0.5s steps pass and 1.0s+ gaps (>=1 missing timestep) fail.
        self.nominal_step_us = 1_000_000.0 / float(self.base_fps)
        self.max_time_gap_us = (float(self.max_frame_gap) + 0.5) * self.nominal_step_us

        self._scene_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self.scenes = self._build_scene_index()

        if self._scene_annotations is not None:
            # 每个进索引的 scene 必须有标注（当前标注对 filtered 集是 100% 覆盖，缺失即数据错位）。
            # 过滤发生在缓存加载之后：磁盘索引缓存保存的是原始扫描结果，标注只做后处理，
            # 因此 cache fingerprint 无需包含标注文件。
            missing = [scene.scene_name for scene in self.scenes if scene.scene_name not in self._scene_annotations]
            if missing:
                raise ValueError(
                    f"{len(missing)} scenes have no entry in annotations_path={self.annotations_path} "
                    f"(examples: {missing[:5]}); annotation file and data_path are out of sync"
                )
            if self.annotations_drop_distorted:
                kept = [scene for scene in self.scenes if not self._scene_annotations[scene.scene_name]["distortion"]]
                dropped = len(self.scenes) - len(kept)
                logger.info(
                    "Counterfactual annotations: dropped %d distorted scenes, kept %d (annotations_path=%s)",
                    dropped,
                    len(kept),
                    self.annotations_path,
                )
                self.scenes = kept
            if self.annotations_require_trajectory_match:
                dropped_false = sum(
                    self._scene_annotations[scene.scene_name]["trajectory_match"] is False for scene in self.scenes
                )
                dropped_null = sum(
                    self._scene_annotations[scene.scene_name]["trajectory_match"] is None for scene in self.scenes
                )
                self.scenes = [
                    scene
                    for scene in self.scenes
                    if self._scene_annotations[scene.scene_name]["trajectory_match"] is True
                ]
                logger.info(
                    "Counterfactual annotations: dropped %d trajectory-mismatched and %d unlabeled scenes, "
                    "kept %d (annotations_path=%s)",
                    dropped_false,
                    dropped_null,
                    len(self.scenes),
                    self.annotations_path,
                )
            if self.annotation_selection == "safe_only":
                kept = [scene for scene in self.scenes if not self._scene_annotations[scene.scene_name]["accident"]]
                logger.info(
                    "Counterfactual annotations: safe_only dropped %d hazard scenes, kept %d " "(annotations_path=%s)",
                    len(self.scenes) - len(kept),
                    len(kept),
                    self.annotations_path,
                )
                self.scenes = kept
            elif self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
                kept = [
                    scene
                    for scene in self.scenes
                    if self._scene_annotations[scene.scene_name]["accident"] is True
                    and self._scene_annotations[scene.scene_name]["accident_type"]
                    in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
                ]
                logger.info(
                    "Counterfactual annotations: exact accident_type allowlist dropped %d scenes, kept %d "
                    "(annotations_path=%s)",
                    len(self.scenes) - len(kept),
                    len(kept),
                    self.annotations_path,
                )
                self.scenes = kept

                expected_scene_names = {
                    scene_name
                    for scene_name, annotation in self._scene_annotations.items()
                    if annotation["trajectory_match"] is True
                    and annotation["accident"] is True
                    and annotation["accident_type"] in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
                }
                actual_scene_names = {scene.scene_name for scene in self.scenes}
                if actual_scene_names != expected_scene_names:
                    missing_from_index = sorted(expected_scene_names - actual_scene_names)
                    unexpected_in_index = sorted(actual_scene_names - expected_scene_names)
                    raise ValueError(
                        "Formal-v2 CF annotation/index cohort mismatch after scene indexing and filtering: "
                        f"expected_count={len(expected_scene_names)}, actual_count={len(actual_scene_names)}, "
                        f"missing_from_index={missing_from_index[:10]}, "
                        f"unexpected_in_index={unexpected_in_index[:10]}. "
                        "Missing expected scenes indicate absent PKLs, camera/observed-frame files, or "
                        "insufficient scene length; these data losses cannot be silently accepted."
                    )

        if self._scene_trajectory_quality is not None:
            if self.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
                actual_scene_names = {scene.scene_name for scene in self.scenes}
                quality_scene_names = set(self._scene_trajectory_quality)
                if quality_scene_names != actual_scene_names:
                    raise ValueError(
                        "Formal-v2 CF trajectory quality scene set must exactly match the selected dataset cohort: "
                        f"dataset_count={len(actual_scene_names)}, quality_count={len(quality_scene_names)}, "
                        f"missing_quality={sorted(actual_scene_names - quality_scene_names)[:10]}, "
                        f"unexpected_quality={sorted(quality_scene_names - actual_scene_names)[:10]}"
                    )
            missing_quality = [
                scene.scene_name for scene in self.scenes if scene.scene_name not in self._scene_trajectory_quality
            ]
            if missing_quality:
                raise ValueError(
                    f"{len(missing_quality)} scenes have no entry in "
                    f"trajectory_quality_path={self.trajectory_quality_path} "
                    f"(examples: {missing_quality[:5]})"
                )

        if not self.scenes:
            raise ValueError(
                "No valid NavSim scenes found. "
                f"data_path={data_path}, sensor_blobs_path={sensor_blobs_path}, "
                f"cameras={self.camera_names}, required_frames={self.min_valid_frames}, "
                f"image_require_policy={self.image_require_policy}"
            )

        if self.token_mode:
            # Scenes can be silently dropped while building the index (missing
            # camera dirs / images / too short); official anchoring requires
            # every listed log to survive. Applies to cache hits as well.
            dropped_logs = sorted(self._filter_log_names - {scene.scene_name for scene in self.scenes})
            if dropped_logs:
                raise ValueError(
                    f"{len(dropped_logs)} scene-filter logs were dropped while building the scene index "
                    f"(missing camera dirs / images / too short). Examples: {dropped_logs[:10]}"
                )

        # Build window entries from scenes (official token anchoring or stride sliding).
        self.windows = self._build_window_index_from_tokens() if self.token_mode else self._build_window_index()

        if not self.windows:
            raise ValueError(
                "No valid sliding windows could be built from scenes. "
                f"scenes={len(self.scenes)}, min_valid_frames={self.min_valid_frames}, "
                f"window_stride={self.window_stride}, image_require_policy={self.image_require_policy}, "
                f"required_image_frames={self.required_image_frames}, tail_seconds={self.tail_seconds}"
            )

        if self.pose_overlay_reader is not None:
            window_scene_names = [self.scenes[window.scene_idx].scene_name for window in self.windows]
            preloaded_scenes = self.pose_overlay_reader.preload_scenes(window_scene_names)
            logger.info(
                "Preloaded NavSim pose overlays: scenes=%d, root=%s",
                preloaded_scenes,
                self.pose_overlay_reader.root,
            )

        logger.info(
            "NavSim dataset ready: scenes=%d, windows=%d, camera=%s, "
            "frames_per_clip=%d, sample_step=%d, window_stride=%d, "
            "image_require_policy=%s, required_image_frames=%d, tail_seconds=%s",
            len(self.scenes),
            len(self.windows),
            ",".join(self.camera_names),
            self.frames_per_clip,
            self.sample_step,
            self.window_stride,
            self.image_require_policy,
            self.required_image_frames,
            self.tail_seconds,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def cvoi_pair_key(self, index: int) -> tuple[str, int]:
        """Return ``(base_scene_id, source window_start_pos)`` without sample I/O."""

        if type(index) is not int or index < 0 or index >= len(self.windows):
            raise IndexError(index)
        window = self.windows[index]
        scene_name = self.scenes[window.scene_idx].scene_name
        match = _COUNTERFACTUAL_SCENE_RE.fullmatch(scene_name)
        if match is not None:
            return match.group("base"), int(match.group("start"))
        return scene_name, int(window.start_pos)

    def cvoi_hazard_label(self, index: int) -> tuple[bool, str]:
        """Return the normalized annotation label without loading scene tensors."""

        if type(index) is not int or index < 0 or index >= len(self.windows):
            raise IndexError(index)
        if self._scene_annotations is None:
            return False, ""
        window = self.windows[index]
        scene_name = self.scenes[window.scene_idx].scene_name
        annotation = self._scene_annotations[scene_name]
        return bool(annotation["accident"]), str(annotation["accident_type"] if annotation["accident"] else "")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        max_retries = 5
        window_idx = int(index)

        for retry in range(max_retries):
            window = self.windows[window_idx]
            scene = self.scenes[window.scene_idx]
            try:
                frames = self._load_scene_frames(scene.pkl_path)
                sampled_frame_indices = self._sampled_frame_indices(window.start_pos)
                image_frame_indices = self._required_image_frame_indices(sampled_frame_indices)
                reduced_frame_indices = sampled_frame_indices[:: self.tubelet_size]
                raw_metadata_valid_mask = self._build_metadata_valid_mask(frames, sampled_frame_indices)
                self._validate_counterfactual_scene_start_metadata(
                    window_start=window.start_pos,
                    metadata_valid_mask=raw_metadata_valid_mask,
                )

                states = actions = ego_dynamics = None
                if self.pose_overlay_reader is not None:
                    # Keep pose-overlay parsing out of the large image allocation window.
                    overlay = self.pose_overlay_reader.build_states_actions_ego_dynamics(
                        scene.scene_name,
                        reduced_frame_indices,
                        action_dim=self.action_dim,
                        dt=self._reduced_frame_dt(reduced_frame_indices),
                    )
                    states = overlay.states
                    actions = overlay.actions
                    ego_dynamics = overlay.ego_dynamics

                buffer, source_image_shapes = self._load_clip_images(scene, frames, image_frame_indices)
                raw_buffer = buffer
                if self.transform is not None:
                    buffer = self._apply_clip_transform(buffer, self.transform)
                else:
                    buffer = self._default_image_tensor(buffer)

                proposal_buffer = None
                if self.proposal_transform is not None:
                    proposal_buffer = self._apply_clip_transform(raw_buffer, self.proposal_transform)

                output_hw = self._infer_output_hw(buffer)
                camera_intrinsics, camera2ego = self._build_camera_metadata(
                    frames,
                    image_frame_indices,
                    source_image_shapes=source_image_shapes,
                    output_hw=output_hw,
                    transform=self.transform,
                )

                if states is None or actions is None or ego_dynamics is None:
                    states = self._build_states(frames, reduced_frame_indices)
                    actions = self._build_actions(states)
                    ego_dynamics = self._build_ego_dynamics(frames, reduced_frame_indices)
                driving_command = self._build_driving_commands(frames, reduced_frame_indices)
                metadata_valid_mask = self._build_metadata_valid_mask(frames, reduced_frame_indices)
                observed_metadata_valid = bool(raw_metadata_valid_mask[: self.num_observed_frames].all())
                num_observed_reduced_steps = (self.num_observed_frames + self.tubelet_size - 1) // self.tubelet_size
                future_metadata_valid_mask = metadata_valid_mask[num_observed_reduced_steps:]
                future_agent_geometry_valid = bool(
                    self.load_agent_annotations
                    and future_metadata_valid_mask.size > 0
                    and future_metadata_valid_mask.all()
                    and self._future_agent_annotations_are_valid(
                        frames,
                        reduced_frame_indices[num_observed_reduced_steps:],
                    )
                )

                # Keep a stable 7D shape for compatibility with existing world-model path.
                extrinsics = np.zeros_like(states, dtype=np.float32)

                # --- agent annotations (for collision detection) ---
                if self.load_agent_annotations:
                    agent_boxes, agent_mask, raw_agent_count = self._build_agent_annotations(
                        frames, reduced_frame_indices
                    )
                    bev_segmentation = self._build_bev_segmentation(agent_boxes, agent_mask)
                else:
                    T_reduced = len(reduced_frame_indices)
                    agent_boxes = np.zeros((T_reduced, self.max_agents, 7), dtype=np.float32)
                    agent_mask = np.zeros((T_reduced, self.max_agents), dtype=np.bool_)
                    bev_segmentation = np.zeros((T_reduced, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
                    raw_agent_count = None

                # Sample-level annotation facts (policy decisions live in value_planning):
                # roots without annotations_path are non-counterfactual data => not accident.
                if self._scene_annotations is not None:
                    scene_annotation = self._scene_annotations[scene.scene_name]
                    cf_annotation_valid = not bool(scene_annotation["distortion"])
                    is_accident = bool(scene_annotation["accident"])
                    accident_type = str(scene_annotation["accident_type"])
                else:
                    cf_annotation_valid = False
                    is_accident = False
                    accident_type = ""
                quality_fields = (
                    {} if self._scene_trajectory_quality is None else self._scene_trajectory_quality[scene.scene_name]
                )

                sample_token = None
                if self.token_mode:
                    anchor_frame_index = int(window.start_pos) + (self.num_observed_frames - 1) * self.sample_step
                    if scene.frame_tokens is None or anchor_frame_index >= len(scene.frame_tokens):
                        raise ValueError(
                            f"Token-anchored sample {scene.scene_name}:{window.start_pos} has no anchor token"
                        )
                    sample_token = scene.frame_tokens[anchor_frame_index]
                    if not isinstance(sample_token, str) or not sample_token:
                        raise ValueError(
                            f"Token-anchored sample {scene.scene_name}:{window.start_pos} has invalid "
                            f"sample_token={sample_token!r}"
                        )
                    stable_sample_id = collate.make_stable_sample_id(
                        "navsim",
                        "default",
                        "token",
                        sample_token,
                    )
                else:
                    stable_sample_id = collate.make_stable_sample_id(
                        "navsim",
                        "default",
                        scene.scene_name,
                        int(window.start_pos),
                    )

                return {
                    "buffer": buffer,
                    "proposal_buffer": proposal_buffer,
                    "actions": actions,
                    "states": states,
                    "extrinsics": extrinsics,
                    "indices": np.asarray(sampled_frame_indices, dtype=np.int64),
                    "is_accident": is_accident,
                    "accident_type": accident_type,
                    "cf_annotation_valid": cf_annotation_valid,
                    "cf_is_hazard": is_accident,
                    "cf_hazard_type": accident_type if is_accident else "",
                    **quality_fields,
                    "future_agent_geometry_valid": future_agent_geometry_valid,
                    "geometry_present": self.load_agent_annotations,
                    "geometry_source": REAL_AGENT_GEOMETRY_SOURCE if self.load_agent_annotations else None,
                    "geometry_coordinate_frame": (
                        REAL_AGENT_COORDINATE_FRAME if self.load_agent_annotations else None
                    ),
                    "coordinate_frame": REAL_AGENT_COORDINATE_FRAME if self.load_agent_annotations else None,
                    "agent_geometry_truncated": False if self.load_agent_annotations else None,
                    "scene_name": scene.scene_name,
                    "pkl_path": scene.pkl_path,
                    "window_start_pos": int(window.start_pos),
                    "stable_sample_id": stable_sample_id,
                    "sample_id": stable_sample_id,
                    **({"sample_token": sample_token} if sample_token is not None else {}),
                    "sampled_frame_indices": np.asarray(sampled_frame_indices, dtype=np.int64),
                    "image_frame_indices": np.asarray(image_frame_indices, dtype=np.int64),
                    "raw_metadata_valid_mask": raw_metadata_valid_mask,
                    "metadata_valid_mask": metadata_valid_mask,
                    "observed_metadata_valid": observed_metadata_valid,
                    "seg_masks": None,
                    "seg_frame_indices": None,
                    "driving_command": driving_command,
                    "ego_dynamics": ego_dynamics,
                    "agent_boxes": agent_boxes,
                    "agent_mask": agent_mask,
                    "raw_agent_count": raw_agent_count,
                    "bev_segmentation": bev_segmentation,
                    "camera_names": list(self.camera_names),
                    "camera_intrinsics": camera_intrinsics,
                    "camera2ego": camera2ego,
                }
            except CameraMetadataError:
                # point 11/12: 相机元数据缺失视为不可重试的致命错误，不换窗口，直接 fail-loud。
                raise
            except AgentGeometryOverflowError:
                # Capacity overflow is a dataset-contract violation. Replacing the
                # requested training window would silently change the cohort.
                raise
            except Exception as exc:
                if retry == max_retries - 1:
                    raise
                logger.warning(
                    "NavSim sample load failed (scene=%s, window_start=%d, retry=%d/%d): %s",
                    scene.scene_name,
                    window.start_pos,
                    retry + 1,
                    max_retries,
                    str(exc),
                )
                if not self.is_validation:
                    window_idx = random.randint(0, len(self.windows) - 1)

        raise RuntimeError("Unreachable NavSim dataset retry state")

    def _build_scene_index(self) -> List[SceneRecord]:
        if not os.path.isdir(self.data_path):
            raise ValueError(f"NavSim data_path does not exist: {self.data_path}")
        if not os.path.isdir(self.sensor_blobs_path):
            raise ValueError(f"NavSim sensor_blobs_path does not exist: {self.sensor_blobs_path}")

        pkl_paths = sorted(glob(os.path.join(self.data_path, "*.pkl")))
        if self.token_mode:
            paths_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in pkl_paths}
            missing_logs = sorted(self._filter_log_names - set(paths_by_stem))
            if missing_logs:
                raise ValueError(
                    f"{len(missing_logs)} scene-filter logs have no pkl under {self.data_path} "
                    f"(examples: {missing_logs[:10]})"
                )
            pkl_paths = sorted(paths_by_stem[name] for name in self._filter_log_names)
        if self.max_scenes is not None:
            pkl_paths = pkl_paths[: int(self.max_scenes)]

        # --- try loading from disk cache ---
        cache_path = self._get_index_cache_path(len(pkl_paths))
        if cache_path is not None:
            cached = self._load_index_cache(cache_path, len(pkl_paths))
            if cached is not None:
                return cached

        # --- build index from scratch (with progress logging) ---
        t0 = time.monotonic()
        total = len(pkl_paths)
        logger.info("Building NavSim scene index from %d pkl files ...", total)

        scenes: List[SceneRecord] = []
        skipped_missing_camera_dir = 0
        skipped_not_enough_frames = 0
        skipped_not_enough_required_images = 0

        for i, pkl_path in enumerate(pkl_paths):
            scene_name = os.path.splitext(os.path.basename(pkl_path))[0]
            camera_dirs = {
                camera_name: os.path.join(self.sensor_blobs_path, scene_name, camera_name)
                for camera_name in self.camera_names
            }
            if any(not os.path.isdir(camera_dir) for camera_dir in camera_dirs.values()):
                skipped_missing_camera_dir += 1
                continue

            image_names_by_camera: Dict[str, set] = {}
            has_empty_camera = False
            for camera_name, camera_dir in camera_dirs.items():
                image_names = {
                    name for name in os.listdir(camera_dir) if name.lower().endswith((".jpg", ".jpeg", ".png"))
                }
                if not image_names:
                    has_empty_camera = True
                    break
                image_names_by_camera[camera_name] = image_names
            if has_empty_camera:
                skipped_missing_camera_dir += 1
                continue

            frames = self._read_scene_frames_no_cache(pkl_path)
            valid_indices = self._compute_valid_frame_indices(frames, image_names_by_camera)
            frame_count = len(frames)

            if frame_count < self.min_valid_frames:
                skipped_not_enough_frames += 1
                continue
            if len(valid_indices) < self.required_image_frames:
                skipped_not_enough_required_images += 1
                continue

            scenes.append(
                SceneRecord(
                    scene_name=scene_name,
                    pkl_path=pkl_path,
                    camera_dir=camera_dirs[self.camera_name],
                    valid_frame_indices=valid_indices,
                    camera_dirs=camera_dirs,
                    frame_count=frame_count,
                    frame_timestamps=self._extract_frame_timestamps(frames),
                    frame_tokens=self._extract_frame_tokens(frames),
                )
            )

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  scene index progress: %d/%d (%.1f%%) elapsed=%.1fs",
                    i + 1,
                    total,
                    100.0 * (i + 1) / total,
                    elapsed,
                )

        elapsed = time.monotonic() - t0
        logger.info(
            "NavSim scene index built: total_pkls=%d, kept=%d, "
            "skipped_missing_camera=%d, skipped_short=%d, "
            "skipped_not_enough_required_images=%d, time=%.1fs",
            total,
            len(scenes),
            skipped_missing_camera_dir,
            skipped_not_enough_frames,
            skipped_not_enough_required_images,
            elapsed,
        )

        # --- persist to disk cache ---
        if cache_path is not None:
            self._save_index_cache(cache_path, scenes, total)

        return scenes

    def _build_window_index(self) -> List[WindowRecord]:
        """Enumerate all sliding-window positions across all scenes.

        Windows are anchored to consecutive image-valid segments within each
        scene.  For each maximal run of raw frames that all have images (length
        >= ``required_image_frames``), the stride clock resets at the segment
        start so every segment contributes windows regardless of its raw offset.
        GT trajectory coverage (``min_valid_frames`` raw frames) is still
        required; segments near the end of a log may yield fewer windows if GT
        runs out before the image segment does.

        Returns
        -------
        List[WindowRecord]
            Flat list of (scene_idx, start_pos) pairs, one per window.
        """
        windows: List[WindowRecord] = []
        rejected_time_gap = 0
        for scene_idx, scene in enumerate(self.scenes):
            frame_count = self._scene_frame_count(scene)
            max_gt_start = frame_count - self.min_valid_frames
            tail_min_start = self._tail_min_window_start(frame_count)
            timestamps = scene.frame_timestamps
            if self.window_start_policy == WINDOW_START_COUNTERFACTUAL_SCENE_START:
                valid_segments = self._image_segment_starts(scene)
                scene_start_is_valid = any(
                    seg_start == 0
                    and min(seg_end - (self.required_image_frames - 1) * self.sample_step, max_gt_start) >= 0
                    for seg_start, seg_end in valid_segments
                )
                if scene_start_is_valid:
                    if self._is_window_time_continuous(timestamps, 0):
                        windows.append(WindowRecord(scene_idx=scene_idx, start_pos=0))
                    else:
                        rejected_time_gap += 1
                continue
            for seg_start, seg_end in self._image_segment_starts(scene):
                # the required_image_frames sampled slots span (required_image_frames-1)*sample_step raw
                # frames, so the last valid start must leave room for the stride (= -required+1 when sample_step=1).
                max_start_in_seg = min(seg_end - (self.required_image_frames - 1) * self.sample_step, max_gt_start)
                for start in range(seg_start, max_start_in_seg + 1, self.window_stride):
                    if start < tail_min_start:
                        continue
                    # Reject windows whose GT span crosses a recording gap; the
                    # raw frame indices are contiguous by construction, so this
                    # is the only place a real temporal jump can be caught.
                    if not self._is_window_time_continuous(timestamps, start):
                        rejected_time_gap += 1
                        continue
                    windows.append(WindowRecord(scene_idx=scene_idx, start_pos=start))

        logger.info(
            "Built window index: %d windows from %d scenes "
            "(stride=%d, min_valid=%d, image_require_policy=%s, required_image_frames=%d, "
            "tail_seconds=%s, max_time_gap=%.0fus, rejected_time_gap=%d)",
            len(windows),
            len(self.scenes),
            self.window_stride,
            self.min_valid_frames,
            self.image_require_policy,
            self.required_image_frames,
            self.tail_seconds,
            self.max_time_gap_us,
            rejected_time_gap,
        )
        return windows

    def _build_window_index_from_tokens(self) -> List[WindowRecord]:
        """Anchor one window per official scene-filter token.

        A token names the official sample's "current" frame (the 4th frame of
        the official 14-frame sample); the window is placed so that frame is
        the last observed frame:
        ``start_pos = token_frame_idx - (num_observed_frames - 1) * sample_step``.
        Data-validity failures (history/future bounds, missing observed
        images, recording gaps) are counted and skipped; a token that cannot
        be located at all is a structural error and raises.
        """
        token_lookup: Dict[str, Tuple[int, int]] = {}
        for scene_idx, scene in enumerate(self.scenes):
            if scene.frame_tokens is None:
                raise ValueError(
                    f"Scene {scene.scene_name} carries no per-frame 'token' field; "
                    "official token anchoring requires token-bearing NavSim pkls"
                )
            for frame_idx, token in enumerate(scene.frame_tokens):
                previous = token_lookup.get(token)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate frame token {token!r} in {scene.scene_name} (frame {frame_idx}) "
                        f"and {self.scenes[previous[0]].scene_name} (frame {previous[1]})"
                    )
                token_lookup[token] = (scene_idx, frame_idx)

        anchor_offset = (self.num_observed_frames - 1) * self.sample_step
        window_span = (self.frames_per_clip - 1) * self.sample_step
        valid_indices_by_scene = [frozenset(scene.valid_frame_indices) for scene in self.scenes]

        windows: List[WindowRecord] = []
        reject_counts = {
            "out_of_bounds_history": 0,
            "out_of_bounds_future": 0,
            "image_missing": 0,
            "outside_tail": 0,
            "time_gap": 0,
        }
        for token in self._filter_tokens:
            located = token_lookup.get(token)
            if located is None:
                raise ValueError(f"Scene-filter token {token!r} was not found in any kept log under {self.data_path}")
            scene_idx, frame_idx = located
            scene = self.scenes[scene_idx]
            start_pos = frame_idx - anchor_offset
            tail_min_start = self._tail_min_window_start(self._scene_frame_count(scene))
            if start_pos < 0:
                reject_counts["out_of_bounds_history"] += 1
                continue
            if start_pos + window_span >= self._scene_frame_count(scene):
                reject_counts["out_of_bounds_future"] += 1
                continue
            if start_pos < tail_min_start:
                reject_counts["outside_tail"] += 1
                continue
            sampled = self._sampled_frame_indices(start_pos)
            required = self._required_image_frame_indices(sampled)
            if not all(idx in valid_indices_by_scene[scene_idx] for idx in required):
                reject_counts["image_missing"] += 1
                continue
            if not self._is_window_time_continuous(scene.frame_timestamps, start_pos):
                reject_counts["time_gap"] += 1
                continue
            windows.append(WindowRecord(scene_idx=scene_idx, start_pos=start_pos))

        total_rejected = sum(reject_counts.values())
        if len(windows) + total_rejected != len(self._filter_tokens):
            raise RuntimeError(
                "Token window accounting mismatch: "
                f"windows={len(windows)} rejected={total_rejected} tokens={len(self._filter_tokens)}"
            )
        self.token_reject_counts = reject_counts

        logger.info(
            "Built window index from official tokens: %d windows from %d scenes "
            "(tokens=%d, rejected: history=%d, future=%d, image_missing=%d, time_gap=%d, "
            "outside_tail=%d, num_observed_frames=%d, sample_step=%d, tail_seconds=%s)",
            len(windows),
            len(self.scenes),
            len(self._filter_tokens),
            reject_counts["out_of_bounds_history"],
            reject_counts["out_of_bounds_future"],
            reject_counts["image_missing"],
            reject_counts["time_gap"],
            reject_counts["outside_tail"],
            self.num_observed_frames,
            self.sample_step,
            self.tail_seconds,
        )
        if total_rejected:
            logger.warning(
                "Official token anchoring rejected %d/%d tokens for data validity: %s",
                total_rejected,
                len(self._filter_tokens),
                reject_counts,
            )
        return windows

    def _image_segment_starts(self, scene: SceneRecord) -> List[tuple]:
        """Return (seg_start, seg_end_inclusive) for each maximal run of
        consecutive raw frames that all have camera images, keeping only
        runs long enough to satisfy ``required_image_frames``."""
        vi = scene.valid_frame_indices
        if not vi:
            return []
        segments: List[tuple] = []
        s = e = vi[0]
        for v in vi[1:]:
            if v == e + 1:
                e = v
            else:
                if e - s + 1 >= self.required_image_frames:
                    segments.append((s, e))
                s = e = v
        if e - s + 1 >= self.required_image_frames:
            segments.append((s, e))
        return segments

    def _scene_frame_count(self, scene: SceneRecord) -> int:
        frame_count = getattr(scene, "frame_count", None)
        if frame_count is not None:
            return int(frame_count)
        if scene.valid_frame_indices:
            return int(max(scene.valid_frame_indices) + 1)
        return 0

    def _validate_counterfactual_scene_start_metadata(
        self,
        *,
        window_start: int,
        metadata_valid_mask: np.ndarray,
    ) -> None:
        if self.window_start_policy != WINDOW_START_COUNTERFACTUAL_SCENE_START:
            return
        observed_valid = metadata_valid_mask[: self.num_observed_frames]
        future_valid = metadata_valid_mask[self.num_observed_frames :]
        if window_start != 0 or not observed_valid.all() or future_valid.size == 0 or future_valid.any():
            raise ValueError(
                "counterfactual_scene_start requires start=0 with a fully valid observed prefix "
                "and metadata-invalid generated future"
            )

    def _tail_min_window_start(self, frame_count: int) -> int:
        """Return the earliest raw start index allowed by ``tail_seconds``."""
        if self.tail_seconds is None:
            return 0
        tail_raw_frames = int(np.ceil(float(self.tail_seconds) * float(self.base_fps)))
        return max(0, int(frame_count) - tail_raw_frames)

    def _sampled_frame_indices(self, start_pos: int) -> List[int]:
        """Return deterministic raw frame indices for a window starting at *start_pos*."""
        positions = int(start_pos) + np.arange(self.frames_per_clip) * self.sample_step
        return [int(p) for p in positions]

    def _reduced_frame_dt(self, reduced_frame_indices: Sequence[int]) -> float:
        """Seconds between adjacent reduced frames."""
        if len(reduced_frame_indices) >= 2:
            raw_delta = int(reduced_frame_indices[1]) - int(reduced_frame_indices[0])
            if raw_delta <= 0:
                raise ValueError(f"reduced_frame_indices must be strictly increasing, got {reduced_frame_indices}")
            return float(raw_delta) / float(self.base_fps)
        return float(self.sample_step * self.tubelet_size) / float(self.base_fps)

    def _required_image_frame_indices(self, sampled_frame_indices: Sequence[int]) -> List[int]:
        """Return the prefix of sampled frames that must have images."""
        return [int(frame_idx) for frame_idx in sampled_frame_indices[: self.required_image_frames]]

    def _extract_frame_timestamps(self, frames: Sequence[Dict[str, Any]]) -> Optional[List[int]]:
        """Capture per-frame timestamps (microseconds) for time-gap detection.

        Returns None if any frame lacks a timestamp, in which case the
        downstream time-continuity check is skipped (fail-open) so datasets
        without timestamps keep working.
        """
        timestamps: List[int] = []
        for frame in frames:
            ts = frame.get("timestamp") if isinstance(frame, dict) else None
            if ts is None:
                if self.timestamp_policy_was_explicit:
                    raise ValueError("explicit timestamp_policy requires every NavSim frame to contain timestamp")
                return None
            timestamps.append(int(ts))
        return timestamps

    def _extract_frame_tokens(self, frames: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
        """Capture per-frame tokens for official token anchoring.

        Returns None if any frame lacks a non-empty "token" (e.g. converted
        nuScenes exports) so stride mode keeps working; token mode raises
        explicitly at lookup time in ``_build_window_index_from_tokens``.
        """
        tokens: List[str] = []
        for frame in frames:
            token = frame.get("token") if isinstance(frame, dict) else None
            if not isinstance(token, str) or not token:
                return None
            tokens.append(token)
        return tokens

    def _is_window_time_continuous(self, timestamps: Optional[Sequence[int]], start_pos: int) -> bool:
        """Return whether the window's raw GT span has no recording gap.

        The window spans raw frames
        ``[start_pos, start_pos + (frames_per_clip - 1) * sample_step]``
        (sample_step==1 for NavSim; >1 when config fps < base_fps). Each
        consecutive raw pair within the span must be within ``max_time_gap_us``;
        a larger delta means missing timesteps and a GT trajectory jump.
        Fail-open when timestamps are unavailable.
        """
        if not timestamps:
            return True
        end = start_pos + (self.frames_per_clip - 1) * self.sample_step + 1
        if end > len(timestamps):
            return True
        for i in range(start_pos, end - 1):
            if timestamps[i + 1] - timestamps[i] > self.max_time_gap_us:
                return False
        return True

    # ------------------------------------------------------------------
    # Scene index disk cache helpers
    # ------------------------------------------------------------------

    def _get_index_cache_path(self, total_pkls: int) -> Optional[str]:
        """Return cache file path, or ``None`` if caching is disabled."""
        if not self.index_cache:
            return None
        fingerprint_data = json.dumps(
            {
                "data_path": os.path.abspath(self.data_path),
                "sensor_blobs_path": os.path.abspath(self.sensor_blobs_path),
                "camera_names": self.camera_names,
                "frames_per_clip": self.frames_per_clip,
                "fps": self.fps,
                "base_fps": self.base_fps,
                "tubelet_size": self.tubelet_size,
                "max_scenes": self.max_scenes,
                "max_frame_gap": self.max_frame_gap,
                "image_require_policy": self.image_require_policy,
                "num_observed_frames": self.num_observed_frames,
                # None in stride mode; content hash of (log_names, tokens) in
                # token mode, so cache survives yaml relocation but not edits.
                "scene_filter_hash": self._scene_filter_hash,
            },
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
        return os.path.join(self.data_path, f".navsim_scene_index_cache_{fingerprint}.pkl")

    def _load_index_cache(self, cache_path: str, current_total_pkls: int) -> Optional[List[SceneRecord]]:
        """Try to load & validate a cached scene index.  Return ``None`` on miss / stale."""
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or payload.get("version") != 4:
                logger.info("Scene index cache version mismatch, will rebuild.")
                return None
            if payload.get("total_pkls") != current_total_pkls:
                logger.info(
                    "Scene index cache stale (cached %s pkls vs current %d), will rebuild.",
                    payload.get("total_pkls"),
                    current_total_pkls,
                )
                return None
            scenes: List[SceneRecord] = payload["scenes"]
            logger.info("Loaded scene index from cache: %s (%d scenes)", cache_path, len(scenes))
            return scenes
        except Exception as exc:
            logger.warning("Failed to load scene index cache %s: %s", cache_path, exc)
            return None

    def _save_index_cache(self, cache_path: str, scenes: List[SceneRecord], total_pkls: int) -> None:
        """Persist the scene index to disk."""
        payload = {
            "version": 4,
            "total_pkls": total_pkls,
            "scenes": scenes,
        }
        try:
            tmp_path = f"{cache_path}.{os.getpid()}.tmp"  # pid-unique: concurrent ranks must not share a tmp
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            logger.info("Saved scene index cache: %s (%d scenes)", cache_path, len(scenes))
        except Exception as exc:
            logger.warning("Failed to save scene index cache %s: %s", cache_path, exc)

    def _compute_valid_frame_indices(
        self,
        frames: Sequence[Dict[str, Any]],
        image_names_by_camera: Dict[str, set],
    ) -> List[int]:
        valid_indices: List[int] = []
        for idx, frame in enumerate(frames):
            is_valid = True
            for camera_name in self.camera_names:
                cam_dict = frame.get("cams", {}).get(camera_name)
                if cam_dict is None:
                    is_valid = False
                    break
                rel_path = cam_dict.get("data_path")
                if not rel_path:
                    is_valid = False
                    break
                image_name = os.path.basename(rel_path)
                if image_name not in image_names_by_camera[camera_name]:
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(idx)
        return valid_indices

    def _read_scene_frames_no_cache(self, pkl_path: str) -> List[Dict[str, Any]]:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if not isinstance(frames, list):
            raise ValueError(f"Unexpected NavSim pickle structure: {pkl_path}")
        return frames

    def _load_scene_frames(self, pkl_path: str) -> List[Dict[str, Any]]:
        cached = self._scene_cache.get(pkl_path)
        if cached is not None:
            self._scene_cache.move_to_end(pkl_path)
            return cached

        frames = self._read_scene_frames_no_cache(pkl_path)
        self._scene_cache[pkl_path] = frames
        self._scene_cache.move_to_end(pkl_path)

        if len(self._scene_cache) > self.cache_size:
            self._scene_cache.popitem(last=False)

        return frames

    def _load_clip_images(
        self,
        scene: SceneRecord,
        frames: Sequence[Dict[str, Any]],
        sampled_frame_indices: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_view_images: List[np.ndarray] = []
        all_view_shapes: List[np.ndarray] = []
        camera_dirs = scene.camera_dirs or {self.camera_name: scene.camera_dir}

        for camera_name in self.camera_names:
            images: List[np.ndarray] = []
            image_shapes: List[Tuple[int, int]] = []
            camera_dir = camera_dirs[camera_name]
            for frame_idx in sampled_frame_indices:
                frame = frames[frame_idx]
                rel_path = frame["cams"][camera_name]["data_path"]
                image_name = os.path.basename(rel_path)
                image_path = os.path.join(camera_dir, image_name)

                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                with Image.open(image_path) as img:
                    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                images.append(rgb)
                image_shapes.append((rgb.shape[0], rgb.shape[1]))

            all_view_images.append(np.stack(images, axis=0))
            all_view_shapes.append(np.asarray(image_shapes, dtype=np.int64))

        if self.multiview_enabled:
            return np.stack(all_view_images, axis=0), np.stack(all_view_shapes, axis=0)
        return all_view_images[0], all_view_shapes[0]

    def _apply_clip_transform(self, buffer: np.ndarray, transform: Any) -> torch.Tensor:
        """Apply a single-view transform to one or more camera clips."""
        if buffer.ndim == 5:
            transformed_views = []
            for view_idx in range(buffer.shape[0]):
                transformed = transform(buffer[view_idx])
                if not torch.is_tensor(transformed):
                    transformed = torch.as_tensor(transformed)
                transformed_views.append(transformed)
            return torch.stack(transformed_views, dim=0)

        transformed = transform(buffer)
        if not torch.is_tensor(transformed):
            transformed = torch.as_tensor(transformed)
        return transformed

    def _default_image_tensor(self, buffer: np.ndarray) -> torch.Tensor:
        """Convert raw uint8 images to model tensor format."""
        if buffer.ndim == 5:
            return torch.from_numpy(buffer).permute(0, 4, 1, 2, 3).float() / 255.0
        return torch.from_numpy(buffer).permute(3, 0, 1, 2).float() / 255.0

    @staticmethod
    def _infer_output_hw(buffer: torch.Tensor) -> Tuple[int, int]:
        if buffer.ndim == 5:
            return int(buffer.shape[-2]), int(buffer.shape[-1])
        if buffer.ndim == 4:
            return int(buffer.shape[-2]), int(buffer.shape[-1])
        raise ValueError(f"Unexpected transformed buffer shape: {tuple(buffer.shape)}")

    def _build_camera_metadata(
        self,
        frames: Sequence[Dict[str, Any]],
        sampled_frame_indices: Sequence[int],
        *,
        source_image_shapes: np.ndarray,
        output_hw: Tuple[int, int],
        transform: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build resized/cropped camera intrinsics and camera-to-ego transforms."""
        if source_image_shapes.ndim == 2:
            source_shapes = source_image_shapes[None]
        else:
            source_shapes = source_image_shapes

        intrinsics = np.zeros((len(self.camera_names), len(sampled_frame_indices), 3, 3), dtype=np.float32)
        camera2ego = np.zeros((len(self.camera_names), len(sampled_frame_indices), 4, 4), dtype=np.float32)

        # camera_intrinsics + camera2ego are consumed ONLY by the multi-view PETR fusion. When multi-view is off
        # (single camera) they are never read downstream, so don't require them in the data — return zeros
        # (unused) instead of failing on a missing 'cam_intrinsic'. Fail-loud is preserved below for the
        # multi-view case, where the intrinsics actually drive the geometry.
        if not self.multiview_enabled:
            return intrinsics, camera2ego

        for view_idx, camera_name in enumerate(self.camera_names):
            for t_idx, frame_idx in enumerate(sampled_frame_indices):
                cam = frames[frame_idx]["cams"][camera_name]
                # fail-loud (point 11): 缺 cam_intrinsic 不得用 identity 内参替代。
                if "cam_intrinsic" not in cam:
                    raise CameraMetadataError(
                        f"NavSim camera '{camera_name}' (frame {frame_idx}) missing 'cam_intrinsic'; "
                        "禁止用 identity 内参替代。"
                    )
                raw_intrinsic = np.asarray(cam["cam_intrinsic"], dtype=np.float32)
                intrinsics[view_idx, t_idx] = self._resize_crop_intrinsics(
                    raw_intrinsic,
                    source_hw=tuple(int(x) for x in source_shapes[view_idx, t_idx]),
                    output_hw=output_hw,
                    transform=transform,
                )
                camera2ego[view_idx, t_idx] = self._camera_to_ego_transform(cam)

        return intrinsics, camera2ego

    @staticmethod
    def _resize_crop_intrinsics(
        intrinsic: np.ndarray,
        *,
        source_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        transform: Any,
    ) -> np.ndarray:
        """Approximate the deterministic crop/resize used by train/eval transforms."""
        src_h, src_w = float(source_hw[0]), float(source_hw[1])
        out_h, out_w = float(output_hw[0]), float(output_hw[1])
        intrinsic = intrinsic.astype(np.float32, copy=True)

        crop_top_bottom = getattr(transform, "crop_top_bottom", None)
        if crop_top_bottom is not None and int(crop_top_bottom) > 0:
            crop = float(crop_top_bottom)
            crop_x = 0.0
            crop_y = crop
            crop_w = src_w
            crop_h = src_h - 2.0 * crop
        else:
            target_aspect = out_w / max(out_h, 1.0)
            source_aspect = src_w / max(src_h, 1.0)
            if source_aspect > target_aspect:
                crop_h = src_h
                crop_w = crop_h * target_aspect
                crop_x = (src_w - crop_w) * 0.5
                crop_y = 0.0
            else:
                crop_w = src_w
                crop_h = crop_w / max(target_aspect, 1e-6)
                crop_x = 0.0
                crop_y = (src_h - crop_h) * 0.5

        if crop_h <= 0 or crop_w <= 0:
            crop_x = crop_y = 0.0
            crop_h, crop_w = src_h, src_w

        scale_x = out_w / crop_w
        scale_y = out_h / crop_h
        intrinsic[0, 0] *= scale_x
        intrinsic[1, 1] *= scale_y
        intrinsic[0, 2] = (intrinsic[0, 2] - crop_x) * scale_x
        intrinsic[1, 2] = (intrinsic[1, 2] - crop_y) * scale_y
        return intrinsic

    @staticmethod
    def _camera_to_ego_transform(cam: Dict[str, Any]) -> np.ndarray:
        for key in ("camera2ego", "cam2ego", "sensor2ego"):
            value = cam.get(key)
            if value is not None:
                matrix = np.asarray(value, dtype=np.float32)
                if matrix.shape != (4, 4):
                    # fail-loud (point 12): 外参字段存在但形状不符直接报错，
                    # 禁止静默跳过该字段继续尝试其他外参来源。
                    raise CameraMetadataError(
                        f"NavSim camera extrinsics '{key}' has shape {tuple(matrix.shape)}, expected (4, 4)"
                    )
                return matrix

        has_s2e_rot = "sensor2ego_rotation" in cam
        has_s2e_trans = "sensor2ego_translation" in cam
        if has_s2e_rot or has_s2e_trans:
            if not (has_s2e_rot and has_s2e_trans):
                # fail-loud (point 12): rotation/translation 只缺一半时直接报错，
                # 禁止用 identity/zeros 补全缺失的那一半。
                raise CameraMetadataError(
                    "NavSim camera has partial sensor2ego extrinsics: "
                    f"sensor2ego_rotation present={has_s2e_rot}, "
                    f"sensor2ego_translation present={has_s2e_trans}"
                )
            return NavSimWorldModelDataset._as_transform(
                cam["sensor2ego_rotation"],
                cam["sensor2ego_translation"],
            )

        has_s2l_rot = "sensor2lidar_rotation" in cam
        has_s2l_trans = "sensor2lidar_translation" in cam
        if has_s2l_rot or has_s2l_trans:
            if not (has_s2l_rot and has_s2l_trans):
                raise CameraMetadataError(
                    "NavSim camera has partial sensor2lidar extrinsics: "
                    f"sensor2lidar_rotation present={has_s2l_rot}, "
                    f"sensor2lidar_translation present={has_s2l_trans}"
                )
            sensor2lidar = NavSimWorldModelDataset._as_transform(
                cam["sensor2lidar_rotation"],
                cam["sensor2lidar_translation"],
            )
            has_l2e_rot = "lidar2ego_rotation" in cam
            has_l2e_trans = "lidar2ego_translation" in cam
            if has_l2e_rot != has_l2e_trans:
                raise CameraMetadataError(
                    "NavSim camera has partial lidar2ego extrinsics: "
                    f"lidar2ego_rotation present={has_l2e_rot}, "
                    f"lidar2ego_translation present={has_l2e_trans}"
                )
            if has_l2e_rot:
                lidar2ego = NavSimWorldModelDataset._as_transform(
                    cam["lidar2ego_rotation"],
                    cam["lidar2ego_translation"],
                )
                return (lidar2ego @ sensor2lidar).astype(np.float32)
            # 原生 NavSim PKL 的 cams 仅含 sensor2lidar_*（devkit Camera dataclass 没有
            # per-cam lidar2ego 字段），lidar2ego 全缺时按既定契约直接以 sensor2lidar
            # 作为 camera2ego（与 eval 侧 navsim_feature_builder._sensor2lidar_transform
            # 一致，保持 train/infer 几何对齐）。部分缺失才属于数据错误（上面已 raise）。
            return sensor2lidar.astype(np.float32)

        # fail-loud (point 12): 相机外参字段全缺时直接报错，禁止用 identity 外参替代。
        raise CameraMetadataError(
            "NavSim camera missing extrinsics: none of (camera2ego/cam2ego/sensor2ego, "
            "sensor2ego_rotation/translation, sensor2lidar_*+lidar2ego_*) present in cam metadata; "
            "禁止用 identity 外参替代。"
        )

    @staticmethod
    def _as_transform(rotation: Any, translation: Any) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        rot = np.asarray(rotation, dtype=np.float32)
        if rot.shape == (9,):
            rot = rot.reshape(3, 3)
        if rot.shape == (3, 3):
            rot_matrix = rot
        elif rot.shape == (4,):
            # NavSim/nuScenes metadata generally stores quaternions as wxyz.
            quat_xyzw = np.asarray([rot[1], rot[2], rot[3], rot[0]], dtype=np.float32)
            rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        else:
            # fail-loud (point 12): 旋转形状不符直接报错，禁止静默退化为 identity 旋转。
            raise CameraMetadataError(
                f"Unsupported camera extrinsics rotation shape {tuple(rot.shape)}; "
                "expected (3, 3), flattened (9,) or quaternion (4,) wxyz"
            )
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)[:3]
        return matrix

    def _build_agent_annotations(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 NavSim frame 的 anns 字段提取 agent bounding boxes。

        Parameters
        ----------
        frames               : NavSim raw frame dicts (PKL)
        reduced_frame_indices : tubelet 降采样后的帧索引

        Returns
        -------
        agent_boxes : [T, max_agents, 7]  float32
            每个 agent 的 [x, y, z, length, width, height, heading]，
            坐标系为该帧自车坐标系。
        agent_mask  : [T, max_agents]     bool
            True 表示该位置有有效 agent。
        raw_agent_count : [T] int64
            每个 reduced frame 在 padding 前的动态 agent 数量。
        """
        T = len(reduced_frame_indices)
        agent_boxes = np.zeros((T, self.max_agents, 7), dtype=np.float32)
        agent_mask = np.zeros((T, self.max_agents), dtype=np.bool_)
        raw_agent_count = np.zeros(T, dtype=np.int64)

        for t_idx, frame_idx in enumerate(reduced_frame_indices):
            frame = frames[frame_idx]
            parsed = self._parse_agent_annotations(frame, frame_idx=frame_idx)
            if parsed is None:
                continue
            gt_boxes, gt_names = parsed

            # 过滤：只保留动态 agent 类型
            keep_indices = [i for i, name in enumerate(gt_names) if name in self._DYNAMIC_AGENT_TYPES]

            if not keep_indices:
                continue

            kept_boxes = gt_boxes[keep_indices]
            if not np.isfinite(kept_boxes).all():
                raise ValueError(f"frame {frame_idx} dynamic agent boxes must be finite")
            if np.any(kept_boxes[:, 3:6] <= 0):
                raise ValueError(f"frame {frame_idx} dynamic agent boxes must have positive dimensions")

            frame_agent_count = int(len(kept_boxes))
            raw_agent_count[t_idx] = frame_agent_count
            if frame_agent_count > self.max_agents:
                raise AgentGeometryOverflowError(
                    f"frame {frame_idx} has {frame_agent_count} dynamic agents, exceeding max_agents={self.max_agents}"
                )
            agent_boxes[t_idx, :frame_agent_count] = kept_boxes
            agent_mask[t_idx, :frame_agent_count] = True

        return agent_boxes, agent_mask, raw_agent_count

    @staticmethod
    def _parse_agent_annotations(
        frame: Dict[str, Any],
        *,
        frame_idx: int,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Return structurally validated agent annotations, or None when absent."""

        if "anns" not in frame or frame["anns"] is None:
            return None
        anns = frame["anns"]
        if not isinstance(anns, dict):
            raise ValueError(f"frame {frame_idx} anns must be a mapping, got {type(anns).__name__}")
        missing = [name for name in ("gt_boxes", "gt_names") if name not in anns]
        if missing:
            raise ValueError(f"frame {frame_idx} anns is missing required fields: {missing}")

        gt_boxes = np.asarray(anns["gt_boxes"], dtype=np.float32)
        if gt_boxes.size == 0:
            gt_boxes = np.empty((0, 7), dtype=np.float32)
        elif gt_boxes.ndim != 2 or gt_boxes.shape[1] != 7:
            raise ValueError(f"frame {frame_idx} anns['gt_boxes'] must have shape [N, 7], got {gt_boxes.shape}")
        raw_names = anns["gt_names"]
        if not isinstance(raw_names, (list, tuple, np.ndarray)):
            raise ValueError(f"frame {frame_idx} anns['gt_names'] must be a sequence")
        gt_names = [str(name) for name in raw_names]
        if len(gt_names) != int(gt_boxes.shape[0]):
            raise ValueError(f"frame {frame_idx} anns has {gt_boxes.shape[0]} boxes but {len(gt_names)} names")
        return gt_boxes, gt_names

    def _future_agent_annotations_are_valid(
        self,
        frames: Sequence[Dict[str, Any]],
        future_frame_indices: Sequence[int],
    ) -> bool:
        """Validate future boxes, ego poses, and timestamp continuity."""

        indices = [int(frame_idx) for frame_idx in future_frame_indices]
        if not indices or any(current <= previous for previous, current in zip(indices, indices[1:])):
            return False

        for frame_idx in indices:
            frame = frames[frame_idx]
            try:
                parsed = self._parse_agent_annotations(frame, frame_idx=frame_idx)
            except (TypeError, ValueError):
                return False
            if parsed is None:
                return False
            gt_boxes, gt_names = parsed
            dynamic_indices = [index for index, name in enumerate(gt_names) if name in self._DYNAMIC_AGENT_TYPES]
            dynamic_boxes = gt_boxes[dynamic_indices]
            if len(dynamic_boxes) > self.max_agents:
                return False
            if not np.isfinite(dynamic_boxes).all() or np.any(dynamic_boxes[:, 3:6] <= 0):
                return False

            translation = frame.get("ego2global_translation")
            rotation = frame.get("ego2global_rotation")
            try:
                translation_array = np.asarray(translation, dtype=np.float64)
                rotation_array = np.asarray(rotation, dtype=np.float64)
            except (TypeError, ValueError):
                return False
            if translation_array.ndim != 1 or translation_array.shape[0] < 3:
                return False
            if rotation_array.shape not in {(4,), (9,), (3, 3)}:
                return False
            if not np.isfinite(translation_array[:3]).all() or not np.isfinite(rotation_array).all():
                return False
            if rotation_array.shape == (4,) and float(np.linalg.norm(rotation_array)) <= 0.0:
                return False

        time_indices = indices
        if indices[0] > 0:
            time_indices = [indices[0] - 1, *indices]
        timestamps: List[int] = []
        for frame_idx in time_indices:
            timestamp = frames[frame_idx].get("timestamp")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, np.integer)):
                return False
            timestamps.append(int(timestamp))
        for pair_index, (previous, current) in enumerate(zip(timestamps, timestamps[1:])):
            raw_index_delta = time_indices[pair_index + 1] - time_indices[pair_index]
            timestamp_delta = current - previous
            if timestamp_delta <= 0 or timestamp_delta > self.max_time_gap_us * max(1, raw_index_delta):
                return False
        return True

    def _build_bev_segmentation(
        self,
        agent_boxes: np.ndarray,
        agent_mask: np.ndarray,
    ) -> np.ndarray:
        """将 agent boxes 栅格化为 per-frame BEV 分割图。

        每帧的 seg map 在该帧的 ego 坐标系下（与 ST-P3/VAD 对齐）。

        Parameters
        ----------
        agent_boxes : [T, max_agents, 7]  各帧 ego 坐标系
        agent_mask  : [T, max_agents]     bool

        Returns
        -------
        bev_seg : [T, BEV_SIZE, BEV_SIZE]  uint8, 1=occupied, 0=free
        """
        T = agent_boxes.shape[0]
        bev_seg = np.zeros((T, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        for t in range(T):
            bev_seg[t] = _rasterize_agents_to_bev(agent_boxes[t], agent_mask[t])
        return bev_seg

    def _build_states(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Build states array from NavSim frames.

        NavSim stores ego poses as UTM global coordinates (``ego2global_translation``,
        magnitude ~10^5–10^6).  Storing these directly as ``float32`` causes significant
        precision loss (up to 0.5 m for the y-component).  To avoid this we:

        1. Read translations in **float64**.
        2. Subtract the **first frame's translation** (scene-centre) so that all
           subsequent positions are O(0–100 m).
        3. Cast to **float32** – now safe because the magnitudes are small.

        The centering is algebraically transparent to all downstream consumers
        (actions, GT trajectory, status features) because they only use
        *differences* between frames.
        """
        # --- collect raw translations in float64 for precision ---
        translations_f64: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        speeds: List[float] = []

        for frame_idx in reduced_frame_indices:
            frame = frames[frame_idx]

            translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
            translations_f64.append(translation)

            quat_wxyz = np.asarray(frame["ego2global_rotation"], dtype=np.float64)
            quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
            roll_pitch_yaw = Rotation.from_quat(quat_xyzw).as_euler("xyz")
            orientations.append(roll_pitch_yaw)

            if "ego_dynamic_state" not in frame:
                # fail-loud (point 11): 缺 ego_dynamic_state 直接报错，禁止用 0 速度伪造
                # states[:, 6]（速度会进 status features / 条件输入）。原生 NavSim PKL 与
                # 两个 nuScenes 转换器均逐帧写入该字段；缺失说明 PKL 需要重新生成。
                raise ValueError(
                    "NavSim frame missing 'ego_dynamic_state' (required for states[:, 6] speed); "
                    "禁止用零速度替代，请重新生成 PKL。"
                )
            speeds.append(float(frame["ego_dynamic_state"][0]))

        # --- centre translations around first frame (float64 arithmetic) ---
        translations_arr = np.stack(translations_f64, axis=0)  # [T, 3] float64
        origin_translation = translations_arr[0].copy()  # first-frame UTM origin
        translations_arr -= origin_translation  # now O(0–100 m)

        # --- assemble final states [T, 7] as float32 ---
        orientations_arr = np.stack(orientations, axis=0)  # [T, 3] float64
        speeds_arr = np.asarray(speeds, dtype=np.float64).reshape(-1, 1)  # [T, 1]
        states = np.concatenate([translations_arr, orientations_arr, speeds_arr], axis=1)  # [T, 7]

        return states.astype(np.float32)

    def _build_metadata_valid_mask(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Return whether action/state metadata is trusted for each reduced timestep."""
        return np.asarray(
            [bool(frames[frame_idx].get("metadata_valid", True)) for frame_idx in reduced_frame_indices],
            dtype=np.bool_,
        )

    def _build_driving_commands(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Extract per-frame driving_command from NavSim raw data.

        Each frame in the NavSim PKL contains a ``driving_command`` field — a
        4-element integer one-hot array from the nuPlan route planner indicating
        the high-level navigation intent (GO_STRAIGHT / TURN_LEFT / TURN_RIGHT /
        U_TURN).

        Parameters
        ----------
        frames               : raw NavSim frame dicts loaded from PKL.
        reduced_frame_indices : indices into *frames* (after tubelet down-sampling).

        Returns
        -------
        np.ndarray
            ``[T, 4]`` float32 one-hot driving commands.

        Raises
        ------
        KeyError
            If any frame is missing the ``driving_command`` field.
        """
        cmds: List[np.ndarray] = []
        for frame_idx in reduced_frame_indices:
            frame = frames[frame_idx]
            # 字段缺失时直接报错，不做静默降级
            cmd = np.asarray(frame["driving_command"], dtype=np.float32)
            cmds.append(cmd[:4])  # 确保只取前 4 维
        return np.stack(cmds, axis=0)  # [T, 4]

    def _build_ego_dynamics(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Extract per-frame ego_dynamic_state [vx, vy, ax, ay] from NavSim raw data.

        Parameters
        ----------
        frames               : raw NavSim frame dicts loaded from PKL.
        reduced_frame_indices : indices into *frames* (after tubelet down-sampling).

        Returns
        -------
        np.ndarray
            ``[T, 4]`` float32 array of ``[vx, vy, ax, ay]``.

        Raises
        ------
        KeyError
            If any frame is missing the ``ego_dynamic_state`` field.
        """
        dynamics: List[np.ndarray] = []
        for frame_idx in reduced_frame_indices:
            frame = frames[frame_idx]
            # 字段缺失时直接报错，不做静默降级
            dyn = np.asarray(frame["ego_dynamic_state"], dtype=np.float32)
            dynamics.append(dyn[:4])  # [vx, vy, ax, ay]
        return np.stack(dynamics, axis=0)  # [T, 4]

    def _build_actions(self, states: np.ndarray) -> np.ndarray:
        if states.shape[0] < 2:
            raise ValueError("Need at least 2 reduced states to build actions")
        return self._build_actions_3d(states)

    def _build_actions_3d(self, states: np.ndarray) -> np.ndarray:
        """Build 3D actions: [dx_ego, dy_ego, d_yaw].

        NavSim states[:3] are global UTM coordinates (ego2global_translation).
        We rotate the global position difference into the ego frame of the
        *current* timestep so that dx ≈ forward displacement and dy ≈ lateral
        displacement, consistent with Mongo Raw's local-coordinate actions.
        """
        t_steps = states.shape[0]
        actions = np.zeros((t_steps - 1, 3), dtype=np.float32)

        for t in range(t_steps - 1):
            # --- global position diff → ego-frame position diff ---
            dx_global = states[t + 1, 0] - states[t, 0]
            dy_global = states[t + 1, 1] - states[t, 1]

            yaw = states[t, 5]  # current ego yaw in world frame
            cos_h = np.cos(-yaw)
            sin_h = np.sin(-yaw)
            dx_ego = cos_h * dx_global - sin_h * dy_global
            dy_ego = sin_h * dx_global + cos_h * dy_global

            # --- yaw diff ---
            d_yaw = states[t + 1, 5] - states[t, 5]
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

            actions[t] = np.asarray([dx_ego, dy_ego, d_yaw], dtype=np.float32)

        return actions


def navsim_world_model_collate_fn(
    batch: Sequence[Dict[str, Any] | MatchedRealCounterfactualPair],
):
    pair_flags = [isinstance(item, MatchedRealCounterfactualPair) for item in batch]
    if any(pair_flags):
        if not all(pair_flags):
            raise ValueError("NavSim collate cannot mix atomic real/CF pairs with ordinary samples")
        batch = [sample for pair in batch for sample in (pair.real, pair.counterfactual)]
    context_frames, actions, states, extrinsics = collate.stack_core(batch)
    seg_targets = [None for _ in batch]

    # NavSim-specific fields (driving_command, ego_dynamics) appended at tuple tail
    driving_command = collate.stack_required(batch, "driving_command")
    ego_dynamics = collate.stack_required(batch, "ego_dynamics")

    # Agent annotations for collision detection (index 7, 8)
    agent_boxes = collate.stack_required(batch, "agent_boxes")
    agent_mask = collate.stack_agent_mask(batch)

    # Pre-computed BEV segmentation maps for collision rate (index 9)
    bev_segmentation = collate.stack_required(batch, "bev_segmentation")

    # Optional proposal encoder frames (index 10), e.g. V-JEPA 256x512 while
    # the main V-JEPA branch keeps its own 256x256 transform.
    proposal_context_frames = collate.stack_proposal_buffer(batch)

    # build_window_metadata provides scene_name/pkl_path/window_start_pos/sampled_frame_indices/
    # image_frame_indices; main's counterfactual + value-planning feature adds the dataset-root +
    # metadata-validity keys (populated per-item above). Keep both.
    metadata = collate.build_window_metadata(batch, "pkl_path")
    required_supervision_metadata = (
        "dataset_domain",
        "dataset_root_name",
        "dataset_root_index",
        "base_scene_id",
        "cf_annotation_valid",
        "cf_is_hazard",
        "cf_hazard_type",
        "future_agent_geometry_valid",
        "geometry_present",
        "geometry_source",
        "geometry_coordinate_frame",
        "agent_geometry_truncated",
        "raw_agent_count",
    )
    for field_name in required_supervision_metadata:
        missing = [index for index, item in enumerate(batch) if field_name not in item]
        if missing:
            raise ValueError(
                f"NavSim collate requires normalized {field_name!r} metadata on every sample; "
                f"missing indices={missing}"
            )
    metadata["dataset_domain"] = [str(item["dataset_domain"]) for item in batch]
    metadata["dataset_root_name"] = [str(item["dataset_root_name"]) for item in batch]
    metadata["dataset_root_index"] = torch.as_tensor(
        [int(item["dataset_root_index"]) for item in batch], dtype=torch.long
    )
    metadata["base_scene_id"] = [str(item["base_scene_id"]) for item in batch]
    metadata["cf_annotation_valid"] = torch.as_tensor(
        [bool(item["cf_annotation_valid"]) for item in batch], dtype=torch.bool
    )
    metadata["cf_is_hazard"] = torch.as_tensor([bool(item["cf_is_hazard"]) for item in batch], dtype=torch.bool)
    metadata["cf_hazard_type"] = [str(item["cf_hazard_type"]) for item in batch]
    quality_transport_fields = (
        "cf_quality_present",
        "cf_quality",
        "cf_progress_score",
        "cf_reverse_risk",
        "cf_comfort_score",
        "cf_path_efficiency",
        "cf_quality_schema",
        "cf_quality_source",
    )
    partial_quality = [
        index
        for index, item in enumerate(batch)
        if any(field_name in item for field_name in quality_transport_fields)
        and not all(field_name in item for field_name in quality_transport_fields)
    ]
    if partial_quality:
        raise ValueError(
            "NavSim collate requires the complete counterfactual quality transport schema; "
            f"partial indices={partial_quality}"
        )
    metadata["cf_quality_present"] = torch.as_tensor(
        [bool(item.get("cf_quality_present", False)) for item in batch], dtype=torch.bool
    )
    for field_name in (
        "cf_quality",
        "cf_progress_score",
        "cf_reverse_risk",
        "cf_comfort_score",
        "cf_path_efficiency",
    ):
        metadata[field_name] = torch.as_tensor(
            [float(item.get(field_name, float("nan"))) for item in batch], dtype=torch.float32
        )
    metadata["cf_quality_schema"] = [item.get("cf_quality_schema") for item in batch]
    metadata["cf_quality_source"] = [item.get("cf_quality_source") for item in batch]
    metadata["future_agent_geometry_valid"] = torch.as_tensor(
        [bool(item["future_agent_geometry_valid"]) for item in batch], dtype=torch.bool
    )
    metadata["geometry_present"] = torch.as_tensor(
        [bool(item["geometry_present"]) for item in batch], dtype=torch.bool
    )
    metadata["geometry_source"] = [item["geometry_source"] for item in batch]
    metadata["geometry_coordinate_frame"] = [item["geometry_coordinate_frame"] for item in batch]
    metadata["coordinate_frame"] = [item.get("coordinate_frame", item["geometry_coordinate_frame"]) for item in batch]
    metadata["agent_geometry_truncated"] = [item["agent_geometry_truncated"] for item in batch]
    metadata["raw_agent_count"] = [
        None if item["raw_agent_count"] is None else torch.as_tensor(item["raw_agent_count"], dtype=torch.long)
        for item in batch
    ]
    # Sample-level counterfactual annotation facts (loader stamps them on every item
    # when the real dataset runs; keep optional so minimal/synthetic batches collate).
    if batch and "is_accident" in batch[0]:
        metadata["is_accident"] = torch.as_tensor([bool(item["is_accident"]) for item in batch], dtype=torch.bool)
        metadata["accident_type"] = [str(item["accident_type"]) for item in batch]
    # metadata-validity masks are populated by the real loader (counterfactual + value-planning path);
    # keep them optional — like the camera_intrinsics block below — so minimal/synthetic batches still collate.
    if batch and "metadata_valid_mask" in batch[0]:
        metadata["metadata_valid_mask"] = torch.stack(
            [torch.as_tensor(item["metadata_valid_mask"], dtype=torch.bool) for item in batch]
        )
        metadata["raw_metadata_valid_mask"] = torch.stack(
            [torch.as_tensor(item["raw_metadata_valid_mask"], dtype=torch.bool) for item in batch]
        )
        metadata["observed_metadata_valid_mask"] = torch.as_tensor(
            [bool(item["observed_metadata_valid"]) for item in batch], dtype=torch.bool
        )
    if batch and "camera_intrinsics" in batch[0]:
        metadata["camera_names"] = list(batch[0].get("camera_names", []))
        metadata["camera_intrinsics"] = torch.stack(
            [torch.from_numpy(item["camera_intrinsics"]).float() for item in batch]
        )
        metadata["camera2ego"] = torch.stack([torch.from_numpy(item["camera2ego"]).float() for item in batch])

    return (
        context_frames,
        actions,
        states,
        extrinsics,
        seg_targets,
        driving_command,
        ego_dynamics,
        agent_boxes,
        agent_mask,
        bev_segmentation,
        proposal_context_frames,
        metadata,
    )


def init_navsim_data(
    data_path: str,
    sensor_blobs_path: str,
    batch_size: int,
    frames_per_clip: int = 20,
    fps: int = 2,
    base_fps: float = 2.0,
    tubelet_size: int = 2,
    transform: Any = None,
    proposal_transform: Any = None,
    num_workers: int = 4,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    rank: int = 0,
    world_size: int = 1,
    camera_name: str = "CAM_F0",
    camera_names: Optional[Sequence[str]] = None,
    max_scenes: Optional[int] = None,
    action_dim: int = 7,
    shuffle: bool = True,
    index_cache: bool = True,
    window_stride: int = 1,
    max_frame_gap: int = 3,
    max_agents: int = NAVSIM_DEFAULT_MAX_AGENTS,
    load_agent_annotations: bool = True,
    image_require_policy: str = IMAGE_REQUIRE_ALL_FRAMES,
    num_observed_frames: Optional[int] = None,
    scene_filter_yaml: Optional[str] = None,
    pose_overlay_path: Optional[str] = None,
    pose_overlay_coord_frame: str = "opencv_first_frame",
    pose_overlay_txt_start_seconds: float = 1.5,
    pose_overlay_required: bool = False,
    tail_seconds: Optional[float] = None,
    counterfactual_tail_seconds: Optional[float] = 5.0,
    window_start_policy: str = WINDOW_START_SLIDING,
    annotations_path: Optional[str] = None,
    annotations_drop_distorted: Optional[bool] = None,
    annotations_require_trajectory_match: bool = False,
    annotations_accident_type_allowlist: Optional[Sequence[str]] = None,
    trajectory_quality_path: Optional[str] = None,
    annotation_selection: str = "all_valid",
    drop_last: bool = True,
    dataset_roots: Optional[Sequence[Dict[str, Any]]] = None,
    balance_dataset_roots: bool = False,
    atomic_real_cf_pairing: bool = False,
    dataset_domain: str = "real",
    dataset_root_name: str = "default",
    counterfactual_supervision_v2: bool = False,
    is_validation: bool = False,
) -> Tuple[DataLoader, "Sampler[int]"]:
    if type(atomic_real_cf_pairing) is not bool:
        raise ValueError("atomic_real_cf_pairing must be an exact boolean")
    roots = list(dataset_roots or [])
    max_agents = resolve_navsim_root_max_agents(
        roots,
        default_max_agents=max_agents,
        field_name="NavSim dataset_roots",
    )

    def _root_tail_seconds(root: Dict[str, Any]) -> Optional[float]:
        if "tail_seconds" in root:
            value = root.get("tail_seconds")
            return None if value is None else float(value)
        if root.get("domain") == "counterfactual":
            return None if counterfactual_tail_seconds is None else float(counterfactual_tail_seconds)
        return None if tail_seconds is None else float(tail_seconds)

    def _build_dataset(root: Optional[Dict[str, Any]] = None, root_index: int = 0) -> RootTaggedDataset:
        is_explicit_root = root is not None
        root = dict(root or {})
        is_formal_root = root.get("effective_runtime_root_schema") == FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA
        if is_formal_root:
            root = validate_formal_v2_navsim_effective_root(root)
        if is_explicit_root and "domain" not in root:
            raise ValueError(f"NavSim dataset root {root_index} requires explicit domain")
        domain = root.get("domain", dataset_domain)
        if domain not in {"real", "counterfactual"}:
            raise ValueError(f"NavSim root domain must be 'real' or 'counterfactual', got {domain!r}")
        root_name = str(root.get("name", dataset_root_name))
        if not root_name:
            raise ValueError(f"NavSim root {root_index} requires a non-empty name")

        def _root_value(field: str, default: Any) -> Any:
            return root[field] if is_formal_root else root.get(field, default)

        if is_formal_root:
            global_timeline = {
                "num_target_frames": frames_per_clip,
                "num_observed_frames": num_observed_frames,
                "fps": fps,
                "base_fps": base_fps,
            }
            for field, global_value in global_timeline.items():
                if root[field] != global_value:
                    raise ValueError(
                        f"Formal-v2 NavSim root {root_name!r}.{field}={root[field]!r} differs from "
                        f"the global runtime value {global_value!r}"
                    )

        root_annotation_selection = _root_value("annotation_selection", annotation_selection)
        root_annotations_path = root.get("annotations_path") if is_explicit_root else annotations_path
        root_drop_distorted = (
            root.get("annotations_drop_distorted") if is_explicit_root else annotations_drop_distorted
        )
        root_require_trajectory_match = (
            root.get("annotations_require_trajectory_match", False)
            if is_explicit_root
            else annotations_require_trajectory_match
        )
        root_accident_type_allowlist = (
            root.get("annotations_accident_type_allowlist")
            if is_explicit_root
            else annotations_accident_type_allowlist
        )
        root_trajectory_quality_path = (
            root.get("trajectory_quality_path") if is_explicit_root else trajectory_quality_path
        )
        if counterfactual_supervision_v2 and domain == "counterfactual":
            if not isinstance(root_annotations_path, str) or not root_annotations_path:
                raise ValueError(
                    f"Counterfactual NavSim root {root_name!r} requires annotations_path under " "cf_supervision_v2"
                )
            if root_drop_distorted is not True:
                raise ValueError(
                    f"Counterfactual NavSim root {root_name!r} requires "
                    "annotations_drop_distorted=true under cf_supervision_v2"
                )
        root_data_path = _root_value("data_path", data_path)
        root_sensor_blobs_path = _root_value("sensor_blobs_path", sensor_blobs_path)
        if not root_data_path or not root_sensor_blobs_path:
            raise ValueError(f"NavSim root requires data_path and sensor_blobs_path, got {root}")
        load_root_agent_annotations = bool(_root_value("load_agent_annotations", load_agent_annotations))
        root_pose_overlay_path = (
            root.get("pose_overlay_path") if is_formal_root else root.get("pose_overlay_path", pose_overlay_path)
        )
        if "pose_overlay_required" in root:
            root_pose_overlay_required = bool(root["pose_overlay_required"])
        else:
            root_pose_overlay_required = bool(pose_overlay_required and root_pose_overlay_path)
        root_num_observed_frames = _root_value("num_observed_frames", num_observed_frames)
        base_dataset = NavSimWorldModelDataset(
            data_path=root_data_path,
            sensor_blobs_path=root_sensor_blobs_path,
            camera_name=_root_value("camera_name", camera_name),
            camera_names=_root_value("camera_names", camera_names),
            frames_per_clip=int(_root_value("num_target_frames", frames_per_clip)),
            fps=int(_root_value("fps", fps)),
            base_fps=_root_value("base_fps", base_fps),
            tubelet_size=tubelet_size,
            transform=transform,
            proposal_transform=proposal_transform,
            max_scenes=_root_value("max_scenes", max_scenes),
            action_dim=action_dim,
            index_cache=bool(root.get("index_cache", index_cache)),
            window_stride=int(_root_value("window_stride", window_stride)),
            max_frame_gap=int(_root_value("max_frame_gap", max_frame_gap)),
            max_agents=int(_root_value("max_agents", max_agents)),
            load_agent_annotations=load_root_agent_annotations,
            image_require_policy=_root_value("image_require_policy", image_require_policy),
            num_observed_frames=(None if root_num_observed_frames is None else int(root_num_observed_frames)),
            scene_filter_yaml=(
                root.get("scene_filter_yaml") if is_formal_root else root.get("scene_filter_yaml", scene_filter_yaml)
            ),
            pose_overlay_path=root_pose_overlay_path,
            pose_overlay_coord_frame=root.get("pose_overlay_coord_frame", pose_overlay_coord_frame),
            pose_overlay_txt_start_seconds=root.get("pose_overlay_txt_start_seconds", pose_overlay_txt_start_seconds),
            pose_overlay_required=root_pose_overlay_required,
            tail_seconds=_root_tail_seconds(root),
            window_start_policy=_root_value("window_start_policy", window_start_policy),
            timestamp_policy=_root_value("timestamp_policy", None),
            annotations_path=root_annotations_path,
            annotations_drop_distorted=root_drop_distorted,
            annotations_require_trajectory_match=root_require_trajectory_match,
            annotations_accident_type_allowlist=root_accident_type_allowlist,
            trajectory_quality_path=root_trajectory_quality_path,
            annotation_selection=root_annotation_selection,
            is_validation=is_validation,
        )
        return RootTaggedDataset(
            base_dataset,
            domain=domain,
            dataset_root_name=root_name,
            dataset_root_index=root_index,
            future_agent_geometry_valid=domain == "real" and load_root_agent_annotations,
            load_agent_annotations=load_root_agent_annotations,
            annotation_selection=root_annotation_selection,
        )

    if roots:
        datasets = []
        root_names = []
        root_repeats = []
        for idx, root in enumerate(roots):
            dataset = _build_dataset(root, root_index=idx)
            datasets.append(dataset)
            root_names.append(str(root.get("name", idx)))
            root_repeats.append(int(root.get("repeat", 1)))
            logger.info(
                "NavSim mixed root ready: name=%s repeat=%d data=%s blobs=%s windows=%d",
                root_names[-1],
                root_repeats[-1],
                root.get("data_path", data_path),
                root.get("sensor_blobs_path", sensor_blobs_path),
                len(dataset),
            )
        if len(datasets) == 1:
            dataset = datasets[0]
        elif atomic_real_cf_pairing:
            if not any(root.annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION for root in datasets):
                raise ValueError(
                    "atomic_real_cf_pairing=true requires the Formal-v2 counterfactual accident allowlist"
                )
            if not balance_dataset_roots:
                raise ValueError("Formal-v2 matched real/CF roots require balance_dataset_roots=true")
            if root_repeats != [1, 1]:
                raise ValueError("Formal-v2 matched real/CF roots require exact 1:1 root repeats")
            dataset = MatchedRealCounterfactualPairDataset(datasets)
            logger.info(
                "Using atomic matched NavSim real/CF pairs: pairs=%d effective_samples=%d",
                len(dataset),
                2 * len(dataset),
            )
        elif balance_dataset_roots:
            dataset = BalancedRootConcatDataset(datasets, root_repeats=root_repeats)
            logger.info(
                "Using balanced NavSim mixed roots: roots=%d repeats=%s virtual_windows=%d root_lengths=%s",
                len(datasets),
                root_repeats,
                len(dataset),
                [len(ds) for ds in datasets],
            )
        else:
            dataset = ConcatDataset(datasets)
            logger.info(
                "Using concatenated NavSim mixed roots: roots=%d windows=%d root_lengths=%s",
                len(datasets),
                len(dataset),
                [len(ds) for ds in datasets],
            )
    else:
        dataset = _build_dataset()

    atomic_pairs = isinstance(dataset, MatchedRealCounterfactualPairDataset)
    if atomic_pairs and (type(batch_size) is not int or batch_size < 2 or batch_size % 2):
        raise ValueError("Formal-v2 atomic real/CF pairing requires an even batch_size >= 2")
    dataloader_batch_size = batch_size // 2 if atomic_pairs else batch_size

    if is_validation:
        if shuffle or drop_last:
            raise ValueError("NavSim validation requires shuffle=False and drop_last=False")
        from app.vjepa_cowa_world_model.training.samplers import ExactDistributedEvalSampler

        sampler = ExactDistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    # 可复现的数据增强 RNG：worker_init_fn 给每个 worker 播种 numpy/random；generator 种子继承
    # setup_distributed 设的全局 torch 种子（config.meta.seed）并按 rank 区分，使重启后增强序列一致。
    from app.vjepa_cowa_world_model.training.loop import make_dataloader_generator, seed_dataloader_worker

    loader = DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        collate_fn=navsim_world_model_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=False if is_validation else drop_last,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(
            rank=rank,
            stream="navsim/validation" if is_validation else "navsim/train",
        ),
    )

    logger.info(
        "NavSim dataloader created: batches=%d, batch_size=%d, workers=%d",
        len(loader),
        dataloader_batch_size,
        num_workers,
    )
    return loader, sampler
