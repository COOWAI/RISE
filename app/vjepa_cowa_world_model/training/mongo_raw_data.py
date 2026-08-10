"""Mongo-backed raw clip dataloader for world-model training."""

import fcntl
import hashlib
import json
import os
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import libclip_container
import numpy as np
import torch
from pymongo import MongoClient
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_cowa_world_model.training import collate

if TYPE_CHECKING:
    from torch.utils.data import Sampler

from src.utils.logging import get_logger

from .config import MongoRawConfig

logger = get_logger(__name__)


DEFAULT_CAMERA_NAMES = {
    "/camera/panorama/1/h264": "CAM_BACK_LEFT",
    "/camera/panorama/2/h264": "CAM_FRONT_LEFT",
    "/camera/panorama/3/h264": "CAM_FRONT",
    "/camera/panorama/4/h264": "CAM_FRONT_RIGHT",
    "/camera/panorama/5/h264": "CAM_BACK_RIGHT",
    "/camera/stereo/back/1/h264": "CAM_STEREO_BACK_1",
    "/camera/stereo/back/2/h264": "CAM_STEREO_BACK_2",
    "/camera/stereo/left/1/h264": "CAM_STEREO_LEFT_1",
    "/camera/stereo/left/2/h264": "CAM_STEREO_LEFT_2",
    "/camera/stereo/right/1/h264": "CAM_STEREO_RIGHT_1",
    "/camera/stereo/right/2/h264": "CAM_STEREO_RIGHT_2",
    "/camera/stereo/front/1/h264": "CAM_STEREO_FRONT_1",
    "/camera/stereo/front/2/h264": "CAM_STEREO_FRONT_2",
    "/camera/surround/front/h264": "CAM_SUR_FRONT",
    "/camera/surround/left/h264": "CAM_SUR_LEFT",
    "/camera/surround/right/h264": "CAM_SUR_RIGHT",
    "/camera/surround/back/h264": "CAM_SUR_BACK",
}


@dataclass
class ClipRecord:
    clip_id: str
    source_id: Optional[str]
    data_path: str


@dataclass
class PreparedClip:
    clip_id: str
    reader: Any
    pose_provider: Any
    main_channel: Any
    selected_camera_topics: List[str]
    camera_channels: Dict[str, Any]
    sampled_main_indices: List[int]
    sampled_camera_indices_by_topic: Dict[str, List[int]]
    states: np.ndarray
    extrinsic_7d_by_topic: Dict[str, np.ndarray]


def _normalize_camera_topic(raw_topic: str) -> str:
    topic = raw_topic.strip()
    if not topic:
        raise ValueError("camera topic cannot be empty")
    if not topic.startswith("/"):
        topic = f"/{topic}"
    if not topic.startswith("/camera/"):
        topic = f"/camera{topic}"
    if not topic.endswith("/h264"):
        topic = f"{topic}/h264"
    return topic


def _process_transform(transform: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = transform[:3]
    roll, pitch, yaw = transform[3:]
    t_cam2ego = np.asarray([x, y, z], dtype=np.float32)
    r_cam2ego = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix().astype(np.float32)
    return t_cam2ego, r_cam2ego


def _compute_velocity_from_positions(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros(0, dtype=np.float32)

    ts = np.asarray(timestamps, dtype=np.float64)
    if ts[0] > 1e15:
        ts = ts / 1e9
    elif ts[0] > 1e12:
        ts = ts / 1e6
    elif ts[0] > 1e9:
        ts = ts / 1e3

    velocities = np.zeros(len(positions), dtype=np.float32)
    for idx in range(len(positions) - 1):
        dt = ts[idx + 1] - ts[idx]
        if dt <= 0 or dt > 1.0:
            continue
        distance = float(np.linalg.norm(positions[idx + 1] - positions[idx]))
        velocities[idx] = distance / float(dt)
    if len(velocities) > 1:
        velocities[-1] = velocities[-2]
    return velocities


def _get_match_dict(
    matched_channel,
    selected_topics: Sequence[str],
    main_length: int,
) -> Dict[int, List[Optional[int]]]:
    matched_dict: Dict[int, List[Optional[int]]] = defaultdict(list)
    for camera_topic in selected_topics:
        for matched_proto in matched_channel:
            if matched_proto.camera != camera_topic:
                continue
            matched_index = 0
            matched_length = len(matched_proto.pairs)
            for main_index in range(main_length):
                if matched_index >= matched_length:
                    matched_dict[main_index].append(None)
                elif main_index == matched_proto.pairs[matched_index].lidar:
                    matched_dict[main_index].append(matched_proto.pairs[matched_index].camera)
                    matched_index += 1
                elif main_index < matched_proto.pairs[matched_index].lidar:
                    matched_dict[main_index].append(None)

    invalid_keys = [key for key, value in matched_dict.items() if None in value]
    for key in invalid_keys:
        matched_dict.pop(key, None)
    return matched_dict


def _resolve_clip_path(doc: Dict[str, Any], cfg: MongoRawConfig) -> Optional[str]:
    data_path = doc.get("data_path")
    if not data_path:
        return None
    storage = doc.get("storage")
    if storage not in (None, "", "default", "e2e", "clipdata"):
        return None
    if storage == "e2e":
        prefix = cfg.e2e_storage_root
    elif storage == "clipdata":
        prefix = cfg.clipdata_storage_root
    else:
        prefix = cfg.default_storage_root
    return os.path.join(prefix, data_path.lstrip("/"))


def _build_query_filter(cfg: MongoRawConfig) -> Dict[str, Any]:
    query_filter: Dict[str, Any] = dict(cfg.query_filter)
    if cfg.require_latest_available_revision:
        query_filter["latest_available_revision"] = {"$exists": True}
    if "vehicle.type" in query_filter:
        return query_filter

    vehicle_types = [vehicle_type for vehicle_type in cfg.vehicle_types if vehicle_type]
    if len(vehicle_types) == 1:
        query_filter["vehicle.type"] = vehicle_types[0]
    elif len(vehicle_types) > 1:
        query_filter["vehicle.type"] = {"$in": vehicle_types}
    elif cfg.vehicle_type:
        query_filter["vehicle.type"] = cfg.vehicle_type
    return query_filter


def _split_records(records: Sequence[ClipRecord], cfg: MongoRawConfig, split: str) -> List[ClipRecord]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    val_ratio = min(max(float(cfg.val_ratio), 0.0), 1.0)
    if val_ratio <= 0.0:
        return list(records) if split == "train" else []
    if val_ratio >= 1.0:
        return list(records) if split == "val" else []

    train_records: List[ClipRecord] = []
    val_records: List[ClipRecord] = []
    threshold = int(val_ratio * 10000)
    for record in records:
        digest = hashlib.md5(f"{record.clip_id}:{cfg.split_seed}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        if bucket < threshold:
            val_records.append(record)
        else:
            train_records.append(record)

    selected = train_records if split == "train" else val_records
    max_count = cfg.max_val_clips if split == "val" else cfg.max_clips
    if max_count is not None:
        selected = selected[: int(max_count)]
    return selected


def _make_cache_key(cfg: MongoRawConfig) -> str:
    """根据查询相关配置生成 deterministic hash 作为缓存文件名的一部分。"""
    key_parts = {
        "database": cfg.database,
        "collection": cfg.collection,
        "query_filter": str(_build_query_filter(cfg)),
        "start_index": cfg.start_index,
        "end_index": cfg.end_index,
        "default_storage_root": cfg.default_storage_root,
        "e2e_storage_root": cfg.e2e_storage_root,
        "clipdata_storage_root": cfg.clipdata_storage_root,
        "val_ratio": cfg.val_ratio,
        "split_seed": cfg.split_seed,
        # the cache stores the TRUNCATED split (_split_records applies these), so they must key it —
        # else changing the limits silently reuses a wrongly-truncated cached split.
        "max_clips": cfg.max_clips,
        "max_val_clips": cfg.max_val_clips,
    }
    raw = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_cache_path(cfg: MongoRawConfig, split: str) -> Optional[str]:
    if not cfg.record_cache_dir:
        return None
    cache_key = _make_cache_key(cfg)
    return os.path.join(cfg.record_cache_dir, f"mongo_records_{cache_key}_{split}.json")


def _load_cached_records(cfg: MongoRawConfig, split: str) -> Optional[List[ClipRecord]]:
    """尝试从本地缓存加载 records，缓存过期或不存在则返回 None。"""
    cache_path = _get_cache_path(cfg, split)
    if cache_path is None or not os.path.exists(cache_path):
        return None
    try:
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime > cfg.record_cache_ttl:
            logger.info(
                "Record cache expired (age=%.0fs, ttl=%ds): %s", time.time() - mtime, cfg.record_cache_ttl, cache_path
            )
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [
            ClipRecord(
                clip_id=item["clip_id"],
                source_id=item.get("source_id"),
                data_path=item["data_path"],
            )
            for item in data
        ]
        logger.info("Loaded %d records from cache (split=%s): %s", len(records), split, cache_path)
        return records
    except Exception as exc:
        logger.warning("Failed to load record cache %s: %s", cache_path, exc)
        return None


def _save_cached_records(cfg: MongoRawConfig, split: str, records: List[ClipRecord]) -> None:
    """将 records 保存到本地缓存文件。"""
    cache_path = _get_cache_path(cfg, split)
    if cache_path is None:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = [{"clip_id": r.clip_id, "source_id": r.source_id, "data_path": r.data_path} for r in records]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Saved %d records to cache (split=%s): %s", len(records), split, cache_path)
    except Exception as exc:
        logger.warning("Failed to save record cache %s: %s", cache_path, exc)


def _load_blacklist(path: Optional[str]) -> set:
    """从 JSON 文件加载坏 clip ID 集合。

    文件格式: {"clip_ids": ["id1", "id2", ...], ...}
    文件不存在或解析失败返回空集。
    """
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        clip_ids = set(data.get("clip_ids", []))
        if clip_ids:
            logger.info("Loaded blacklist with %d clip IDs from %s", len(clip_ids), path)
        return clip_ids
    except Exception as exc:
        logger.warning("Failed to load blacklist %s: %s", path, exc)
        return set()


def _append_to_blacklist(path: Optional[str], clip_id: str) -> None:
    """原子地将一个坏 clip ID 追加到黑名单 JSON 文件。

    使用 fcntl.flock 文件锁保证多 worker 并发写安全。
    异常时仅 log warning，不中断训练。
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 使用排他锁保证并发安全
        with open(path, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                else:
                    data = {"clip_ids": []}
                existing = set(data.get("clip_ids", []))
                if clip_id in existing:
                    return  # 已存在，无需重复添加
                existing.add(clip_id)
                data["clip_ids"] = sorted(existing)
                data["updated_at"] = datetime.now(timezone.utc).isoformat()
                f.seek(0)
                f.truncate()
                json.dump(data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        logger.warning("Failed to append clip %s to blacklist %s: %s", clip_id, path, exc)


def query_mongo_clip_records(cfg: MongoRawConfig, split: str) -> List[ClipRecord]:
    # 尝试从本地缓存加载
    cached = _load_cached_records(cfg, split)
    if cached is not None:
        return cached

    mongo_uri = cfg.mongo_uri
    if not mongo_uri and cfg.mongo_uri_env:
        mongo_uri = os.environ.get(cfg.mongo_uri_env)
    if not mongo_uri:
        raise ValueError("data.mongo_raw.mongo_uri or data.mongo_raw.mongo_uri_env must be configured")

    query_filter = _build_query_filter(cfg)
    logger.info("Mongo query filter: %s", query_filter)
    client = MongoClient(mongo_uri)
    try:
        collection = client[cfg.database][cfg.collection]
        cursor = collection.find(
            query_filter,
            {
                "source_id": 1,
                "data_path": 1,
                "storage": 1,
            },
        ).sort("_id", 1)

        docs = list(cursor)
    finally:
        client.close()

    start_index = max(0, int(cfg.start_index))
    end_index = cfg.end_index if cfg.end_index is None else max(start_index, int(cfg.end_index))
    docs = docs[start_index:end_index]

    records: List[ClipRecord] = []
    missing_paths = 0
    missing_files = 0
    unknown_storage = 0
    for doc in docs:
        clip_id = str(doc.get("_id", ""))
        resolved_path = _resolve_clip_path(doc, cfg)
        if not clip_id:
            continue
        if not doc.get("data_path"):
            missing_paths += 1
            continue
        if not resolved_path:
            unknown_storage += 1
            continue
        if not os.path.exists(resolved_path):
            missing_files += 1
            continue
        records.append(
            ClipRecord(
                clip_id=clip_id,
                source_id=str(doc.get("source_id")) if doc.get("source_id") is not None else None,
                data_path=resolved_path,
            )
        )

    split_records = _split_records(records, cfg, split=split)
    logger.info(
        (
            "Mongo raw clip query complete: fetched=%d, valid=%d, missing_paths=%d, "
            "missing_files=%d, unknown_storage=%d, split=%s, selected=%d"
        ),
        len(docs),
        len(records),
        missing_paths,
        missing_files,
        unknown_storage,
        split,
        len(split_records),
    )

    # 保存到本地缓存
    _save_cached_records(cfg, split, split_records)

    return split_records


class MongoRawWorldModelDataset(Dataset):
    """World-model dataset that reads raw clips online."""

    def __init__(
        self,
        mongo_cfg: MongoRawConfig,
        split: str,
        frames_per_clip: int,
        fps: int,
        tubelet_size: int,
        transform: Any = None,
        action_dim: int = 3,
        records: Optional[List[ClipRecord]] = None,
        is_validation: bool = False,
    ):
        self.mongo_cfg = mongo_cfg
        self.split = split
        self.frames_per_clip = int(frames_per_clip)
        self.fps = int(max(1, fps))
        self.tubelet_size = int(max(1, tubelet_size))
        self.transform = transform
        self.action_dim = int(action_dim)
        self.is_validation = bool(is_validation)
        self.max_retries = max(1, int(mongo_cfg.max_retries))

        if not mongo_cfg.camera_topics:
            raise ValueError("data.mongo_raw.camera_topics must contain at least one raw camera topic")

        self.camera_topics = [_normalize_camera_topic(topic) for topic in mongo_cfg.camera_topics]
        self.camera_name_map = dict(DEFAULT_CAMERA_NAMES)
        self.camera_name_map.update(mongo_cfg.extra_camera_mappings)

        self.base_fps = int(max(1, mongo_cfg.base_fps))
        self.source_fps = int(max(1, mongo_cfg.source_fps))
        self.sample_step = max(1, self.base_fps // self.fps)
        self.min_base_frames = 1 + (self.frames_per_clip - 1) * self.sample_step

        if records is not None:
            self.records = records
        else:
            self.records = query_mongo_clip_records(mongo_cfg, split=split)
        if not self.records:
            raise ValueError(f"No Mongo raw clips found for split={split}")

        self._clip_cache: "OrderedDict[str, PreparedClip]" = OrderedDict()
        self._failed_clip_ids: set = set()

        logger.info(
            (
                "Mongo raw dataset ready: split=%s, clips=%d, base_fps=%d, "
                "target_fps=%d, sample_step=%d, camera_topics=%s"
            ),
            split,
            len(self.records),
            self.base_fps,
            self.fps,
            self.sample_step,
            self.camera_topics,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_index = int(index)
        for retry in range(self.max_retries):
            record = self.records[record_index]
            if record.clip_id in self._failed_clip_ids:
                if self.is_validation:
                    raise RuntimeError(
                        f"Validation clip {record.clip_id!r} was marked failed before bounded same-index retry"
                    )
                record_index = random.randint(0, len(self.records) - 1)
                continue
            try:
                prepared = self._get_prepared_clip(record)
                return self._build_sample(prepared)
            except Exception as exc:
                self._clip_cache.pop(record.clip_id, None)
                if not self.is_validation:
                    self._failed_clip_ids.add(record.clip_id)
                    _append_to_blacklist(self.mongo_cfg.blacklist_path, record.clip_id)
                if retry == self.max_retries - 1:
                    raise
                logger.warning(
                    "Mongo raw sample load failed (clip=%s, retry=%d/%d): %s",
                    record.clip_id,
                    retry + 1,
                    self.max_retries,
                    str(exc),
                )
                if not self.is_validation:
                    record_index = random.randint(0, len(self.records) - 1)

        raise RuntimeError("Unreachable Mongo raw dataset retry state")

    def _get_prepared_clip(self, record: ClipRecord) -> PreparedClip:
        cached = self._clip_cache.get(record.clip_id)
        if cached is not None:
            self._clip_cache.move_to_end(record.clip_id)
            return cached

        prepared = self._prepare_clip(record)
        self._clip_cache[record.clip_id] = prepared
        self._clip_cache.move_to_end(record.clip_id)
        if len(self._clip_cache) > max(1, int(self.mongo_cfg.cache_size)):
            self._clip_cache.popitem(last=False)
        return prepared

    def _prepare_clip(self, record: ClipRecord) -> PreparedClip:
        if not os.path.exists(record.data_path):
            raise FileNotFoundError(f"Raw clip file does not exist: {record.data_path}")

        reader = libclip_container.DataClipReader(record.data_path)
        reader.ReadDriveData()
        all_topics = reader.ListTopics()

        if self.mongo_cfg.pose_topic not in all_topics:
            raise ValueError(f"Pose topic not found in clip {record.clip_id}: {self.mongo_cfg.pose_topic}")
        if self.mongo_cfg.match_topic not in all_topics:
            raise ValueError(f"Match topic not found in clip {record.clip_id}: {self.mongo_cfg.match_topic}")
        if self.mongo_cfg.main_topic not in all_topics:
            raise ValueError(f"Main topic not found in clip {record.clip_id}: {self.mongo_cfg.main_topic}")

        selected_camera_topics = [topic for topic in self.camera_topics if topic in all_topics]
        if len(selected_camera_topics) < len(self.camera_topics):
            missing = sorted(set(self.camera_topics) - set(selected_camera_topics))
            raise ValueError(f"Missing required camera topics for clip {record.clip_id}: {missing}")

        pose_provider = libclip_container.PoseProvider(reader.GetChannel(self.mongo_cfg.pose_topic))
        main_channel = reader.GetChannel(self.mongo_cfg.main_topic)
        matched_channel = reader.GetChannel(self.mongo_cfg.match_topic)
        main_length = len(main_channel)
        if main_length == 0:
            raise ValueError(f"Empty main channel for clip {record.clip_id}")

        match_dict = _get_match_dict(matched_channel, selected_camera_topics, main_length)
        matched_main_indices = sorted(match_dict.keys())
        if len(matched_main_indices) < self.min_base_frames:
            raise ValueError(
                (
                    f"Clip {record.clip_id} has insufficient matched frames: "
                    f"{len(matched_main_indices)} < {self.min_base_frames}"
                )
            )

        source_to_base = max(1, self.source_fps // self.base_fps)
        sampled_positions = list(range(0, len(matched_main_indices), source_to_base))
        sampled_main_indices = [
            matched_main_indices[pos] for pos in sampled_positions if pos < len(matched_main_indices)
        ]
        if len(sampled_main_indices) < self.min_base_frames:
            raise ValueError(
                (
                    f"Clip {record.clip_id} has insufficient base-fps frames: "
                    f"{len(sampled_main_indices)} < {self.min_base_frames}"
                )
            )

        camera_channels = {topic: reader.GetChannel(topic) for topic in selected_camera_topics}
        extrinsic_7d_by_topic: Dict[str, np.ndarray] = {}
        for camera_topic in selected_camera_topics:
            camera_channel = camera_channels[camera_topic]
            if len(camera_channel) == 0:
                raise ValueError(f"Camera channel is empty for clip {record.clip_id}: {camera_topic}")

            transform = camera_channel[0].transform
            if transform is None:
                raise ValueError(f"Camera transform missing for clip {record.clip_id}: {camera_topic}")
            t_cam2ego, r_cam2ego = _process_transform(transform)
            euler = Rotation.from_matrix(r_cam2ego).as_euler("xyz").astype(np.float32)
            extrinsic_7d_by_topic[camera_topic] = np.concatenate(
                [t_cam2ego, euler, np.asarray([0.0], dtype=np.float32)], axis=0
            )

        sampled_camera_indices_by_topic: Dict[str, List[int]] = {topic: [] for topic in selected_camera_topics}
        positions: List[np.ndarray] = []
        timestamps: List[int] = []
        states: List[np.ndarray] = []
        for main_index in sampled_main_indices:
            camera_indices = match_dict.get(main_index)
            if not camera_indices:
                raise ValueError(f"No camera indices found for main index {main_index} in clip {record.clip_id}")

            for topic, camera_index in zip(selected_camera_topics, camera_indices):
                if camera_index is None:
                    raise ValueError(f"Missing matched camera index for clip {record.clip_id}, topic {topic}")
                sampled_camera_indices_by_topic[topic].append(int(camera_index))

            main_timestamp = main_channel[main_index].timestamp
            pose = pose_provider.GetTransforms(main_timestamp)
            # Keep float64 for precision (matches offline H5 pipeline)
            translation = pose[:3, 3].copy()  # float64
            euler_pose = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")  # float64
            positions.append(translation)
            timestamps.append(int(main_timestamp))
            states.append(np.concatenate([translation, euler_pose, np.asarray([0.0], dtype=np.float64)], axis=0))

        states_np_f64 = np.stack(states, axis=0)  # float64
        velocities = _compute_velocity_from_positions(
            np.stack(positions, axis=0),
            np.asarray(timestamps, dtype=np.int64),
        )
        states_np_f64[:, 6] = velocities
        # Cast to float32 only at storage time
        states_np = states_np_f64.astype(np.float32)

        return PreparedClip(
            clip_id=record.clip_id,
            reader=reader,
            pose_provider=pose_provider,
            main_channel=main_channel,
            selected_camera_topics=selected_camera_topics,
            camera_channels=camera_channels,
            sampled_main_indices=sampled_main_indices,
            sampled_camera_indices_by_topic=sampled_camera_indices_by_topic,
            states=states_np,
            extrinsic_7d_by_topic=extrinsic_7d_by_topic,
        )

    def _sample_base_window_start(self, prepared: PreparedClip) -> int:
        max_start = len(prepared.sampled_main_indices) - self.min_base_frames
        if max_start <= 0:
            return 0
        if self.split == "train":
            return random.randint(0, max_start)
        return max_start // 2

    def _decode_clip_frames(
        self,
        prepared: PreparedClip,
        camera_topic: str,
        base_window_indices: Sequence[int],
    ) -> np.ndarray:
        camera_channel = prepared.camera_channels[camera_topic]
        images: List[np.ndarray] = []

        for base_idx in base_window_indices:
            camera_index = prepared.sampled_camera_indices_by_topic[camera_topic][base_idx]
            image = camera_channel.GetMat(camera_index)
            if image is None:
                raise ValueError(
                    f"GetMat returned None for camera_index={camera_index} "
                    f"in clip {prepared.clip_id}, topic={camera_topic}"
                )
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(f"Unexpected image shape from raw clip: {image.shape}")
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image = image[:, :, ::-1].copy()
            images.append(image)

        return np.stack(images, axis=0)

    def _build_actions(self, states: np.ndarray) -> np.ndarray:
        if states.shape[0] < 2:
            raise ValueError("Need at least 2 states to build actions")
        return self._build_actions_3d(states)

    @staticmethod
    def _build_actions_3d(states: np.ndarray) -> np.ndarray:
        """Build 3D actions: [dx, dy, d_yaw].

        Mongo Raw states are already in local coordinates, so position diffs
        are directly used without additional rotation.
        """
        actions = np.zeros((states.shape[0] - 1, 3), dtype=np.float32)
        for idx in range(states.shape[0] - 1):
            dx = states[idx + 1, 0] - states[idx, 0]
            dy = states[idx + 1, 1] - states[idx, 1]
            d_yaw = states[idx + 1, 5] - states[idx, 5]
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
            actions[idx] = np.asarray([dx, dy, d_yaw], dtype=np.float32)
        return actions

    def _build_sample(self, prepared: PreparedClip) -> Dict[str, Any]:
        start = self._sample_base_window_start(prepared)
        positions = start + np.arange(self.frames_per_clip) * self.sample_step
        base_window_indices = [int(pos) for pos in positions]
        if self.split == "train":
            camera_topic = random.choice(prepared.selected_camera_topics)
        else:
            camera_topic = prepared.selected_camera_topics[0]

        if base_window_indices[-1] >= len(prepared.sampled_main_indices):
            raise ValueError(
                (
                    f"Window exceeds prepared clip length for clip {prepared.clip_id}: "
                    f"{base_window_indices[-1]} >= {len(prepared.sampled_main_indices)}"
                )
            )

        buffer = self._decode_clip_frames(prepared, camera_topic, base_window_indices)
        if self.transform is not None:
            buffer_tensor = self.transform(buffer)
            if not torch.is_tensor(buffer_tensor):
                buffer_tensor = torch.as_tensor(buffer_tensor)
        else:
            buffer_tensor = torch.from_numpy(buffer).permute(3, 0, 1, 2).float() / 255.0

        states = prepared.states[base_window_indices].astype(np.float32)
        actions = self._build_actions(states)
        extrinsics = np.tile(
            prepared.extrinsic_7d_by_topic[camera_topic][None, :],
            (len(base_window_indices), 1),
        ).astype(np.float32)

        return {
            "buffer": buffer_tensor,
            "actions": actions,
            "states": states,
            "extrinsics": extrinsics,
            "indices": np.asarray(base_window_indices, dtype=np.int64),
            "seg_masks": None,
            "seg_frame_indices": None,
            "stable_sample_id": collate.make_stable_sample_id(
                "mongo_raw",
                self.split,
                prepared.clip_id,
                start,
            ),
            "sample_id": collate.make_stable_sample_id(
                "mongo_raw",
                self.split,
                prepared.clip_id,
                start,
            ),
        }


def mongo_raw_world_model_collate_fn(batch: Sequence[Dict[str, Any]]):
    context_frames, actions, states, extrinsics = collate.stack_core(batch)
    seg_targets = [None for _ in batch]
    metadata = collate.build_stable_sample_metadata(batch)
    return (
        context_frames,
        actions,
        states,
        extrinsics,
        seg_targets,
        None,
        None,
        None,
        None,
        None,
        None,
        metadata,
    )


def init_mongo_raw_data(
    mongo_cfg: MongoRawConfig,
    split: str,
    batch_size: int,
    frames_per_clip: int,
    fps: int,
    tubelet_size: int,
    transform: Any = None,
    num_workers: int = 4,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    rank: int = 0,
    world_size: int = 1,
    action_dim: int = 3,
    shuffle: bool = True,
    drop_last: bool = True,
    is_validation: bool = False,
) -> Tuple[DataLoader, "Sampler[int]"]:
    # Only rank 0 queries MongoDB and checks file existence, then broadcast to all ranks.
    if world_size > 1:
        import torch.distributed as dist

        if rank == 0:
            records = query_mongo_clip_records(mongo_cfg, split=split)
            record_data = [(r.clip_id, r.source_id, r.data_path) for r in records]
        else:
            record_data = None
        broadcast_list = [record_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        record_data = broadcast_list[0]
        records = [ClipRecord(clip_id=cid, source_id=sid, data_path=dp) for cid, sid, dp in record_data]
    else:
        records = query_mongo_clip_records(mongo_cfg, split=split)

    # 预过滤已知坏 clip
    blacklisted = _load_blacklist(mongo_cfg.blacklist_path)
    if blacklisted:
        before = len(records)
        records = [r for r in records if r.clip_id not in blacklisted]
        removed = before - len(records)
        if removed > 0:
            logger.info(
                "Blacklist filtered: %d -> %d clips (removed %d, split=%s)",
                before,
                len(records),
                removed,
                split,
            )

    dataset = MongoRawWorldModelDataset(
        mongo_cfg=mongo_cfg,
        split=split,
        frames_per_clip=frames_per_clip,
        fps=fps,
        tubelet_size=tubelet_size,
        transform=transform,
        action_dim=action_dim,
        records=records,
        is_validation=is_validation,
    )

    if is_validation:
        if split != "val":
            raise ValueError(f"Mongo raw exact validation sampler requires split='val', got {split!r}")
        if shuffle or drop_last:
            raise ValueError("Mongo raw validation requires shuffle=False and drop_last=False")
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

    # 可复现的数据增强 RNG（见 navsim_data 同处说明）：worker_init_fn 播种 numpy/random，
    # generator 种子继承全局 torch 种子（config.meta.seed）并按 rank 区分。
    from app.vjepa_cowa_world_model.training.loop import make_dataloader_generator, seed_dataloader_worker

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=mongo_raw_world_model_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=False if is_validation else drop_last,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(
            rank=rank,
            stream="mongo_raw/validation" if is_validation else "mongo_raw/train",
        ),
    )

    logger.info(
        "Mongo raw dataloader created: split=%s, batches=%d, batch_size=%d, workers=%d",
        split,
        len(loader),
        batch_size,
        num_workers,
    )
    return loader, sampler
