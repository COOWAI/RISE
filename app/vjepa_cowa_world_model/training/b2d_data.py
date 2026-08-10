"""Bench2Drive dataloader utilities for world-model training."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_cowa_world_model.training import collate

if TYPE_CHECKING:
    from torch.utils.data import Sampler

from app.vjepa_cowa_world_model.utils.metrics import BEV_SIZE, _rasterize_agents_to_bev
from src.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_REQUIRE_ALL_FRAMES = "all_frames"
IMAGE_REQUIRE_OBSERVED_ONLY = "observed_only"
VALID_IMAGE_REQUIRE_POLICIES = {IMAGE_REQUIRE_ALL_FRAMES, IMAGE_REQUIRE_OBSERVED_ONLY}


def get_b2d_index_cache_path(
    *,
    ann_file: str,
    data_root: str,
    camera_name: str,
    frames_per_clip: int,
    fps: int,
    base_fps: int,
    tubelet_size: int,
    max_scenes: Optional[int],
    verify_image_exists: bool,
    image_require_policy: str = IMAGE_REQUIRE_ALL_FRAMES,
    num_observed_frames: Optional[int] = None,
    index_cache: bool = True,
    index_cache_dir: Optional[str] = None,
) -> Optional[str]:
    """Return the Bench2Drive scene-index cache path for a dataset config."""
    if not index_cache:
        return None

    ann_stat = os.stat(ann_file)
    image_policy = str(image_require_policy or IMAGE_REQUIRE_ALL_FRAMES).lower()
    observed_frames = int(num_observed_frames) if num_observed_frames is not None else int(frames_per_clip)
    fingerprint_data = json.dumps(
        {
            "ann_file": os.path.abspath(ann_file),
            "ann_size": ann_stat.st_size,
            "ann_mtime_ns": ann_stat.st_mtime_ns,
            "data_root": os.path.abspath(data_root),
            "camera_name": camera_name,
            "frames_per_clip": int(frames_per_clip),
            "fps": int(fps),
            "base_fps": int(base_fps),
            "tubelet_size": int(tubelet_size),
            "max_scenes": max_scenes,
            "verify_image_exists": bool(verify_image_exists),
            "image_require_policy": image_policy,
            "num_observed_frames": observed_frames,
        },
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    cache_dir = index_cache_dir or os.environ.get("B2D_INDEX_CACHE_DIR") or os.path.dirname(os.path.abspath(ann_file))
    return os.path.join(cache_dir, f".b2d_world_model_index_cache_{fingerprint}.pkl")


@dataclass
class B2DSceneRecord:
    scene_name: str
    valid_indices: List[int]
    frame_indices: List[int]
    valid_image_frame_indices: List[int]


@dataclass
class B2DWindowRecord:
    scene_idx: int
    start_pos: int


class Bench2DriveWorldModelDataset(Dataset):
    """Dataset that adapts Bench2Drive PKL indices to the world-model batch contract."""

    def __init__(
        self,
        ann_file: str,
        data_root: str,
        camera_name: str = "CAM_FRONT",
        frames_per_clip: int = 20,
        fps: int = 2,
        base_fps: int = 10,
        tubelet_size: int = 1,
        transform: Any = None,
        proposal_transform: Any = None,
        max_scenes: Optional[int] = None,
        action_dim: int = 3,
        window_stride: int = 1,
        max_frame_gap: int = 6,
        max_agents: int = 50,
        load_agent_annotations: bool = True,
        command_dim: int = 6,
        index_cache: bool = True,
        index_cache_dir: Optional[str] = None,
        verify_image_exists: bool = False,
        image_require_policy: str = IMAGE_REQUIRE_ALL_FRAMES,
        num_observed_frames: Optional[int] = None,
        max_load_retries: int = 5,
        index_cache_wait_seconds: int = 300,
        rank: int = 0,
        is_validation: bool = False,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.camera_name = camera_name
        self.frames_per_clip = int(frames_per_clip)
        self.fps = int(max(1, fps))
        self.base_fps = int(max(1, base_fps))
        self.tubelet_size = int(max(1, tubelet_size))
        self.transform = transform
        self.proposal_transform = proposal_transform
        self.max_scenes = max_scenes
        self.action_dim = int(action_dim)
        self.window_stride = int(max(1, window_stride))
        self.sample_step = max(1, int(round(float(self.base_fps) / float(self.fps))))
        self.max_frame_gap = int(max(max_frame_gap, self.sample_step))
        self.max_agents = int(max_agents)
        self.load_agent_annotations = bool(load_agent_annotations)
        self.command_dim = int(max(4, command_dim))
        # 条件必需特征缺失时的 warning 只打一次（per worker），避免逐样本刷屏。
        self._warned_missing_command = False
        self._warned_missing_dynamics = False
        self._warned_speed_from_translation = False
        self.index_cache = bool(index_cache)
        self.index_cache_dir = index_cache_dir
        self.verify_image_exists = bool(verify_image_exists)
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
        self.max_load_retries = int(max(1, max_load_retries))
        self.index_cache_wait_seconds = int(max(0, index_cache_wait_seconds))
        self.rank = int(rank)
        self.is_validation = bool(is_validation)
        self.min_valid_frames = 1 + (self.frames_per_clip - 1) * self.sample_step
        self.data_infos: Optional[List[Dict[str, Any]]] = None

        self.scenes, self.windows = self._load_or_build_indices()
        if not self.scenes:
            raise ValueError(
                "No valid Bench2Drive scenes found. "
                f"ann_file={ann_file}, data_root={data_root}, camera={camera_name}"
            )
        if not self.windows:
            raise ValueError(
                "No valid Bench2Drive sliding windows could be built. "
                f"scenes={len(self.scenes)}, frames_per_clip={self.frames_per_clip}, sample_step={self.sample_step}"
            )

        logger.info(
            "Bench2Drive dataset ready: scenes=%d, windows=%d, camera=%s, frames_per_clip=%d, fps=%d, sample_step=%d",
            len(self.scenes),
            len(self.windows),
            self.camera_name,
            self.frames_per_clip,
            self.fps,
            self.sample_step,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["data_infos"] = None
        return state

    def _ensure_data_infos_loaded(self) -> List[Dict[str, Any]]:
        if self.data_infos is None:
            self.data_infos = self._load_infos()
        return self.data_infos

    def release_data_infos(self) -> None:
        self.data_infos = None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        window_idx = int(index)
        for retry in range(self.max_load_retries):
            window = self.windows[window_idx]
            scene = self.scenes[window.scene_idx]
            try:
                data_infos = self._ensure_data_infos_loaded()
                sampled_indices = self._window_info_indices(scene.valid_indices, window.start_pos)
                sampled_infos = [data_infos[idx] for idx in sampled_indices]
                image_infos = sampled_infos[: self.required_image_frames]

                buffer = self._load_clip_images(image_infos)
                raw_buffer = buffer
                if self.transform is not None:
                    buffer = self.transform(buffer)
                    if not torch.is_tensor(buffer):
                        buffer = torch.as_tensor(buffer)
                else:
                    buffer = torch.from_numpy(buffer).permute(3, 0, 1, 2).float() / 255.0

                proposal_buffer = None
                if self.proposal_transform is not None:
                    proposal_buffer = self.proposal_transform(raw_buffer)
                    if not torch.is_tensor(proposal_buffer):
                        proposal_buffer = torch.as_tensor(proposal_buffer)

                reduced_infos = sampled_infos[:: self.tubelet_size]
                states = self._build_states(reduced_infos)
                actions = self._build_actions(states)
                driving_command = self._build_driving_commands(reduced_infos)
                ego_dynamics = self._build_ego_dynamics(reduced_infos)
                extrinsics = np.zeros_like(states, dtype=np.float32)

                if self.load_agent_annotations:
                    agent_boxes, agent_mask = self._build_agent_annotations(reduced_infos)
                    bev_segmentation = self._build_bev_segmentation(agent_boxes, agent_mask)
                else:
                    t_reduced = len(reduced_infos)
                    agent_boxes = np.zeros((t_reduced, self.max_agents, 7), dtype=np.float32)
                    agent_mask = np.zeros((t_reduced, self.max_agents), dtype=np.bool_)
                    bev_segmentation = np.zeros((t_reduced, BEV_SIZE, BEV_SIZE), dtype=np.uint8)

                sampled_frame_indices = np.asarray([int(info["frame_idx"]) for info in sampled_infos], dtype=np.int64)
                image_frame_indices = np.asarray([int(info["frame_idx"]) for info in image_infos], dtype=np.int64)
                stable_sample_id = collate.make_stable_sample_id(
                    "bench2drive",
                    os.path.splitext(os.path.basename(self.ann_file))[0],
                    scene.scene_name,
                    int(window.start_pos),
                )
                return {
                    "buffer": buffer,
                    "proposal_buffer": proposal_buffer,
                    "actions": actions,
                    "states": states,
                    "extrinsics": extrinsics,
                    "indices": sampled_frame_indices,
                    "scene_name": scene.scene_name,
                    "ann_file": self.ann_file,
                    "window_start_pos": int(window.start_pos),
                    "stable_sample_id": stable_sample_id,
                    "sample_id": stable_sample_id,
                    "sampled_frame_indices": sampled_frame_indices,
                    "image_frame_indices": image_frame_indices,
                    "seg_masks": None,
                    "seg_frame_indices": None,
                    "driving_command": driving_command,
                    "ego_dynamics": ego_dynamics,
                    "agent_boxes": agent_boxes,
                    "agent_mask": agent_mask,
                    "bev_segmentation": bev_segmentation,
                }
            except Exception as exc:
                if retry == self.max_load_retries - 1:
                    raise
                logger.warning(
                    "Bench2Drive sample load failed (scene=%s, window_start=%d, retry=%d/%d): %s",
                    scene.scene_name,
                    window.start_pos,
                    retry + 1,
                    self.max_load_retries,
                    str(exc),
                )
                if not self.is_validation:
                    window_idx = random.randint(0, len(self.windows) - 1)

        raise RuntimeError("Unreachable Bench2Drive dataset retry state")

    def _load_infos(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.ann_file):
            raise ValueError(f"Bench2Drive ann_file does not exist: {self.ann_file}")
        with open(self.ann_file, "rb") as handle:
            data_infos = pickle.load(handle)
        if not isinstance(data_infos, list):
            raise ValueError(f"Unexpected Bench2Drive pickle structure: {self.ann_file}")
        return data_infos

    def _load_or_build_indices(self) -> Tuple[List[B2DSceneRecord], List[B2DWindowRecord]]:
        cache_path = self._get_index_cache_path()
        scenes: Optional[List[B2DSceneRecord]] = None
        if cache_path is not None:
            scenes = self._load_index_cache(cache_path)
            if self.rank != 0 and scenes is None:
                scenes = self._wait_for_index_cache(cache_path)
                if scenes is None:
                    logger.warning(
                        "Bench2Drive index cache was not ready after %ds on rank=%d; building locally.",
                        self.index_cache_wait_seconds,
                        self.rank,
                    )

        if scenes is None:
            scenes = self._build_scene_index()
            if cache_path is not None:
                self._save_index_cache(cache_path, scenes)
        self.scenes = scenes
        return scenes, self._build_window_index()

    def _build_scene_index(self) -> List[B2DSceneRecord]:
        data_infos = self._ensure_data_infos_loaded()
        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx, info in enumerate(data_infos):
            folder = str(info.get("folder", ""))
            if folder:
                grouped[folder].append(idx)

        scenes: List[B2DSceneRecord] = []
        skipped_missing_camera = 0
        skipped_short = 0
        skipped_not_enough_required_images = 0
        scene_names = sorted(grouped.keys())
        if self.max_scenes is not None:
            scene_names = scene_names[: int(self.max_scenes)]

        t0 = time.monotonic()
        total = len(scene_names)
        logger.info("Building Bench2Drive scene index from %d scenes ...", total)
        for scene_pos, scene_name in enumerate(scene_names):
            sorted_indices = sorted(grouped[scene_name], key=lambda idx: int(data_infos[idx]["frame_idx"]))
            info_indices: List[int] = []
            frame_indices: List[int] = []
            valid_image_frame_indices: List[int] = []
            for idx in sorted_indices:
                info = data_infos[idx]
                frame_idx = int(info["frame_idx"])
                info_indices.append(idx)
                frame_indices.append(frame_idx)
                if self._has_valid_camera(info):
                    valid_image_frame_indices.append(frame_idx)

            if not valid_image_frame_indices:
                skipped_missing_camera += 1
                continue
            if len(info_indices) < self.min_valid_frames:
                skipped_short += 1
                continue
            if len(valid_image_frame_indices) < self.required_image_frames:
                skipped_not_enough_required_images += 1
                continue

            scenes.append(
                B2DSceneRecord(
                    scene_name=scene_name,
                    valid_indices=info_indices,
                    frame_indices=frame_indices,
                    valid_image_frame_indices=valid_image_frame_indices,
                )
            )
            if (scene_pos + 1) % 100 == 0 or (scene_pos + 1) == total:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  Bench2Drive scene index progress: %d/%d (%.1f%%) elapsed=%.1fs",
                    scene_pos + 1,
                    total,
                    100.0 * (scene_pos + 1) / max(1, total),
                    elapsed,
                )

        logger.info(
            "Bench2Drive scene index built: kept=%d, skipped_missing_camera=%d, skipped_short=%d, "
            "skipped_not_enough_required_images=%d, time=%.1fs",
            len(scenes),
            skipped_missing_camera,
            skipped_short,
            skipped_not_enough_required_images,
            time.monotonic() - t0,
        )
        return scenes

    def _build_window_index(self) -> List[B2DWindowRecord]:
        windows: List[B2DWindowRecord] = []
        total_candidates = 0
        rejected_gap = 0
        rejected_missing_required_images = 0
        for scene_idx, scene in enumerate(self.scenes):
            max_start = len(scene.valid_indices) - self.min_valid_frames
            valid_image_frame_indices = set(scene.valid_image_frame_indices)
            for start in range(0, max_start + 1, self.window_stride):
                total_candidates += 1
                sampled_frame_indices = self._window_info_indices(scene.frame_indices, start)
                if not self._is_window_continuous(scene, start):
                    rejected_gap += 1
                    continue
                if not self._has_required_images(valid_image_frame_indices, sampled_frame_indices):
                    rejected_missing_required_images += 1
                    continue
                windows.append(B2DWindowRecord(scene_idx=scene_idx, start_pos=start))

        logger.info(
            "Built Bench2Drive window index: %d windows from %d scenes "
            "(stride=%d, min_valid=%d, max_frame_gap=%d, image_require_policy=%s, "
            "required_image_frames=%d, rejected_gap=%d, rejected_missing_required_images=%d/%d candidates)",
            len(windows),
            len(self.scenes),
            self.window_stride,
            self.min_valid_frames,
            self.max_frame_gap,
            self.image_require_policy,
            self.required_image_frames,
            rejected_gap,
            rejected_missing_required_images,
            total_candidates,
        )
        return windows

    def _window_info_indices(self, valid_indices: Sequence[int], start_pos: int) -> List[int]:
        positions = start_pos + np.arange(self.frames_per_clip) * self.sample_step
        return [int(valid_indices[int(pos)]) for pos in positions]

    def _required_image_frame_indices(self, sampled_frame_indices: Sequence[int]) -> List[int]:
        return [int(frame_idx) for frame_idx in sampled_frame_indices[: self.required_image_frames]]

    def _has_required_images(self, valid_image_frame_indices: set, sampled_frame_indices: Sequence[int]) -> bool:
        return all(
            frame_idx in valid_image_frame_indices
            for frame_idx in self._required_image_frame_indices(sampled_frame_indices)
        )

    def _is_window_continuous(self, scene: B2DSceneRecord, start_pos: int) -> bool:
        positions = start_pos + np.arange(self.frames_per_clip) * self.sample_step
        for i in range(len(positions) - 1):
            raw_gap = int(scene.frame_indices[int(positions[i + 1])]) - int(scene.frame_indices[int(positions[i])])
            if raw_gap > self.max_frame_gap:
                return False
        return True

    def _has_valid_camera(self, info: Dict[str, Any]) -> bool:
        cam_info = info.get("sensors", {}).get(self.camera_name)
        if cam_info is None:
            return False
        rel_path = cam_info.get("data_path")
        if not rel_path:
            return False
        if not self.verify_image_exists:
            return True
        return os.path.exists(os.path.join(self.data_root, rel_path))

    def _get_index_cache_path(self) -> Optional[str]:
        return get_b2d_index_cache_path(
            ann_file=self.ann_file,
            data_root=self.data_root,
            camera_name=self.camera_name,
            frames_per_clip=self.frames_per_clip,
            fps=self.fps,
            base_fps=self.base_fps,
            tubelet_size=self.tubelet_size,
            max_scenes=self.max_scenes,
            verify_image_exists=self.verify_image_exists,
            image_require_policy=self.image_require_policy,
            num_observed_frames=self.num_observed_frames,
            index_cache=self.index_cache,
            index_cache_dir=self.index_cache_dir,
        )

    def _load_index_cache(self, cache_path: str) -> Optional[List[B2DSceneRecord]]:
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or payload.get("version") != 5:
                logger.info("Bench2Drive index cache version mismatch, will rebuild.")
                return None
            scenes = payload.get("scenes")
            if not isinstance(scenes, list):
                logger.info("Bench2Drive index cache is malformed, will rebuild.")
                return None
            if any(not hasattr(scene, "frame_indices") for scene in scenes):
                logger.info("Bench2Drive index cache lacks frame_indices, will rebuild.")
                return None
            if any(not hasattr(scene, "valid_image_frame_indices") for scene in scenes):
                logger.info("Bench2Drive index cache lacks valid_image_frame_indices, will rebuild.")
                return None
            logger.info("Loaded Bench2Drive scene index cache: %s (scenes=%d)", cache_path, len(scenes))
            return scenes
        except Exception as exc:
            logger.warning("Failed to load Bench2Drive index cache %s: %s", cache_path, exc)
            return None

    def _wait_for_index_cache(self, cache_path: str) -> Optional[List[B2DSceneRecord]]:
        if self.index_cache_wait_seconds <= 0:
            return None
        deadline = time.monotonic() + self.index_cache_wait_seconds
        next_log = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            cached = self._load_index_cache(cache_path)
            if cached is not None:
                return cached
            now = time.monotonic()
            if now >= next_log:
                logger.info("Waiting for Bench2Drive index cache on rank=%d: %s", self.rank, cache_path)
                next_log = now + 30.0
            time.sleep(2.0)
        return None

    def _save_index_cache(self, cache_path: str, scenes: List[B2DSceneRecord]) -> None:
        payload = {"version": 5, "scenes": scenes}
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            logger.info("Saved Bench2Drive scene index cache: %s (scenes=%d)", cache_path, len(scenes))
        except Exception as exc:
            logger.warning("Failed to save Bench2Drive index cache %s: %s", cache_path, exc)

    def _load_clip_images(self, infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        images: List[np.ndarray] = []
        for info in infos:
            rel_path = info["sensors"][self.camera_name]["data_path"]
            image_path = os.path.join(self.data_root, rel_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Bench2Drive image not found: {image_path}")
            with Image.open(image_path) as img:
                images.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
        return np.stack(images, axis=0)

    def _build_states(self, infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        translations = []
        yaws = []
        for info in infos:
            translations.append(np.asarray(info["ego_translation"], dtype=np.float64))
            yaws.append(float(info["ego_yaw"]))

        translations_arr = np.stack(translations, axis=0)
        # states[:, 6] 速度是无条件必需量（status velocity / 条件输入的兜底数据源），
        # 缺 ego_vel 时禁止伪造零速度，改用 ego_translation 按 frame_idx 差分推导（真实数据）。
        speeds = self._build_speeds(infos, translations_arr)
        translations_arr -= translations_arr[0:1]
        yaws_arr = np.asarray(yaws, dtype=np.float64).reshape(-1, 1)
        speeds_arr = np.asarray(speeds, dtype=np.float64).reshape(-1, 1)
        zeros = np.zeros((len(infos), 3), dtype=np.float64)
        states = np.concatenate([translations_arr[:, :3], zeros[:, :2], yaws_arr, speeds_arr], axis=1)
        return states.astype(np.float32)

    def _build_actions(self, states: np.ndarray) -> np.ndarray:
        if states.shape[0] < 2:
            return np.zeros((0, self.action_dim), dtype=np.float32)

        actions = np.zeros((states.shape[0] - 1, 3), dtype=np.float32)
        for t in range(states.shape[0] - 1):
            dx_global = states[t + 1, 0] - states[t, 0]
            dy_global = states[t + 1, 1] - states[t, 1]
            yaw = states[t, 5]
            cos_h = np.cos(-yaw)
            sin_h = np.sin(-yaw)
            dx_ego = cos_h * dx_global - sin_h * dy_global
            dy_ego = sin_h * dx_global + cos_h * dy_global
            d_yaw = states[t + 1, 5] - states[t, 5]
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
            actions[t] = np.asarray([dx_ego, dy_ego, d_yaw], dtype=np.float32)

        if self.action_dim == 3:
            return actions
        padded = np.zeros((actions.shape[0], self.action_dim), dtype=np.float32)
        padded[:, :3] = actions
        return padded

    def _build_speeds(self, infos: Sequence[Dict[str, Any]], translations_arr: np.ndarray) -> List[float]:
        """计算每帧标量速度（states[:, 6]）。

        优先用 ego_vel 的平面范数；任一帧缺 ego_vel 时整窗改用 ego_translation
        按 frame_idx 时间差分推导（位移/时间，真实数据），禁止伪造零速度。
        """
        if all("ego_vel" in info for info in infos):
            return [float(np.linalg.norm(np.asarray(info["ego_vel"], dtype=np.float64)[:2])) for info in infos]

        if len(infos) < 2:
            raise ValueError("B2D 窗口帧数 < 2 且缺 'ego_vel'，无法推导速度（禁止伪造零速度）")
        if not self._warned_speed_from_translation:
            logger.warning(
                "B2D info 缺 'ego_vel'；states[:, 6] 速度改用 ego_translation 差分推导（真实数据，非伪造零速度）。"
            )
            self._warned_speed_from_translation = True

        frame_indices = np.asarray([float(info["frame_idx"]) for info in infos], dtype=np.float64)
        delta_t = np.diff(frame_indices) / float(self.base_fps)
        if np.any(delta_t <= 0):
            raise ValueError(f"B2D frame_idx 非严格递增，无法差分推导速度: {frame_indices.tolist()}")
        dists = np.linalg.norm(np.diff(translations_arr[:, :2], axis=0), axis=1)
        speeds = dists / delta_t  # [T-1]，speeds[i] = 帧 i→i+1 的平均速度（后向差分对齐到帧 i+1）
        speeds = np.concatenate([speeds[:1], speeds])  # 首帧复用第一段速度
        return [float(s) for s in speeds]

    def _build_driving_commands(self, infos: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        """构建 [T, command_dim] one-hot 导航指令。

        driving_command 是**条件必需**特征（仅 use_drive_command=True 的配置消费）。
        B2D 数据不保证逐帧含 ``command_near``：缺失时禁止伪造（旧代码 .get(-1) 会把
        缺失映射成 LANEFOLLOW one-hot），返回 None、由消费端（status_features /
        predictor_aux）在配置需要时 fail-loud。注意 -1 (CARLA VOID) 是合法**值**，
        与字段缺失不同，仍走既有 VOID→LANEFOLLOW 映射。
        """
        for info in infos:
            if "command_near" not in info:
                if not self._warned_missing_command:
                    logger.warning(
                        "B2D info 缺 'command_near' 字段；driving_command 特征不可用（不伪造）。"
                        "若配置 use_drive_command=True，下游将报错。"
                    )
                    self._warned_missing_command = True
                return None
        commands = [self._command_to_one_hot(info["command_near"]) for info in infos]
        return np.stack(commands, axis=0).astype(np.float32)

    def _build_ego_dynamics(self, infos: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        """构建 [T, 4] 真实 [vx, vy, ax, ay]。

        ego_dynamics 是**条件必需**特征（state_dim>=8 必需）。B2D 数据不保证逐帧含
        ``ego_vel``/``ego_accel``：缺失时禁止伪造零速度（会把车伪造成静止），返回
        None、由消费端在 state_dim>=8 时 fail-loud（state_dim=7 回退 states 真实速度差分）。
        """
        for info in infos:
            if "ego_vel" not in info or "ego_accel" not in info:
                if not self._warned_missing_dynamics:
                    logger.warning(
                        "B2D info 缺 'ego_vel'/'ego_accel' 字段；ego_dynamics 特征不可用（不伪造零速度）。"
                        "若配置 state_dim>=8，下游将报错。"
                    )
                    self._warned_missing_dynamics = True
                return None
        dynamics: List[np.ndarray] = []
        for info in infos:
            ego_vel = np.asarray(info["ego_vel"], dtype=np.float32)
            ego_accel = np.asarray(info["ego_accel"], dtype=np.float32)
            dynamics.append(np.asarray([ego_vel[0], ego_vel[1], ego_accel[0], ego_accel[1]], dtype=np.float32))
        return np.stack(dynamics, axis=0)

    def _command_to_one_hot(self, command: Any) -> np.ndarray:
        try:
            command_idx = int(command)
        except (TypeError, ValueError) as exc:
            # fail-loud: 字段存在但值无法解析属于数据损坏（不同于字段缺失），
            # 禁止静默当成 VOID→LANEFOLLOW。
            raise ValueError(f"B2D command_near 值非法（无法解析为 int）: {command!r}") from exc
        if command_idx < 0:
            # CARLA RoadOption.VOID(-1) 是合法值，沿用既有 VOID→LANEFOLLOW 语义。
            command_idx = 4
        command_idx -= 1
        one_hot = np.zeros(self.command_dim, dtype=np.float32)
        if 0 <= command_idx < self.command_dim:
            one_hot[command_idx] = 1.0
        return one_hot

    def _build_agent_annotations(self, infos: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        t_steps = len(infos)
        agent_boxes = np.zeros((t_steps, self.max_agents, 7), dtype=np.float32)
        agent_mask = np.zeros((t_steps, self.max_agents), dtype=np.bool_)

        for t_idx, info in enumerate(infos):
            gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
            if gt_boxes.size == 0:
                continue
            lidar2ego = np.asarray(info["sensors"]["LIDAR_TOP"]["lidar2ego"], dtype=np.float64)
            n_keep = min(gt_boxes.shape[0], self.max_agents)
            for box_idx in range(n_keep):
                center_ego, heading_ego = self._lidar_box_to_ego(gt_boxes[box_idx], lidar2ego)
                agent_boxes[t_idx, box_idx] = np.asarray(
                    [
                        center_ego[0],
                        center_ego[1],
                        center_ego[2],
                        gt_boxes[box_idx, 3],
                        gt_boxes[box_idx, 4],
                        gt_boxes[box_idx, 5],
                        heading_ego,
                    ],
                    dtype=np.float32,
                )
            agent_mask[t_idx, :n_keep] = True

        return agent_boxes, agent_mask

    def _build_bev_segmentation(self, agent_boxes: np.ndarray, agent_mask: np.ndarray) -> np.ndarray:
        bev_seg = np.zeros((agent_boxes.shape[0], BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        for t_idx in range(agent_boxes.shape[0]):
            bev_seg[t_idx] = _rasterize_agents_to_bev(agent_boxes[t_idx], agent_mask[t_idx])
        return bev_seg

    @staticmethod
    def _lidar_box_to_ego(box_lidar: np.ndarray, lidar2ego: np.ndarray) -> Tuple[np.ndarray, float]:
        center_lidar = np.asarray([box_lidar[0], box_lidar[1], box_lidar[2], 1.0], dtype=np.float64)
        center_ego = (lidar2ego @ center_lidar)[:3]

        yaw = float(box_lidar[6])
        cos_h = np.cos(yaw)
        sin_h = np.sin(yaw)
        box_pose_lidar = np.eye(4, dtype=np.float64)
        box_pose_lidar[0, 0] = cos_h
        box_pose_lidar[0, 1] = -sin_h
        box_pose_lidar[1, 0] = sin_h
        box_pose_lidar[1, 1] = cos_h
        box_pose_lidar[:3, 3] = box_lidar[:3]

        box_pose_ego = lidar2ego @ box_pose_lidar
        heading_ego = float(np.arctan2(box_pose_ego[1, 0], box_pose_ego[0, 0]))
        return center_ego.astype(np.float32), heading_ego


def _stack_optional_field(batch: Sequence[Dict[str, Any]], key: str) -> Optional[torch.Tensor]:
    """堆叠条件必需特征；任一样本缺失则整个 batch 槽位为 None（禁止伪造占位）。

    None 经 load_clips/val_command 透传到消费端，由 status_features/predictor_aux
    在配置需要该特征时 fail-loud。
    """
    values = [item[key] for item in batch]
    if all(v is not None for v in values):
        return torch.stack([torch.from_numpy(v) for v in values])
    if any(v is not None for v in values) and key not in _WARNED_MIXED_OPTIONAL_FIELDS:
        logger.warning("B2D batch 内 '%s' 部分样本缺失（数据不一致），该 batch 槽位整体置 None。", key)
        _WARNED_MIXED_OPTIONAL_FIELDS.add(key)
    return None


_WARNED_MIXED_OPTIONAL_FIELDS: set = set()


def b2d_world_model_collate_fn(batch: Sequence[Dict[str, Any]]):
    context_frames, actions, states, extrinsics = collate.stack_core(batch)

    seg_targets = [None for _ in batch]
    driving_command = _stack_optional_field(batch, "driving_command")
    ego_dynamics = _stack_optional_field(batch, "ego_dynamics")
    agent_boxes = collate.stack_required(batch, "agent_boxes")
    agent_mask = collate.stack_agent_mask(batch)
    bev_segmentation = collate.stack_required(batch, "bev_segmentation")

    proposal_context_frames = collate.stack_proposal_buffer(batch)

    metadata = collate.build_window_metadata(batch, "ann_file")

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


def init_bench2drive_data(
    ann_file: str,
    data_root: str,
    batch_size: int,
    frames_per_clip: int = 20,
    fps: int = 2,
    base_fps: int = 10,
    tubelet_size: int = 1,
    transform: Any = None,
    proposal_transform: Any = None,
    num_workers: int = 4,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    rank: int = 0,
    world_size: int = 1,
    camera_name: str = "CAM_FRONT",
    max_scenes: Optional[int] = None,
    action_dim: int = 3,
    shuffle: bool = True,
    window_stride: int = 1,
    max_frame_gap: int = 6,
    max_agents: int = 50,
    load_agent_annotations: bool = True,
    command_dim: int = 6,
    index_cache: bool = True,
    index_cache_dir: Optional[str] = None,
    verify_image_exists: bool = False,
    image_require_policy: str = IMAGE_REQUIRE_ALL_FRAMES,
    num_observed_frames: Optional[int] = None,
    max_load_retries: int = 5,
    index_cache_wait_seconds: int = 300,
    drop_last: bool = True,
    is_validation: bool = False,
) -> Tuple[DataLoader, "Sampler[int]"]:
    dataset = Bench2DriveWorldModelDataset(
        ann_file=ann_file,
        data_root=data_root,
        camera_name=camera_name,
        frames_per_clip=frames_per_clip,
        fps=fps,
        base_fps=base_fps,
        tubelet_size=tubelet_size,
        transform=transform,
        proposal_transform=proposal_transform,
        max_scenes=max_scenes,
        action_dim=action_dim,
        window_stride=window_stride,
        max_frame_gap=max_frame_gap,
        max_agents=max_agents,
        load_agent_annotations=load_agent_annotations,
        command_dim=command_dim,
        index_cache=index_cache,
        index_cache_dir=index_cache_dir,
        verify_image_exists=verify_image_exists,
        image_require_policy=image_require_policy,
        num_observed_frames=num_observed_frames,
        max_load_retries=max_load_retries,
        index_cache_wait_seconds=index_cache_wait_seconds,
        rank=rank,
        is_validation=is_validation,
    )

    if is_validation:
        if shuffle or drop_last:
            raise ValueError("Bench2Drive validation requires shuffle=False and drop_last=False")
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
    if num_workers > 0:
        dataset.release_data_infos()

    # 可复现的数据增强 RNG（见 navsim_data 同处说明）：worker_init_fn 播种 numpy/random，
    # generator 种子继承全局 torch 种子（config.meta.seed）并按 rank 区分。
    from app.vjepa_cowa_world_model.training.loop import make_dataloader_generator, seed_dataloader_worker

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=b2d_world_model_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=False if is_validation else drop_last,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(
            rank=rank,
            stream="b2d/validation" if is_validation else "b2d/train",
        ),
    )
    logger.info(
        "Bench2Drive dataloader created: batches=%d, batch_size=%d, workers=%d",
        len(loader),
        batch_size,
        num_workers,
    )
    return loader, sampler
