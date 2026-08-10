# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
数据加载模块

提供数据增强和数据加载器的创建功能。
"""

import os
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader, DistributedSampler, Sampler

from app.vjepa_cowa_world_model.training.video_transforms import make_transforms
from src.utils.logging import get_logger

from ..utils.planner_training import resolve_validation_target_timeline, resolve_validation_timestep_sec
from .config import (
    TrainingConfig,
    resolve_bench2drive_image_require_policy,
    resolve_main_encoder_frame_stride,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_navsim_image_require_policy,
    resolve_proposal_encoder_backbone,
)
from .navsim_data import init_navsim_data
from .validation_rng import VALIDATION_RNG_CONTRACT_VERSION
from .vjepa_transforms import VJEPAImageTransform

logger = get_logger(__name__)


def _hw_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value), int(value)]
    return [int(value[0]), int(value[1])]


def _vjepa_validation_transform_semantics(resolution: Any, crop_top_bottom: int) -> Dict[str, Any]:
    return {
        "type": "vjepa",
        "resolution": _hw_list(resolution),
        "crop_top_bottom": int(crop_top_bottom),
        "interpolation": "bilinear",
        "align_corners": False,
        "normalization": "imagenet_rgb_unit_v1",
    }


def _main_validation_transform_semantics(config: TrainingConfig) -> Dict[str, Any]:
    if config.model.backbone == "dinov2_img_encoder":
        return {
            "type": "dinov2",
            "resolution": _hw_list(config.model.dinov2_resolution),
            "resize_policy": "cover_then_center_crop_v1",
            "frame_selection": "end_of_chunk",
            "frame_selection_stage": "encoder_adapter",
            "frame_stride": config.model.dinov2_frame_stride,
            "interpolation": "bilinear",
            "align_corners": False,
            "normalization": "imagenet_rgb_255_v1",
        }
    if config.model.backbone == "vjepa_img_encoder":
        return _vjepa_validation_transform_semantics(
            config.model.vjepa_resolution,
            config.model.vjepa_crop_top_bottom,
        )
    return {
        "type": "vjepa_deterministic",
        "crop_size": _hw_list(config.data.crop_size),
        "resize_policy": "cover_then_center_crop_v1",
        "interpolation": "bilinear",
        "align_corners": False,
        "normalization": "imagenet_rgb_255_v1",
    }


def _proposal_validation_transform_semantics(config: TrainingConfig) -> Dict[str, Any]:
    proposal = config.proposal
    if not proposal.enabled or not proposal.use_separate_encoder:
        return {"enabled": False}
    if resolve_proposal_encoder_backbone(config) != "vjepa_img_encoder":
        return {"enabled": False}
    return {
        "enabled": True,
        "transform": _vjepa_validation_transform_semantics(
            proposal.vjepa_resolution,
            proposal.vjepa_crop_top_bottom,
        ),
    }


def resolve_navsim_validation_data_semantics(config: TrainingConfig) -> Dict[str, Any]:
    """Resolve the exact path-independent NavSim arguments that define validation samples."""

    navsim = config.data.navsim
    if navsim is None:
        raise ValueError("NavSim validation semantics require data.navsim configuration")
    camera_names = list(navsim.camera_names) if config.multiview.enabled else [navsim.camera_name]
    frame_stride = resolve_main_encoder_frame_stride(config)
    total_steps = resolve_main_encoder_num_time_steps(config, config.data.num_target_frames)
    observed_steps = resolve_main_encoder_num_observed_steps(config)
    future_steps = total_steps - observed_steps
    if future_steps < 1:
        raise ValueError(
            "NavSim validation requires at least one future predictor step, "
            f"got total_steps={total_steps}, observed_steps={observed_steps}"
        )
    planner_target_timeline = resolve_validation_target_timeline(
        num_target_frames=config.data.num_target_frames,
        num_observed_frames=config.train.num_observed_frames,
        predictor_inference_consistent=config.train.predictor_inference_consistent,
        predictor_no_aux_input=config.train.predictor_no_aux_input,
    )
    return {
        "frames_per_clip": config.data.num_target_frames,
        "fps": config.data.fps,
        "num_observed_frames": config.train.num_observed_frames,
        "action_dim": config.train.action_dim,
        "predictor_timeline": {
            "frame_stride": frame_stride,
            "total_steps": total_steps,
            "observed_steps": observed_steps,
            "future_steps": future_steps,
        },
        "planner_target_timeline": planner_target_timeline,
        "metric_timestep_sec": resolve_validation_timestep_sec(
            fps=config.data.fps,
            diff_dt=config.planner.diff_dt,
        ),
        "validation_transform": _main_validation_transform_semantics(config),
        "proposal_transform": _proposal_validation_transform_semantics(config),
        "validation_rng": {
            "version": VALIDATION_RNG_CONTRACT_VERSION,
            "base_seed": config.meta.seed,
            "stable_across_epochs": config.meta.val_stable_noise,
        },
        "camera_name": navsim.camera_name,
        "camera_names": camera_names,
        "max_scenes": navsim.max_val_scenes,
        "window_stride": navsim.val_window_stride if navsim.val_window_stride is not None else navsim.window_stride,
        "max_frame_gap": navsim.max_frame_gap,
        "max_agents": navsim.max_agents,
        "load_agent_annotations": navsim.load_agent_annotations,
        "image_require_policy": resolve_navsim_image_require_policy(config),
        "tail_seconds": navsim.val_tail_seconds,
        "counterfactual_tail_seconds": navsim.counterfactual_tail_seconds,
        "scene_filter_enabled": bool(navsim.val_scene_filter_yaml),
        "pose_overlay_enabled": bool(navsim.val_pose_overlay_path),
        "pose_overlay_coord_frame": navsim.pose_overlay_coord_frame,
        "pose_overlay_required": navsim.pose_overlay_required,
    }


def _is_navsim_enabled(config: TrainingConfig) -> bool:
    navsim = config.data.navsim
    return navsim is not None and navsim.enabled


def _is_bench2drive_enabled(config: TrainingConfig) -> bool:
    bench2drive = config.data.bench2drive
    return bench2drive is not None and bench2drive.enabled


def _is_mongo_raw_enabled(config: TrainingConfig) -> bool:
    mongo_raw = config.data.mongo_raw
    return mongo_raw is not None and mongo_raw.enabled


# Static registry of the world-model loaders, in priority/documentation order. There is no decorator
# magic and no dynamic discovery: to add a dataset, add one entry here (and its init body in
# create_train/val_dataloader). The loaders themselves stay separate and format-specific (invariant #3).
_WORLD_MODEL_DATASET_PREDICATES = {
    "mongo_raw": _is_mongo_raw_enabled,
    "bench2drive": _is_bench2drive_enabled,
    "navsim": _is_navsim_enabled,
}


def _select_world_model_dataset(config: TrainingConfig) -> Optional[str]:
    """Return the single enabled world-model dataset name, or ``None`` for the legacy seg fallback.

    Exactly-one / zero enabled is byte-identical to the previous ``if/elif`` priority dispatch. More
    than one enabled is a misconfiguration and now fails loud (the old chain silently picked the first
    by priority). Both ``create_train_dataloader`` and ``create_val_dataloader`` select through here.
    """
    enabled = [name for name, is_enabled in _WORLD_MODEL_DATASET_PREDICATES.items() if is_enabled(config)]
    if len(enabled) > 1:
        raise ValueError(
            f"Multiple datasets enabled ({enabled}); enable exactly one of "
            f"{list(_WORLD_MODEL_DATASET_PREDICATES)} (or none for the legacy seg loader)."
        )
    return enabled[0] if enabled else None


def create_transforms(config: TrainingConfig) -> Any:
    """
    创建数据增强 transform

    Args:
        config: 训练配置

    Returns:
        Any: transform 对象
    """
    if config.model.backbone == "vjepa_img_encoder":
        return VJEPAImageTransform(
            resolution=config.model.vjepa_resolution,
            crop_top_bottom=config.model.vjepa_crop_top_bottom,
        )

    transform = make_transforms(
        random_horizontal_flip=config.data_aug.horizontal_flip,
        random_resize_aspect_ratio=config.data_aug.random_resize_aspect_ratio,
        random_resize_scale=config.data_aug.random_resize_scale,
        reprob=config.data_aug.reprob,
        auto_augment=config.data_aug.auto_augment,
        motion_shift=config.data_aug.motion_shift,
        crop_size=config.data.crop_size,
    )
    return transform


def create_validation_transforms(config: TrainingConfig) -> Any:
    """Create the deterministic image transform required by validation."""
    if config.model.backbone == "vjepa_img_encoder":
        return VJEPAImageTransform(
            resolution=config.model.vjepa_resolution,
            crop_top_bottom=config.model.vjepa_crop_top_bottom,
        )

    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=config.data_aug.random_resize_aspect_ratio,
        random_resize_scale=config.data_aug.random_resize_scale,
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=config.data.crop_size,
        deterministic=True,
    )


def create_proposal_transforms(config: TrainingConfig) -> Optional[Any]:
    """Create an optional second transform for an independent proposal encoder."""
    proposal_cfg = getattr(config, "proposal", None)
    if proposal_cfg is None or not proposal_cfg.enabled or not proposal_cfg.use_separate_encoder:
        return None
    if resolve_proposal_encoder_backbone(config) != "vjepa_img_encoder":
        return None
    return VJEPAImageTransform(
        resolution=proposal_cfg.vjepa_resolution,
        crop_top_bottom=proposal_cfg.vjepa_crop_top_bottom,
    )


def create_train_dataloader(
    config: TrainingConfig, rank: int, world_size: int, transform: Any = None
) -> Tuple[DataLoader, DistributedSampler]:
    """
    创建训练数据加载器

    Args:
        config: 训练配置
        rank: 当前进程的 rank
        world_size: 进程总数
        transform: 数据增强 transform (可选，如果为 None 则自动创建)

    Returns:
        Tuple[DataLoader, DistributedSampler]: (dataloader, sampler)
    """
    if transform is None:
        transform = create_transforms(config)
    proposal_transform = create_proposal_transforms(config)

    selected = _select_world_model_dataset(config)
    if selected == "mongo_raw":
        from .mongo_raw_data import init_mongo_raw_data

        mongo_raw = config.data.mongo_raw
        if mongo_raw is None:
            raise ValueError("Unexpected mongo_raw config state")

        if config.segmentation.use_segmentation:
            raise ValueError("Mongo raw online loader does not support segmentation in the first version")

        logger.info(
            "Initializing Mongo raw training dataset from database=%s collection=%s vehicle_type=%s vehicle_types=%s",
            mongo_raw.database,
            mongo_raw.collection,
            mongo_raw.vehicle_type,
            mongo_raw.vehicle_types,
        )

        loader, sampler = init_mongo_raw_data(
            mongo_cfg=mongo_raw,
            split="train",
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            tubelet_size=1,
            transform=transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            action_dim=config.train.action_dim,
            shuffle=True,
            is_validation=False,
        )
    elif selected == "bench2drive":
        from .b2d_data import init_bench2drive_data

        bench2drive = config.data.bench2drive
        if bench2drive is None:
            raise ValueError("Unexpected bench2drive config state")

        if not bench2drive.ann_file or not bench2drive.data_root:
            raise ValueError("data.bench2drive.ann_file and data.bench2drive.data_root must be configured")

        logger.info(
            "Initializing Bench2Drive training dataset from ann_file=%s, data_root=%s",
            bench2drive.ann_file,
            bench2drive.data_root,
        )

        loader, sampler = init_bench2drive_data(
            ann_file=bench2drive.ann_file,
            data_root=bench2drive.data_root,
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            base_fps=bench2drive.base_fps,
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            camera_name=bench2drive.camera_name,
            max_scenes=bench2drive.max_scenes,
            action_dim=config.train.action_dim,
            shuffle=True,
            window_stride=bench2drive.window_stride,
            max_frame_gap=bench2drive.max_frame_gap,
            max_agents=bench2drive.max_agents,
            load_agent_annotations=bench2drive.load_agent_annotations,
            command_dim=bench2drive.command_dim,
            index_cache=bench2drive.index_cache,
            index_cache_dir=bench2drive.index_cache_dir,
            verify_image_exists=bench2drive.verify_image_exists,
            image_require_policy=resolve_bench2drive_image_require_policy(config),
            num_observed_frames=config.train.num_observed_frames,
            max_load_retries=bench2drive.max_load_retries,
            index_cache_wait_seconds=bench2drive.index_cache_wait_seconds,
            is_validation=False,
        )
    elif selected == "navsim":
        navsim = config.data.navsim
        if navsim is None:
            raise ValueError("Unexpected navsim config state")

        if not navsim.train_roots and (not navsim.data_path or not navsim.sensor_blobs_path):
            raise ValueError("data.navsim.data_path and data.navsim.sensor_blobs_path must be configured")

        if navsim.train_roots:
            logger.info(
                "Initializing balanced=%s NavSim mixed training roots: %d roots",
                navsim.balance_train_roots,
                len(navsim.train_roots),
            )
        else:
            logger.info(
                "Initializing NavSim training dataset from logs=%s, blobs=%s",
                navsim.data_path,
                navsim.sensor_blobs_path,
            )

        loader, sampler = init_navsim_data(
            data_path=navsim.data_path,
            sensor_blobs_path=navsim.sensor_blobs_path,
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            camera_name=navsim.camera_name,
            camera_names=navsim.camera_names if getattr(config.multiview, "enabled", False) else None,
            max_scenes=navsim.max_scenes,
            action_dim=config.train.action_dim,
            shuffle=True,
            index_cache=navsim.index_cache,
            window_stride=navsim.window_stride,
            max_frame_gap=navsim.max_frame_gap,
            max_agents=navsim.max_agents,
            load_agent_annotations=navsim.load_agent_annotations,
            image_require_policy=resolve_navsim_image_require_policy(config),
            num_observed_frames=config.train.num_observed_frames,
            scene_filter_yaml=navsim.scene_filter_yaml,
            pose_overlay_path=navsim.pose_overlay_path,
            pose_overlay_coord_frame=navsim.pose_overlay_coord_frame,
            pose_overlay_required=navsim.pose_overlay_required,
            tail_seconds=navsim.tail_seconds,
            counterfactual_tail_seconds=navsim.counterfactual_tail_seconds,
            dataset_roots=navsim.train_roots,
            balance_dataset_roots=navsim.balance_train_roots,
            atomic_real_cf_pairing=config.counterfactual_supervision.hazard_negative_pairing.enabled,
            counterfactual_supervision_v2=config.counterfactual_supervision.enabled,
            is_validation=False,
        )
    else:
        dataset_path = config.data.dataset_path
        if dataset_path is None:
            raise ValueError("Training dataset path is not configured")

        logger.info(f"Initializing training dataset from: {dataset_path}")

        from app.vjepa_cowa_world_model.training.seg_data import init_data_only_seg

        loader, sampler = init_data_only_seg(
            data_path=dataset_path,
            batch_size=config.data.batch_size,
            fps=config.data.fps,
            camera_views=config.data.camera_views,
            camera_frame=config.data.camera_frame,
            frames_per_clip=config.data.num_target_frames,
            stereo_view=config.data.stereo_view,
            tubelet_size=1,  # 训练时不使用 tubelet，保持与验证一致
            transform=transform,
            collator=None,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            load_segmentation=config.segmentation.use_segmentation,
            seg_data_root=config.segmentation.seg_data_root,
            crop_size=config.data.crop_size,
            action_dim=config.train.action_dim,
            is_train=True,
            is_validation=False,
        )

    logger.info(f"Training dataset initialized with {len(loader)} batches")

    return loader, sampler


def create_val_dataloader(
    config: TrainingConfig,
    rank: int,
    world_size: int,
    transform: Any = None,
    validation_domain: Optional[str] = None,
) -> Tuple[Optional[DataLoader], Optional[Sampler[int]]]:
    """
    创建验证数据加载器

    Args:
        config: 训练配置
        rank: 当前进程的 rank
        world_size: 进程总数
        transform: 数据增强 transform (可选，如果为 None 则自动创建)

    Returns:
        Tuple[Optional[DataLoader], Optional[Sampler[int]]]: (dataloader, exact validation sampler)
    """
    validation_scopes = {None, "real", "counterfactual", "matched_real_counterfactual"}
    if validation_domain not in validation_scopes:
        raise ValueError(
            "validation_domain must be None, 'real', 'counterfactual', or "
            f"'matched_real_counterfactual', got {validation_domain!r}"
        )
    selected = _select_world_model_dataset(config)
    if validation_domain is not None and selected != "navsim":
        raise ValueError("validation_domain filtering is supported only for NavSim validation roots")
    if selected == "mongo_raw":
        from .mongo_raw_data import init_mongo_raw_data

        mongo_raw = config.data.mongo_raw
        if mongo_raw is None:
            raise ValueError("Unexpected mongo_raw config state")

        if config.segmentation.use_segmentation:
            raise ValueError("Mongo raw online loader does not support segmentation in the first version")

        if transform is None:
            transform = create_validation_transforms(config)
        if not bool(getattr(transform, "is_validation_transform", False)):
            raise ValueError("Validation requires a deterministic transform from create_validation_transforms()")

        logger.info(
            "Initializing Mongo raw validation dataset from database=%s collection=%s "
            "vehicle_type=%s vehicle_types=%s",
            mongo_raw.database,
            mongo_raw.collection,
            mongo_raw.vehicle_type,
            mongo_raw.vehicle_types,
        )

        try:
            loader, sampler = init_mongo_raw_data(
                mongo_cfg=mongo_raw,
                split="val",
                batch_size=config.data.batch_size,
                frames_per_clip=config.data.num_target_frames,
                fps=config.data.fps,
                tubelet_size=1,
                transform=transform,
                num_workers=config.data.num_workers,
                world_size=world_size,
                pin_mem=config.data.pin_mem,
                persistent_workers=config.data.persistent_workers,
                rank=rank,
                action_dim=config.train.action_dim,
                shuffle=False,
                drop_last=False,
                is_validation=True,
            )
        except ValueError as exc:
            raise ValueError(
                f"Mongo-raw validation dataset is configured but could not be built: {exc}. "
                "Fix the val config, or disable validation explicitly (empty val paths)."
            ) from exc
    elif selected == "bench2drive":
        from .b2d_data import init_bench2drive_data

        bench2drive = config.data.bench2drive
        if bench2drive is None:
            raise ValueError("Unexpected bench2drive config state")

        if not bench2drive.val_ann_file or not bench2drive.data_root:
            logger.warning("Bench2Drive val_ann_file or data_root is not configured. Validation will be skipped.")
            return None, None

        if not os.path.exists(bench2drive.val_ann_file) or not os.path.exists(bench2drive.data_root):
            raise FileNotFoundError(
                "Bench2Drive validation paths are configured but do not exist "
                f"(ann_file={bench2drive.val_ann_file!r}, data_root={bench2drive.data_root!r}). "
                "Fix them, or clear them to intentionally train without validation."
            )

        if transform is None:
            transform = create_validation_transforms(config)
        if not bool(getattr(transform, "is_validation_transform", False)):
            raise ValueError("Validation requires a deterministic transform from create_validation_transforms()")
        proposal_transform = create_proposal_transforms(config)

        logger.info(
            "Initializing Bench2Drive validation dataset from ann_file=%s, data_root=%s",
            bench2drive.val_ann_file,
            bench2drive.data_root,
        )

        loader, sampler = init_bench2drive_data(
            ann_file=bench2drive.val_ann_file,
            data_root=bench2drive.data_root,
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            base_fps=bench2drive.base_fps,
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            camera_name=bench2drive.camera_name,
            max_scenes=bench2drive.max_val_scenes,
            action_dim=config.train.action_dim,
            shuffle=False,
            window_stride=(
                bench2drive.val_window_stride
                if bench2drive.val_window_stride is not None
                else bench2drive.window_stride
            ),
            max_frame_gap=bench2drive.max_frame_gap,
            max_agents=bench2drive.max_agents,
            load_agent_annotations=bench2drive.load_agent_annotations,
            command_dim=bench2drive.command_dim,
            index_cache=bench2drive.index_cache,
            index_cache_dir=bench2drive.index_cache_dir,
            verify_image_exists=bench2drive.verify_image_exists,
            image_require_policy=resolve_bench2drive_image_require_policy(config),
            num_observed_frames=config.train.num_observed_frames,
            max_load_retries=bench2drive.max_load_retries,
            index_cache_wait_seconds=bench2drive.index_cache_wait_seconds,
            drop_last=False,
            is_validation=True,
        )
    elif selected == "navsim":
        navsim = config.data.navsim
        if navsim is None:
            raise ValueError("Unexpected navsim config state")
        validation_data_semantics = resolve_navsim_validation_data_semantics(config)

        val_roots = list(navsim.val_roots or [])
        if validation_domain in {"real", "counterfactual"}:
            if not val_roots:
                raise ValueError("validation_domain filtering requires data.navsim.val_roots")
            val_roots = [root for root in val_roots if root.get("domain") == validation_domain]
            if not val_roots:
                raise ValueError(f"data.navsim.val_roots contains no {validation_domain!r} validation root")
        elif validation_domain == "matched_real_counterfactual":
            if not val_roots:
                raise ValueError("matched_real_counterfactual validation requires data.navsim.val_roots")
            domains = [root.get("domain") for root in val_roots]
            if len(val_roots) != 2 or sorted(domains) != ["counterfactual", "real"]:
                raise ValueError(
                    "matched_real_counterfactual validation requires exactly one real and one counterfactual root"
                )
        val_data_path = navsim.val_data_path
        val_sensor_blobs_path = navsim.val_sensor_blobs_path

        # point 13: 区分"未配置"与"已配置但路径不存在"。
        # 未配置（空/None）→ 保留 (None, None) 跳过语义：smoke / 无验证训练显式依赖此契约
        # （见 create_val_dataloader 调用方 train_world_model.py 的 predictor_validation_enabled，
        #  以及 navsim-world_model-smoke.yaml 的 val_data_path: ""）。
        if not val_roots and (not val_data_path or not val_sensor_blobs_path):
            logger.info("NavSim val paths not configured (empty/None); validation will be skipped.")
            return None, None

        # fail-loud（point 13 核心）：已配置 val 路径却不存在（多为拼错路径），直接报错，
        # 禁止 warning 后静默跳过——那会让"看似训练正常但从未验证"的 run 被误判为健康。
        if val_roots:
            missing_root_paths = [
                (root["name"], field_name, root[field_name])
                for root in val_roots
                for field_name in ("data_path", "sensor_blobs_path")
                if not os.path.exists(root[field_name])
            ]
            if missing_root_paths:
                raise FileNotFoundError(f"NavSim val_roots contain missing paths: {missing_root_paths}")
        elif not os.path.exists(val_data_path) or not os.path.exists(val_sensor_blobs_path):
            raise FileNotFoundError(
                "NavSim validation path 已配置但不存在:\n"
                f"  logs: {val_data_path}\n"
                f"  sensor_blobs: {val_sensor_blobs_path}"
            )

        if transform is None:
            transform = create_validation_transforms(config)
        if not bool(getattr(transform, "is_validation_transform", False)):
            raise ValueError("Validation requires a deterministic transform from create_validation_transforms()")
        proposal_transform = create_proposal_transforms(config)

        if val_roots:
            logger.info("Initializing NavSim multi-root validation dataset: roots=%s", [r["name"] for r in val_roots])
        else:
            logger.info(
                "Initializing NavSim validation dataset from logs=%s, blobs=%s",
                val_data_path,
                val_sensor_blobs_path,
            )

        loader, sampler = init_navsim_data(
            data_path=val_data_path,
            sensor_blobs_path=val_sensor_blobs_path,
            batch_size=config.data.batch_size,
            frames_per_clip=validation_data_semantics["frames_per_clip"],
            fps=validation_data_semantics["fps"],
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            camera_name=validation_data_semantics["camera_name"],
            camera_names=validation_data_semantics["camera_names"],
            max_scenes=validation_data_semantics["max_scenes"],
            action_dim=config.train.action_dim,
            shuffle=False,
            index_cache=navsim.index_cache,
            window_stride=validation_data_semantics["window_stride"],
            max_frame_gap=validation_data_semantics["max_frame_gap"],
            max_agents=validation_data_semantics["max_agents"],
            load_agent_annotations=validation_data_semantics["load_agent_annotations"],
            image_require_policy=validation_data_semantics["image_require_policy"],
            num_observed_frames=validation_data_semantics["num_observed_frames"],
            scene_filter_yaml=navsim.val_scene_filter_yaml,
            pose_overlay_path=navsim.val_pose_overlay_path,
            pose_overlay_coord_frame=validation_data_semantics["pose_overlay_coord_frame"],
            pose_overlay_required=validation_data_semantics["pose_overlay_required"],
            tail_seconds=validation_data_semantics["tail_seconds"],
            counterfactual_tail_seconds=validation_data_semantics["counterfactual_tail_seconds"],
            annotations_path=navsim.val_annotations_path,
            annotations_drop_distorted=navsim.val_annotations_drop_distorted,
            annotation_selection=navsim.val_annotation_selection,
            dataset_domain=navsim.val_domain or "real",
            dataset_root_name="validation",
            dataset_roots=val_roots,
            balance_dataset_roots=validation_domain == "matched_real_counterfactual",
            atomic_real_cf_pairing=validation_domain == "matched_real_counterfactual",
            counterfactual_supervision_v2=config.counterfactual_supervision.enabled,
            drop_last=False,
            is_validation=True,
        )
    else:
        val_dataset_path = config.data.val_dataset_path

        if val_dataset_path is None or not os.path.exists(val_dataset_path):
            if val_dataset_path:
                raise FileNotFoundError(
                    f"Validation dataset path is configured but does not exist: {val_dataset_path!r}. "
                    "Fix the path, or leave val_dataset_path empty to intentionally train without validation."
                )
            return None, None

        if transform is None:
            transform = create_validation_transforms(config)
        if not bool(getattr(transform, "is_validation_transform", False)):
            raise ValueError("Validation requires a deterministic transform from create_validation_transforms()")

        logger.info(f"Initializing validation dataset from: {val_dataset_path}")

        from app.vjepa_cowa_world_model.training.seg_data import init_data_only_seg

        loader, sampler = init_data_only_seg(
            data_path=val_dataset_path,
            batch_size=config.data.batch_size,
            fps=config.data.fps,
            camera_views=config.data.camera_views,
            camera_frame=config.data.camera_frame,
            frames_per_clip=config.data.num_target_frames,
            stereo_view=config.data.stereo_view,
            tubelet_size=1,  # 验证时不使用 tubelet，保持与训练一致
            transform=transform,
            collator=None,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            load_segmentation=False,  # 验证时不需要分割标注
            seg_data_root="",
            crop_size=config.data.crop_size,
            action_dim=config.train.action_dim,
            shuffle=False,
            is_train=False,
            is_validation=True,
            drop_last=False,
        )

    logger.info(f"Validation dataset initialized with {len(loader)} batches")

    return loader, sampler


def resolve_planner_validation_domain(config: TrainingConfig) -> Optional[str]:
    """Keep legacy checkpoint metrics real-only when a configured suite is disabled."""

    navsim = config.data.navsim
    has_cf_multi_roots = (
        navsim is not None
        and bool(navsim.val_roots)
        and {root.get("domain") for root in navsim.val_roots} == {"real", "counterfactual"}
    )
    cvoi = getattr(config, "cvoi", None)
    cvoi_planner_training = bool(getattr(cvoi, "enabled", False)) and str(getattr(cvoi, "stage", "")) in {
        "unguided_planner",
        "guided_planner",
    }
    if (
        has_cf_multi_roots
        and (bool(config.counterfactual_supervision.enabled) or cvoi_planner_training)
        and not bool(config.validation_suite.enabled)
    ):
        return "real"
    return None


def calculate_iterations_per_epoch(config: TrainingConfig, dataloader: DataLoader) -> int:
    """
    计算每个 epoch 的迭代次数。

    默认根据 dataloader 长度自动计算 ipe（推荐），无需在 YAML 中手动维护。
    当 batch_size 或 world_size（节点数/GPU数）变化时，ipe 会自动适配。

    若 YAML 中显式设置了 ipe（非 null），则作为手动覆盖使用，
    并在与 dataloader 长度不一致时输出警告日志。

    Args:
        config: 训练配置
        dataloader: 数据加载器

    Returns:
        int: 每个 epoch 的迭代次数
    """
    dataset_len = len(dataloader)
    config_ipe = config.optimization.ipe

    if config_ipe is not None:
        # 显式覆盖：使用配置值，但在不一致时发出警告
        if config_ipe != dataset_len:
            logger.warning(
                f"Config ipe ({config_ipe}) differs from dataloader length ({dataset_len}). "
                f"Using config override. Set ipe to null in YAML for auto-detection."
            )
        ipe = config_ipe
    else:
        # 自动计算（推荐）：ipe = len(dataloader) = ceil(dataset_size / world_size) / batch_size
        ipe = dataset_len
        logger.info(f"Auto-detected ipe from dataloader: {ipe}")

    logger.info(f"iterations per epoch: {ipe} (dataloader length: {dataset_len})")

    return ipe
