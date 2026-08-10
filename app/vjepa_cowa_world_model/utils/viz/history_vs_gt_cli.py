# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

"""离线诊断脚本：可视化 observed action history 与 future GT trajectory。"""

import argparse
import importlib.util
import os
from pathlib import Path

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_local_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[2] / relative_path  # wm package root
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if not (spec.loader is not None):
        raise AssertionError("assertion failed: spec.loader is not None")
    spec.loader.exec_module(module)
    return module


_status_mod = _load_local_module("history_gt_status_features", "utils/status_features.py")
_vis_mod = _load_local_module("history_gt_visualization_utils", "utils/visualization.py")


def _resolve_action_history_dt(config) -> float:
    diff_dt = getattr(config.planner, "diff_dt", None)
    if diff_dt is not None and float(diff_dt) > 0:
        return float(diff_dt)

    fps = getattr(config.data, "fps", None)
    if fps is not None and float(fps) > 0:
        return 1.0 / float(fps)

    return 1.0


def _resolve_status_helper(helper_name: str):
    # Fail-loud: visualization must use the same status/trajectory helpers as training. If the
    # shared status_features module stops exposing one, raise rather than silently diverging onto
    # a local reimplementation (exactly the train/viz inconsistency this path exists to prevent).
    helper = getattr(_status_mod, helper_name, None)
    if helper is None:
        raise AttributeError(
            f"status_features does not expose {helper_name!r}; visualization requires the same "
            "helper training uses — refusing to fall back to a local reimplementation."
        )
    return helper


def extract_actions_states_from_sample(sample, sample_index: int = 0):
    """从 dataloader batch tuple 中提取单个样本的 actions/states。"""
    if len(sample) < 3:
        raise ValueError("Expected sample tuple to contain at least context/actions/states")

    actions = sample[1]
    states = sample[2]
    if actions.ndim != 3:
        raise ValueError(f"Expected batched actions [B, T, A], got {tuple(actions.shape)}")
    if states.ndim != 3:
        raise ValueError(f"Expected batched states [B, T, D], got {tuple(states.shape)}")
    if sample_index < 0 or sample_index >= actions.shape[0] or sample_index >= states.shape[0]:
        raise IndexError(
            f"sample_index={sample_index} is out of range for batch size "
            f"actions={actions.shape[0]}, states={states.shape[0]}"
        )

    return actions[sample_index], states[sample_index]


def build_history_gt_from_sample(
    sample,
    num_observed_frames: int,
    num_poses: int = None,
    sample_index: int = 0,
    action_history_dim: int = 3,
    action_history_dt: float = 1.0,
):
    """从训练/验证 dataloader 的 batch tuple 构建单样本 history / GT 轨迹。"""
    sample_actions, sample_states = extract_actions_states_from_sample(sample, sample_index=sample_index)
    batched_actions = sample_actions.unsqueeze(0)
    batched_states = sample_states.unsqueeze(0)

    history_builder = _resolve_status_helper("build_observed_action_trajectory_history")
    gt_builder = _resolve_status_helper("build_future_gt_trajectory_from_states")

    history = history_builder(
        batched_actions,
        num_observed_frames=num_observed_frames,
        action_history_dim=action_history_dim,
        dt=action_history_dt,
    )[0]
    gt = gt_builder(
        batched_states,
        num_observed_frames=num_observed_frames,
        num_poses=num_poses,
    )[0]
    return history, gt


def load_training_config_from_yaml(config_path: str):
    """加载 YAML 并解析为 TrainingConfig。"""
    from app.vjepa_cowa_world_model.training.config import parse_training_config

    with open(config_path, "r") as handle:
        raw_config = yaml.safe_load(handle)
    return parse_training_config(raw_config)


def create_visualization_loader(config, split: str):
    """按训练配置构建可视化所需的 dataloader。"""
    from app.vjepa_cowa_world_model.training.data import (
        create_train_dataloader,
        create_transforms,
        create_val_dataloader,
        create_validation_transforms,
    )

    transform = create_transforms(config)
    validation_transform = create_validation_transforms(config)
    train_loader = train_sampler = None
    val_loader = val_sampler = None

    if split in {"train", "auto"}:
        train_loader, train_sampler = create_train_dataloader(config, rank=0, world_size=1, transform=transform)
    if split in {"val", "auto"}:
        val_loader, val_sampler = create_val_dataloader(
            config,
            rank=0,
            world_size=1,
            transform=validation_transform,
        )

    if split == "train":
        return train_loader, train_sampler, "train"
    if split == "val":
        if val_loader is None:
            raise ValueError("Validation dataloader is unavailable for this config")
        return val_loader, val_sampler, "val"

    if val_loader is not None:
        return val_loader, val_sampler, "val"
    if train_loader is not None:
        return train_loader, train_sampler, "train"
    raise ValueError("No dataloader could be created from the provided config")


def get_batch_from_loader(loader, batch_index: int):
    """获取 dataloader 的第 N 个 batch。"""
    if loader is None:
        raise ValueError("loader is None")
    if batch_index < 0:
        raise ValueError(f"batch_index must be >= 0, got {batch_index}")

    for current_index, sample in enumerate(loader):
        if current_index == batch_index:
            return sample
    available_batches = current_index + 1 if "current_index" in locals() else 0
    raise IndexError(f"batch_index={batch_index} exceeds available batches={available_batches}")


def save_raw_clip_video_for_sample(
    loader,
    batch_index: int,
    sample_index: int,
    output_dir: str,
    file_prefix: str,
    fps: int,
):
    """导出与当前 batch/sample 对应的原始 RGB clip 视频。"""
    import cv2

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise ValueError("loader does not expose a dataset")

    batch_size = getattr(loader, "batch_size", None)
    if batch_size is None or batch_size < 1:
        raise ValueError(f"loader.batch_size must be >= 1, got {batch_size}")

    dataset_index = batch_index * batch_size + sample_index
    if dataset_index < 0 or dataset_index >= len(dataset):
        raise IndexError(f"Resolved dataset_index={dataset_index} is out of range for dataset size {len(dataset)}")

    if not hasattr(dataset, "windows") or not hasattr(dataset, "scenes"):
        raise ValueError("Raw clip export currently supports datasets exposing windows/scenes metadata only")

    window = dataset.windows[dataset_index]
    scene = dataset.scenes[window.scene_idx]
    frames = dataset._load_scene_frames(scene.pkl_path)
    sampled_frame_indices = dataset._window_frame_indices(scene.valid_frame_indices, window.start_pos)
    clip = dataset._load_clip_images(scene, frames, sampled_frame_indices)

    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"Expected clip shape [T, H, W, 3], got {tuple(clip.shape)}")

    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{file_prefix}_raw.mp4")
    height, width = int(clip.shape[1]), int(clip.shape[2])

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(fps)),
        (width, height),
    )
    try:
        for frame in clip:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return {
        "video_path": video_path,
        "scene_name": scene.scene_name,
        "sampled_frame_indices": list(sampled_frame_indices),
        "dataset_index": dataset_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize observed action history vs GT future trajectory")
    parser.add_argument("--config", type=str, required=True, help="Training YAML config used to build the dataloader")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save diagnostic figures")
    parser.add_argument(
        "--split",
        type=str,
        choices=["auto", "train", "val"],
        default="auto",
        help="Which dataloader split to visualize; auto prefers val then falls back to train",
    )
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index inside the chosen dataloader")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index inside the selected batch")
    parser.add_argument(
        "--num-observed-frames",
        type=int,
        default=None,
        help="Override the observed frame count; defaults to config.train.num_encoder_frames",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=None,
        help="Optional truncation length for future GT poses; default uses the full future horizon",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=None,
        help="Optional output file prefix; default uses the trajectory directory name",
    )
    parser.add_argument(
        "--save-raw-video",
        action="store_true",
        help="Also export the raw RGB clip corresponding to the selected batch/sample as an mp4",
    )
    parser.add_argument(
        "--raw-video-fps",
        type=int,
        default=None,
        help="Optional fps override for the exported raw video; defaults to config.data.fps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_training_config_from_yaml(args.config)
    loader, _sampler, resolved_split = create_visualization_loader(config, split=args.split)
    sample = get_batch_from_loader(loader, batch_index=args.batch_index)

    num_observed_frames = args.num_observed_frames
    if num_observed_frames is None:
        num_observed_frames = int(getattr(config.train, "num_encoder_frames", config.train.num_observed_frames))

    num_poses = args.num_poses
    if num_poses is None:
        num_poses = max(0, int(config.data.num_target_frames) - int(num_observed_frames))

    history, gt = build_history_gt_from_sample(
        sample,
        num_observed_frames=num_observed_frames,
        num_poses=num_poses,
        sample_index=args.sample_index,
        action_history_dim=int(getattr(config.planner, "action_history_dim", 3)),
        action_history_dt=_resolve_action_history_dt(config),
    )

    if gt.shape[0] == 0:
        raise ValueError(
            "No future GT poses available after the observed window. "
            "Reduce --num-observed-frames or provide a longer trajectory."
        )

    file_prefix = args.file_prefix or f"{resolved_split}_b{args.batch_index}_s{args.sample_index}"
    saved_paths = _vis_mod.visualize_history_vs_gt(
        history_traj=history.unsqueeze(0),
        gt_traj=gt.unsqueeze(0),
        output_dir=args.output_dir,
        file_prefix=file_prefix,
        limit=1,
    )
    logger.info(
        "Saved %d history-vs-gt diagnostic figure(s) to %s using %s dataloader batch %d sample %d",
        len(saved_paths),
        args.output_dir,
        resolved_split,
        args.batch_index,
        args.sample_index,
    )
    for save_path in saved_paths:
        logger.info("  %s", save_path)

    if args.save_raw_video:
        raw_video_meta = save_raw_clip_video_for_sample(
            loader=loader,
            batch_index=args.batch_index,
            sample_index=args.sample_index,
            output_dir=args.output_dir,
            file_prefix=file_prefix,
            fps=args.raw_video_fps if args.raw_video_fps is not None else int(config.data.fps),
        )
        logger.info(
            "Saved raw clip video to %s (scene=%s, dataset_index=%d, frame_indices=%s)",
            raw_video_meta["video_path"],
            raw_video_meta["scene_name"],
            raw_video_meta["dataset_index"],
            raw_video_meta["sampled_frame_indices"],
        )


if __name__ == "__main__":
    main()
