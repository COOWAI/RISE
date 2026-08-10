"""Seg-only dataset pipeline (standalone legacy; the original app/vjepa_cowa/cowa.py source is gone)."""

import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from decord import VideoReader
from scipy.spatial.transform import Rotation

import src.datasets.utils.video.transforms as video_transforms
from app.vjepa_cowa_world_model.training import collate
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AutonomousDrivingDatasetOnlySeg(torch.utils.data.Dataset):
    """自动驾驶视频数据集 - 兼容V-JEPA训练 + 分割标注"""

    def __init__(
        self,
        data_path,
        camera_views=["CAM_FRONT"],
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        frameskip=2,
        load_segmentation=True,  # 新增：是否加载分割标注
        seg_data_root="/path/to/segmentation/annotations",  # 新增：分割数据根目录
        crop_size=256,
        num_seg_sample=4,
        action_dim=7,  # action 维度: 7 (机器人) 或 4 (自动驾驶)
        is_train=True,
        is_validation=False,
    ):
        """
        Args:
            data_path: train.txt文件路径
            camera_views: 要使用的相机列表
            frames_per_clip: 每个clip的帧数
            fps: 目标帧率
            transform: 数据增强
            camera_frame: 是否转换到相机坐标系
            frameskip: 帧间隔（tubelet_size）
            load_segmentation: 是否加载分割标注
            seg_data_root: 分割标注数据根目录
            action_dim: action 维度 (7 或 4)
        """
        self.data_path = data_path
        self.camera_views = camera_views
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.frameskip = frameskip
        self.load_segmentation = load_segmentation
        self.seg_data_root = seg_data_root
        self.crop_size = crop_size
        self.num_seg_sample = num_seg_sample
        self.action_dim = action_dim
        self.is_train = is_train
        self.is_validation = bool(is_validation)
        self.dataset_root_name = os.path.splitext(os.path.basename(data_path))[0]
        if not self.dataset_root_name:
            raise ValueError(f"Cannot derive stable dataset root name from data_path={data_path!r}")
        # 加载样本列表
        with open(data_path, "r") as f:
            self.samples = [line.strip() for line in f.readlines()]

        logger.info(f"加载了 {len(self.samples)} 个训练样本")
        if self.load_segmentation:
            logger.info(f"将从 {self.seg_data_root} 加载分割标注")
        logger.info(f"Action维度: {action_dim} ({'自动驾驶' if action_dim == 4 else '机器人'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]

        max_retries = 5
        for retry in range(max_retries):
            try:
                data = self.load_clip(path)
                if not isinstance(data, dict):
                    raise TypeError(f"load_clip must return a dict, got {type(data).__name__}")
                stable_sample_id = collate.make_stable_sample_id(
                    "seg",
                    self.dataset_root_name,
                    int(index),
                    os.path.basename(os.path.normpath(path)),
                )
                data["stable_sample_id"] = stable_sample_id
                data["sample_id"] = stable_sample_id
                return data
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"加载失败 {path} (retry {retry + 1}/{max_retries}): {e}")
                    if not self.is_validation:
                        index = np.random.randint(len(self))
                        path = self.samples[index]
                    logger.warning(f"error is {e}")
                else:
                    raise e

    def load_segmentation_masks_robust(
        self, trajectory_name, indices, color_tolerance=10, visualize=False, save_vis_path=None
    ):
        """鲁棒的mask加载方法（带可视化）"""

        mask_video_path = os.path.join(self.seg_data_root, trajectory_name, "mask_video.mp4")

        if not os.path.exists(mask_video_path):
            logger.warning(f"警告：分割标注不存在: {mask_video_path}")
            return None

        try:
            # 读取视频
            # target_index = indices[0]
            vr = VideoReader(mask_video_path, num_threads=0)
            vlen = len(vr)
            safe_indices = [idx for idx in indices if idx < vlen]
            if len(safe_indices) == 0:
                logger.warning(f"warning indices over mask video range{vlen}")
            vr.seek(0)
            mask_frames = vr.get_batch(safe_indices).asnumpy()  # [T, H, W, C]

            if self.transform is not None:
                buf = torch.tensor(mask_frames, dtype=torch.uint8)
                buf = buf.permute(3, 0, 1, 2)

                buf = self.transform.spatial_transform(
                    images=buf,
                    target_height=self.transform.crop_size,
                    target_width=self.transform.crop_size,
                    scale=self.transform.random_resize_scale,
                    ratio=self.transform.random_resize_aspect_ratio,
                )

                # 可能需要水平翻转
                if self.transform.random_horizontal_flip:
                    buf, _ = video_transforms.horizontal_flip(0.5, buf)

                mask_frames = buf.permute(1, 2, 3, 0).numpy()

            # 判断是否为黑色：所有通道都为0
            # is_black = np.all(mask_frames == 0, axis=-1, keepdims=True)  # [T, H, W, 1]
            is_black = np.all(mask_frames <= color_tolerance, axis=-1, keepdims=True)
            # 创建二值mask：黑色=255，有色=1
            binary_mask = np.where(is_black, 255, 1).astype(np.uint8)  # [T, H, W, 1]

            # 扩展为单通道mask格式 [T, 1, H, W] 以兼容后续可能的处理
            binary_mask = binary_mask.squeeze(-1)  # [T, H, W]
            masks_np = binary_mask[:, np.newaxis, :, :]  # [T, 1, H, W]
            # save_vis_path = "/path/to/rise/results/segmentation/preview.png"
            # 如果需要可视化
            if visualize:
                # 创建可视化用的彩色图（可选：将255显示为白色，1显示为红色）
                vis_frames = np.repeat(binary_mask[:, :, :, np.newaxis], 3, axis=-1)  # [T, H, W, 3]
                # 255 -> 白色 [255,255,255], 1 -> 红色 [255,0,0]
                vis_frames[binary_mask == 255] = [255, 255, 255]
                vis_frames[binary_mask == 1] = [255, 0, 0]

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(mask_frames[0])  # 原始mask
                plt.title("Original Mask")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(vis_frames[0])  # 二值化结果
                plt.title("Binary Mask (255=black, 1=color)")
                plt.axis("off")

                if save_vis_path:
                    plt.savefig(save_vis_path, bbox_inches="tight", dpi=150)
                    logger.info(f"可视化结果已保存: {save_vis_path}")
                plt.show()
            return torch.from_numpy(masks_np)

        except Exception as e:
            logger.warning(f"加载分割标注失败 {mask_video_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def load_clip(self, path):
        """加载单个clip的数据（包含分割标注）"""
        trajectory_name = os.path.basename(path)
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # 2. 随机选择一个可用的相机
        available_cameras = [cam for cam in self.camera_views if f"{cam.lower()}_mp4_path" in metadata]

        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")

        camera_view = self._select_camera_view(available_cameras)

        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, "r") as f:
            # 加载状态 [T, 7]: [x, y, z, roll, pitch, yaw, velocity]
            cartesian_pos = f["observation/robot_state/cartesian_position"][:]
            velocity = f["observation/robot_state/gripper_position"][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)

            extrinsics = f[f"observation/camera_extrinsics/{camera_view}_left"][:]

        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        vr = VideoReader(video_path, num_threads=0)
        vlen = len(vr)

        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.frames_per_clip * fstp

        if vlen < nframes:
            nframes = vlen
            fstp = max(1, vlen // self.frames_per_clip)

        if self.is_train:
            start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        else:
            start_frame = max(0, (vlen - nframes) // 2)
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[: self.frames_per_clip]

        if len(indices) < self.frames_per_clip:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(
                indices, (0, self.frames_per_clip - len(indices)), mode="constant", constant_values=last_index
            )

        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

        if self.transform is not None:
            buffer = self.transform(buffer)

        # 9. 提取对应的状态和外参
        states = states[indices][:: self.frameskip]
        extrinsics = extrinsics[indices][:: self.frameskip]

        # 10. 计算动作（根据 action_dim 选择版本）
        if self.action_dim == 4:
            actions = self.compute_actions_4d(states)
        else:
            actions = self.compute_actions(states)

        # ==================== 新增：加载分割标注 ====================
        seg_masks_tensor = None
        seg_indices_tensor = None
        if self.load_segmentation:
            clip_len = len(indices)
            valid_choices = list(range(0, clip_len, self.frameskip))

            # if len(valid_choices) >= self.num_seg_sample:
            #     selected_relative_indices = sorted(random.sample(valid_choices,self.num_seg_sample))
            # else:
            selected_relative_indices = valid_choices
            target_mask_indices = [indices[i] for i in selected_relative_indices]

            # 3. 只加载这几帧
            # Returns: [K, N, H, W] (K = num_seg_samples)
            sampled_masks = self.load_segmentation_masks_robust(trajectory_name, target_mask_indices)

            if sampled_masks is not None:
                if len(sampled_masks) == len(selected_relative_indices):
                    seg_masks_tensor = sampled_masks
                    # 记录这几帧对应的是 Clip 中的第几个时间步 (0~15)
                    # 这对后续 Loss 切片至关重要
                    seg_indices_tensor = torch.tensor(selected_relative_indices, dtype=torch.long)
                else:
                    logger.info(
                        "Warning: Mask frame count mismatch. "
                        f"Req: {len(target_mask_indices)}, Got: {len(sampled_masks)}"
                    )
                    # 简单处理：截断
                    min_len = min(len(sampled_masks), len(selected_relative_indices))
                    seg_masks_tensor = sampled_masks[:min_len]
                    seg_indices_tensor = torch.tensor(selected_relative_indices[:min_len], dtype=torch.long)

        return {
            "buffer": buffer,
            "actions": actions,
            "states": states,
            "extrinsics": extrinsics,
            "indices": indices,
            "seg_masks": seg_masks_tensor,  # [K, N, H, W] 这里的K很小(如4)
            "seg_frame_indices": seg_indices_tensor,  # [K]
        }

    def _select_camera_view(self, available_cameras):
        if not available_cameras:
            raise ValueError("available_cameras must be non-empty")
        if self.is_train:
            return available_cameras[np.random.randint(len(available_cameras))]
        return available_cameras[0]

    def compute_actions(self, states):
        """
        计算动作序列 - 7维版本（机器人领域）

        Args:
            states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]

        Returns:
            actions: [T-1, 7] - [dx, dy, dz, d_roll, d_pitch, d_yaw, velocity]
        """
        T = len(states)
        actions = np.zeros((T - 1, 7))

        for t in range(T - 1):
            # 位置差分
            xyz_diff = states[t + 1, :3] - states[t, :3]

            # 旋转差分
            R1 = Rotation.from_euler("xyz", states[t, 3:6]).as_matrix()
            R2 = Rotation.from_euler("xyz", states[t + 1, 3:6]).as_matrix()
            R_diff = R2 @ R1.T
            angle_diff = Rotation.from_matrix(R_diff).as_euler("xyz")

            # 使用当前帧的速度
            velocity = states[t, 6]

            actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])

        return actions

    def compute_actions_4d(self, states):
        """
        计算动作序列 - 4维版本（自动驾驶领域）

        与 7 维版本的区别：
        - 去掉 z 轴差分（自动驾驶主要在平面运动）
        - 去掉 roll/pitch 差分（车辆姿态相对稳定）
        - 只保留 yaw 差分（航向角变化是关键）
        - 与 drive_command 的 4 维对齐

        Args:
            states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]

        Returns:
            actions: [T-1, 4] - [dx, dy, dyaw, velocity]
        """
        T = len(states)
        actions = np.zeros((T - 1, 4))

        for t in range(T - 1):
            # 位置差分（只用 x, y）
            dx = states[t + 1, 0] - states[t, 0]
            dy = states[t + 1, 1] - states[t, 1]

            # yaw 差分（直接相减后归一化到 [-pi, pi]）
            dyaw = states[t + 1, 5] - states[t, 5]
            dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))

            # 使用当前帧的速度
            velocity = states[t, 6]

            actions[t] = np.array([dx, dy, dyaw, velocity])

        return actions


def autonomous_driving_collate_fn_with_only_seg(batch):
    """处理分割标注的collate函数"""
    context_frames = torch.stack([item["buffer"] for item in batch])
    actions = torch.stack([torch.from_numpy(item["actions"]) for item in batch])
    states = torch.stack([torch.from_numpy(item["states"]) for item in batch])
    extrinsics = torch.stack([torch.from_numpy(item["extrinsics"]) for item in batch])

    # 处理分割标注
    seg_targets = []
    for item in batch:
        seg_mask = item.get("seg_masks", None)  # [T, N, H, W]
        indices = item.get("seg_frame_indices", None)
        if seg_mask is not None:
            seg_targets.append((seg_mask, indices))
        else:
            # 空target
            seg_targets.append(None)

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


def init_data_only_seg(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    camera_views=["CAM_FRONT"],
    transform=None,
    collator=None,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    rank=0,
    world_size=1,
    load_segmentation=True,  # 新增
    seg_data_root="/path/to/segmentation/annotations",  # 新增
    crop_size=256,
    num_seg_sample=4,
    action_dim=7,  # action 维度: 7 (机器人) 或 4 (自动驾驶)
    shuffle=True,
    is_train=True,
    is_validation=False,
    drop_last=True,
    **kwargs,
):
    """
    初始化自动驾驶数据加载器
    """
    dataset = AutonomousDrivingDatasetOnlySeg(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        camera_views=camera_views,
        transform=transform,
        frameskip=kwargs.get("tubelet_size", 2),
        camera_frame=kwargs.get("camera_frame", False),
        load_segmentation=load_segmentation,  # 新增
        seg_data_root=seg_data_root,  # 新增
        crop_size=crop_size,
        num_seg_sample=4,
        action_dim=action_dim,
        is_train=is_train,
        is_validation=is_validation,
    )

    logger.info(f"创建数据集: {len(dataset)} 个样本")

    # 使用支持分割标注的collate函数
    if collator is None:
        collator = autonomous_driving_collate_fn_with_only_seg

    # 分布式采样器
    if is_validation:
        if is_train:
            raise ValueError("Seg validation requires is_train=False")
        if shuffle or drop_last:
            raise ValueError("Seg validation requires shuffle=False and drop_last=False")
        from app.vjepa_cowa_world_model.training.samplers import ExactDistributedEvalSampler

        sampler = ExactDistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
        )

    # 数据加载器
    from app.vjepa_cowa_world_model.training.loop import make_dataloader_generator

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=False if is_validation else drop_last,
        prefetch_factor=kwargs.get("prefetch_factor", 2 if num_workers > 0 else None),
        generator=make_dataloader_generator(
            rank=rank,
            stream="seg/validation" if is_validation else "seg/train",
        ),
    )

    logger.info(f"数据加载器创建完成: batch_size={batch_size}, num_workers={num_workers}")

    return loader, sampler
