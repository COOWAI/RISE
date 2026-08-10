"""
VJEPAFeatureBuilder — 将 NavSim AgentInput 转换为世界模型所需的 tensor 格式。

AgentInput 包含 num_history_frames 帧的 ego_statuses + cameras，
本 builder 从中提取:
    - "video_clip"          : [C, T, H, W]  — ImageNet-normalized 前视摄像头序列
    - "proposal_video_clip" : [C, T, H, W]  — 可选，proposal encoder 专用分辨率
    - "states"              : [T, 7]        — [x, y, z, roll, pitch, yaw, speed]
    - "actions"             : [T-1, action_dim] — 帧间差分动作
    - "extrinsics"          : [T, 7]        — 全零 (世界模型不使用外参)
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from navsim.common.dataclasses import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
from scipy.spatial.transform import Rotation
from torchvision import transforms as T

from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import (
    observation_key,
    observation_key_tensor,
    unsigned_seed_tensor,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_sample_seed
from app.vjepa_cowa_world_model.training.vjepa_transforms import VJEPAImageTransform

logger = logging.getLogger(__name__)


def _normalize_crop_size(crop_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(crop_size, int):
        size = int(crop_size)
        return size, size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        return int(crop_size[0]), int(crop_size[1])
    raise ValueError(f"crop_size must be an int or a 2-element sequence, got {crop_size!r}")


class VJEPAFeatureBuilder(AbstractFeatureBuilder):
    """从 AgentInput 构建世界模型推理所需特征。"""

    def __init__(
        self,
        crop_size: int | Tuple[int, int] = 256,
        camera_name: str = "cam_f0",
        camera_names: Optional[List[str]] = None,
        action_dim: int = 3,
        crop_top_bottom: Optional[int] = None,
        proposal_crop_size: Optional[int | Tuple[int, int]] = None,
        proposal_camera_name: Optional[str] = None,
        proposal_crop_top_bottom: Optional[int] = None,
        official_cvoi_identity: bool = False,
        cvoi_evaluation_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if type(official_cvoi_identity) is not bool:
            raise TypeError(f"official_cvoi_identity must be a bool, got {type(official_cvoi_identity).__name__}")
        if official_cvoi_identity:
            if type(cvoi_evaluation_seed) is not int:
                raise TypeError(
                    "cvoi_evaluation_seed must be an unsigned 64-bit integer when official CVoI identity is enabled"
                )
            if cvoi_evaluation_seed < 0 or cvoi_evaluation_seed >= 1 << 64:
                raise ValueError(
                    "cvoi_evaluation_seed must be an unsigned 64-bit integer in [0, 2**64) when official CVoI "
                    f"identity is enabled, got {cvoi_evaluation_seed}"
                )
        elif cvoi_evaluation_seed is not None:
            raise ValueError("cvoi_evaluation_seed must be None when official CVoI identity is disabled")
        self._official_cvoi_identity = official_cvoi_identity
        self._cvoi_evaluation_seed = cvoi_evaluation_seed
        self.crop_size = crop_size
        self.crop_height, self.crop_width = _normalize_crop_size(crop_size)
        self.camera_name = camera_name
        self.camera_names = list(camera_names) if camera_names is not None else [camera_name]
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera")
        self.multiview_enabled = len(self.camera_names) > 1
        self.action_dim = action_dim
        self._main_transform = None
        if crop_top_bottom is not None:
            self._main_transform = VJEPAImageTransform(
                resolution=(self.crop_height, self.crop_width),
                crop_top_bottom=crop_top_bottom,
            )
        self._emit_proposal_video_clip = (
            proposal_crop_size is not None or proposal_camera_name is not None or proposal_crop_top_bottom is not None
        )
        self.proposal_camera_name = proposal_camera_name or camera_name
        if proposal_crop_size is None:
            self.proposal_crop_height, self.proposal_crop_width = self.crop_height, self.crop_width
        else:
            self.proposal_crop_height, self.proposal_crop_width = _normalize_crop_size(proposal_crop_size)
        self._proposal_transform = None
        if proposal_crop_top_bottom is not None:
            self._proposal_transform = VJEPAImageTransform(
                resolution=(self.proposal_crop_height, self.proposal_crop_width),
                crop_top_bottom=proposal_crop_top_bottom,
            )

        # ImageNet 标准化 (与训练 pipeline 保持一致)
        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def get_unique_name(self) -> str:
        return "vjepa_world_model_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        从 AgentInput 构建推理特征。

        Parameters
        ----------
        agent_input : AgentInput
            包含 ego_statuses, cameras, lidars 的历史帧列表。

        Returns
        -------
        dict
            "video_clip"       : [C, T, H, W]
            "states"           : [T, 7]
            "actions"          : [T-1, action_dim]
            "extrinsics"       : [T, 7]
            "driving_command"  : [T, 4]  — one-hot 导航指令（条件必需，不可用时省略该 key）
            "ego_dynamics"     : [T, 4]  — [vx, vy, ax, ay]（条件必需，不可用时省略该 key）
        """
        cvoi_identity_features: Optional[Dict[str, torch.Tensor]] = None
        if self._official_cvoi_identity:
            cameras = getattr(agent_input, "cameras", None)
            if not isinstance(cameras, (list, tuple)) or not cameras:
                raise ValueError("Official CVoI identity requires a non-empty AgentInput.cameras history")
            cam_f0 = getattr(cameras[-1], "cam_f0", None)
            raw_image = None if cam_f0 is None else getattr(cam_f0, "image", None)
            if raw_image is None:
                raise ValueError("Official CVoI identity requires the raw decoded final-history cam_f0 image")
            key = observation_key(raw_image)
            sample_seed = cvoi_sample_seed(self._cvoi_evaluation_seed, key)
            cvoi_identity_features = {
                "cvoi_observation_key": observation_key_tensor(key),
                "cvoi_rng_seed_bytes": unsigned_seed_tensor(sample_seed),
            }

        # ====== 1. 视频帧 ======
        if self.multiview_enabled:
            video_tensor = self._build_multiview_video_tensor(
                agent_input,
                camera_names=self.camera_names,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                transform=self._main_transform,
            )
            camera_intrinsics, camera2ego = self._build_multiview_camera_metadata(
                agent_input,
                camera_names=self.camera_names,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                transform=self._main_transform,
            )
        elif self._main_transform is not None:
            video_tensor = self._build_vjepa_video_tensor(
                agent_input,
                camera_name=self.camera_name,
                transform=self._main_transform,
            )
            camera_intrinsics, camera2ego = self._build_multiview_camera_metadata(
                agent_input,
                camera_names=[self.camera_name],
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                transform=self._main_transform,
            )
        else:
            video_tensor = self._build_video_tensor(
                agent_input,
                camera_name=self.camera_name,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
            )
            camera_intrinsics, camera2ego = self._build_multiview_camera_metadata(
                agent_input,
                camera_names=[self.camera_name],
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                transform=None,
            )

        # ====== 2. 状态 (ego states) ======
        states = self._build_states(agent_input)  # [T, 7]

        # ====== 3. 动作 (帧间差分) ======
        actions = self._build_actions(states)  # [T-1, action_dim]

        # ====== 4. 外参 (placeholder) ======
        extrinsics = np.zeros_like(states, dtype=np.float32)  # [T, 7]

        # ====== 5. 导航指令 (driving command) ======
        driving_command = self._build_driving_commands(agent_input)  # [T, 4] or None

        # ====== 6. 自车动力学 (ego dynamics) ======
        ego_dynamics = self._build_ego_dynamics(agent_input)  # [T, 4] or None

        features = {
            "video_clip": video_tensor,
            "states": torch.from_numpy(states).float(),
            "actions": torch.from_numpy(actions).float(),
            "extrinsics": torch.from_numpy(extrinsics).float(),
            "camera_intrinsics": torch.from_numpy(camera_intrinsics).float(),
            "camera2ego": torch.from_numpy(camera2ego).float(),
        }
        # 条件必需特征：不可用时省略 key（消费端统一用 features.get(...)，
        # 配置确实需要时由 status_features 的 fail-loud 把关），禁止用 zeros 占位。
        if driving_command is not None:
            features["driving_command"] = torch.from_numpy(driving_command).float()
        if ego_dynamics is not None:
            features["ego_dynamics"] = torch.from_numpy(ego_dynamics).float()
        if self._emit_proposal_video_clip:
            if self._proposal_transform is not None:
                features["proposal_video_clip"] = self._build_vjepa_video_tensor(
                    agent_input,
                    camera_name=self.proposal_camera_name,
                    transform=self._proposal_transform,
                )
            elif (
                self.proposal_camera_name == self.camera_name
                and self.proposal_crop_height == self.crop_height
                and self.proposal_crop_width == self.crop_width
            ):
                features["proposal_video_clip"] = video_tensor
            else:
                features["proposal_video_clip"] = self._build_video_tensor(
                    agent_input,
                    camera_name=self.proposal_camera_name,
                    crop_height=self.proposal_crop_height,
                    crop_width=self.proposal_crop_width,
                )
        if cvoi_identity_features is not None:
            features.update(cvoi_identity_features)
        return features

    # ------------------------------------------------------------------
    #  内部方法
    # ------------------------------------------------------------------

    def _build_raw_images(self, agent_input: AgentInput, *, camera_name: str) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for frame_idx in range(len(agent_input.ego_statuses)):
            cam = getattr(agent_input.cameras[frame_idx], camera_name)
            if cam is None or cam.image is None:
                raise ValueError(f"Camera {camera_name} image is None at frame {frame_idx}")
            images.append(cam.image)
        return images

    def _get_camera(self, agent_input: AgentInput, *, frame_idx: int, camera_name: str):
        cam = getattr(agent_input.cameras[frame_idx], camera_name)
        if cam is None:
            raise ValueError(f"Camera {camera_name} is None at frame {frame_idx}")
        return cam

    def _build_multiview_video_tensor(
        self,
        agent_input: AgentInput,
        *,
        camera_names: List[str],
        crop_height: int,
        crop_width: int,
        transform: Optional[VJEPAImageTransform],
    ) -> torch.Tensor:
        view_tensors = []
        for camera_name in camera_names:
            if transform is not None:
                view_tensors.append(
                    self._build_vjepa_video_tensor(agent_input, camera_name=camera_name, transform=transform)
                )
            else:
                view_tensors.append(
                    self._build_video_tensor(
                        agent_input,
                        camera_name=camera_name,
                        crop_height=crop_height,
                        crop_width=crop_width,
                    )
                )
        return torch.stack(view_tensors, dim=0)

    def _build_video_tensor(
        self,
        agent_input: AgentInput,
        *,
        camera_name: str,
        crop_height: int,
        crop_width: int,
    ) -> torch.Tensor:
        """构建单路 ImageNet-normalized 视频 clip，输出 [C, T, H, W]。"""
        images: List[np.ndarray] = []
        for img in self._build_raw_images(agent_input, camera_name=camera_name):
            h, w = img.shape[:2]
            target_h = w // 2
            if target_h < h:
                crop_top = (h - target_h) // 2
                img = img[crop_top : crop_top + target_h]
            img = cv2.resize(img, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
            images.append(img)

        video = np.stack(images, axis=0)
        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0
        for t in range(video_tensor.shape[1]):
            video_tensor[:, t] = self._normalize(video_tensor[:, t])
        return video_tensor

    def _build_vjepa_video_tensor(
        self,
        agent_input: AgentInput,
        *,
        camera_name: str,
        transform: VJEPAImageTransform,
    ) -> torch.Tensor:
        """构建 V-JEPA 训练一致输入 clip。"""
        images = self._build_raw_images(agent_input, camera_name=camera_name)
        return transform(np.stack(images, axis=0))

    def _build_multiview_camera_metadata(
        self,
        agent_input: AgentInput,
        *,
        camera_names: List[str],
        crop_height: int,
        crop_width: int,
        transform: Optional[VJEPAImageTransform],
    ) -> Tuple[np.ndarray, np.ndarray]:
        intrinsics = np.zeros((len(camera_names), len(agent_input.ego_statuses), 3, 3), dtype=np.float32)
        camera2ego = np.zeros((len(camera_names), len(agent_input.ego_statuses), 4, 4), dtype=np.float32)
        for view_idx, camera_name in enumerate(camera_names):
            for frame_idx in range(len(agent_input.ego_statuses)):
                cam = self._get_camera(agent_input, frame_idx=frame_idx, camera_name=camera_name)
                image = cam.image
                if image is None:
                    raise ValueError(f"Camera {camera_name} image is None at frame {frame_idx}")
                # fail-loud: 缺 intrinsics 不得用 identity 内参替代 (与训练侧 navsim_data._build_multiview_camera_metadata
                # 一致)。np.eye(3) 是伪标定，会让该视角几何失真并污染 PETR 多视位置编码，造成 train/infer skew。
                raw_intrinsic_value = getattr(cam, "intrinsics", None)
                if raw_intrinsic_value is None:
                    raise ValueError(
                        f"Camera {camera_name} missing intrinsics at frame {frame_idx}; "
                        "禁止用 identity 内参替代 (train/infer geometry skew)."
                    )
                raw_intrinsic = np.asarray(raw_intrinsic_value, dtype=np.float32)
                intrinsics[view_idx, frame_idx] = self._resize_crop_intrinsics(
                    raw_intrinsic,
                    source_hw=(image.shape[0], image.shape[1]),
                    output_hw=(crop_height, crop_width),
                    transform=transform,
                )
                camera2ego_value = getattr(cam, "camera2ego", None)
                if camera2ego_value is not None:
                    camera2ego[view_idx, frame_idx] = np.asarray(camera2ego_value, dtype=np.float32)
                else:
                    # The NavSim `Camera` dataclass exposes sensor2lidar_{rotation,translation}, NOT
                    # camera2ego. Training builds camera2ego = lidar2ego @ sensor2lidar, falling back
                    # to sensor2lidar when lidar2ego is absent (navsim_data._camera_to_ego_transform).
                    # lidar2ego is unavailable on the inference Camera, so use sensor2lidar directly.
                    # The previous np.eye(4) fallback made EVERY view geometrically identical, which
                    # corrupted the PETR multi-view position embedding at eval only (train/infer skew).
                    camera2ego[view_idx, frame_idx] = self._sensor2lidar_transform(cam)
        return intrinsics, camera2ego

    @staticmethod
    def _sensor2lidar_transform(cam) -> np.ndarray:
        """Build a 4x4 sensor->lidar transform from a NavSim Camera (per-view mounting geometry)."""
        rotation = getattr(cam, "sensor2lidar_rotation", None)
        translation = getattr(cam, "sensor2lidar_translation", None)
        if rotation is None or translation is None:
            # fail-loud: 缺外参直接报错。退化为 identity 会让多视角几何静默错位，
            # 在 eval 端损坏 PETR 多视角位置编码（train/infer 偏斜）。
            raise ValueError(
                "NavSim Camera missing sensor2lidar extrinsics "
                f"(rotation present={rotation is not None}, translation present={translation is not None}); "
                "refusing to fall back to identity."
            )
        matrix = np.eye(4, dtype=np.float32)
        rot = np.asarray(rotation, dtype=np.float32)
        if rot.shape == (9,):
            rot = rot.reshape(3, 3)
        if rot.shape == (3, 3):
            matrix[:3, :3] = rot
        elif rot.shape == (4,):
            # Quaternion (wxyz, matching navsim_data._as_transform); scipy expects xyzw.
            quat_xyzw = np.asarray([rot[1], rot[2], rot[3], rot[0]], dtype=np.float32)
            matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        else:
            # fail-loud: 形状不符直接报错，禁止静默保持 identity 旋转。
            raise ValueError(
                f"Unsupported sensor2lidar_rotation shape {tuple(rot.shape)}; "
                "expected (3, 3), flattened (9,) or quaternion (4,) wxyz"
            )
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(-1)[:3]
        return matrix

    @staticmethod
    def _resize_crop_intrinsics(
        intrinsic: np.ndarray,
        *,
        source_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        transform: Optional[VJEPAImageTransform],
    ) -> np.ndarray:
        src_h, src_w = float(source_hw[0]), float(source_hw[1])
        out_h, out_w = float(output_hw[0]), float(output_hw[1])
        intrinsic = intrinsic.astype(np.float32, copy=True)

        if transform is not None and transform.crop_top_bottom > 0:
            crop_x = 0.0
            crop_y = float(transform.crop_top_bottom)
            crop_w = src_w
            crop_h = src_h - 2.0 * crop_y
        else:
            target_h = src_w / 2.0
            if target_h < src_h:
                crop_x = 0.0
                crop_y = (src_h - target_h) / 2.0
                crop_w = src_w
                crop_h = target_h
            else:
                crop_x = 0.0
                crop_y = 0.0
                crop_w = src_w
                crop_h = src_h

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

    def _build_states(self, agent_input: AgentInput) -> np.ndarray:
        """
        构建 [T, 7] 状态向量: [x, y, z, roll, pitch, yaw, speed]

        注意: AgentInput 的 ego_pose 是 *相对于最后一帧* 的局部坐标 (x, y, heading)，
        ego_velocity 是 [vx, vy]，ego_acceleration 是 [ax, ay]。

        为了与训练数据格式对齐 (全局坐标 → 帧间差分)，
        这里直接使用 ego_pose 的 x, y, heading 作为 x, y, yaw，
        其余维度 (z, roll, pitch) 填零，速度使用 vx（纵向速度），与训练数据 ego_dynamic_state[0] 一致。
        """
        states_list: List[np.ndarray] = []
        for frame_idx, ego_status in enumerate(agent_input.ego_statuses):
            if ego_status is None:
                # fail-loud: states 是评测的无条件必需输入（actions 差分 / status 速度 / 几何
                # 全部由它派生），全零伪造会静默毁掉整条评测链路。
                raise ValueError(
                    f"AgentInput.ego_statuses[{frame_idx}] is None; ego states 缺失。"
                    "states 是评测必需输入，禁止用 zeros 伪造。"
                )

            pose = np.asarray(ego_status.ego_pose, dtype=np.float32)  # [x, y, heading]
            velocity = np.asarray(ego_status.ego_velocity, dtype=np.float32)  # [vx, vy]
            speed = float(velocity[0])  # vx (纵向速度)，与训练一致

            # [x, y, z=0, roll=0, pitch=0, yaw=heading, speed]
            state = np.array(
                [pose[0], pose[1], 0.0, 0.0, 0.0, pose[2], speed],
                dtype=np.float32,
            )
            states_list.append(state)

        return np.stack(states_list, axis=0)

    def _build_actions(self, states: np.ndarray) -> np.ndarray:
        """与 NavSimWorldModelDataset 的 _build_actions_3d 保持一致。"""
        t_steps = states.shape[0]
        if t_steps < 2:
            return np.zeros((0, 3), dtype=np.float32)
        return self._build_actions_3d(states)

    def _build_actions_3d(self, states: np.ndarray) -> np.ndarray:
        """Build 3D actions: [dx_ego, dy_ego, d_yaw]."""
        t_steps = states.shape[0]
        actions = np.zeros((t_steps - 1, 3), dtype=np.float32)
        for t in range(t_steps - 1):
            dx_global = states[t + 1, 0] - states[t, 0]
            dy_global = states[t + 1, 1] - states[t, 1]

            yaw = states[t, 5]
            cos_h = np.cos(-yaw)
            sin_h = np.sin(-yaw)
            dx_ego = cos_h * dx_global - sin_h * dy_global
            dy_ego = sin_h * dx_global + cos_h * dy_global

            d_yaw = states[t + 1, 5] - states[t, 5]
            d_yaw = float(np.arctan2(np.sin(d_yaw), np.cos(d_yaw)))
            actions[t] = np.asarray([dx_ego, dy_ego, d_yaw], dtype=np.float32)
        return actions

    def _build_driving_commands(self, agent_input: AgentInput) -> Optional[np.ndarray]:
        """
        提取原始 driving_command。

        driving_command 是**条件必需**特征（仅 use_drive_command=True 的配置消费）。
        缺失时不伪造（禁止 zeros）也不在此处硬报错，而是返回 None、由消费端把关：
        use_drive_command=True 时 status_features._prepare_drive_command_* 会 fail-loud，
        use_drive_command=False 的配置则正常评测。

        Returns
        -------
        Optional[np.ndarray]
            [T, 4] float32, one-hot: [GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, U_TURN]；
            任一帧 ego_status 缺失时返回 None。
        """
        cmds = []
        for frame_idx, ego_status in enumerate(agent_input.ego_statuses):
            if ego_status is None:
                logger.warning(
                    "AgentInput.ego_statuses[%d] is None; driving_command 特征不可用（不伪造 zeros）。"
                    "若配置 use_drive_command=True，下游 status_features 将报错。",
                    frame_idx,
                )
                return None
            cmds.append(np.asarray(ego_status.driving_command, dtype=np.float32))
        return np.stack(cmds, axis=0)

    def _build_ego_dynamics(self, agent_input: AgentInput) -> Optional[np.ndarray]:
        """
        提取原始 ego_dynamics [vx, vy, ax, ay]。

        与训练数据 PKL 中的 ego_dynamic_state 字段对齐。

        ego_dynamics 是**条件必需**特征（state_dim>=8 必需；state_dim=7 时可由 states
        速度差分替代）。缺失时不伪造（禁止 zeros，vy/ay 伪造成 0 会改变 status 语义）
        也不在此处硬报错，而是返回 None、由消费端把关：state_dim>=8 时
        status_features._prepare_drive_command_8 会 fail-loud。

        Returns
        -------
        Optional[np.ndarray]
            [T, 4] float32；任一帧 ego_status 缺失时返回 None。
        """
        dynamics = []
        for frame_idx, ego_status in enumerate(agent_input.ego_statuses):
            if ego_status is None:
                logger.warning(
                    "AgentInput.ego_statuses[%d] is None; ego_dynamics 特征不可用（不伪造 zeros）。"
                    "若配置 state_dim>=8，下游 status_features 将报错。",
                    frame_idx,
                )
                return None
            vel = np.asarray(ego_status.ego_velocity, dtype=np.float32)  # [vx, vy]
            acc = np.asarray(ego_status.ego_acceleration, dtype=np.float32)  # [ax, ay]
            dynamics.append(np.array([vel[0], vel[1], acc[0], acc[1]], dtype=np.float32))
        return np.stack(dynamics, axis=0)
