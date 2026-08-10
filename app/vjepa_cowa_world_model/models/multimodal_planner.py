# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
统一的多模态时序 Planner

通过配置参数支持多种训练模式:
- use_temporal=True, use_z_context=False  → train_giant_first (时序预测)
- use_temporal=False, use_z_context=False → train_mm (单帧预测)
- use_temporal=False, use_z_context=True  → train_context (context 单帧)
- use_temporal=True, use_z_context=True   → train_command (context 时序)
"""

import math

import torch
import torch.nn as nn

# Observed-token vocabulary + normalizer live in the canonical planner_contracts module
# (single source of truth, shared with diffusion_planner and training/configs/common).
from app.vjepa_cowa_world_model.models.planner_contracts import (
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    PLANNER_OBSERVED_TOKEN_NONE,
)
from app.vjepa_cowa_world_model.models.planner_contracts import (  # noqa: E402
    normalize_observed_token_mode as _normalize_observed_token_mode,
)
from app.vjepa_cowa_world_model.models.trajectory_head import TrajectoryHead


class MultiModalTemporalPlanner(nn.Module):
    """
    多模态 Planner (支持单帧预测和时序预测)
    输出 K 条轨迹 + K 个置信度 logit。
    损失计算在外部 wta_loss() 中完成，forward 只负责推理。

    通过 use_temporal 参数控制:
    - use_temporal=False: 单帧预测 (只使用第一帧)
    - use_temporal=True: 时序预测 (使用所有历史帧)

    通过 use_z_context 参数控制 planner 输入来源:
    - use_z_context=False: 使用 z_ar (predictor 预测的未来表征)
    - use_z_context=True: 使用 z_context (encoder 编码的第一帧/当前帧)
      此模式下强制走单帧路径，推理时只需编码当前帧即可。

    通过 use_observed_tokens 参数控制是否将观测帧 tokens 与预测 tokens 拼接:
    - use_observed_tokens=False: 仅使用 z_ar (预测 tokens)
    - use_observed_tokens=True: 将 z_observed (encoder 编码的观测帧) 与 z_ar 拼接后
      一起输入 temporal memory，让 planner 同时看到已观测和预测的表征。
      此模式需要 use_temporal=True 且 use_z_context=False。

    时序对齐约束 (use_temporal=True 时启用):
    - 使用 additive distance bias 在 cross-attention logit 上软约束
      第 i 个 query 更关注第 i 个时间步的 memory token
    - 不引入额外可学习参数，不与已有 PE 冲突
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        tf_d_model: int = 256,
        tf_d_ffn: int = 1024,
        tf_num_layers: int = 3,
        tf_num_head: int = 8,
        tf_dropout: float = 0.0,
        tokens_per_frame: int = 256,
        num_poses: int = 7,
        num_time_steps: int = 7,
        num_context_frames: int = 1,
        status_dim: int = 7,
        use_spatial_tokens: bool = False,
        num_modes: int = 6,
        use_temporal: bool = False,
        use_time_aligned_bias: bool = True,
        time_aligned_bias_strength: float = 5.0,
        use_z_context: bool = False,
        use_status_for_planner: bool = True,
        use_observed_tokens: bool = False,
        observed_token_mode=None,
        num_observed_frames: int = 1,
        use_action_history: bool = False,
        action_history_dim: int = 3,
        enable_rl_actor_critic: bool = False,
        rl_action_dim: int = 2,
        command_dim: int = 0,
    ):
        """
        Parameters
        ----------
        encoder_dim               : Encoder 输出维度
        tf_d_model               : Transformer 模型维度
        tf_d_ffn                 : Transformer FFN 维度
        tf_num_layers            : Transformer 层数
        tf_num_head              : Transformer 注意力头数
        tf_dropout               : Dropout 率
        tokens_per_frame         : 每帧 token 数
        num_poses                : 预测轨迹点数
        num_time_steps           : 时序预测的帧数 (use_temporal=True 时使用)
        num_context_frames       : Context 帧数 (use_z_context=True 时使用)
        status_dim               : 状态特征维度
        use_spatial_tokens       : 是否保留空间 token
        num_modes                : 轨迹模态数 K
        use_temporal             : 是否使用时序模式
        use_time_aligned_bias    : 是否使用时序对齐 bias
        time_aligned_bias_strength: 时序对齐 bias 初始强度
        use_z_context            : 是否使用 z_context 作为输入
        use_status_for_planner   : 是否使用 status 特征
        use_observed_tokens      : 是否将观测帧 tokens 与预测 tokens 拼接输入 planner
        observed_token_mode      : "none" | "concat" | "concat_type_embed"
        num_observed_frames      : 观测帧数 (use_observed_tokens=True 时使用)
        use_action_history       : 是否追加独立的历史轨迹 token 分支
        action_history_dim       : 历史轨迹 token 的每步维度
        command_dim              : 分类特征维度 (>0 时将 status 前 command_dim 维作为
                                   导航指令独立嵌入，其余作为运动学特征独立嵌入;
                                   =0 时保持旧行为，单个 MLP 处理全部 status)
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.tf_d_model = tf_d_model
        self.tokens_per_frame = tokens_per_frame
        self.num_poses = num_poses
        self.num_time_steps = num_time_steps
        self.num_context_frames = num_context_frames
        self.use_spatial_tokens = use_spatial_tokens
        self.num_modes = num_modes
        self.use_temporal = use_temporal
        self.use_time_aligned_bias = use_time_aligned_bias
        self.use_z_context = use_z_context
        self.use_status_for_planner = use_status_for_planner
        self.observed_token_mode = _normalize_observed_token_mode(observed_token_mode, use_observed_tokens)
        self.use_observed_tokens = self.observed_token_mode != PLANNER_OBSERVED_TOKEN_NONE
        self.num_observed_frames = num_observed_frames
        self.use_action_history = use_action_history
        self.action_history_dim = action_history_dim
        self.command_dim = command_dim

        # 校验: use_observed_tokens 需要 use_temporal=True 且 use_z_context=False
        if self.use_observed_tokens:
            if not (use_temporal):
                raise AssertionError("use_observed_tokens=True requires use_temporal=True")
            if not (not use_z_context):
                raise AssertionError("use_observed_tokens=True and use_z_context=True are mutually exclusive")
        self.enable_rl_actor_critic = enable_rl_actor_critic
        self.rl_action_dim = rl_action_dim

        # ==================== 时序对齐：可学习的 bias 强度 ====================
        if use_temporal and use_time_aligned_bias:
            init_log_strength = math.log(max(time_aligned_bias_strength, 0.01))
            self.log_time_aligned_bias_strength = nn.Parameter(torch.tensor(init_log_strength, dtype=torch.float32))
        else:
            self.register_buffer(
                "log_time_aligned_bias_strength", torch.tensor(0.0, dtype=torch.float32), persistent=False
            )

        # ==================== 共享组件 ====================
        self.query_embedding = nn.Embedding(num_modes * num_poses, tf_d_model)

        # ==================== 时序对齐：Query 时间步索引 ====================
        # use_observed_tokens 时，query 时间步偏移 num_observed_frames
        # 这样 query 0 对应 memory 中的第 num_observed_frames 个时间步 (即第一个预测步)
        obs_offset = num_observed_frames if self.use_observed_tokens else 0
        query_step_idx = (torch.arange(num_poses, dtype=torch.long) + obs_offset).repeat(num_modes)
        self.register_buffer("query_step_idx", query_step_idx, persistent=False)

        self.transformer = nn.Transformer(
            d_model=tf_d_model,
            nhead=tf_num_head,
            num_encoder_layers=tf_num_layers,
            num_decoder_layers=tf_num_layers,
            dim_feedforward=tf_d_ffn,
            dropout=tf_dropout,
            batch_first=True,
        )

        trajectory_head_cls = TrajectoryHead
        self.trajectory_heads = nn.ModuleList(
            [trajectory_head_cls(num_poses, tf_d_ffn, tf_d_model) for _ in range(num_modes)]
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(num_modes * tf_d_model, tf_d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(tf_d_ffn, num_modes),
        )

        if enable_rl_actor_critic:
            self.rl_actor_head = nn.Sequential(
                nn.Linear(tf_d_model, tf_d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(tf_d_ffn, rl_action_dim),
            )
            self.rl_value_head = nn.Sequential(
                nn.Linear(tf_d_model, tf_d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(tf_d_ffn, 1),
            )
            self.rl_log_std = nn.Parameter(torch.full((rl_action_dim,), -0.5, dtype=torch.float32))

        # ==================== 根据 use_temporal 选择不同结构 ====================
        if use_temporal:
            if use_spatial_tokens:
                self.temporal_fc = nn.Linear(encoder_dim, tf_d_model)
            else:
                self.temporal_fc = nn.Linear(encoder_dim * tokens_per_frame, tf_d_model)

            # 确定时间步数：use_z_context 时用 num_context_frames，use_observed_tokens 时加上 num_observed_frames
            temporal_steps = num_context_frames if use_z_context else num_time_steps
            if self.use_observed_tokens:
                temporal_steps = num_observed_frames + num_time_steps
            num_action_tokens = num_observed_frames if use_action_history else 0
            # command_dim > 0 时 status 拆为 command + kinematics 两个 token，否则 1 个
            _num_status_tokens = (2 if command_dim > 0 else 1) if use_status_for_planner else 0
            num_kv = (
                (tokens_per_frame if use_spatial_tokens else 1) * temporal_steps
                + num_action_tokens
                + _num_status_tokens
            )
            self.temporal_embedding = nn.Embedding(num_kv, tf_d_model)

        # 单帧路径的层：use_temporal=False 时必需；
        # use_temporal=True + use_z_context=True 时也需要（z_context 只有第一帧，走单帧路径）
        if not use_temporal or use_z_context:
            self.image_fc = nn.Linear(encoder_dim, tf_d_model)
            _num_status_kv = (2 if command_dim > 0 else 1) if use_status_for_planner else 0
            _num_action_kv = num_observed_frames if use_action_history else 0
            if use_spatial_tokens:
                num_keyval = tokens_per_frame + _num_action_kv + _num_status_kv
            else:
                num_keyval = 1 + _num_action_kv + _num_status_kv

            self.keyval_embedding = nn.Embedding(num_keyval, tf_d_model)

        if self.observed_token_mode == PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED:
            self.observed_source_embedding = nn.Embedding(2, encoder_dim)

        if use_action_history:
            self.action_history_encoding = nn.Sequential(
                nn.LayerNorm(action_history_dim),
                nn.Linear(action_history_dim, 128),
                nn.ReLU(),
                nn.Linear(128, tf_d_model),
            )

        if use_status_for_planner:
            if command_dim > 0:
                # 拆分嵌入：分类（导航指令）与连续（运动学）各自独立编码
                kinematics_dim = status_dim - command_dim
                self.command_encoding = nn.Sequential(
                    nn.Linear(command_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, tf_d_model),
                )
                self.kinematics_encoding = nn.Sequential(
                    nn.LayerNorm(kinematics_dim),
                    nn.Linear(kinematics_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, tf_d_model),
                )
            else:
                # 旧行为：单个 MLP 处理全部 status
                self.status_encoding = nn.Sequential(
                    nn.Linear(status_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, tf_d_model),
                )

    # ─────────────────────────────────────────────────────────────────
    def _build_memory_temporal(
        self,
        z_input: torch.Tensor,
        status_feature: torch.Tensor,
        num_steps: int,
        action_history: torch.Tensor = None,
    ) -> tuple:
        """构建时序预测的 memory 和每个 token 的时间步索引。

        Parameters
        ----------
        z_input        : [B, num_steps * tokens_per_frame, D]
        status_feature : [B, status_dim]
        num_steps      : 时间步数

        Returns
        -------
        memory         : [B, M, d]
        memory_step_idx: [M] 每个 memory token 的时间步索引，-1 表示不参与对齐
        """
        B = z_input.shape[0]
        T = num_steps
        P = self.tokens_per_frame
        D = self.encoder_dim

        expected_tokens = T * P
        if not (z_input.ndim == 3):
            raise AssertionError(f"Expected z_input shape [B, N, D], got ndim={z_input.ndim}")
        if not (z_input.shape[1] == expected_tokens):
            raise AssertionError(
                f"Planner memory reshape mismatch: got N={z_input.shape[1]}, "
                f"expected num_steps*tokens_per_frame={T}*{P}={expected_tokens}"
            )
        if not (z_input.shape[2] == D):
            raise AssertionError(f"Planner channel mismatch: got D={z_input.shape[2]}, expected encoder_dim={D}")
        z_reshaped = z_input.view(B, T, P, D)

        # ==================== 时序对齐：Memory 时间步索引 ====================
        if self.use_spatial_tokens:
            feat = self.temporal_fc(z_reshaped).view(B, T * P, -1)
            memory_step_idx = torch.arange(T, device=z_input.device, dtype=torch.long).repeat_interleave(P)
        else:
            feat = self.temporal_fc(z_reshaped.reshape(B, T, P * D))
            memory_step_idx = torch.arange(T, device=z_input.device, dtype=torch.long)

        T_feat = feat.shape[1]

        feat = feat + self.temporal_embedding.weight[:T_feat].unsqueeze(0)
        memory_tokens = [feat]
        memory_step_parts = [memory_step_idx]
        next_embedding_idx = T_feat

        if self.use_action_history:
            action_tokens = self._encode_action_history(action_history)
            num_action_tokens = action_tokens.shape[1]
            action_tokens = action_tokens + self.temporal_embedding.weight[
                next_embedding_idx : next_embedding_idx + num_action_tokens
            ].unsqueeze(0)
            memory_tokens.append(action_tokens)
            memory_step_parts.append(torch.full((num_action_tokens,), -1, device=z_input.device, dtype=torch.long))
            next_embedding_idx += num_action_tokens

        if self.use_status_for_planner:
            if self.command_dim > 0:
                # 拆分模式：分类（导航指令）和连续（运动学）分别编码为独立 token
                cmd_feat = self.command_encoding(status_feature[:, : self.command_dim]).unsqueeze(1)
                kin_feat = self.kinematics_encoding(status_feature[:, self.command_dim :]).unsqueeze(1)
                cmd_feat = cmd_feat + self.temporal_embedding.weight[
                    next_embedding_idx : next_embedding_idx + 1
                ].unsqueeze(0)
                kin_feat = kin_feat + self.temporal_embedding.weight[
                    next_embedding_idx + 1 : next_embedding_idx + 2
                ].unsqueeze(0)

                memory_tokens.extend([cmd_feat, kin_feat])
                memory_step_parts.append(torch.full((2,), -1, device=z_input.device, dtype=torch.long))
            else:
                # 旧行为：单个 status token
                status = self.status_encoding(status_feature).unsqueeze(1)
                status = status + self.temporal_embedding.weight[
                    next_embedding_idx : next_embedding_idx + 1
                ].unsqueeze(0)

                memory_tokens.append(status)
                memory_step_parts.append(torch.full((1,), -1, device=z_input.device, dtype=torch.long))

        return torch.cat(memory_tokens, dim=1), torch.cat(memory_step_parts, dim=0)

    # ─────────────────────────────────────────────────────────────────
    def _build_memory_single(
        self,
        z_input: torch.Tensor,
        status_feature: torch.Tensor,
        action_history: torch.Tensor = None,
    ) -> torch.Tensor:
        """构建单帧预测的 memory。

        Parameters
        ----------
        z_input        : [B, tokens_per_frame, D]  单帧 encoder 输出
        status_feature : [B, status_dim]
        """
        z_frame = z_input[:, : self.tokens_per_frame]

        if self.use_spatial_tokens:
            img_feat = self.image_fc(z_frame)
        else:
            z_pooled = z_frame.mean(dim=1)
            img_feat = self.image_fc(z_pooled)
            img_feat = img_feat.unsqueeze(1)

        keyval_parts = [img_feat]

        if self.use_action_history:
            keyval_parts.append(self._encode_action_history(action_history))

        if self.use_status_for_planner:
            if self.command_dim > 0:
                cmd_encoded = self.command_encoding(status_feature[:, : self.command_dim]).unsqueeze(1)
                kin_encoded = self.kinematics_encoding(status_feature[:, self.command_dim :]).unsqueeze(1)
                keyval_parts.extend([cmd_encoded, kin_encoded])
            else:
                status_encoded = self.status_encoding(status_feature)
                status_encoded = status_encoded.unsqueeze(1)
                keyval_parts.append(status_encoded)

        keyval = torch.cat(keyval_parts, dim=1)

        num_keyval = keyval.shape[1]
        keyval = keyval + self.keyval_embedding.weight[:num_keyval, :].unsqueeze(0)

        return keyval

    def _encode_action_history(self, action_history: torch.Tensor) -> torch.Tensor:
        if not self.use_action_history:
            raise RuntimeError("_encode_action_history() called when use_action_history=False")
        if not (action_history is not None):
            raise AssertionError("use_action_history=True but action_history is None")
        if not (action_history.ndim == 3):
            raise AssertionError(f"Expected action_history shape [B, T_obs, D], got ndim={action_history.ndim}")
        if not (action_history.shape[-1] == self.action_history_dim):
            raise AssertionError(
                f"action_history dim mismatch: got D={action_history.shape[-1]}, expected {self.action_history_dim}"
            )
        if not (action_history.shape[1] <= self.num_observed_frames):
            raise AssertionError(
                f"action_history length mismatch: got T={action_history.shape[1]}, "
                f"expected <= {self.num_observed_frames}"
            )
        return self.action_history_encoding(action_history)

    # ─────────────────────────────────────────────────────────────────
    def _build_time_aligned_memory_bias(
        self,
        query_step_idx: torch.Tensor,
        memory_step_idx: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """构建时间对齐 bias [Q, M]，加到 decoder cross-attn logit 上。

        核心思想：
        - 第 i 个 query 负责预测第 i 个未来时间步的轨迹点
        - 应该让第 i 个 query 更关注第 i 个历史时间步的 memory token
        - 时间步距离越远，attention 权重应该越低（通过负 bias 实现）

        Args:
            query_step_idx:  [Q]  每个 query 的时间步索引
            memory_step_idx: [M]  每个 memory token 的时间步索引，-1 表示不参与对齐
            dtype: 输出 tensor 的数据类型

        Returns:
            bias: [Q, M] 的 attention bias 矩阵
        """
        Q = query_step_idx.shape[0]
        M = memory_step_idx.shape[0]

        time_aligned_bias_strength = torch.exp(self.log_time_aligned_bias_strength)

        if (not self.use_time_aligned_bias) or time_aligned_bias_strength <= 0:
            return torch.zeros(Q, M, device=query_step_idx.device, dtype=dtype)

        q = query_step_idx.to(torch.float32).unsqueeze(1)  # [Q, 1]
        m = memory_step_idx.to(torch.float32).unsqueeze(0)  # [1, M]
        distance = (q - m).abs()  # [Q, M]

        # 归一化因子：使用总时间步数（含 observed_frames 如果启用）
        total_steps = max(self.num_time_steps, self.num_context_frames)
        if self.use_observed_tokens:
            total_steps = self.num_observed_frames + self.num_time_steps
        norm = max(1.0, float(total_steps - 1))

        bias = -time_aligned_bias_strength * (distance / norm)  # [Q, M]

        status_mask = memory_step_idx.eq(-1).unsqueeze(0)  # [1, M]
        bias = torch.where(status_mask, torch.zeros_like(bias), bias)

        return bias.to(dtype=dtype)

    def _apply_observed_source_embeddings(
        self,
        z_observed: torch.Tensor,
        z_predicted: torch.Tensor,
    ) -> tuple:
        if self.observed_token_mode != PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED:
            return z_observed, z_predicted
        observed_embedding = self.observed_source_embedding.weight[0].to(dtype=z_observed.dtype).view(1, 1, -1)
        predicted_embedding = self.observed_source_embedding.weight[1].to(dtype=z_predicted.dtype).view(1, 1, -1)
        return z_observed + observed_embedding, z_predicted + predicted_embedding

    def _prepare_planner_input(
        self,
        z_ar: torch.Tensor,
        z_context: torch.Tensor = None,
        z_observed: torch.Tensor = None,
    ) -> tuple:
        # 根据 use_z_context 开关选择 planner 的实际输入
        if self.use_z_context:
            if not (z_context is not None):
                raise AssertionError(
                    "use_z_context=True but z_context is None. "
                    "Please pass z_context (first-frame encoder output) to planner.forward()."
                )
            planner_input = z_context
            num_steps = self.num_context_frames
        else:
            planner_input = z_ar
            num_steps = self.num_time_steps

        if not self.use_observed_tokens:
            return planner_input, num_steps

        if not (z_observed is not None):
            raise AssertionError(
                "observed_token_mode requires z_observed. "
                "Please pass z_observed (observed frames encoder output) to planner.forward()."
            )
        expected_observed_tokens = self.num_observed_frames * self.tokens_per_frame
        if not (z_observed.shape[1] == expected_observed_tokens):
            raise AssertionError(
                f"Observed token mismatch: got N={z_observed.shape[1]}, "
                f"expected num_observed_frames*tokens_per_frame="
                f"{self.num_observed_frames}*{self.tokens_per_frame}={expected_observed_tokens}"
            )
        z_observed, planner_input = self._apply_observed_source_embeddings(z_observed, planner_input)
        planner_input = torch.cat([z_observed, planner_input], dim=1)
        num_steps = self.num_observed_frames + num_steps
        return planner_input, num_steps

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        z_ar: torch.Tensor,
        status_feature: torch.Tensor,
        z_context: torch.Tensor = None,
        z_observed: torch.Tensor = None,
        action_history: torch.Tensor = None,
    ) -> dict:
        """
        Parameters
        ----------
        z_ar           : [B, N, D]  autoregressive prediction tokens (use_z_context=False 时使用)
        status_feature : [B, status_dim]
        z_context      : [B, N', D] encoder output tokens (optional, used when use_z_context=True)
        z_observed     : [B, num_obs*tokens_per_frame, D] 观测帧 encoder tokens
                         (optional, used when use_observed_tokens=True)
        action_history : [B, T_obs, action_history_dim] 观测段累计历史轨迹 tokens
             (optional, used when use_action_history=True)

        Returns
        -------
        dict with keys:
            "trajectories" : [B, K, num_poses, 3]   (x, y, yaw)
            "confidences"  : [B, K]                  unnormalized logits
        """
        planner_input, num_steps = self._prepare_planner_input(z_ar, z_context=z_context, z_observed=z_observed)

        B = planner_input.shape[0]
        K = self.num_modes

        # use_z_context=True 时，根据 num_context_frames 决定是否走时序路径
        if self.use_z_context:
            use_temporal_path = self.num_context_frames > 1
        else:
            use_temporal_path = self.use_temporal

        if use_temporal_path:
            # 时序模式：构建 memory 并返回时间步索引用于计算对齐 bias
            memory, memory_step_idx = self._build_memory_temporal(
                planner_input,
                status_feature,
                num_steps,
                action_history=action_history,
            )

            # ==================== 时序对齐：构建 Attention Bias ====================
            memory_bias = self._build_time_aligned_memory_bias(
                self.query_step_idx,  # [K*num_poses] 每个 query 的时间步
                memory_step_idx,  # [M] 每个 memory token 的时间步
                dtype=memory.dtype,
            )
        else:
            # 单帧模式
            memory = self._build_memory_single(planner_input, status_feature, action_history=action_history)
            memory_bias = None

        query = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)

        if memory_bias is not None:
            query_out = self.transformer(src=memory, tgt=query, memory_mask=memory_bias)
        else:
            query_out = self.transformer(src=memory, tgt=query)

        query_out = query_out.view(B, K, self.num_poses, self.tf_d_model)

        traj_list = []
        for k in range(K):
            head_out = self.trajectory_heads[k](query_out[:, k])
            if isinstance(head_out, dict):
                traj_k = head_out["trajectory"]
            else:
                traj_k = head_out
            traj_list.append(traj_k)

        trajs = torch.stack(traj_list, dim=1)

        conf_feat = query_out.mean(dim=2)
        conf_logits = self.confidence_head(conf_feat.reshape(B, K * self.tf_d_model))

        output = {
            "trajectories": trajs,
            "confidences": conf_logits,
        }

        if self.enable_rl_actor_critic:
            planner_feature = query_out.mean(dim=(1, 2))
            output["planner_feature"] = planner_feature
            output["policy_mean"] = self.rl_actor_head(planner_feature)
            output["policy_log_std"] = self.rl_log_std.unsqueeze(0).expand(B, -1)
            output["value"] = self.rl_value_head(planner_feature).squeeze(-1)

        return output
