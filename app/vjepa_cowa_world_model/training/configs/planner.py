"""Split from training/config.py (verbatim node moves). Part: planner."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.vjepa_cowa_world_model.training.configs.common import PLANNER_OBSERVED_TOKEN_NONE

# Canonical valid values for planner.planner_type — shared with the planner factory's dispatch + errors.
PLANNER_TYPES = ("transformer", "diffusion")


@dataclass
class MultiViewConfig:
    """NavSim 多视角融合配置。"""

    enabled: bool = False
    fusion_type: str = "petr_cross_attn"
    output_mode: str = "fused"  # "fused" | "per_view"
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.0
    load_from_predictor_checkpoint: bool = False
    freeze_fusion: bool = False


@dataclass
class PredictorDiTConfig:
    """Latent DiT predictor 配置。"""

    objective: str = "flow_matching"  # "flow_matching" | "x0_prediction"
    conditioning_mode: str = "mean"
    max_condition_steps: int = 128
    num_inference_steps: int = 8
    sampler_type: str = "heun"
    schedule_type: str = "cosine"
    temperature: float = 1.0
    hidden_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    x0_loss_weight: float = 0.0
    bottleneck_dim: Optional[int] = None
    use_anchor_frame: bool = False
    metadata_condition_dropout: float = 0.0
    metadata_guidance_scale: float = 1.0
    metadata_conditioning_policy: str = "auto"
    masked_inpainting_enabled: bool = False
    masked_train_full_prefix_prob: float = 0.25
    masked_train_min_prefix_steps: int = 1
    masked_train_max_non_full_prefix_steps: Optional[int] = None
    masked_sample_return_full: bool = False
    joint_action_enabled: bool = False
    joint_action_loss_weight: float = 0.0
    joint_action_dim: int = 3
    joint_action_scale: Tuple[float, ...] = (8.0, 4.0, 1.0)
    joint_action_state_dim: int = 7
    joint_action_noise_mode: str = "shared"  # "shared" | "decoupled"
    joint_action_state_mode: str = "last_observed"
    joint_action_guidance_mode: str = "cond_only"
    joint_action_inference_noise_mode: str = "shared"  # "shared" | "decoupled"
    joint_video_final_noise: float = 0.0


@dataclass
class PlannerConfig:
    """Planner 配置：轨迹预测模块相关设置"""

    use_planner: bool = True
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0
    planner_loss_weight: float = 1.0
    use_spatial_tokens: bool = False
    use_temporal: bool = False
    temporal_alignment: bool = True
    z_ar_mode: str = "full"
    planner_input_source: str = "z_ar"  # "z_ar" (自回归展开) 或 "z_tf" (teacher forcing)
    num_modes: int = 6
    num_context_frames: int = 1
    conf_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    horizon_reg_loss_seconds: List[float] = field(default_factory=list)
    horizon_reg_loss_weights: List[float] = field(default_factory=list)
    horizon_reg_loss_normalize: bool = True
    states_mode: str = "first"
    use_status_for_planner: bool = True
    use_states_for_planner: bool = True
    use_z_context: bool = False
    # 观测 tokens 配置
    observed_token_mode: str = PLANNER_OBSERVED_TOKEN_NONE  # none | concat | concat_type_embed
    use_observed_tokens: bool = False  # 是否将观测帧 encoder tokens 与预测 tokens 拼接后输入 planner
    use_action_history_for_planner: bool = False
    action_history_dim: int = 3
    latent_dit_action_source: str = "planner"  # "planner" | "joint_action"
    policy_output_source: str = "planner"  # "planner" | "joint_action"
    enable_rl_actor_critic: bool = False
    rl_action_dim: int = 2
    # WTA 损失配置
    wta_loss_version: str = "v1"
    wta_temperature: float = 1.0
    wta_alpha: float = 5.0  # WTA soft-assignment sharpness (was hardcoded 5.0 in the loss dispatch)
    wta_global_batch_norm: bool = True
    cover_loss_weight: float = 0.1
    # aWTA (v3) 专用参数
    awta_init_temperature: float = 8.0
    awta_exp_base: float = 0.984
    awta_min_temperature: float = 0.1
    # Planner 类型选择
    planner_type: str = "transformer"  # one of PLANNER_TYPES
    refinement_core_type: Optional[str] = None  # Stage-3 refinement core; None 时继承 planner_type
    # Diffusion planner 专用参数
    diff_hidden_dim: int = 256
    diff_num_layers: int = 4
    diff_num_heads: int = 8
    diff_dropout: float = 0.0
    diff_mlp_ratio: float = 4.0
    diff_sde_beta_min: float = 0.1
    diff_sde_beta_max: float = 20.0
    diff_inference_steps: int = 2  # DPM-Solver++ 采样步数
    diff_num_samples: int = 6  # K 个噪声样本用于多模态
    diff_traj_dim: int = 6  # 轨迹维度 (x, y, vx, vy, cos_yaw, sin_yaw)
    diff_dt: float = 0.2  # 帧间时间间隔 (秒), 用于速度计算
    diff_trajectory_token_mode: str = "single_token"  # "single_token" 或 "per_pose_token"
    diff_adaln_version: str = (
        "legacy"  # "legacy"=旧 6-param adaLN, "v2"=9-param full adaLN, "v3"=legacy 结构但去掉 cross/mlp2 残差
    )
    diff_use_last_frame_only: bool = True  # True: 仅用 z_ar 最后一帧做 cross-attn (AR causal → 最后帧含全部信息)
    diff_interleave_predictor_sampling: bool = False  # 推理时 predictor 每产出一帧 prefix，就推进若干 diffusion step
    diff_train_prefix_conditioning: bool = False
    diff_train_min_prefix_frames: int = 1
    diff_train_full_prefix_prob: float = 0.25
    diff_train_max_non_full_prefix_frames: Optional[int] = None
    diff_num_modes: int = 1
    diff_independent_modes: bool = False  # True: B->B*K independent processing (anti-collapse); False: joint XTR
    # Architectural anti-collapse: expand K modes into the DiT sequence dimension
    # (with learnable mode embeddings) so every block can diversify modes via
    # self-attention.  Only meaningful for per_pose_token + num_modes>1; ignored
    # otherwise.  Default False preserves checkpoint compatibility.
    diff_mode_token_expansion: bool = False
    diff_use_anchor_frame: bool = False
    # Diffusion seed initialization (inference x_T/x_t init)
    # "gaussian": pure noise (legacy behavior)
    # "kinematic": constant-velocity prior + optional mode spread + noise
    diff_init_traj_strategy: str = "gaussian"
    diff_init_traj_noise_scale: float = 1.0
    diff_init_traj_yaw_span_deg: float = 30.0
    diff_init_traj_speed_scale_span: float = 0.2
    diff_cls_loss_weight: float = 1.0
    diff_reg_loss_weight: float = 1.0
    diff_vel_loss_weight: float = 0.5
    diff_yaw_loss_weight: float = 0.5
    # 轨迹生成框架：vp_diffusion 保持原 DPM-Solver++ 路径；flow_matching 启用 flow planner
    diff_generation_framework: str = "vp_diffusion"  # "vp_diffusion" | "flow_matching"
    diff_flow_matching_variant: str = "rectified"  # "rectified"(方案A) | "scheduler"(方案C/Wan-style)
    diff_flow_shift: float = 1.0
    diff_flow_sampler: str = "euler"  # "euler" | "heun"
    diff_flow_timestep_sampling: str = "logit_normal"  # "logit_normal" | "uniform"
    # Hybrid WTA (aWTA reg + XTR gated soft-CE) 专用超参
    diff_conf_temperature: float = 1.5
    diff_cls_th: float = 2.0
    diff_cls_ignore: float = 0.2
    # Status 拆分嵌入：将分类（导航指令 one-hot）与连续（运动学）分量独立嵌入
    split_status_embedding: bool = True
    # Planner 侧是否保留 drive_command 4 维；None 时继承 train.use_drive_command
    use_drive_command: Optional[bool] = None
    # Planner status 维度（与 predictor.state_dim 解耦）：
    #   0  = 继承 train.state_dim（向后兼容，默认）
    #   12 = cmd(4) + dyn(4) + pose(4, x_local/y_local/sin_yaw/cos_yaw)
    status_dim: int = 0


@dataclass
class ProposalConfig:
    """独立 proposal provider 配置。"""

    enabled: bool = False
    provider_type: str = "transformer"  # transformer | diffusion | history_kinematic
    checkpoint: Optional[str] = None
    use_separate_encoder: bool = False
    encoder_backbone: Optional[str] = None
    encoder_model_name: Optional[str] = None
    encoder_checkpoint: Optional[str] = None
    encoder_checkpoint_key: str = "encoder"
    encoder_freeze: bool = True
    vjepa_resolution: Tuple[int, int] = (256, 512)
    vjepa_crop_top_bottom: int = 28
    vjepa_num_frames: int = 2
    vjepa_checkpoint_key: Optional[str] = None
    vjepa_use_grid_mask: bool = True
    vjepa_use_causal_attention: bool = True
    freeze: bool = True
    num_modes: int = 6
    provider_num_modes: Optional[int] = None
    log_metrics_only: bool = True
    use_z_context: bool = True
    temporal_alignment: bool = True
    runtime_normalize_reps: Optional[bool] = None
    use_token_ae: Optional[bool] = None
    history_temperature: float = 1.0
    hidden_dim: int = 256
    manual_mode_expansion: bool = False
    manual_lateral_offsets: Optional[List[float]] = None
    manual_yaw_offsets_deg: Optional[List[float]] = None
    manual_speed_scales: Optional[List[float]] = None
    manual_ramp_power: float = 1.5
    manual_confidence_temperature: float = 1.0


@dataclass
class TokenAEConfig:
    """Token AE 配置"""

    enabled: bool = False
    num_latent_tokens: int = 64
    num_heads: int = 16
    encoder_depth: int = 4
    decoder_depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    encoder_mode: str = "parallel"
    loss_type: str = "smooth_l1"
    cos_loss_weight: float = 0.25
    latent_reg_weight: float = 0.0
    pos_embed_type: str = "sincos"
    input_grid_size: Optional[Tuple[int, int]] = None
    latent_grid_size: Optional[Tuple[int, int]] = None
    temporal_depth: int = 0
    temporal_num_heads: Optional[int] = None
    temporal_mlp_ratio: Optional[float] = None
    temporal_causal: bool = True
    temporal_mode: str = "index"
    temporal_pos_embed_type: str = "none"
    input_frame_mode: str = "all_frames"
    temporal_loss_weight: float = 0.0
