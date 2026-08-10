"""Split from training/config.py (verbatim node moves). Part: core."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MetaConfig:
    """元配置：训练相关的通用设置"""

    folder: str = ""
    seed: int = 0
    # 确定性策略：True 时启用 cudnn.deterministic 并关闭 cudnn.benchmark（可复现、稍慢）；
    # False(默认) 保留 cudnn.benchmark=True（更快但不完全可复现）。在 setup_distributed 中生效。
    deterministic: bool = False
    # 验证采样确定性：True(默认)时验证种子不含 epoch，使每个样本在所有 epoch 用同一
    # 采样噪声 → 验证曲线只随权重变化，消除 epoch 间扩散采样抖动(单 epoch best 不再是"运气点")。
    # 设为 False 恢复旧行为(每个 epoch 用不同噪声，可复现但抖动)。
    val_stable_noise: bool = True
    dtype: str = "bfloat16"
    resume_checkpoint: Optional[str] = None
    pretrain_repo: Optional[str] = None
    pretrain_checkpoint: Optional[str] = None
    pretrain_checkpoint_full: Optional[str] = None
    predictor_checkpoint: Optional[str] = (
        None  # predictor 独立 checkpoint，优先于 pretrain_checkpoint_full 中的 predictor
    )
    value_checkpoint: Optional[str] = None  # stage2 value_head checkpoint，用于 latent_guidance / stage3 warm-start
    planner_value_checkpoint: Optional[str] = None  # stage3 planner+value checkpoint，用于 stage4 warm-start
    # 显式覆盖 predictor/refiner 的 normalize_reps（None=按 checkpoint 元数据/实验默认解析，见
    # resolve_predictor_runtime_normalize_reps）。设为 True/False 可避免 checkpoint 静默覆盖实验 YAML。
    predictor_runtime_normalize_reps: Optional[bool] = None
    ae_checkpoint: Optional[str] = None
    load_encoder: bool = True
    load_predictor: bool = False
    load_seg: bool = True
    load_planner: bool = True
    context_encoder_key: str = "encoder"
    target_encoder_key: str = "target_encoder"
    save_every_freq: int = -1
    # Additional 1-based checkpoint epochs required by a downstream selector. These are unioned with
    # save_every_freq rather than replacing the ordinary cadence.
    selection_checkpoint_epochs: Tuple[int, ...] = ()
    # Periodic e{N}.pt milestones only when epoch >= save_from_epoch (latest.pt still every checkpoint_freq).
    # Default 0 keeps historical behavior. auto_research full runs set 100 to match eval_from_epoch.
    save_from_epoch: int = 0
    skip_batches: int = -1
    use_sdpa: bool = False
    sync_gc: bool = False
    val_freq: int = 5  # 验证频率
    resume_broadcast: bool = False  # resume 时是否通过 broadcast 分发 checkpoint（False 则每个 rank 独立读盘）
    resume_model_only: bool = False  # 仅加载模型权重，跳过 optimizer/scheduler/epoch（用于分阶段训练）
    auto_resume_latest: bool = False


# Canonical valid values for the model-type config fields — the single source of truth shared by the
# dataclasses below and the model factories' dispatch + fail-loud error messages.
BACKBONE_TYPES = ("vjepa2", "vjepa2.1", "vjepa_img_encoder", "dinov2_img_encoder")
PREDICTOR_TYPES = ("ac_transformer", "latent_dit")


@dataclass
class ModelConfig:
    """模型配置：模型架构相关设置"""

    model_name: str = ""
    backbone: str = "vjepa2"  # one of BACKBONE_TYPES
    vjepa_resolution: Tuple[int, int] = (256, 512)
    vjepa_crop_top_bottom: int = 28
    vjepa_num_frames: int = 2
    vjepa_checkpoint_key: str = "target_encoder"
    vjepa_use_grid_mask: bool = True
    vjepa_use_causal_attention: bool = True
    dinov2_model_name: str = "vit_large_patch14_reg4_dinov2"
    dinov2_resolution: Tuple[int, int] = (224, 448)
    dinov2_frame_stride: int = 2
    dinov2_forward_chunk_size: int = 16
    patch_size: int = 16
    pred_depth: int = 12
    pred_num_heads: Optional[int] = None
    pred_embed_dim: int = 384
    pred_is_frame_causal: bool = True
    uniform_power: bool = False
    use_rope: bool = False
    use_silu: bool = False
    use_pred_silu: bool = False
    wide_silu: bool = True
    use_extrinsics: bool = False
    use_mask_tokens: bool = False
    zero_init_mask_tokens: bool = True
    compile_model: bool = False
    use_activation_checkpointing: bool = False


@dataclass
class TrainConfig:
    """训练配置：训练策略相关设置"""

    encoder_train: bool = False
    seg_head: bool = True
    encoder_ema: bool = False
    perceiver_ema: bool = True
    predictor_train: bool = True
    # 解冻 predictor，仅用 planner 梯度（经 z_ar）微调它，且 *不* 加 jepa_loss。
    # 复用 planner_only_mode 的「无 jepa 监督 / 跳过 target encoder」通路，额外把 predictor 设为可训练。
    # 默认 False = 现有行为不变。配合 optimization.predictor_lr_scale 控制 predictor 学习率。
    predictor_planner_finetune: bool = False
    use_states_for_predictor: bool = True
    action_dim: int = 7
    state_dim: int = 7  # IC状态维度: 7=drive_command_7, 8=drive_command_8
    command_dim: int = 0  # >0: 拆分 state[0:command_dim] 为 command token，剩余为 kinematics token
    use_drive_command: bool = False  # point 23: 默认 False；True 时 status 向量保留 drive_command 的 4 维 one-hot
    predictor_inference_consistent: bool = False
    # "auto" | "full" | "inference_consistent" | "mask_future" | "no_state" | "no_aux"
    predictor_aux_policy: str = "auto"
    use_parallel_predictor: bool = False
    predictor_supervision_mode: str = "auto"  # "auto" | "tf" | "ar" | "tf_ar"
    predictor_loss_scope: str = "auto"  # "auto" | "next_step" | "future_only"
    predictor_use_z_ar_supervision: bool = True
    predictor_validation_enabled: bool = True
    predictor_static_graph: bool = False
    # Opt-in memory path for the narrow EMA predictor-only TF+AR envelope. It backpropagates
    # teacher-forcing loss before constructing the autoregressive graph, while preserving one
    # optimizer/scheduler step. The parser rejects every unsupported graph consumer.
    predictor_split_tf_ar_backward: bool = False
    reuse_context_as_target_when_frozen: bool = False
    predictor_no_aux_input: bool = False
    # Raw-frame semantics for diffusion / world-model generation tasks:
    # - num_encoder_frames: how many observed image frames are exposed to the frozen encoder
    #
    # `num_observed_frames` is kept for backward compatibility with older training
    # scripts that still use the previous name.
    num_encoder_frames: int = 2
    num_observed_frames: int = 2
    predictor_type: str = "ac_transformer"  # one of PREDICTOR_TYPES
    latent_dit_planner_input: str = "train_helper"  # "train_helper" | "sample"


@dataclass
class PredictorDynamicRolloutConfig:
    """ac_transformer cumulative rollout-prefix distribution."""

    enabled: bool = False
    full_prefix_prob: float = 0.25
    min_prefix_steps: int = 1
    max_non_full_prefix_steps: Optional[int] = None
    max_horizon: Optional[int] = None
    horizon_probabilities: Optional[Tuple[float, ...]] = None


@dataclass
class ValidationSuiteConfig:
    """Multi-domain full + deterministic cumulative-prefix validation."""

    enabled: bool = False
    protocol_version: str = "dynamic_rollout_validation_v2"
    horizons: List[int] = field(default_factory=list)
    expected_weights: List[float] = field(default_factory=list)
    primary_domain: str = "real"
    primary_cohort: str = "all"
    primary_protocol: str = "full"

    @property
    def expected_weight_by_horizon(self) -> Dict[int, float]:
        return dict(zip(self.horizons, self.expected_weights))


@dataclass
class EMAConfig:
    """EMA 配置：指数移动平均相关设置"""

    ema_start: float = 0.996
    ema_end: float = 0.999

    @property
    def ema_range(self) -> Tuple[float, float]:
        return (self.ema_start, self.ema_end)


@dataclass
class SegmentationConfig:
    """分割配置：分割模块相关设置"""

    use_segmentation: bool = True
    seg_loss_weight: float = 1.0
    seg_data_root: str = "/path/to/segmentation/annotations"
    num_classes: int = 2  # seg head class count (was a hardcoded `# 占位` in segmentation.py)
    loss_seg_weight: float = 2.0  # seg vs dice loss balance (were hardcoded in segmentation.py)
    loss_dice_weight: float = 5.0


@dataclass
class LossConfig:
    """损失函数配置"""

    auto_steps: Optional[int] = None
    loss_exp: float = 2.0
    normalize_reps: bool = True


@dataclass
class OptimizationConfig:
    """优化器配置"""

    ipe: Optional[int] = None
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    epochs: int = 100
    # Scheduler horizon may exceed the training stop epoch for selected checkpoints.
    schedule_epochs: Optional[int] = None
    anneal: int = 1
    warmup: int = 10
    start_lr: float = 0.0001
    lr: float = 0.0001
    final_lr: float = 0.0
    enc_lr_scale: float = 1.0
    # predictor 参数学习率缩放（相对 base lr）。微调预训练 predictor 时通常取 <1（如 0.1）。
    # 注意：仅 WarmupCosineSchedule 生效；anneal/cooldown 用的 LinearDecaySchedule 会忽略 lr_scale。
    predictor_lr_scale: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0  # gradient-clipping max-norm (was hardcoded 1.0 in the training lines)
    optimizer: str = "adamw"  # only 'adamw' is currently supported; other values fail loud
    is_anneal: bool = False
    anneal_ckpt: Optional[str] = None
    resume_anneal: bool = False
    ipe_scale: float = 1.0
