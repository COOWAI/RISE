"""Split from training/config.py (verbatim node moves). Part: reward."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardConfig:
    """Predictor-token reward model 配置"""

    enabled: bool = False
    hidden_dim: int = 512
    dropout: float = 0.1
    horizon_seconds: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    near_miss_distance: float = 2.0
    near_miss_weight: float = 0.75
    comfort_weight: float = 0.20
    train_roots: List[Dict[str, Any]] = field(default_factory=list)
    val_roots: List[Dict[str, Any]] = field(default_factory=list)
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None


@dataclass
class RewardSelectorConfig:
    """Reward-based mode selector 配置 (方向 B / Step 1)。

    启用后，planner 的 WTA winner 不再按纯轨迹 L2 选择，而是按 composite reward
    (higher=better) 的 argmax 选择；reward 由 losses.reward_selector 计算并作为
    mode_scores 传入 wta_loss*。各权重影响监督语义（哪个 mode 被回归 / 作为 conf
    label），按 CLAUDE.md fail-loud 约定：enabled 时必须显式提供，不允许静默默认。
    """

    enabled: bool = False
    world_model_weight: float = 1.0  # PRIMARY: per-mode latent match (predictor rollout vs real future latent)
    trajectory_error_weight: float = 0.0  # shaping: ADE vs GT
    comfort_weight: float = 0.0  # shaping: candidate accel/yaw/jerk
    collision_weight: float = 0.0  # shaping: counterfactual collision
    offroad_weight: float = 0.0  # shaping: requires drivable map (unwired → raises)


@dataclass
class WorldModelAuxConfig:
    """World-model 预测准度辅助监督 (路线 Phase 1 / doc 方向 D)。

    三个独立门控组件，加在 predictor 训练上（默认全关 → 旧行为不变）：
    - multistep_discount: AR rollout (sloss) 的逐步 λ^k 折扣加权 (doc 9.2)。
      None=关; λ∈(0,1]; λ=1.0 等价于现行均权（数值 allclose）。
    - reward_head_weight: reward/risk 辅助头**联合**训练 (doc 9.3)——梯度流入
      predictor，迫使 world model 学决策相关结构。与 train_reward_model（冻结
      predictor 上训头）本质不同。label 复用 compute_safety_reward_labels。
    - contrastive_weight: ranking 约束 (doc 9.4)——GT 轨迹条件下 rollout 的未来
      latent 应比反事实轨迹（无参数 kinematic 网格生成）更接近真实未来 latent。
      正负样本走同一条 rollout_predictor_modes 路径，只有轨迹不同。
    """

    multistep_discount: Optional[float] = None
    reward_head_weight: float = 0.0
    reward_head_hidden_dim: int = 512
    contrastive_weight: float = 0.0
    contrastive_num_negatives: int = 4
    contrastive_margin: float = 0.1


@dataclass
class WMTrajOptConfig:
    """Phase 3: world-model 导向的候选轨迹优化器配置(推理/验证期)。

    候选轨迹经可微 rollout + 冻结 reward head(Phase 1 联合训练产物,不依赖
    h_target)打分,做 N 步信赖域梯度上升。reward head 从 world-model
    checkpoint 的 ``wm_reward_head``(extra_state)加载,缺失即 fail-loud。
    """

    enabled: bool = False
    steps: int = 5
    lr: float = 0.1
    trust_radius_xy: float = 1.0  # 米, L∞
    trust_radius_yaw: float = 0.2  # 弧度, L∞
    comfort_weight: float = 0.1


@dataclass
class RLConfig:
    """闭环 RL 配置"""

    enabled: bool = False
    algo: str = "ppo"
    status_mode: str = "current_only"
    hugsim_repo_root: Optional[str] = None
    scenario_path: Optional[str] = None
    scenario_manifest: Optional[str] = None
    base_path: Optional[str] = None
    camera_path: Optional[str] = None
    kinematic_path: Optional[str] = None
    camera_name: str = "CAM_FRONT"
    output_subdir: str = "hugsim_rl"
    eval_checkpoint: Optional[str] = None
    rollout_steps: int = 128
    max_episode_steps: int = 400
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    reward_scale: float = 1.0
    rl_loss_weight: float = 1.0
    supervised_loss_weight: float = 0.0
    supervised_warmup_epochs: int = 0
    supervised_batches_per_epoch: int = 0
    normalize_advantage: bool = True
    deterministic_eval: bool = True
    eval_episodes: int = 1
    wheel_base: float = 2.7
    kinematic_dt: float = 0.25


@dataclass
class ValuePlanningConfig:
    """Variant A Method 1 value-planning 配置."""

    enabled: bool = False
    variant: str = "a_method1"
    prefix_steps: int = 2
    gamma: float = 0.99
    lambda_return: float = 0.8
    bootstrap_horizon: int = 3
    progress_weight: float = 1.0
    comfort_weight: float = 0.2
    value_loss_weight: float = 1.0
    td_loss_weight: float = 1.0
    safe_floor_weight: float = 0.05
    episode_ranking_weight: float = 0.1
    episode_ranking_margin: float = 1.0
    validation_calibration_weight: float = 1.0
    validation_ranking_weight: float = 1.0
    srpo_shaping_weight: float = 0.0
    srpo_rho_mode: str = "nearest_accident_latent"
    srpo_potential_based: bool = True
    pred_consistency_weight: float = 0.0
    target_tau: float = 0.995


@dataclass
class ValueGuidanceConfig:
    """Latent-side value guidance 配置.

    Guidance 作用在 predictor 产生的 ``z_planner_input`` 上，不改变 predictor
    参数，也不引入 candidate-conditioned rollout。
    """

    enabled: bool = False
    steps: int = 1
    step_size: float = 0.05
    max_delta_norm: float = 0.25
    objective: str = "last"
    detach_output: bool = True


@dataclass
class BudgetControllerConfig:
    """Continuous compute-budget controller 配置."""

    enabled: bool = False
    mode: str = "oracle_distillation"
    policy_dist: str = "beta"
    lambda_compute: float = 0.0
    oracle_budget_grid: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    schedule: Dict[str, Any] = field(default_factory=dict)
    hidden_dim: int = 128
    feature_dim: int = 0
    min_concentration: float = 1.0
    oracle_path: Optional[str] = None
    oracle_output_path: Optional[str] = None
    output_checkpoint: Optional[str] = None
    controller_checkpoint: Optional[str] = None
    bc_mse_weight: float = 0.1
    grpo_num_samples_per_scene: int = 4
    grpo_bc_weight: float = 0.0
    grpo_reward_interp: str = "linear"
    online_reward_source: str = "world4drive_l2_avg"
    online_resume_checkpoint: Optional[str] = None
