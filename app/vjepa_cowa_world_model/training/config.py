"""
配置解析模块

提供结构化的配置数据类，用于解析训练配置 YAML 文件。
"""

# Facade: definitions live in training/configs/*; import surface is unchanged.
# flake8: noqa: F401
from app.vjepa_cowa_world_model.training.configs.common import (
    _PROPOSAL_PRETRAIN_YAML_NAMES,
    NAVSIM_IMAGE_REQUIRE_ALL_FRAMES,
    NAVSIM_IMAGE_REQUIRE_AUTO,
    NAVSIM_IMAGE_REQUIRE_OBSERVED_ONLY,
    NAVSIM_IMAGE_REQUIRE_POLICIES,
    PLANNER_OBSERVED_TOKEN_CONCAT,
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    PLANNER_OBSERVED_TOKEN_MODES,
    PLANNER_OBSERVED_TOKEN_NONE,
    _get_adjacent_pretrain_config,
    _get_adjacent_pretrain_value,
    _get_encoder_static_attr,
    _get_nested_value,
    _get_proposal_pretrain_value,
    _is_vjepa_proposal_encoder_config,
    _load_proposal_pretrain_config,
    _parse_hw_resolution,
    _resolve_observed_frames_strict,
    _validate_vjepa_main_token_ae,
    apply_multiview_image_size_multiplier,
    apply_multiview_token_multiplier,
    compute_tokens_per_frame,
    is_dinov2_main_encoder_config,
    is_factory_pretrained_main_encoder_config,
    is_multiview_per_view_output,
    is_vjepa_main_encoder_config,
    normalize_image_size,
    resolve_effective_tokens_per_frame,
    resolve_main_encoder_frame_stride,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_predictor_img_size,
    resolve_main_encoder_raw_tokens_per_frame,
    resolve_main_encoder_tokens_per_frame,
    resolve_multiview_num_views,
    resolve_predictor_runtime_normalize_reps,
    resolve_proposal_encoder_backbone,
    resolve_proposal_num_time_steps,
    resolve_proposal_runtime_normalize_reps,
    resolve_proposal_tokens_per_frame,
    resolve_proposal_use_token_ae,
)
from app.vjepa_cowa_world_model.training.configs.core import (
    EMAConfig,
    LossConfig,
    MetaConfig,
    ModelConfig,
    OptimizationConfig,
    PredictorDynamicRolloutConfig,
    SegmentationConfig,
    TrainConfig,
    ValidationSuiteConfig,
)
from app.vjepa_cowa_world_model.training.configs.counterfactual import CounterfactualSupervisionConfig
from app.vjepa_cowa_world_model.training.configs.cvoi import (
    CVOI_DUAL_VALUE_NAVSIM_E120_SCHEMA,
    CVOI_DUAL_VALUE_SCHEMA,
    CVOI_FULL_STATE_WARMSTART_CONFIG_SCHEMA,
    CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1,
    CVOI_TRAINING_STAGES,
    CVoIConfig,
    CvoiFullStateWarmstartConfig,
    CvoiPinnedArtifactConfig,
    cvoi_uses_managed_predictor_initialization,
    is_cvoi_formal_v2_navsim_e120_profile,
    resolve_cvoi_formal_v2_navsim_e120_planner_stage,
    validate_cvoi_config,
)
from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import (
    CVOI_CF_FIELD_WEIGHTS,
    CVOI_EVALUATION_SEED,
    CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA,
    CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES,
    CVOI_FORMAL_V2_NAVSIM_E120_INITIALIZATION_MODE,
    CVOI_FORMAL_V2_NAVSIM_E120_PREFIX_MODES,
    CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL,
    CVOI_FORMAL_V2_NAVSIM_E120_VALUE_MECHANISMS,
    CVOI_PREFIX_POLICY_PARAMETERS,
    CvoiFormalV2NavSimE120AblationSignature,
    parse_cvoi_ablation_signature,
)
from app.vjepa_cowa_world_model.training.configs.data import (
    Bench2DriveConfig,
    DataAugConfig,
    DataConfig,
    MongoRawConfig,
    NavSimConfig,
    validate_cvoi_navsim_geometry_contract,
    validate_navsim_cvoi_geometry_contract,
)
from app.vjepa_cowa_world_model.training.configs.lewm import LeWMConfig  # back-compat alias of WorldModelConfig
from app.vjepa_cowa_world_model.training.configs.lewm import Stage2Config  # back-compat alias of RefinementConfig
from app.vjepa_cowa_world_model.training.configs.lewm import Stage3Config  # back-compat alias of RefinementGatedConfig
from app.vjepa_cowa_world_model.training.configs.lewm import (
    RefinementConfig,
    RefinementGatedConfig,
    StageLossWeightsConfig,
    WorldModelConfig,
)
from app.vjepa_cowa_world_model.training.configs.parse import (
    TrainingConfig,
    _normalize_predictor_aux_policy,
    _normalize_predictor_loss_scope,
    _normalize_predictor_supervision_mode,
    _validate_explicit_frozen_encoder_config,
    _validate_lewm_normalize_reps_config,
    _validate_multiview_config,
    _validate_reuse_context_as_target_config,
    _validate_vjepa_parallel_predictor_config,
    normalize_planner_observed_token_mode,
    parse_training_config,
    resolve_bench2drive_image_require_policy,
    resolve_image_require_policy,
    resolve_navsim_image_require_policy,
    resolve_planner_observed_token_mode,
    resolve_planner_use_observed_tokens,
)
from app.vjepa_cowa_world_model.training.configs.planner import (
    MultiViewConfig,
    PlannerConfig,
    PredictorDiTConfig,
    ProposalConfig,
    TokenAEConfig,
)
from app.vjepa_cowa_world_model.training.configs.reward import (
    BudgetControllerConfig,
    RewardConfig,
    RewardSelectorConfig,
    RLConfig,
    ValueGuidanceConfig,
    ValuePlanningConfig,
    WMTrajOptConfig,
    WorldModelAuxConfig,
)
