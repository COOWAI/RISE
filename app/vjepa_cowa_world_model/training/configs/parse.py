"""Split from training/config.py (verbatim node moves). Part: parse."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from app.vjepa_cowa_world_model.training.budget_control import BudgetSchedule
from app.vjepa_cowa_world_model.training.configs.common import (
    IMAGE_REQUIRE_ALL_FRAMES,
    IMAGE_REQUIRE_AUTO,
    IMAGE_REQUIRE_OBSERVED_ONLY,
    IMAGE_REQUIRE_POLICIES,
    NAVSIM_IMAGE_REQUIRE_AUTO,
    NAVSIM_IMAGE_REQUIRE_POLICIES,
    PLANNER_OBSERVED_TOKEN_CONCAT,
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    PLANNER_OBSERVED_TOKEN_MODES,
    PLANNER_OBSERVED_TOKEN_NONE,
    _get_nested_value,
    _parse_hw_resolution,
    is_vjepa_main_encoder_config,
    normalize_image_size,
    resolve_main_encoder_tokens_per_frame,
    resolve_multiview_num_views,
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
from app.vjepa_cowa_world_model.training.configs.counterfactual import (
    CounterfactualHazardNegativePairingConfig,
    CounterfactualSupervisionConfig,
)
from app.vjepa_cowa_world_model.training.configs.cvoi import (
    CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1,
    CVoIConfig,
    parse_cvoi_config,
    validate_cvoi_config,
    validate_cvoi_cross_section,
)
from app.vjepa_cowa_world_model.training.configs.data import (
    NAVSIM_DEFAULT_MAX_AGENTS,
    Bench2DriveConfig,
    DataAugConfig,
    DataConfig,
    MongoRawConfig,
    NavSimConfig,
    require_positive_navsim_max_agents,
    resolve_navsim_root_max_agents,
)
from app.vjepa_cowa_world_model.training.configs.lewm import (
    RefinementConfig,
    RefinementGatedConfig,
    StageLossWeightsConfig,
    WorldModelConfig,
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
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_MAX_AGENTS
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST,
    FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION,
)
from app.vjepa_cowa_world_model.training.predictor_split_backward import validate_predictor_split_tf_ar_backward_config
from app.vjepa_cowa_world_model.training.prefix_schedule import normalize_horizon_probabilities
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _reject_removed_prefix_fields(section_name: str, values: Dict[str, Any], replacements: Dict[str, str]) -> None:
    """Reject v1 prefix keys instead of silently ignoring or aliasing their semantics."""
    removed = [field for field in replacements if field in values]
    if not removed:
        return
    details = ", ".join(f"{field!r} -> {replacements[field]!r}" for field in removed)
    raise ValueError(f"{section_name} contains removed prefix semantic field(s): {details}")


def _validate_prefix_probability(field_path: str, value: float) -> None:
    """Fail fast unless a prefix probability is finite and within the unit interval."""
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_path} must be finite and in [0, 1], got {value}")


def _parse_selection_checkpoint_epochs(value: object) -> tuple[int, ...]:
    """Parse strict, unique, increasing 1-based selector checkpoint epochs."""

    if type(value) not in (list, tuple):
        raise TypeError(
            "meta.selection_checkpoint_epochs must be a list or tuple of positive 1-based integers, "
            f"got {type(value).__name__}"
        )
    epochs = tuple(value)
    if any(type(epoch) is not int for epoch in epochs):
        raise TypeError("meta.selection_checkpoint_epochs must contain only integers (booleans are not accepted)")
    if any(epoch <= 0 for epoch in epochs):
        raise ValueError("meta.selection_checkpoint_epochs must contain only positive 1-based integers")
    if any(current >= following for current, following in zip(epochs, epochs[1:])):
        raise ValueError("meta.selection_checkpoint_epochs must be unique and strictly increasing")
    return epochs


def _validate_selection_checkpoint_epochs_bound(
    selection_checkpoint_epochs: tuple[int, ...],
    *,
    optimization_epochs: object,
) -> None:
    if not selection_checkpoint_epochs:
        return
    if type(optimization_epochs) is not int or optimization_epochs <= 0:
        raise ValueError(
            "meta.selection_checkpoint_epochs requires optimization.epochs to be a positive integer, "
            f"got {optimization_epochs!r}"
        )
    if selection_checkpoint_epochs[-1] > optimization_epochs:
        raise ValueError(
            "meta.selection_checkpoint_epochs must not exceed optimization.epochs: "
            f"last={selection_checkpoint_epochs[-1]}, optimization.epochs={optimization_epochs}"
        )


def _parse_positive_dinov2_int(value: object, field_path: str) -> int:
    """Parse a DINOv2 positive integer without coercing incompatible YAML values."""

    if type(value) is not int:
        raise ValueError(f"{field_path} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field_path} must be positive, got {value!r}")
    return value


def _parse_dinov2_resolution(value: object) -> tuple[int, int]:
    """Validate the DINOv2 image resolution before using the shared resolution parser."""

    field_path = "model.dinov2_resolution"
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{field_path} must be a two-element sequence of positive integers, got {value!r}")
    return _parse_hw_resolution(value, field_path)


@dataclass
class TrainingConfig:
    """完整的训练配置"""

    method: str = ""
    meta: MetaConfig = field(default_factory=MetaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    multiview: MultiViewConfig = field(default_factory=MultiViewConfig)
    predictor_dynamic_rollout: PredictorDynamicRolloutConfig = field(default_factory=PredictorDynamicRolloutConfig)
    validation_suite: ValidationSuiteConfig = field(default_factory=ValidationSuiteConfig)
    predictor_dit: PredictorDiTConfig = field(default_factory=PredictorDiTConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    counterfactual_supervision: CounterfactualSupervisionConfig = field(
        default_factory=CounterfactualSupervisionConfig
    )
    data: DataConfig = field(default_factory=DataConfig)
    data_aug: DataAugConfig = field(default_factory=DataAugConfig)
    token_ae: TokenAEConfig = field(default_factory=TokenAEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    refinement_gated: RefinementGatedConfig = field(default_factory=RefinementGatedConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    reward_selector: RewardSelectorConfig = field(default_factory=RewardSelectorConfig)
    wm_aux: WorldModelAuxConfig = field(default_factory=WorldModelAuxConfig)
    value_planning: ValuePlanningConfig = field(default_factory=ValuePlanningConfig)
    value_guidance: ValueGuidanceConfig = field(default_factory=ValueGuidanceConfig)
    budget_controller: BudgetControllerConfig = field(default_factory=BudgetControllerConfig)
    cvoi: CVoIConfig = field(default_factory=CVoIConfig)
    traj_opt: WMTrajOptConfig = field(default_factory=WMTrajOptConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @property
    def dtype(self) -> torch.dtype:
        """获取 PyTorch dtype"""
        dtype_str = self.meta.dtype.lower()
        if dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float16":
            return torch.float16
        else:
            return torch.float32

    @property
    def mixed_precision(self) -> bool:
        """是否使用混合精度"""
        return self.meta.dtype.lower() in ("bfloat16", "float16")

    @property
    def effective_tokens_per_frame(self) -> int:
        """运行时每帧有效 token 数，Token AE 开启时返回压缩后的 latent token 数。"""
        return resolve_main_encoder_tokens_per_frame(self)


def resolve_image_require_policy(config: TrainingConfig, dataset_key: str) -> str:
    """Resolve a dataset's image-availability policy from training mode.

    Predictor supervision needs future image tokens, so predictor-only and joint predictor+planner training
    must require all frames to have images. Planner-only training only needs observed-frame images; future
    supervision comes from states/actions/trajectory labels.
    """
    dataset_cfg = getattr(getattr(config, "data", None), dataset_key, None)
    raw_policy = getattr(dataset_cfg, "image_require_policy", IMAGE_REQUIRE_AUTO)
    policy = str(raw_policy or IMAGE_REQUIRE_AUTO).lower()
    if policy not in IMAGE_REQUIRE_POLICIES:
        raise ValueError(
            f"data.{dataset_key}.image_require_policy must be one of "
            f"{sorted(IMAGE_REQUIRE_POLICIES)}, got {raw_policy!r}"
        )
    if policy != IMAGE_REQUIRE_AUTO:
        return policy
    if bool(getattr(config.train, "predictor_train", True)):
        return IMAGE_REQUIRE_ALL_FRAMES
    if bool(getattr(config.planner, "use_planner", False)):
        return IMAGE_REQUIRE_OBSERVED_ONLY
    return IMAGE_REQUIRE_ALL_FRAMES


def resolve_navsim_image_require_policy(config: TrainingConfig) -> str:
    """Back-compat wrapper: NavSim image-require policy."""
    return resolve_image_require_policy(config, "navsim")


def resolve_bench2drive_image_require_policy(config: TrainingConfig) -> str:
    """Back-compat wrapper: Bench2Drive image-require policy."""
    return resolve_image_require_policy(config, "bench2drive")


def _resolve_image_require_policies_in_place(config: TrainingConfig) -> None:
    """Resolve `auto` image_require_policy at parse time and store the concrete value.

    The policy depends only on config (predictor_train / use_planner), so resolving it here makes the
    stored/logged config value the *effective* one (no more silent "auto" whose meaning is only decided
    later at dataloader init). Idempotent: the data-loader callers re-run the resolver and get this
    concrete value back unchanged.
    """
    navsim = getattr(getattr(config, "data", None), "navsim", None)
    if navsim is not None:
        resolved = resolve_navsim_image_require_policy(config)
        if str(navsim.image_require_policy) != resolved:
            logger.info(
                "data.navsim.image_require_policy: %s -> %s (resolved at parse)",
                navsim.image_require_policy,
                resolved,
            )
        navsim.image_require_policy = resolved
    bench2drive = getattr(getattr(config, "data", None), "bench2drive", None)
    if bench2drive is not None:
        resolved = resolve_bench2drive_image_require_policy(config)
        if str(bench2drive.image_require_policy) != resolved:
            logger.info(
                "data.bench2drive.image_require_policy: %s -> %s (resolved at parse)",
                bench2drive.image_require_policy,
                resolved,
            )
        bench2drive.image_require_policy = resolved


def _validate_reuse_context_as_target_config(config: TrainingConfig) -> None:
    if not config.train.reuse_context_as_target_when_frozen:
        return

    if config.world_model.enabled or config.method in ("lewm", "le-wm", "le_wm"):
        raise ValueError("LEWM method cannot use train.reuse_context_as_target_when_frozen")

    if config.multiview.enabled:
        raise ValueError("train.reuse_context_as_target_when_frozen cannot be true when multiview.enabled is true")

    enabled_encoder_flags = []
    if config.train.encoder_train:
        enabled_encoder_flags.append("train.encoder_train")
    if config.train.encoder_ema:
        enabled_encoder_flags.append("train.encoder_ema")
    if enabled_encoder_flags:
        raise ValueError(
            "train.reuse_context_as_target_when_frozen requires train.encoder_train=False and "
            "train.encoder_ema=False; got " + ", ".join(f"{flag}=True" for flag in enabled_encoder_flags)
        )


def _validate_explicit_frozen_encoder_config(
    config: TrainingConfig,
    *,
    encoder_train_configured: bool,
) -> None:
    if not encoder_train_configured or config.train.encoder_train:
        return
    if config.world_model.enabled or config.method in ("lewm", "le-wm", "le_wm"):
        return
    if config.multiview.enabled:
        return
    if config.train.encoder_ema:
        raise ValueError("train.encoder_train=False requires train.encoder_ema=False")
    if not config.train.reuse_context_as_target_when_frozen:
        raise ValueError("train.encoder_train=False requires " "train.reuse_context_as_target_when_frozen=True")


def _validate_vjepa_parallel_predictor_config(config: TrainingConfig) -> None:
    if not is_vjepa_main_encoder_config(config):
        return
    if config.model.vjepa_use_causal_attention:
        return
    if config.train.use_parallel_predictor:
        return
    raise ValueError("model.vjepa_use_causal_attention=False requires train.use_parallel_predictor=True")


def _normalize_predictor_supervision_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "z_tf": "tf",
        "tf_only": "tf",
        "teacher_forcing": "tf",
        "teacher_forced": "tf",
        "z_ar": "ar",
        "ar_only": "ar",
        "autoregressive": "ar",
        "both": "tf_ar",
        "tf+ar": "tf_ar",
        "tf_ar_supervision": "tf_ar",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "tf", "ar", "tf_ar"}:
        raise ValueError("train.predictor_supervision_mode must be one of ['tf', 'ar', 'tf_ar'], got " f"{value!r}")
    return mode


def _normalize_predictor_aux_policy(value: Any) -> str:
    policy = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "none": "no_aux",
        "zero": "no_aux",
        "zero_aux": "no_aux",
        "nostate": "no_state",
        "zero_state": "no_state",
        "ic": "inference_consistent",
        "infer_consistent": "inference_consistent",
        "future_mask": "mask_future",
    }
    policy = aliases.get(policy, policy)
    if policy not in {"auto", "full", "inference_consistent", "mask_future", "no_state", "no_aux"}:
        raise ValueError(
            "train.predictor_aux_policy must be one of "
            "['auto', 'full', 'inference_consistent', 'mask_future', 'no_state', 'no_aux'], got "
            f"{value!r}"
        )
    return policy


def _normalize_predictor_loss_scope(value: Any) -> str:
    scope = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "next": "next_step",
        "all": "next_step",
        "all_next": "next_step",
        "all_next_steps": "next_step",
        "teacher_forcing": "next_step",
        "tf": "next_step",
        "future": "future_only",
        "future_tokens": "future_only",
        "future_steps": "future_only",
        "observed_future": "future_only",
        "inference_consistent": "future_only",
        "ic": "future_only",
    }
    scope = aliases.get(scope, scope)
    if scope not in {"auto", "next_step", "future_only"}:
        raise ValueError(
            "train.predictor_loss_scope must be one of ['auto', 'next_step', 'future_only'], got " f"{value!r}"
        )
    return scope


def normalize_planner_observed_token_mode(value: Any = None, use_observed_tokens: Optional[Any] = None) -> str:
    """Normalize planner observed-token ablation mode.

    ``use_observed_tokens`` is accepted only as a backward-compatible source
    when the new enum is not provided.
    """
    explicit_mode = value is not None
    if explicit_mode:
        mode = str(value).strip().lower().replace("-", "_")
    elif use_observed_tokens is not None:
        mode = PLANNER_OBSERVED_TOKEN_CONCAT if bool(use_observed_tokens) else PLANNER_OBSERVED_TOKEN_NONE
    else:
        mode = PLANNER_OBSERVED_TOKEN_NONE

    aliases = {
        "false": PLANNER_OBSERVED_TOKEN_NONE,
        "off": PLANNER_OBSERVED_TOKEN_NONE,
        "no": PLANNER_OBSERVED_TOKEN_NONE,
        "disabled": PLANNER_OBSERVED_TOKEN_NONE,
        "true": PLANNER_OBSERVED_TOKEN_CONCAT,
        "on": PLANNER_OBSERVED_TOKEN_CONCAT,
        "simple": PLANNER_OBSERVED_TOKEN_CONCAT,
        "with_observed": PLANNER_OBSERVED_TOKEN_CONCAT,
        "concat_source": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
        "source_embed": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
        "source_embedding": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
        "type_embed": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
        "type_embedding": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    }
    mode = aliases.get(mode, mode)
    if mode not in PLANNER_OBSERVED_TOKEN_MODES:
        raise ValueError(
            "planner.observed_token_mode must be one of " f"{sorted(PLANNER_OBSERVED_TOKEN_MODES)}, got {value!r}"
        )
    if explicit_mode and use_observed_tokens is not None:
        expected_use_observed = mode != PLANNER_OBSERVED_TOKEN_NONE
        if bool(use_observed_tokens) != expected_use_observed:
            raise ValueError(
                "planner.use_observed_tokens conflicts with "
                f"planner.observed_token_mode={mode!r}; remove the legacy boolean or set it to "
                f"{expected_use_observed}"
            )
    return mode


def resolve_planner_observed_token_mode(config: Any) -> str:
    """Resolve observed-token mode from dataclass or raw-dict config."""
    planner = _get_nested_value(config, "planner", default=None)
    explicit_mode = _get_nested_value(config, "planner", "observed_token_mode", default=None)
    legacy_bool = _get_nested_value(config, "planner", "use_observed_tokens", default=None)
    if isinstance(planner, dict):
        return normalize_planner_observed_token_mode(
            explicit_mode if "observed_token_mode" in planner else None,
            legacy_bool if "use_observed_tokens" in planner else None,
        )
    if explicit_mode in (None, PLANNER_OBSERVED_TOKEN_NONE) and legacy_bool is not None:
        return normalize_planner_observed_token_mode(None, legacy_bool)
    return normalize_planner_observed_token_mode(explicit_mode, None)


def resolve_planner_use_observed_tokens(config: Any) -> bool:
    """Return whether the resolved observed-token mode consumes observed tokens."""
    return resolve_planner_observed_token_mode(config) != PLANNER_OBSERVED_TOKEN_NONE


def _validate_lewm_normalize_reps_config(config: TrainingConfig) -> None:
    if not config.world_model.enabled and config.method not in ("lewm", "le-wm", "le_wm"):
        return
    if config.loss.normalize_reps is False:
        return
    raise ValueError("LEWM method requires loss.normalize_reps=False")


def _validate_multiview_config(config: TrainingConfig) -> None:
    """When multiview is enabled, ensure the geometry-driven PETR fusion can actually run.

    The fused tokens depend entirely on per-view camera geometry (camera_intrinsics + camera2ego,
    sourced from camera_metadata). Catch the common misconfigurations at config-parse time, before
    any data is loaded, rather than letting the fusion silently degrade.
    """
    if not bool(getattr(config.multiview, "enabled", False)):
        return
    fusion_type = str(getattr(config.multiview, "fusion_type", ""))
    if fusion_type != "petr_cross_attn":
        raise ValueError(
            f"multiview.enabled=true requires multiview.fusion_type='petr_cross_attn', got {fusion_type!r}"
        )
    num_views = resolve_multiview_num_views(config)
    if num_views < 2:
        camera_names = _get_nested_value(config, "data", "navsim", "camera_names", default=None)
        raise ValueError(
            "multiview.enabled=true requires data.navsim.camera_names to list at least 2 cameras "
            "so the PETR fusion receives per-view camera_intrinsics/camera2ego; "
            f"got camera_names={camera_names!r} (num_views={num_views})."
        )


def _validate_policy_output_source_config(config: TrainingConfig) -> None:
    source = str(getattr(config.planner, "policy_output_source", "planner")).lower()
    if source not in {"planner", "joint_action"}:
        raise ValueError("planner.policy_output_source must be one of ['planner', 'joint_action'], " f"got {source!r}")
    config.planner.policy_output_source = source
    if source != "joint_action":
        return
    if not bool(config.planner.use_planner):
        raise ValueError(
            "planner.policy_output_source='joint_action' requires planner.use_planner=True so the "
            "policy/trajectory objective and validation pipeline are enabled."
        )
    if str(getattr(config.train, "predictor_type", "")).lower() != "latent_dit":
        raise ValueError("planner.policy_output_source='joint_action' requires train.predictor_type='latent_dit'")
    if not bool(getattr(config.predictor_dit, "joint_action_enabled", False)):
        raise ValueError(
            "planner.policy_output_source='joint_action' requires predictor_dit.joint_action_enabled=True"
        )
    if int(getattr(config.predictor_dit, "joint_action_dim", 3)) != 3:
        raise ValueError(
            "planner.policy_output_source='joint_action' requires predictor_dit.joint_action_dim=3 "
            "for ego-frame [dx, dy, dyaw] actions"
        )


# Pre-restructure config section keys -> new canonical names. parse_training_config remaps these at the
# top so old YAMLs + params-pretrain.yaml sidecars keep loading (fail-loud if a file mixes old and new).
_LEGACY_CONFIG_SECTION_ALIASES = {
    "lewm": "world_model",
    "le-wm": "world_model",
    "le_wm": "world_model",
    "stage2": "refinement",
    "stage3": "refinement_gated",
}


def _migrate_legacy_config_sections(args: Dict[str, Any]) -> Dict[str, Any]:
    """Rename old top-level config sections to new names in-place; fail-loud on an old+new conflict."""
    for old_key, new_key in _LEGACY_CONFIG_SECTION_ALIASES.items():
        if old_key not in args:
            continue
        if new_key in args:
            raise ValueError(
                f"Config has both the legacy section '{old_key}' and its new name '{new_key}'. "
                f"Remove '{old_key}' (it was renamed to '{new_key}')."
            )
        args[new_key] = args.pop(old_key)
        logger.warning("Config section '%s' is deprecated; auto-migrated to '%s'.", old_key, new_key)
    return args


# The predictor-unification rename (docs/NAME_MIGRATION.md) nests the predictor knobs under a `predictor:`
# section — predictor.method (lewm|ema) + predictor.lewm (the SIGReg/projector config). They map onto the flat
# `method` + `world_model` keys the parser consumes, so config readers (config.method / config.world_model) are
# unchanged. The old flat keys keep working; fail-loud on an old+new conflict or an unknown predictor.* key.
_KNOWN_PREDICTOR_KEYS = frozenset({"method", "lewm"})


def _migrate_predictor_section(args: Dict[str, Any]) -> Dict[str, Any]:
    """Map canonical predictor.method / predictor.lewm onto flat method / world_model; fail-loud on conflicts
    or unknown predictor.* keys. Runs before the legacy-section migration so a predictor.lewm and a legacy
    top-level lewm/le-wm both normalize to world_model and conflict-check against each other."""
    predictor = args.get("predictor")
    if predictor is None:
        return args
    if not isinstance(predictor, dict):
        raise ValueError(
            f"`predictor` must be a mapping with predictor.method / predictor.lewm; got {type(predictor).__name__}."
        )
    unknown = set(predictor) - _KNOWN_PREDICTOR_KEYS
    if unknown:
        raise ValueError(f"Unknown predictor.* key(s) {sorted(unknown)}; valid keys: {sorted(_KNOWN_PREDICTOR_KEYS)}.")
    if "method" in predictor:
        new_method = predictor["method"]
        for old_key in ("method", "training_method", "predictor_method"):
            if old_key in args and str(args[old_key]) != str(new_method):
                raise ValueError(
                    f"Config sets both predictor.method={new_method!r} and the deprecated top-level "
                    f"{old_key}={args[old_key]!r}; keep only predictor.method."
                )
        args["method"] = new_method
    if "lewm" in predictor:
        new_world_model = predictor["lewm"]
        if "world_model" in args and args["world_model"] != new_world_model:
            raise ValueError(
                "Config sets both predictor.lewm and a top-level world_model/lewm section; keep only predictor.lewm."
            )
        args["world_model"] = new_world_model
    args.pop("predictor")
    return args


# Launcher/SLURM scalars + top-level sections consumed by app.main / scaffold / the training lines /
# standalone validators (predictor_lora, reward_eval) rather than by parse_training_config itself —
# allowed at the top level even though they are not TrainingConfig fields.
_KNOWN_TOP_LEVEL_EXTRAS = frozenset(
    # launcher/dispatcher metadata + sections validated by standalone validators (NOT by parse_training_config).
    # `stage` / `stage_inputs` are the cf staged-training stage name + per-stage input spec read by the cf
    # dispatcher tool.
    {
        "app",
        "folder",
        "cpus_per_task",
        "mem_per_gpu",
        "nodes",
        "tasks_per_node",
        "predictor_lora",
        "reward_eval",
        "stage",
        "stage_inputs",
    }
)


def _reject_unknown_config_keys(args: Dict[str, Any]) -> None:
    """Fail loud on ANY config key that maps to no dataclass field — a top-level section name (catches
    ``optimzation:`` / ``plannner:``) or a key inside any section (recursing into nested dataclass sections
    like ``data.navsim``). Without this a typo'd / stale key is silently dropped and the structural default
    is used (CLAUDE.md fail-loud). Runs after the legacy-section migration so old names are already remapped.
    """
    import typing
    from dataclasses import MISSING, fields, is_dataclass

    def _nested_dataclass(field_obj, owner_type):
        """The dataclass nested under this field — via default_factory, or an ``Optional[X]`` /
        bare-dataclass type annotation (e.g. ``data.navsim: Optional[NavSimConfig]``) — else None."""
        factory = field_obj.default_factory
        if factory is not MISSING and is_dataclass(factory):
            return factory
        try:
            hint = typing.get_type_hints(owner_type).get(field_obj.name)
        except Exception:
            return None
        for cand in typing.get_args(hint) or ((hint,) if hint is not None else ()):
            if cand is not type(None) and is_dataclass(cand):
                return cand
        return None

    def _check(prefix: str, cfg, dataclass_type) -> None:
        if not isinstance(cfg, dict):
            return
        fmap = {f.name: f for f in fields(dataclass_type)}
        for key, value in cfg.items():
            if key not in fmap:
                raise ValueError(f"Unknown config key '{prefix}{key}'; valid keys here: {sorted(fmap)}.")
            if isinstance(value, dict):
                nested = _nested_dataclass(fmap[key], dataclass_type)
                if prefix == "cvoi." and key == "ablation_signature":
                    from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import (
                        CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA,
                        CvoiFormalV2NavSimE120AblationSignature,
                    )

                    schema = value.get("schema")
                    if schema != CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA:
                        raise ValueError(
                            "cvoi.ablation_signature.schema must be exactly "
                            f"{CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA!r}, got {schema!r}"
                        )
                    nested = CvoiFormalV2NavSimE120AblationSignature
                if nested is not None:
                    _check(f"{prefix}{key}.", value, nested)

    if not isinstance(args, dict):
        return
    top = {f.name: f for f in fields(TrainingConfig)}
    valid_top = set(top) | _KNOWN_TOP_LEVEL_EXTRAS
    for key, value in args.items():
        if key in _KNOWN_TOP_LEVEL_EXTRAS:
            continue
        if key not in top:
            raise ValueError(f"Unknown top-level config section '{key}'; valid: {sorted(valid_top)}.")
        if isinstance(value, dict):
            nested = _nested_dataclass(top[key], TrainingConfig)
            if nested is not None:
                _check(f"{key}.", value, nested)


def parse_training_config(
    args: Dict[str, Any],
    *,
    _allow_evaluation_value_guidance: bool = False,
) -> TrainingConfig:
    """
    从原始 args 字典解析配置

    Args:
        args: 从 YAML 文件加载的配置字典

    Returns:
        TrainingConfig: 结构化的配置对象
    """
    if not isinstance(_allow_evaluation_value_guidance, bool):
        raise TypeError("_allow_evaluation_value_guidance must be a bool")
    raw_cvoi = args.get("cvoi") if isinstance(args, dict) else None
    raw_data = args.get("data") if isinstance(args, dict) else None
    raw_navsim = raw_data.get("navsim") if isinstance(raw_data, dict) else None
    if (
        isinstance(raw_cvoi, dict)
        and raw_cvoi.get("protocol_version") == CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1
        and raw_cvoi.get("stage") != "gate_distillation"
        and (
            not isinstance(raw_navsim, dict)
            or raw_navsim.get("max_agents") != FORMAL_V2_NAVSIM_MAX_AGENTS
            or type(raw_navsim.get("max_agents")) is not int
        )
    ):
        raise ValueError(
            "Formal-v2 NavSim e120 requires explicit " f"data.navsim.max_agents={FORMAL_V2_NAVSIM_MAX_AGENTS}"
        )
    args = _migrate_predictor_section(args)
    args = _migrate_legacy_config_sections(args)
    cfgs_value_planning_raw = args.get("value_planning")
    removed_value_planning_fields = {
        "crash_reward",
        "accident_ranking_weight",
        "accident_root_names",
        "accident_terminal_types",
    }
    if isinstance(cfgs_value_planning_raw, dict):
        stale_fields = sorted(removed_value_planning_fields.intersection(cfgs_value_planning_raw))
        if stale_fields:
            field_name = stale_fields[0]
            raise ValueError(
                f"value_planning.{field_name} was removed by cf_supervision_v2; "
                "use episode-level ranking targets and explicit counterfactual supervision masks"
            )
    if isinstance(cfgs_value_planning_raw, dict) and "no_clone_accident_roots" in cfgs_value_planning_raw:
        raise ValueError(
            "value_planning.no_clone_accident_roots was removed by cf_supervision_v2; "
            "planner imitation now always uses counterfactual_supervision.imitation_policy"
        )
    predictor_dit_args = args.get("predictor_dit")
    if isinstance(predictor_dit_args, dict) and "masked_condition_known_prefix" in predictor_dit_args:
        raise ValueError(
            "predictor_dit.masked_condition_known_prefix was removed in prefix v2; remove this field because "
            "masked prefixes are always cumulative future [0,h)"
        )
    _reject_unknown_config_keys(args)
    # 解析 meta 配置
    cfgs_meta = args.get("meta", {})
    meta = MetaConfig(
        folder=args.get("folder", ""),
        seed=cfgs_meta.get("seed", 0),
        deterministic=cfgs_meta.get("deterministic", False),
        val_stable_noise=cfgs_meta.get("val_stable_noise", True),
        dtype=cfgs_meta.get("dtype", "bfloat16"),
        resume_checkpoint=cfgs_meta.get("resume_checkpoint"),
        pretrain_repo=cfgs_meta.get("pretrain_repo"),
        pretrain_checkpoint=cfgs_meta.get("pretrain_checkpoint"),
        pretrain_checkpoint_full=cfgs_meta.get("pretrain_checkpoint_full"),
        predictor_checkpoint=cfgs_meta.get("predictor_checkpoint"),
        value_checkpoint=cfgs_meta.get("value_checkpoint"),
        planner_value_checkpoint=cfgs_meta.get("planner_value_checkpoint"),
        predictor_runtime_normalize_reps=cfgs_meta.get("predictor_runtime_normalize_reps"),
        ae_checkpoint=cfgs_meta.get("ae_checkpoint"),
        load_encoder=cfgs_meta.get("load_encoder", True),
        load_predictor=cfgs_meta.get("load_predictor", False),
        load_seg=cfgs_meta.get("load_seg", True),
        load_planner=cfgs_meta.get("load_planner", True),
        context_encoder_key=cfgs_meta.get("context_encoder_key", "encoder"),
        target_encoder_key=cfgs_meta.get("target_encoder_key", "target_encoder"),
        save_every_freq=cfgs_meta.get("save_every_freq", -1),
        selection_checkpoint_epochs=_parse_selection_checkpoint_epochs(
            cfgs_meta.get("selection_checkpoint_epochs", ())
        ),
        save_from_epoch=int(cfgs_meta.get("save_from_epoch", 0) or 0),
        skip_batches=cfgs_meta.get("skip_batches", -1),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        sync_gc=cfgs_meta.get("sync_gc", False),
        val_freq=cfgs_meta.get("val_freq", 5),
        resume_broadcast=cfgs_meta.get("resume_broadcast", False),
        resume_model_only=cfgs_meta.get("resume_model_only", False),
        auto_resume_latest=cfgs_meta.get("auto_resume_latest", False),
    )

    # 解析 model 配置
    cfgs_model = args.get("model", {})
    vjepa_resolution = _parse_hw_resolution(
        cfgs_model.get("vjepa_resolution", (256, 512)),
        "model.vjepa_resolution",
    )
    dinov2_resolution = _parse_dinov2_resolution(
        cfgs_model.get("dinov2_resolution", (224, 448)),
    )
    dinov2_frame_stride = _parse_positive_dinov2_int(
        cfgs_model.get("dinov2_frame_stride", 2),
        "model.dinov2_frame_stride",
    )
    dinov2_forward_chunk_size = _parse_positive_dinov2_int(
        cfgs_model.get("dinov2_forward_chunk_size", 16),
        "model.dinov2_forward_chunk_size",
    )
    model = ModelConfig(
        model_name=cfgs_model.get("model_name", ""),
        backbone=cfgs_model.get("backbone", "vjepa2"),
        vjepa_resolution=vjepa_resolution,
        vjepa_crop_top_bottom=cfgs_model.get("vjepa_crop_top_bottom", 28),
        vjepa_num_frames=cfgs_model.get("vjepa_num_frames", 2),
        vjepa_checkpoint_key=cfgs_model.get("vjepa_checkpoint_key", "target_encoder"),
        vjepa_use_grid_mask=cfgs_model.get("vjepa_use_grid_mask", True),
        vjepa_use_causal_attention=cfgs_model.get("vjepa_use_causal_attention", True),
        dinov2_model_name=cfgs_model.get("dinov2_model_name", "vit_large_patch14_reg4_dinov2"),
        dinov2_resolution=dinov2_resolution,
        dinov2_frame_stride=dinov2_frame_stride,
        dinov2_forward_chunk_size=dinov2_forward_chunk_size,
        patch_size=cfgs_model.get("patch_size", 16),
        pred_depth=cfgs_model.get("pred_depth", 12),
        pred_num_heads=cfgs_model.get("pred_num_heads"),
        pred_embed_dim=cfgs_model.get("pred_embed_dim", 384),
        pred_is_frame_causal=cfgs_model.get("pred_is_frame_causal", True),
        uniform_power=cfgs_model.get("uniform_power", False),
        use_rope=cfgs_model.get("use_rope", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        use_extrinsics=cfgs_model.get("use_extrinsics", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", False),
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        compile_model=cfgs_model.get("compile_model", False),
        use_activation_checkpointing=cfgs_model.get("use_activation_checkpointing", False),
    )

    cfgs_world_model = args.get("world_model", {})

    # 解析 train 配置
    cfgs_train = args.get("train", {})
    # num_encoder_frames is the legacy name for num_observed_frames; prefer the canonical key and fail loud
    # on an explicit old+new conflict (don't let the legacy key silently clobber the canonical one).
    _legacy_nef = cfgs_train.get("num_encoder_frames")
    _canonical_nof = cfgs_train.get("num_observed_frames")
    if _legacy_nef is not None and _canonical_nof is not None and int(_legacy_nef) != int(_canonical_nof):
        raise ValueError(
            f"train.num_encoder_frames ({_legacy_nef}) conflicts with train.num_observed_frames "
            f"({_canonical_nof}); set only the canonical train.num_observed_frames."
        )
    observed_frames = _canonical_nof if _canonical_nof is not None else (_legacy_nef if _legacy_nef is not None else 2)
    predictor_use_z_ar_supervision_default = not bool(cfgs_world_model.get("enabled", False))
    raw_supervision_mode = _normalize_predictor_supervision_mode(cfgs_train.get("predictor_supervision_mode", "auto"))
    if raw_supervision_mode == "auto":
        predictor_use_z_ar_supervision = cfgs_train.get(
            "predictor_use_z_ar_supervision",
            predictor_use_z_ar_supervision_default,
        )
        predictor_supervision_mode = "tf_ar" if bool(predictor_use_z_ar_supervision) else "tf"
    else:
        predictor_supervision_mode = raw_supervision_mode
        predictor_use_z_ar_supervision = predictor_supervision_mode in ("ar", "tf_ar")
        if "predictor_use_z_ar_supervision" in cfgs_train and bool(
            cfgs_train["predictor_use_z_ar_supervision"]
        ) != bool(predictor_use_z_ar_supervision):
            raise ValueError(
                "train.predictor_use_z_ar_supervision conflicts with "
                f"train.predictor_supervision_mode={predictor_supervision_mode!r}"
            )
    if raw_supervision_mode == "auto":
        logger.info(
            "train.predictor_supervision_mode: auto -> %s (z_ar_supervision=%s)",
            predictor_supervision_mode,
            bool(predictor_use_z_ar_supervision),
        )
    predictor_aux_policy = _normalize_predictor_aux_policy(cfgs_train.get("predictor_aux_policy", "auto"))
    raw_loss_scope = _normalize_predictor_loss_scope(cfgs_train.get("predictor_loss_scope", "auto"))
    if raw_loss_scope == "auto":
        predictor_inference_consistent = bool(cfgs_train.get("predictor_inference_consistent", False))
        predictor_loss_scope = "future_only" if predictor_inference_consistent else "next_step"
    else:
        predictor_loss_scope = raw_loss_scope
        predictor_inference_consistent = bool(
            cfgs_train.get("predictor_inference_consistent", predictor_loss_scope == "future_only")
        )
    if raw_loss_scope == "auto":
        logger.info(
            "train.predictor_loss_scope: auto -> %s (inference_consistent=%s)",
            predictor_loss_scope,
            bool(predictor_inference_consistent),
        )
    predictor_split_tf_ar_backward = cfgs_train.get("predictor_split_tf_ar_backward", False)
    if type(predictor_split_tf_ar_backward) is not bool:
        raise ValueError(
            "train.predictor_split_tf_ar_backward must be a bool, " f"got {predictor_split_tf_ar_backward!r}"
        )
    train = TrainConfig(
        encoder_train=cfgs_train.get("encoder_train", False),
        seg_head=cfgs_train.get("seg_head", True),
        encoder_ema=cfgs_train.get("encoder_ema", False),
        perceiver_ema=cfgs_train.get("perceiver_ema", True),
        predictor_train=cfgs_train.get("predictor_train", True),
        predictor_planner_finetune=cfgs_train.get("predictor_planner_finetune", False),
        use_states_for_predictor=cfgs_train.get("use_states_for_predictor", True),
        action_dim=cfgs_train.get("action_dim", 7),
        state_dim=cfgs_train.get("state_dim", 7),
        command_dim=cfgs_train.get("command_dim", 0),
        use_drive_command=cfgs_train.get("use_drive_command", False),  # point 23: 默认 False
        predictor_inference_consistent=predictor_inference_consistent,
        predictor_aux_policy=predictor_aux_policy,
        use_parallel_predictor=cfgs_train.get("use_parallel_predictor", False),
        predictor_supervision_mode=predictor_supervision_mode,
        predictor_loss_scope=predictor_loss_scope,
        predictor_use_z_ar_supervision=predictor_use_z_ar_supervision,
        predictor_validation_enabled=cfgs_train.get("predictor_validation_enabled", True),
        predictor_static_graph=cfgs_train.get("predictor_static_graph", False),
        predictor_split_tf_ar_backward=predictor_split_tf_ar_backward,
        reuse_context_as_target_when_frozen=cfgs_train.get("reuse_context_as_target_when_frozen", False),
        predictor_no_aux_input=cfgs_train.get("predictor_no_aux_input", False),
        num_encoder_frames=observed_frames,
        num_observed_frames=observed_frames,
        predictor_type=cfgs_train.get("predictor_type", "ac_transformer"),
        latent_dit_planner_input=cfgs_train.get("latent_dit_planner_input", "train_helper"),
    )

    cfgs_predictor_dynamic_rollout = args.get("predictor_dynamic_rollout", {}) or {}
    _reject_removed_prefix_fields(
        "predictor_dynamic_rollout",
        cfgs_predictor_dynamic_rollout,
        {
            "full_horizon_prob": "full_prefix_prob",
            "min_future_steps": "min_prefix_steps",
            "max_future_steps": "max_non_full_prefix_steps",
        },
    )
    raw_dynamic_max_non_full_prefix_steps = cfgs_predictor_dynamic_rollout.get(
        "max_non_full_prefix_steps",
        None,
    )
    dynamic_max_non_full_prefix_steps = (
        None if raw_dynamic_max_non_full_prefix_steps is None else int(raw_dynamic_max_non_full_prefix_steps)
    )
    raw_horizon_probabilities = cfgs_predictor_dynamic_rollout.get("horizon_probabilities")
    horizon_probabilities = (
        None
        if raw_horizon_probabilities is None
        else normalize_horizon_probabilities(
            raw_horizon_probabilities,
            field_name="predictor_dynamic_rollout.horizon_probabilities",
        )
    )
    dynamic_rollout = PredictorDynamicRolloutConfig(
        enabled=bool(cfgs_predictor_dynamic_rollout.get("enabled", False)),
        full_prefix_prob=float(cfgs_predictor_dynamic_rollout.get("full_prefix_prob", 0.25)),
        min_prefix_steps=int(cfgs_predictor_dynamic_rollout.get("min_prefix_steps", 1)),
        max_non_full_prefix_steps=dynamic_max_non_full_prefix_steps,
        max_horizon=(
            None
            if cfgs_predictor_dynamic_rollout.get("max_horizon") is None
            else int(cfgs_predictor_dynamic_rollout["max_horizon"])
        ),
        horizon_probabilities=horizon_probabilities,
    )
    _validate_prefix_probability(
        "predictor_dynamic_rollout.full_prefix_prob",
        dynamic_rollout.full_prefix_prob,
    )
    if dynamic_rollout.min_prefix_steps < 0:
        raise ValueError(
            "predictor_dynamic_rollout.min_prefix_steps must be >= 0, " f"got {dynamic_rollout.min_prefix_steps}"
        )
    if dynamic_rollout.max_non_full_prefix_steps is not None:
        if dynamic_rollout.max_non_full_prefix_steps < 0:
            raise ValueError(
                "predictor_dynamic_rollout.max_non_full_prefix_steps must be >= 0 when set, "
                f"got {dynamic_rollout.max_non_full_prefix_steps}"
            )
        if dynamic_rollout.min_prefix_steps > dynamic_rollout.max_non_full_prefix_steps:
            raise ValueError(
                "predictor_dynamic_rollout.min_prefix_steps must be <= "
                "predictor_dynamic_rollout.max_non_full_prefix_steps, got min_prefix_steps="
                f"{dynamic_rollout.min_prefix_steps}, max_non_full_prefix_steps="
                f"{dynamic_rollout.max_non_full_prefix_steps}"
            )
    if dynamic_rollout.max_horizon is not None:
        if dynamic_rollout.max_horizon <= 0:
            raise ValueError("predictor_dynamic_rollout.max_horizon must be positive when set")
        if (
            dynamic_rollout.max_non_full_prefix_steps is not None
            and dynamic_rollout.max_non_full_prefix_steps != dynamic_rollout.max_horizon - 1
        ):
            raise ValueError(
                "predictor_dynamic_rollout.max_non_full_prefix_steps must equal max_horizon - 1 when "
                "max_horizon is explicit"
            )
        if (
            dynamic_rollout.horizon_probabilities is not None
            and len(dynamic_rollout.horizon_probabilities) != dynamic_rollout.max_horizon + 1
        ):
            raise ValueError("predictor_dynamic_rollout.horizon_probabilities must contain max_horizon + 1 entries")

    cfgs_validation_suite = args.get("validation_suite", {}) or {}
    validation_suite = ValidationSuiteConfig(
        enabled=bool(cfgs_validation_suite.get("enabled", False)),
        protocol_version=str(cfgs_validation_suite.get("protocol_version", "dynamic_rollout_validation_v2")),
        horizons=list(cfgs_validation_suite.get("horizons", []) or []),
        expected_weights=list(cfgs_validation_suite.get("expected_weights", []) or []),
        primary_domain=str(cfgs_validation_suite.get("primary_domain", "real")),
        primary_cohort=str(cfgs_validation_suite.get("primary_cohort", "all")),
        primary_protocol=str(cfgs_validation_suite.get("primary_protocol", "full")),
    )
    if validation_suite.enabled:
        if not meta.val_stable_noise:
            raise ValueError(
                "validation_suite requires meta.val_stable_noise=true because its sample-keyed common-random "
                "contract is intentionally stable across validation epochs"
            )
        raw_cvoi_profile = args.get("cvoi", {})
        navsim_h4_profile = (
            isinstance(raw_cvoi_profile, dict)
            and raw_cvoi_profile.get("protocol_version") == CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1
        )
        expected_validation_protocol = (
            "dynamic_rollout_validation_navsim_h4_v3" if navsim_h4_profile else "dynamic_rollout_validation_v2"
        )
        if validation_suite.protocol_version != expected_validation_protocol:
            raise ValueError(
                f"validation_suite.protocol_version must be {expected_validation_protocol!r}, "
                f"got {validation_suite.protocol_version!r}"
            )
        normalized_horizons = []
        for horizon in validation_suite.horizons:
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                raise TypeError(f"validation_suite.horizons must contain integers, got {horizon!r}")
            if horizon < 0:
                raise ValueError(f"validation_suite.horizons must be non-negative, got {horizon}")
            normalized_horizons.append(int(horizon))
        if not normalized_horizons:
            raise ValueError("validation_suite.horizons must not be empty when enabled")
        if any(current >= following for current, following in zip(normalized_horizons, normalized_horizons[1:])):
            raise ValueError(
                "validation_suite.horizons must be unique and strictly increasing, " f"got {normalized_horizons}"
            )
        validation_suite.horizons = normalized_horizons
        if len(validation_suite.expected_weights) != len(validation_suite.horizons):
            raise ValueError(
                "validation_suite.expected_weights must have the same length as horizons, "
                f"got {len(validation_suite.expected_weights)} and {len(validation_suite.horizons)}"
            )
        normalized_weights = []
        for weight in validation_suite.expected_weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"validation_suite.expected_weights must contain real numbers, got {weight!r}")
            normalized_weight = float(weight)
            if not math.isfinite(normalized_weight):
                raise ValueError("validation_suite.expected_weights must be finite, " f"got {normalized_weight}")
            if normalized_weight < 0.0:
                raise ValueError("validation_suite.expected_weights must be non-negative, " f"got {normalized_weight}")
            normalized_weights.append(normalized_weight)
        if not math.isclose(sum(normalized_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "validation_suite.expected_weights must sum to exactly 1 and are not implicitly "
                f"renormalized, got {sum(normalized_weights)}"
            )
        validation_suite.expected_weights = normalized_weights
        if navsim_h4_profile:
            if validation_suite.horizons != [0, 1, 2, 3, 4]:
                raise ValueError("NavSim H4 validation_suite.horizons must be exactly [0, 1, 2, 3, 4]")
            if dynamic_rollout.max_horizon != 4:
                raise ValueError("NavSim H4 validation requires predictor_dynamic_rollout.max_horizon=4")
            if tuple(validation_suite.expected_weights) != dynamic_rollout.horizon_probabilities:
                raise ValueError(
                    "NavSim H4 validation_suite.expected_weights must exactly match "
                    "predictor_dynamic_rollout.horizon_probabilities"
                )
        primary_slice = (
            validation_suite.primary_domain,
            validation_suite.primary_cohort,
            validation_suite.primary_protocol,
        )
        if primary_slice != ("real", "all", "full"):
            raise ValueError(
                "validation_suite checkpoint primary slice must remain real/all/full, "
                f"got {'/'.join(primary_slice)}"
            )

    cfgs_multiview = args.get("multiview", {}) or {}
    multiview_output_mode = str(cfgs_multiview.get("output_mode", "fused")).lower()
    if multiview_output_mode not in ("fused", "per_view"):
        raise ValueError(f"multiview.output_mode must be 'fused' or 'per_view', got {multiview_output_mode!r}")
    multiview = MultiViewConfig(
        enabled=bool(cfgs_multiview.get("enabled", False)),
        fusion_type=cfgs_multiview.get("fusion_type", "petr_cross_attn"),
        output_mode=multiview_output_mode,
        hidden_dim=int(cfgs_multiview.get("hidden_dim", 256)),
        num_heads=int(cfgs_multiview.get("num_heads", 8)),
        dropout=float(cfgs_multiview.get("dropout", 0.0)),
        load_from_predictor_checkpoint=bool(cfgs_multiview.get("load_from_predictor_checkpoint", False)),
        freeze_fusion=bool(cfgs_multiview.get("freeze_fusion", False)),
    )

    cfgs_predictor_dit = args.get("predictor_dit", {}) or {}
    _reject_removed_prefix_fields(
        "predictor_dit",
        cfgs_predictor_dit,
        {
            "masked_train_full_horizon_prob": "masked_train_full_prefix_prob",
            "masked_train_min_future_steps": "masked_train_min_prefix_steps",
            "masked_train_max_future_steps": "masked_train_max_non_full_prefix_steps",
        },
    )
    raw_bottleneck_dim = cfgs_predictor_dit.get("bottleneck_dim", None)
    if raw_bottleneck_dim is None or int(raw_bottleneck_dim) <= 0:
        bottleneck_dim = None
    else:
        bottleneck_dim = int(raw_bottleneck_dim)
    raw_masked_train_max_non_full_prefix_steps = cfgs_predictor_dit.get(
        "masked_train_max_non_full_prefix_steps",
        None,
    )
    if raw_masked_train_max_non_full_prefix_steps is None:
        masked_train_max_non_full_prefix_steps = None
    else:
        masked_train_max_non_full_prefix_steps = int(raw_masked_train_max_non_full_prefix_steps)
    raw_joint_action_scale = cfgs_predictor_dit.get("joint_action_scale", (8.0, 4.0, 1.0))
    joint_action_scale = tuple(float(x) for x in raw_joint_action_scale)
    predictor_dit = PredictorDiTConfig(
        objective=cfgs_predictor_dit.get("objective", "flow_matching"),
        conditioning_mode=cfgs_predictor_dit.get("conditioning_mode", "mean"),
        max_condition_steps=int(cfgs_predictor_dit.get("max_condition_steps", 128)),
        num_inference_steps=int(cfgs_predictor_dit.get("num_inference_steps", 8)),
        sampler_type=cfgs_predictor_dit.get("sampler_type", "heun"),
        schedule_type=cfgs_predictor_dit.get("schedule_type", "cosine"),
        temperature=float(cfgs_predictor_dit.get("temperature", 1.0)),
        hidden_dim=int(cfgs_predictor_dit.get("hidden_dim", 512)),
        depth=int(cfgs_predictor_dit.get("depth", 6)),
        num_heads=int(cfgs_predictor_dit.get("num_heads", 8)),
        dropout=float(cfgs_predictor_dit.get("dropout", 0.0)),
        x0_loss_weight=float(cfgs_predictor_dit.get("x0_loss_weight", 0.0)),
        bottleneck_dim=bottleneck_dim,
        use_anchor_frame=bool(cfgs_predictor_dit.get("use_anchor_frame", False)),
        metadata_condition_dropout=float(cfgs_predictor_dit.get("metadata_condition_dropout", 0.0)),
        metadata_guidance_scale=float(cfgs_predictor_dit.get("metadata_guidance_scale", 1.0)),
        metadata_conditioning_policy=cfgs_predictor_dit.get("metadata_conditioning_policy", "auto"),
        masked_inpainting_enabled=bool(cfgs_predictor_dit.get("masked_inpainting_enabled", False)),
        masked_train_full_prefix_prob=float(cfgs_predictor_dit.get("masked_train_full_prefix_prob", 0.25)),
        masked_train_min_prefix_steps=int(cfgs_predictor_dit.get("masked_train_min_prefix_steps", 1)),
        masked_train_max_non_full_prefix_steps=masked_train_max_non_full_prefix_steps,
        masked_sample_return_full=bool(cfgs_predictor_dit.get("masked_sample_return_full", False)),
        joint_action_enabled=bool(cfgs_predictor_dit.get("joint_action_enabled", False)),
        joint_action_loss_weight=float(cfgs_predictor_dit.get("joint_action_loss_weight", 0.0)),
        joint_action_dim=int(cfgs_predictor_dit.get("joint_action_dim", 3)),
        joint_action_scale=joint_action_scale,
        joint_action_state_dim=int(cfgs_predictor_dit.get("joint_action_state_dim", 7)),
        joint_action_noise_mode=str(cfgs_predictor_dit.get("joint_action_noise_mode", "shared")).lower(),
        joint_action_state_mode=str(cfgs_predictor_dit.get("joint_action_state_mode", "last_observed")).lower(),
        joint_action_guidance_mode=str(cfgs_predictor_dit.get("joint_action_guidance_mode", "cond_only")).lower(),
        joint_action_inference_noise_mode=str(
            cfgs_predictor_dit.get("joint_action_inference_noise_mode", "shared")
        ).lower(),
        joint_video_final_noise=float(cfgs_predictor_dit.get("joint_video_final_noise", 0.0)),
    )
    _validate_prefix_probability(
        "predictor_dit.masked_train_full_prefix_prob",
        predictor_dit.masked_train_full_prefix_prob,
    )
    if predictor_dit.masked_train_min_prefix_steps < 0:
        raise ValueError(
            "predictor_dit.masked_train_min_prefix_steps must be >= 0, "
            f"got {predictor_dit.masked_train_min_prefix_steps}"
        )
    if (
        predictor_dit.masked_inpainting_enabled
        and predictor_dit.masked_train_full_prefix_prob < 1.0
        and predictor_dit.masked_train_min_prefix_steps == 0
    ):
        raise ValueError(
            "masked Latent-DiT predictor supervision cannot include h=0; "
            "set predictor_dit.masked_train_min_prefix_steps >= 1"
        )
    if predictor_dit.masked_train_max_non_full_prefix_steps is not None:
        if predictor_dit.masked_train_max_non_full_prefix_steps < 0:
            raise ValueError(
                "predictor_dit.masked_train_max_non_full_prefix_steps must be >= 0 when set, "
                f"got {predictor_dit.masked_train_max_non_full_prefix_steps}"
            )
        if predictor_dit.masked_train_min_prefix_steps > predictor_dit.masked_train_max_non_full_prefix_steps:
            raise ValueError(
                "predictor_dit.masked_train_min_prefix_steps must be <= "
                "predictor_dit.masked_train_max_non_full_prefix_steps"
            )

    # 解析 EMA 配置
    cfgs_ema = args.get("ema", {})
    ema = EMAConfig(
        ema_start=cfgs_ema.get("ema_start", 0.996),
        ema_end=cfgs_ema.get("ema_end", 0.999),
    )

    # 解析 segmentation 配置
    cfgs_seg = args.get("segmentation", {})
    segmentation = SegmentationConfig(
        use_segmentation=cfgs_seg.get("use_segmentation", True),
        seg_loss_weight=cfgs_seg.get("seg_loss_weight", 1.0),
        seg_data_root=cfgs_seg.get("seg_data_root", "/path/to/segmentation/annotations"),
        num_classes=cfgs_seg.get("num_classes", 2),
        loss_seg_weight=cfgs_seg.get("loss_seg_weight", 2.0),
        loss_dice_weight=cfgs_seg.get("loss_dice_weight", 5.0),
    )

    # 解析 planner 配置
    cfgs_planner = args.get("planner", {})
    observed_token_mode = normalize_planner_observed_token_mode(
        cfgs_planner.get("observed_token_mode"),
        cfgs_planner.get("use_observed_tokens") if "use_observed_tokens" in cfgs_planner else None,
    )
    planner = PlannerConfig(
        use_planner=cfgs_planner.get("use_planner", True),
        tf_d_model=cfgs_planner.get("tf_d_model", 256),
        tf_d_ffn=cfgs_planner.get("tf_d_ffn", 1024),
        tf_num_layers=cfgs_planner.get("tf_num_layers", 3),
        tf_num_head=cfgs_planner.get("tf_num_head", 8),
        tf_dropout=cfgs_planner.get("tf_dropout", 0.0),
        planner_loss_weight=cfgs_planner.get("planner_loss_weight", 1.0),
        use_spatial_tokens=cfgs_planner.get("use_spatial_tokens", False),
        use_temporal=cfgs_planner.get("use_temporal", False),
        temporal_alignment=cfgs_planner.get("temporal_alignment", True),
        z_ar_mode=cfgs_planner.get("z_ar_mode", "full"),
        planner_input_source=cfgs_planner.get("planner_input_source", "z_ar"),
        num_modes=cfgs_planner.get("num_modes", 6),
        num_context_frames=cfgs_planner.get("num_context_frames", 1),
        conf_loss_weight=cfgs_planner.get("conf_loss_weight", 1.0),
        reg_loss_weight=cfgs_planner.get("reg_loss_weight", 1.0),
        horizon_reg_loss_seconds=cfgs_planner.get("horizon_reg_loss_seconds", []),
        horizon_reg_loss_weights=cfgs_planner.get("horizon_reg_loss_weights", []),
        horizon_reg_loss_normalize=cfgs_planner.get("horizon_reg_loss_normalize", True),
        states_mode=cfgs_planner.get("states_mode", cfgs_planner.get("status_mode", "first")),
        use_status_for_planner=cfgs_planner.get("use_status_for_planner", cfgs_planner.get("use_status", True)),
        use_states_for_planner=cfgs_planner.get("use_states_for_planner", True),
        use_z_context=cfgs_planner.get("use_z_context", False),
        observed_token_mode=observed_token_mode,
        use_observed_tokens=observed_token_mode != PLANNER_OBSERVED_TOKEN_NONE,
        use_action_history_for_planner=cfgs_planner.get("use_action_history_for_planner", False),
        action_history_dim=cfgs_planner.get("action_history_dim", 3),
        latent_dit_action_source=cfgs_planner.get("latent_dit_action_source", "planner"),
        policy_output_source=str(cfgs_planner.get("policy_output_source", "planner")).lower(),
        enable_rl_actor_critic=cfgs_planner.get("enable_rl_actor_critic", False),
        rl_action_dim=cfgs_planner.get("rl_action_dim", 2),
        wta_loss_version=cfgs_planner.get("wta_loss_version", "v1"),
        wta_temperature=cfgs_planner.get("wta_temperature", 1.0),
        wta_alpha=cfgs_planner.get("wta_alpha", 5.0),
        wta_global_batch_norm=cfgs_planner.get("wta_global_batch_norm", True),
        cover_loss_weight=cfgs_planner.get("cover_loss_weight", 0.1),
        awta_init_temperature=cfgs_planner.get("awta_init_temperature", 8.0),
        awta_exp_base=cfgs_planner.get("awta_exp_base", 0.984),
        awta_min_temperature=cfgs_planner.get("awta_min_temperature", 0.1),
        planner_type=cfgs_planner.get("planner_type", "transformer"),
        refinement_core_type=cfgs_planner.get("refinement_core_type"),
        diff_hidden_dim=cfgs_planner.get("diff_hidden_dim", 256),
        diff_num_layers=cfgs_planner.get("diff_num_layers", 4),
        diff_num_heads=cfgs_planner.get("diff_num_heads", 8),
        diff_dropout=cfgs_planner.get("diff_dropout", 0.0),
        diff_mlp_ratio=cfgs_planner.get("diff_mlp_ratio", 4.0),
        diff_sde_beta_min=cfgs_planner.get("diff_sde_beta_min", 0.1),
        diff_sde_beta_max=cfgs_planner.get("diff_sde_beta_max", 20.0),
        diff_inference_steps=cfgs_planner.get("diff_inference_steps", 2),
        diff_num_samples=cfgs_planner.get("diff_num_samples", 6),
        diff_traj_dim=cfgs_planner.get("diff_traj_dim", 6),
        diff_dt=cfgs_planner.get("diff_dt", 0.2),
        diff_trajectory_token_mode=cfgs_planner.get("diff_trajectory_token_mode", "single_token"),
        diff_adaln_version=cfgs_planner.get("diff_adaln_version", "legacy"),
        diff_use_last_frame_only=cfgs_planner.get("diff_use_last_frame_only", True),
        diff_interleave_predictor_sampling=cfgs_planner.get("diff_interleave_predictor_sampling", False),
        diff_train_prefix_conditioning=cfgs_planner.get("diff_train_prefix_conditioning", False),
        diff_train_min_prefix_frames=cfgs_planner.get("diff_train_min_prefix_frames", 1),
        diff_train_full_prefix_prob=cfgs_planner.get("diff_train_full_prefix_prob", 0.25),
        diff_train_max_non_full_prefix_frames=cfgs_planner.get(
            "diff_train_max_non_full_prefix_frames",
            None,
        ),
        diff_num_modes=cfgs_planner.get("diff_num_modes", 1),
        diff_independent_modes=cfgs_planner.get("diff_independent_modes", False),
        diff_mode_token_expansion=cfgs_planner.get("diff_mode_token_expansion", False),
        diff_use_anchor_frame=cfgs_planner.get("diff_use_anchor_frame", False),
        diff_init_traj_strategy=cfgs_planner.get("diff_init_traj_strategy", "gaussian"),
        diff_init_traj_noise_scale=cfgs_planner.get("diff_init_traj_noise_scale", 1.0),
        diff_init_traj_yaw_span_deg=cfgs_planner.get("diff_init_traj_yaw_span_deg", 30.0),
        diff_init_traj_speed_scale_span=cfgs_planner.get("diff_init_traj_speed_scale_span", 0.2),
        diff_cls_loss_weight=cfgs_planner.get("diff_cls_loss_weight", 1.0),
        diff_reg_loss_weight=cfgs_planner.get("diff_reg_loss_weight", 1.0),
        diff_vel_loss_weight=cfgs_planner.get("diff_vel_loss_weight", 0.5),
        diff_yaw_loss_weight=cfgs_planner.get("diff_yaw_loss_weight", 0.5),
        diff_generation_framework=cfgs_planner.get("diff_generation_framework", "vp_diffusion"),
        diff_flow_matching_variant=cfgs_planner.get("diff_flow_matching_variant", "rectified"),
        diff_flow_shift=cfgs_planner.get("diff_flow_shift", 1.0),
        diff_flow_sampler=cfgs_planner.get("diff_flow_sampler", "euler"),
        diff_flow_timestep_sampling=cfgs_planner.get("diff_flow_timestep_sampling", "logit_normal"),
        diff_conf_temperature=cfgs_planner.get("diff_conf_temperature", 1.5),
        diff_cls_th=cfgs_planner.get("diff_cls_th", 2.0),
        diff_cls_ignore=cfgs_planner.get("diff_cls_ignore", 0.2),
        split_status_embedding=cfgs_planner.get("split_status_embedding", True),
        use_drive_command=cfgs_planner.get("use_drive_command"),
        status_dim=cfgs_planner.get("status_dim", 0),
    )
    _validate_prefix_probability(
        "planner.diff_train_full_prefix_prob",
        planner.diff_train_full_prefix_prob,
    )
    if int(planner.diff_train_min_prefix_frames) < 0:
        raise ValueError(
            "planner.diff_train_min_prefix_frames must be >= 0, " f"got {planner.diff_train_min_prefix_frames}"
        )
    if planner.diff_train_max_non_full_prefix_frames is not None:
        planner.diff_train_max_non_full_prefix_frames = int(planner.diff_train_max_non_full_prefix_frames)
        if planner.diff_train_max_non_full_prefix_frames < 0:
            raise ValueError(
                "planner.diff_train_max_non_full_prefix_frames must be >= 0 when set, "
                f"got {planner.diff_train_max_non_full_prefix_frames}"
            )
        if planner.diff_train_min_prefix_frames > planner.diff_train_max_non_full_prefix_frames:
            raise ValueError(
                "planner.diff_train_min_prefix_frames must be <= " "planner.diff_train_max_non_full_prefix_frames"
            )
    planner_prefix_can_sample_h0 = (
        bool(planner.diff_train_prefix_conditioning)
        and float(planner.diff_train_full_prefix_prob) < 1.0
        and int(planner.diff_train_min_prefix_frames) == 0
    )
    if planner_prefix_can_sample_h0 and not (
        planner.use_z_context or planner.use_observed_tokens or planner.use_action_history_for_planner
    ):
        raise ValueError(
            "planner internal h=0 prefix conditioning requires at least one non-future context token source: "
            "use_z_context, observed tokens, or action_history"
        )

    # 解析独立 proposal 配置
    cfgs_proposal = args.get("proposal", {}) or {}
    proposal_vjepa_resolution = _parse_hw_resolution(
        cfgs_proposal.get("vjepa_resolution", model.vjepa_resolution),
        "proposal.vjepa_resolution",
    )
    proposal = ProposalConfig(
        enabled=bool(cfgs_proposal.get("enabled", False)),
        provider_type=cfgs_proposal.get("provider_type", cfgs_planner.get("planner_type", "transformer")),
        checkpoint=cfgs_proposal.get("checkpoint"),
        use_separate_encoder=bool(cfgs_proposal.get("use_separate_encoder", False)),
        encoder_backbone=cfgs_proposal.get("encoder_backbone"),
        encoder_model_name=cfgs_proposal.get("encoder_model_name", model.model_name),
        encoder_checkpoint=cfgs_proposal.get("encoder_checkpoint"),
        encoder_checkpoint_key=cfgs_proposal.get("encoder_checkpoint_key", "encoder"),
        encoder_freeze=bool(cfgs_proposal.get("encoder_freeze", True)),
        vjepa_resolution=proposal_vjepa_resolution,
        vjepa_crop_top_bottom=cfgs_proposal.get("vjepa_crop_top_bottom", model.vjepa_crop_top_bottom),
        vjepa_num_frames=cfgs_proposal.get("vjepa_num_frames", model.vjepa_num_frames),
        vjepa_checkpoint_key=cfgs_proposal.get("vjepa_checkpoint_key"),
        vjepa_use_grid_mask=cfgs_proposal.get("vjepa_use_grid_mask", model.vjepa_use_grid_mask),
        vjepa_use_causal_attention=cfgs_proposal.get(
            "vjepa_use_causal_attention",
            model.vjepa_use_causal_attention,
        ),
        freeze=bool(cfgs_proposal.get("freeze", True)),
        num_modes=int(cfgs_proposal.get("num_modes", cfgs_planner.get("num_modes", 6))),
        provider_num_modes=(
            int(cfgs_proposal["provider_num_modes"]) if cfgs_proposal.get("provider_num_modes") is not None else None
        ),
        log_metrics_only=bool(cfgs_proposal.get("log_metrics_only", True)),
        use_z_context=bool(cfgs_proposal.get("use_z_context", cfgs_planner.get("use_z_context", True))),
        temporal_alignment=bool(cfgs_proposal.get("temporal_alignment", cfgs_planner.get("temporal_alignment", True))),
        runtime_normalize_reps=cfgs_proposal.get("runtime_normalize_reps"),
        use_token_ae=cfgs_proposal.get("use_token_ae"),
        history_temperature=float(cfgs_proposal.get("history_temperature", 1.0)),
        hidden_dim=int(cfgs_proposal.get("hidden_dim", cfgs_planner.get("tf_d_model", 256))),
        manual_mode_expansion=bool(cfgs_proposal.get("manual_mode_expansion", False)),
        manual_lateral_offsets=cfgs_proposal.get("manual_lateral_offsets"),
        manual_yaw_offsets_deg=cfgs_proposal.get("manual_yaw_offsets_deg"),
        manual_speed_scales=cfgs_proposal.get("manual_speed_scales"),
        manual_ramp_power=float(cfgs_proposal.get("manual_ramp_power", 1.5)),
        manual_confidence_temperature=float(cfgs_proposal.get("manual_confidence_temperature", 1.0)),
    )

    cfgs_counterfactual_supervision = args.get("counterfactual_supervision", {}) or {}
    enabled_value = cfgs_counterfactual_supervision.get("enabled", False)
    if not isinstance(enabled_value, bool):
        raise ValueError("counterfactual_supervision.enabled must be a bool, " f"got {enabled_value!r}")
    raw_hazard_negative_pairing = cfgs_counterfactual_supervision.get("hazard_negative_pairing", {}) or {}
    if not isinstance(raw_hazard_negative_pairing, dict):
        raise ValueError("counterfactual_supervision.hazard_negative_pairing must be a mapping")
    pairing_enabled = raw_hazard_negative_pairing.get("enabled", False)
    if type(pairing_enabled) is not bool:
        raise ValueError("counterfactual_supervision.hazard_negative_pairing.enabled must be a bool")
    hazard_negative_pairing = CounterfactualHazardNegativePairingConfig(
        enabled=pairing_enabled,
        hazard_domain=raw_hazard_negative_pairing.get("hazard_domain", "counterfactual"),
        hazard_accident_types=list(
            raw_hazard_negative_pairing.get("hazard_accident_types", ["自车行为引起", "非自车行为引起"]) or []
        ),
        safe_negative_domain=raw_hazard_negative_pairing.get("safe_negative_domain", "real"),
        safe_negative_semantics=raw_hazard_negative_pairing.get(
            "safe_negative_semantics",
            "factual_real_sample",
        ),
        pairing_key=list(raw_hazard_negative_pairing.get("pairing_key", ["base_scene_id", "window_start_pos"]) or []),
        same_source_identity_required=raw_hazard_negative_pairing.get("same_source_identity_required", True),
        cross_scene_pairing_forbidden=raw_hazard_negative_pairing.get("cross_scene_pairing_forbidden", True),
        unmatched_pair_is_failure=raw_hazard_negative_pairing.get("unmatched_pair_is_failure", True),
        fallback_pairing_forbidden=raw_hazard_negative_pairing.get("fallback_pairing_forbidden", True),
        relabel_non_ego_hazard_as_safe_forbidden=raw_hazard_negative_pairing.get(
            "relabel_non_ego_hazard_as_safe_forbidden",
            True,
        ),
    )
    counterfactual_supervision = CounterfactualSupervisionConfig(
        enabled=enabled_value,
        protocol_version=str(cfgs_counterfactual_supervision.get("protocol_version", "cf_supervision_v2")),
        imitation_policy=str(cfgs_counterfactual_supervision.get("imitation_policy", "real_and_cf_safe")),
        world_model_policy=str(cfgs_counterfactual_supervision.get("world_model_policy", "all_valid")),
        value_policy=str(cfgs_counterfactual_supervision.get("value_policy", "all_valid")),
        ego_hazard_types=list(
            cfgs_counterfactual_supervision.get("ego_hazard_types", ["自车行为引起", "非自车行为引起"]) or []
        ),
        cf_sample_accident_type_allowlist=list(
            cfgs_counterfactual_supervision.get(
                "cf_sample_accident_type_allowlist",
                ["自车行为引起", "非自车行为引起"],
            )
            or []
        ),
        cf_sample_filter_mode=str(cfgs_counterfactual_supervision.get("cf_sample_filter_mode", "strict_allowlist")),
        retained_accident_types=list(
            cfgs_counterfactual_supervision.get(
                "retained_accident_types",
                ["自车行为引起", "非自车行为引起"],
            )
            or []
        ),
        retained_accident_types_are_all_hazards=cfgs_counterfactual_supervision.get(
            "retained_accident_types_are_all_hazards",
            True,
        ),
        hazard_negative_pairing=hazard_negative_pairing,
        hazard_target_mode=str(cfgs_counterfactual_supervision.get("hazard_target_mode", "episode_ranking")),
    )
    expected_counterfactual_values = {
        "protocol_version": "cf_supervision_v2",
        "imitation_policy": "real_and_cf_safe",
        "world_model_policy": "all_valid",
        "value_policy": "all_valid",
        "cf_sample_filter_mode": "strict_allowlist",
        "hazard_target_mode": "episode_ranking",
    }
    for field_name, expected_value in expected_counterfactual_values.items():
        actual_value = getattr(counterfactual_supervision, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"counterfactual_supervision.{field_name} must be {expected_value!r}, " f"got {actual_value!r}"
            )
    expected_ego_hazard_types = ["自车行为引起", "非自车行为引起"]
    for field_name in (
        "ego_hazard_types",
        "cf_sample_accident_type_allowlist",
        "retained_accident_types",
    ):
        actual = getattr(counterfactual_supervision, field_name)
        if type(actual) is not list or actual != expected_ego_hazard_types:
            raise ValueError(
                f"counterfactual_supervision.{field_name} must be exactly "
                f"{expected_ego_hazard_types!r}, got {actual!r}"
            )
    if counterfactual_supervision.retained_accident_types_are_all_hazards is not True:
        raise ValueError("counterfactual_supervision.retained_accident_types_are_all_hazards must be exactly true")
    pairing_expected = {
        "hazard_domain": "counterfactual",
        "hazard_accident_types": expected_ego_hazard_types,
        "safe_negative_domain": "real",
        "safe_negative_semantics": "factual_real_sample",
        "pairing_key": ["base_scene_id", "window_start_pos"],
        "same_source_identity_required": True,
        "cross_scene_pairing_forbidden": True,
        "unmatched_pair_is_failure": True,
        "fallback_pairing_forbidden": True,
        "relabel_non_ego_hazard_as_safe_forbidden": True,
    }
    for field_name, expected_value in pairing_expected.items():
        actual = getattr(counterfactual_supervision.hazard_negative_pairing, field_name)
        if actual != expected_value or type(actual) is not type(expected_value):
            raise ValueError(
                f"counterfactual_supervision.hazard_negative_pairing.{field_name} must be exactly "
                f"{expected_value!r}, got {actual!r}"
            )

    # 解析 data 配置
    cfgs_data = args.get("data", {})
    cfgs_navsim = cfgs_data.get("navsim")
    cfgs_bench2drive = cfgs_data.get("bench2drive")
    cfgs_mongo_raw = cfgs_data.get("mongo_raw")

    navsim = None
    if isinstance(cfgs_navsim, dict):
        front_only = cfgs_navsim.get("front_only", True)
        default_camera_name = "CAM_F0" if front_only else "CAM_F0"
        camera_name = cfgs_navsim.get("camera_name", default_camera_name)
        camera_names = list(cfgs_navsim.get("camera_names", [camera_name]) or [camera_name])

        def _parse_scene_filter_yaml(yaml_key: str, stride_key: str, truncate_key: str) -> Optional[str]:
            """官方 token 锚定的 yaml 路径解析：文件必须存在，且与 stride/截断键互斥。

            stride 键允许显式写 1（与默认等价、无歧义）；截断键只允许缺省。
            """
            yaml_path = cfgs_navsim.get(yaml_key)
            if yaml_path is None:
                return None
            if not isinstance(yaml_path, str) or not yaml_path:
                raise ValueError(f"data.navsim.{yaml_key} must be a non-empty path, got {yaml_path!r}")
            if not Path(yaml_path).is_file():
                raise ValueError(f"data.navsim.{yaml_key} does not exist: {yaml_path}")
            stride_value = cfgs_navsim.get(stride_key)
            if stride_value not in (None, 1):
                raise ValueError(
                    f"data.navsim.{stride_key}={stride_value!r} is ignored under official token "
                    f"anchoring; remove it when data.navsim.{yaml_key} is set"
                )
            truncate_value = cfgs_navsim.get(truncate_key)
            if truncate_value is not None:
                raise ValueError(
                    f"data.navsim.{truncate_key}={truncate_value!r} cannot be combined with "
                    f"data.navsim.{yaml_key}; official token anchoring requires full log coverage "
                    "(use a truncated scene-filter yaml for smoke runs)"
                )
            return yaml_path

        scene_filter_yaml = _parse_scene_filter_yaml("scene_filter_yaml", "window_stride", "max_scenes")
        val_scene_filter_yaml = _parse_scene_filter_yaml(
            "val_scene_filter_yaml", "val_window_stride", "max_val_scenes"
        )

        navsim_max_agents = require_positive_navsim_max_agents(
            cfgs_navsim.get("max_agents", NAVSIM_DEFAULT_MAX_AGENTS),
            field_name="data.navsim.max_agents",
        )
        train_roots = []
        raw_train_roots = cfgs_navsim.get("train_roots", []) or []
        if not isinstance(raw_train_roots, list):
            raise ValueError("data.navsim.train_roots must be a list of root mappings")
        for root_index, raw_root in enumerate(raw_train_roots):
            if not isinstance(raw_root, dict):
                raise ValueError(
                    f"data.navsim.train_roots[{root_index}] must be a mapping, " f"got {type(raw_root).__name__}"
                )
            root = dict(raw_root)
            domain = root.get("domain")
            if domain not in {"real", "counterfactual"}:
                raise ValueError(
                    f"data.navsim.train_roots[{root_index}].domain must be 'real' or "
                    f"'counterfactual', got {domain!r}"
                )
            annotation_selection = root.get("annotation_selection", "all_valid")
            if annotation_selection not in {
                "all_valid",
                "safe_only",
                FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION,
            }:
                raise ValueError(
                    f"data.navsim.train_roots[{root_index}].annotation_selection must be "
                    f"'all_valid', 'safe_only', or {FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}, "
                    f"got {annotation_selection!r}"
                )
            root["annotation_selection"] = annotation_selection
            require_trajectory_match = root.get("annotations_require_trajectory_match", False)
            if "annotations_require_trajectory_match" in root and not isinstance(require_trajectory_match, bool):
                raise ValueError(
                    f"data.navsim.train_roots[{root_index}].annotations_require_trajectory_match "
                    f"must be boolean, got {require_trajectory_match!r}"
                )
            if require_trajectory_match and not root.get("annotations_path"):
                raise ValueError(
                    f"data.navsim.train_roots[{root_index}].annotations_require_trajectory_match=true "
                    "requires annotations_path"
                )
            if annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
                expected_allowlist = list(FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST)
                if domain != "counterfactual":
                    raise ValueError(
                        f"data.navsim.train_roots[{root_index}] Formal-v2 accident allowlist is counterfactual-only"
                    )
                if (
                    type(root.get("annotations_accident_type_allowlist")) is not list
                    or root.get("annotations_accident_type_allowlist") != expected_allowlist
                ):
                    raise ValueError(
                        f"data.navsim.train_roots[{root_index}].annotations_accident_type_allowlist must be "
                        f"exactly {expected_allowlist!r}"
                    )
                if require_trajectory_match is not True:
                    raise ValueError(
                        f"data.navsim.train_roots[{root_index}].annotations_require_trajectory_match must be true "
                        "for the Formal-v2 accident allowlist"
                    )
                if root.get("annotations_drop_distorted") is not True:
                    raise ValueError(
                        f"data.navsim.train_roots[{root_index}].annotations_drop_distorted must be true "
                        "for the Formal-v2 accident allowlist"
                    )
            elif "annotations_accident_type_allowlist" in root:
                raise ValueError(
                    f"data.navsim.train_roots[{root_index}].annotations_accident_type_allowlist requires "
                    f"annotation_selection={FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}"
                )
            if "max_agents" in root:
                require_positive_navsim_max_agents(
                    root["max_agents"],
                    field_name=f"data.navsim.train_roots[{root_index}].max_agents",
                )
            if domain == "counterfactual" and counterfactual_supervision.enabled:
                annotations_path = root.get("annotations_path")
                if not isinstance(annotations_path, str) or not annotations_path:
                    raise ValueError(
                        f"data.navsim.train_roots[{root_index}].annotations_path must be a "
                        "non-empty path for cf_supervision_v2"
                    )
                if root.get("annotations_drop_distorted") is not True:
                    raise ValueError(
                        f"data.navsim.train_roots[{root_index}].annotations_drop_distorted must "
                        "be true for cf_supervision_v2"
                    )
            train_roots.append(root)

        val_roots = []
        raw_val_roots = cfgs_navsim.get("val_roots", []) or []
        if not isinstance(raw_val_roots, list):
            raise ValueError("data.navsim.val_roots must be a list of root mappings")
        if raw_val_roots:
            direct_validation_fields = (
                "val_data_path",
                "val_sensor_blobs_path",
                "val_domain",
                "val_annotations_path",
                "val_annotations_drop_distorted",
                "val_annotation_selection",
                "val_pose_overlay_path",
                "val_tail_seconds",
                "val_scene_filter_yaml",
            )
            configured_direct_fields = [field for field in direct_validation_fields if field in cfgs_navsim]
            if configured_direct_fields:
                raise ValueError(
                    "data.navsim.val_roots cannot be combined with direct validation fields; "
                    f"remove {configured_direct_fields}"
                )
        seen_val_root_names = set()
        seen_val_dataset_ids = set()
        for root_index, raw_root in enumerate(raw_val_roots):
            if not isinstance(raw_root, dict):
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}] must be a mapping, " f"got {type(raw_root).__name__}"
                )
            root = dict(raw_root)
            name = root.get("name")
            if not isinstance(name, str) or not name or name in seen_val_root_names:
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}].name must be unique and non-empty, got {name!r}"
                )
            seen_val_root_names.add(name)
            dataset_id = root.get("dataset_id")
            if validation_suite.enabled:
                if not isinstance(dataset_id, str) or not dataset_id or dataset_id in seen_val_dataset_ids:
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}].dataset_id must be unique and non-empty "
                        f"for validation_suite, got {dataset_id!r}"
                    )
                seen_val_dataset_ids.add(dataset_id)
            domain = root.get("domain")
            if domain not in {"real", "counterfactual"}:
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}].domain must be 'real' or "
                    f"'counterfactual', got {domain!r}"
                )
            for path_field in ("data_path", "sensor_blobs_path"):
                if not isinstance(root.get(path_field), str) or not root[path_field]:
                    raise ValueError(f"data.navsim.val_roots[{root_index}].{path_field} must be a non-empty path")
            annotation_selection = root.get("annotation_selection", "all_valid")
            if annotation_selection not in {
                "all_valid",
                "safe_only",
                FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION,
            }:
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}].annotation_selection must be "
                    f"'all_valid', 'safe_only', or {FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}, "
                    f"got {annotation_selection!r}"
                )
            if validation_suite.enabled and annotation_selection not in {
                "all_valid",
                FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION,
            }:
                raise ValueError(
                    "validation_suite requires every data.navsim.val_roots entry to use "
                    "annotation_selection='all_valid'"
                )
            root["annotation_selection"] = annotation_selection
            require_trajectory_match = root.get("annotations_require_trajectory_match", False)
            if "annotations_require_trajectory_match" in root and not isinstance(require_trajectory_match, bool):
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}].annotations_require_trajectory_match "
                    f"must be boolean, got {require_trajectory_match!r}"
                )
            if require_trajectory_match and not root.get("annotations_path"):
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}].annotations_require_trajectory_match=true "
                    "requires annotations_path"
                )
            if annotation_selection == FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION:
                expected_allowlist = list(FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST)
                if domain != "counterfactual":
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}] Formal-v2 accident allowlist is counterfactual-only"
                    )
                if (
                    type(root.get("annotations_accident_type_allowlist")) is not list
                    or root.get("annotations_accident_type_allowlist") != expected_allowlist
                ):
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}].annotations_accident_type_allowlist must be "
                        f"exactly {expected_allowlist!r}"
                    )
                if require_trajectory_match is not True:
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}].annotations_require_trajectory_match must be true "
                        "for the Formal-v2 accident allowlist"
                    )
                if root.get("annotations_drop_distorted") is not True:
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}].annotations_drop_distorted must be true "
                        "for the Formal-v2 accident allowlist"
                    )
            elif "annotations_accident_type_allowlist" in root:
                raise ValueError(
                    f"data.navsim.val_roots[{root_index}].annotations_accident_type_allowlist requires "
                    f"annotation_selection={FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION!r}"
                )
            requires_counterfactual_cohorts = validation_suite.enabled or counterfactual_supervision.enabled
            if domain == "counterfactual" and requires_counterfactual_cohorts:
                contract_name = "validation_suite" if validation_suite.enabled else "cf_supervision_v2"
                if not isinstance(root.get("annotations_path"), str) or not root["annotations_path"]:
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}].annotations_path must be a non-empty path "
                        f"for {contract_name}"
                    )
                if root.get("annotations_drop_distorted") is not True:
                    raise ValueError(
                        f"data.navsim.val_roots[{root_index}].annotations_drop_distorted must be true "
                        f"for {contract_name}"
                    )
            val_roots.append(root)
        navsim_max_agents = resolve_navsim_root_max_agents(
            [*train_roots, *val_roots],
            default_max_agents=navsim_max_agents,
            field_name="data.navsim mixed roots",
        )
        if validation_suite.enabled:
            val_domains = {root["domain"] for root in val_roots}
            if val_domains != {"real", "counterfactual"}:
                raise ValueError(
                    "validation_suite requires data.navsim.val_roots with both real and counterfactual "
                    f"domains, got {sorted(val_domains)}"
                )

        val_domain = cfgs_navsim.get("val_domain")
        if val_domain is not None and val_domain not in {"real", "counterfactual"}:
            raise ValueError("data.navsim.val_domain must be 'real' or 'counterfactual', " f"got {val_domain!r}")
        has_direct_val_path = bool(cfgs_navsim.get("val_data_path") or cfgs_navsim.get("val_sensor_blobs_path"))
        if counterfactual_supervision.enabled and has_direct_val_path and val_domain is None:
            raise ValueError("data.navsim.val_domain is required for direct validation under cf_supervision_v2")
        val_annotation_selection = cfgs_navsim.get("val_annotation_selection", "all_valid")
        if val_annotation_selection not in {"all_valid", "safe_only"}:
            raise ValueError(
                "data.navsim.val_annotation_selection must be 'all_valid' or 'safe_only', "
                f"got {val_annotation_selection!r}"
            )
        if counterfactual_supervision.enabled and val_domain == "counterfactual":
            val_annotations_path = cfgs_navsim.get("val_annotations_path")
            if not isinstance(val_annotations_path, str) or not val_annotations_path:
                raise ValueError(
                    "data.navsim.val_annotations_path must be a non-empty path for "
                    "counterfactual validation under cf_supervision_v2"
                )
            if cfgs_navsim.get("val_annotations_drop_distorted") is not True:
                raise ValueError(
                    "data.navsim.val_annotations_drop_distorted must be true for "
                    "counterfactual validation under cf_supervision_v2"
                )

        navsim = NavSimConfig(
            enabled=cfgs_navsim.get("enabled", True),
            data_path=cfgs_navsim.get("data_path", ""),
            sensor_blobs_path=cfgs_navsim.get("sensor_blobs_path", ""),
            train_roots=train_roots,
            balance_train_roots=bool(cfgs_navsim.get("balance_train_roots", False)),
            val_roots=val_roots,
            val_data_path=cfgs_navsim.get("val_data_path"),
            val_sensor_blobs_path=cfgs_navsim.get("val_sensor_blobs_path"),
            val_domain=val_domain,
            val_annotation_selection=val_annotation_selection,
            camera_name=camera_name,
            camera_names=camera_names,
            num_history_frames=cfgs_navsim.get("num_history_frames"),
            image_require_policy=cfgs_navsim.get("image_require_policy", NAVSIM_IMAGE_REQUIRE_AUTO),
            max_scenes=cfgs_navsim.get("max_scenes"),
            max_val_scenes=cfgs_navsim.get("max_val_scenes"),
            index_cache=cfgs_navsim.get("index_cache", True),
            window_stride=cfgs_navsim.get("window_stride", 1),
            val_window_stride=cfgs_navsim.get("val_window_stride", None),
            tail_seconds=cfgs_navsim.get("tail_seconds", None),
            val_tail_seconds=cfgs_navsim.get("val_tail_seconds", None),
            counterfactual_tail_seconds=cfgs_navsim.get("counterfactual_tail_seconds", 5.0),
            max_frame_gap=cfgs_navsim.get("max_frame_gap", 3),
            max_agents=navsim_max_agents,
            load_agent_annotations=bool(cfgs_navsim.get("load_agent_annotations", True)),
            scene_filter_yaml=scene_filter_yaml,
            val_scene_filter_yaml=val_scene_filter_yaml,
            pose_overlay_path=cfgs_navsim.get("pose_overlay_path"),
            val_pose_overlay_path=cfgs_navsim.get("val_pose_overlay_path"),
            pose_overlay_coord_frame=cfgs_navsim.get("pose_overlay_coord_frame", "opencv_first_frame"),
            pose_overlay_required=bool(cfgs_navsim.get("pose_overlay_required", False)),
            val_annotations_path=cfgs_navsim.get("val_annotations_path"),
            val_annotations_drop_distorted=cfgs_navsim.get("val_annotations_drop_distorted"),
        )

    bench2drive = None
    if isinstance(cfgs_bench2drive, dict):
        b2d_image_require_policy = str(
            cfgs_bench2drive.get("image_require_policy", NAVSIM_IMAGE_REQUIRE_AUTO) or NAVSIM_IMAGE_REQUIRE_AUTO
        ).lower()
        if b2d_image_require_policy not in NAVSIM_IMAGE_REQUIRE_POLICIES:
            raise ValueError(
                "data.bench2drive.image_require_policy must be one of "
                f"{sorted(NAVSIM_IMAGE_REQUIRE_POLICIES)}, got {b2d_image_require_policy!r}"
            )
        bench2drive = Bench2DriveConfig(
            enabled=cfgs_bench2drive.get("enabled", True),
            data_root=cfgs_bench2drive.get("data_root"),
            ann_file=cfgs_bench2drive.get("ann_file"),
            val_ann_file=cfgs_bench2drive.get("val_ann_file"),
            camera_name=cfgs_bench2drive.get("camera_name", "CAM_FRONT"),
            base_fps=int(cfgs_bench2drive.get("base_fps", 10)),
            image_require_policy=b2d_image_require_policy,
            max_scenes=cfgs_bench2drive.get("max_scenes"),
            max_val_scenes=cfgs_bench2drive.get("max_val_scenes"),
            window_stride=int(cfgs_bench2drive.get("window_stride", 1)),
            val_window_stride=cfgs_bench2drive.get("val_window_stride"),
            max_frame_gap=int(cfgs_bench2drive.get("max_frame_gap", 6)),
            max_agents=int(cfgs_bench2drive.get("max_agents", 50)),
            load_agent_annotations=bool(cfgs_bench2drive.get("load_agent_annotations", True)),
            command_dim=int(cfgs_bench2drive.get("command_dim", 6)),
            index_cache=bool(cfgs_bench2drive.get("index_cache", True)),
            index_cache_dir=cfgs_bench2drive.get("index_cache_dir"),
            verify_image_exists=bool(cfgs_bench2drive.get("verify_image_exists", False)),
            max_load_retries=int(cfgs_bench2drive.get("max_load_retries", 5)),
            index_cache_wait_seconds=int(cfgs_bench2drive.get("index_cache_wait_seconds", 300)),
        )

    mongo_raw = None
    if isinstance(cfgs_mongo_raw, dict):
        mongo_raw = MongoRawConfig(
            enabled=cfgs_mongo_raw.get("enabled", True),
            mongo_uri=cfgs_mongo_raw.get("mongo_uri"),
            mongo_uri_env=cfgs_mongo_raw.get("mongo_uri_env"),
            database=cfgs_mongo_raw.get("database", "e2e-data-platform-prod"),
            collection=cfgs_mongo_raw.get("collection", "clip"),
            vehicle_type=cfgs_mongo_raw.get("vehicle_type"),
            vehicle_types=list(cfgs_mongo_raw.get("vehicle_types", []) or []),
            require_latest_available_revision=cfgs_mongo_raw.get("require_latest_available_revision", True),
            query_filter=cfgs_mongo_raw.get("query_filter", {}) or {},
            start_index=cfgs_mongo_raw.get("start_index", 0),
            end_index=cfgs_mongo_raw.get("end_index"),
            max_clips=cfgs_mongo_raw.get("max_clips"),
            max_val_clips=cfgs_mongo_raw.get("max_val_clips"),
            val_ratio=cfgs_mongo_raw.get("val_ratio", 0.05),
            split_seed=cfgs_mongo_raw.get("split_seed", 0),
            source_fps=cfgs_mongo_raw.get("source_fps", 10),
            base_fps=cfgs_mongo_raw.get("base_fps", 5),
            main_topic=cfgs_mongo_raw.get("main_topic", "/main/ruby/lidar_points"),
            pose_topic=cfgs_mongo_raw.get("pose_topic", "/pose/odom"),
            match_topic=cfgs_mongo_raw.get("match_topic", "/match"),
            camera_topics=cfgs_mongo_raw.get("camera_topics", []),
            default_storage_root=cfgs_mongo_raw.get("default_storage_root", "/path/to/mongo/default-storage"),
            e2e_storage_root=cfgs_mongo_raw.get("e2e_storage_root", "/path/to/mongo/e2e-storage"),
            clipdata_storage_root=cfgs_mongo_raw.get("clipdata_storage_root", "/path/to/mongo/clipdata-storage"),
            cache_size=cfgs_mongo_raw.get("cache_size", 8),
            max_retries=cfgs_mongo_raw.get("max_retries", 5),
            extra_camera_mappings=cfgs_mongo_raw.get("extra_camera_mappings", {}) or {},
            record_cache_dir=cfgs_mongo_raw.get("record_cache_dir"),
            record_cache_ttl=cfgs_mongo_raw.get("record_cache_ttl", 604800),
            blacklist_path=cfgs_mongo_raw.get("blacklist_path"),
        )

    raw_crop_size = cfgs_data.get("crop_size", 256)
    if cfgs_model.get("backbone", "vjepa2") == "dinov2_img_encoder":
        if (
            not isinstance(raw_crop_size, (list, tuple))
            or len(raw_crop_size) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in raw_crop_size)
        ):
            raise ValueError(
                "data.crop_size must be a two-element sequence of positive integers for DINOv2, "
                f"got {raw_crop_size!r}"
            )

    data = DataConfig(
        datasets=cfgs_data.get("datasets", []),
        val_datasets=cfgs_data.get("val_datasets"),
        dataset_fpcs=cfgs_data.get("dataset_fpcs", []),
        batch_size=cfgs_data.get("batch_size", 4),
        tubelet_size=cfgs_data.get("tubelet_size", 2),
        use_tubelet_repeat=cfgs_data.get("use_tubelet_repeat", True),
        fps=cfgs_data.get("fps", 5),
        crop_size=normalize_image_size(raw_crop_size),
        patch_size=cfgs_data.get("patch_size", 16),
        num_target_frames=cfgs_data.get("num_target_frames", 16),
        pin_mem=cfgs_data.get("pin_mem", True),
        num_workers=cfgs_data.get("num_workers", 8),
        persistent_workers=cfgs_data.get("persistent_workers", True),
        camera_frame=cfgs_data.get("camera_frame", False),
        camera_views=cfgs_data.get("camera_views", ["left_mp4_path"]),
        stereo_view=cfgs_data.get("stereo_view", False),
        navsim=navsim,
        bench2drive=bench2drive,
        mongo_raw=mongo_raw,
    )

    cvoi = parse_cvoi_config(args.get("cvoi", {}))
    validate_cvoi_config(cvoi, data.navsim)
    if dynamic_rollout.horizon_probabilities is not None:
        dynamic_horizon_steps = (
            int(cvoi.max_horizon) if cvoi.enabled else int(data.num_target_frames) - int(train.num_observed_frames)
        )
        if dynamic_horizon_steps <= 0:
            raise ValueError(
                "predictor_dynamic_rollout.horizon_probabilities requires a positive rollout horizon, "
                f"got H={dynamic_horizon_steps}"
            )
        dynamic_rollout.horizon_probabilities = normalize_horizon_probabilities(
            dynamic_rollout.horizon_probabilities,
            horizon_steps=dynamic_horizon_steps,
            field_name="predictor_dynamic_rollout.horizon_probabilities",
        )

    # 解析 data_aug 配置
    cfgs_data_aug = args.get("data_aug", {})
    data_aug = DataAugConfig(
        horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
        motion_shift=cfgs_data_aug.get("motion_shift", False),
        reprob=cfgs_data_aug.get("reprob", 0.0),
        auto_augment=cfgs_data_aug.get("auto_augment", False),
    )

    # 解析 token_ae 配置
    cfgs_token_ae = args.get("token_ae", {})
    token_ae = TokenAEConfig(
        enabled=cfgs_token_ae.get("enabled", False),
        num_latent_tokens=cfgs_token_ae.get("num_latent_tokens", 64),
        num_heads=cfgs_token_ae.get("num_heads", 16),
        encoder_depth=cfgs_token_ae.get("encoder_depth", 4),
        decoder_depth=cfgs_token_ae.get("decoder_depth", 4),
        mlp_ratio=cfgs_token_ae.get("mlp_ratio", 4.0),
        dropout=cfgs_token_ae.get("dropout", 0.0),
        encoder_mode=cfgs_token_ae.get("encoder_mode", "parallel"),
        loss_type=cfgs_token_ae.get("loss_type", "smooth_l1"),
        cos_loss_weight=cfgs_token_ae.get("cos_loss_weight", 0.25),
        latent_reg_weight=cfgs_token_ae.get("latent_reg_weight", 0.0),
        pos_embed_type=cfgs_token_ae.get("pos_embed_type", "sincos"),
        input_grid_size=cfgs_token_ae.get("input_grid_size"),
        latent_grid_size=cfgs_token_ae.get("latent_grid_size"),
        temporal_depth=cfgs_token_ae.get("temporal_depth", 0),
        temporal_num_heads=cfgs_token_ae.get("temporal_num_heads"),
        temporal_mlp_ratio=cfgs_token_ae.get("temporal_mlp_ratio"),
        temporal_causal=cfgs_token_ae.get("temporal_causal", True),
        temporal_mode=cfgs_token_ae.get("temporal_mode", "index"),
        temporal_pos_embed_type=cfgs_token_ae.get("temporal_pos_embed_type", "none"),
        input_frame_mode=cfgs_token_ae.get("input_frame_mode", "all_frames"),
        temporal_loss_weight=cfgs_token_ae.get("temporal_loss_weight", 0.0),
    )

    # 解析 loss 配置
    cfgs_loss = args.get("loss", {})
    method_name = str(args.get("method", args.get("training_method", args.get("predictor_method", "")))).lower()
    normalize_reps_default = (
        False if bool(cfgs_world_model.get("enabled", False)) or method_name in ("lewm", "le-wm", "le_wm") else True
    )
    loss = LossConfig(
        auto_steps=cfgs_loss.get("auto_steps"),
        loss_exp=cfgs_loss.get("loss_exp", 2.0),
        normalize_reps=cfgs_loss.get("normalize_reps", normalize_reps_default),
    )

    # 解析 RL 配置
    cfgs_rl = args.get("rl", {})
    rl = RLConfig(
        enabled=cfgs_rl.get("enabled", False),
        algo=cfgs_rl.get("algo", "ppo"),
        status_mode=cfgs_rl.get("status_mode", "current_only"),
        hugsim_repo_root=cfgs_rl.get("hugsim_repo_root"),
        scenario_path=cfgs_rl.get("scenario_path"),
        scenario_manifest=cfgs_rl.get("scenario_manifest"),
        base_path=cfgs_rl.get("base_path"),
        camera_path=cfgs_rl.get("camera_path"),
        kinematic_path=cfgs_rl.get("kinematic_path"),
        camera_name=cfgs_rl.get("camera_name", "CAM_FRONT"),
        output_subdir=cfgs_rl.get("output_subdir", "hugsim_rl"),
        eval_checkpoint=cfgs_rl.get("eval_checkpoint"),
        rollout_steps=int(cfgs_rl.get("rollout_steps", 128)),
        max_episode_steps=int(cfgs_rl.get("max_episode_steps", 400)),
        ppo_epochs=int(cfgs_rl.get("ppo_epochs", 4)),
        mini_batch_size=int(cfgs_rl.get("mini_batch_size", 32)),
        gamma=float(cfgs_rl.get("gamma", 0.99)),
        gae_lambda=float(cfgs_rl.get("gae_lambda", 0.95)),
        clip_eps=float(cfgs_rl.get("clip_eps", 0.2)),
        value_clip_eps=float(cfgs_rl.get("value_clip_eps", 0.2)),
        vf_coef=float(cfgs_rl.get("vf_coef", 0.5)),
        ent_coef=float(cfgs_rl.get("ent_coef", 0.01)),
        lr=float(cfgs_rl.get("lr", 3e-4)),
        weight_decay=float(cfgs_rl.get("weight_decay", 0.01)),
        max_grad_norm=float(cfgs_rl.get("max_grad_norm", 1.0)),
        reward_scale=float(cfgs_rl.get("reward_scale", 1.0)),
        rl_loss_weight=float(cfgs_rl.get("rl_loss_weight", 1.0)),
        supervised_loss_weight=float(cfgs_rl.get("supervised_loss_weight", 0.0)),
        supervised_warmup_epochs=int(cfgs_rl.get("supervised_warmup_epochs", 0)),
        supervised_batches_per_epoch=int(cfgs_rl.get("supervised_batches_per_epoch", 0)),
        normalize_advantage=bool(cfgs_rl.get("normalize_advantage", True)),
        deterministic_eval=bool(cfgs_rl.get("deterministic_eval", True)),
        eval_episodes=int(cfgs_rl.get("eval_episodes", 1)),
        wheel_base=float(cfgs_rl.get("wheel_base", 2.7)),
        kinematic_dt=float(cfgs_rl.get("kinematic_dt", 0.25)),
    )

    # 解析 predictor reward model 配置
    cfgs_reward = args.get("reward", {})
    reward = RewardConfig(
        enabled=bool(cfgs_reward.get("enabled", False)),
        hidden_dim=int(cfgs_reward.get("hidden_dim", 512)),
        dropout=float(cfgs_reward.get("dropout", 0.1)),
        horizon_seconds=list(cfgs_reward.get("horizon_seconds", [1, 2, 3, 4])),
        near_miss_distance=float(cfgs_reward.get("near_miss_distance", 2.0)),
        near_miss_weight=float(cfgs_reward.get("near_miss_weight", 0.75)),
        comfort_weight=float(cfgs_reward.get("comfort_weight", 0.20)),
        train_roots=list(cfgs_reward.get("train_roots", []) or []),
        val_roots=list(cfgs_reward.get("val_roots", []) or []),
        max_train_batches=cfgs_reward.get("max_train_batches"),
        max_val_batches=cfgs_reward.get("max_val_batches"),
    )

    # 解析 reward-based mode selector 配置 (方向 B / Step 1)
    cfgs_reward_selector = args.get("reward_selector", {})
    if bool(cfgs_reward_selector.get("enabled", False)):
        # fail-loud：reward 权重直接决定监督语义（winner/conf label），
        # 启用时必须显式提供，禁止静默默认（CLAUDE.md 编码约定 第 1 条）。
        _required_rs_keys = [
            "world_model_weight",
            "trajectory_error_weight",
            "comfort_weight",
            "collision_weight",
            "offroad_weight",
        ]
        _missing_rs = [k for k in _required_rs_keys if k not in cfgs_reward_selector]
        if _missing_rs:
            raise KeyError(
                "reward_selector.enabled=true 需显式提供权重 "
                f"{_missing_rs}（fail-loud：监督相关字段不允许静默默认）"
            )
        reward_selector = RewardSelectorConfig(
            enabled=True,
            world_model_weight=float(cfgs_reward_selector["world_model_weight"]),
            trajectory_error_weight=float(cfgs_reward_selector["trajectory_error_weight"]),
            comfort_weight=float(cfgs_reward_selector["comfort_weight"]),
            collision_weight=float(cfgs_reward_selector["collision_weight"]),
            offroad_weight=float(cfgs_reward_selector["offroad_weight"]),
        )
    else:
        reward_selector = RewardSelectorConfig()

    # 解析 world-model 辅助监督配置 (Phase 1 / doc 方向 D)
    cfgs_wm_aux = args.get("wm_aux", {})
    _wm_discount = cfgs_wm_aux.get("multistep_discount")
    if _wm_discount is not None:
        _wm_discount = float(_wm_discount)
        if not 0.0 < _wm_discount <= 1.0:
            raise ValueError(f"wm_aux.multistep_discount must be in (0, 1], got {_wm_discount}")
    _wm_reward_w = float(cfgs_wm_aux.get("reward_head_weight", 0.0))
    _wm_contrastive_w = float(cfgs_wm_aux.get("contrastive_weight", 0.0))
    if _wm_reward_w < 0.0 or _wm_contrastive_w < 0.0:
        raise ValueError(
            f"wm_aux weights must be >= 0, got reward_head_weight={_wm_reward_w}, "
            f"contrastive_weight={_wm_contrastive_w}"
        )
    _wm_num_neg = int(cfgs_wm_aux.get("contrastive_num_negatives", 4))
    if _wm_contrastive_w > 0.0 and _wm_num_neg < 1:
        raise ValueError(f"wm_aux.contrastive_num_negatives must be >= 1, got {_wm_num_neg}")
    wm_aux = WorldModelAuxConfig(
        multistep_discount=_wm_discount,
        reward_head_weight=_wm_reward_w,
        reward_head_hidden_dim=int(cfgs_wm_aux.get("reward_head_hidden_dim", 512)),
        contrastive_weight=_wm_contrastive_w,
        contrastive_num_negatives=_wm_num_neg,
        contrastive_margin=float(cfgs_wm_aux.get("contrastive_margin", 0.1)),
    )

    # 解析 value-planning 配置。保留 Variant A Method 1，同时允许 latent-side guidance
    # 复用同一个 TemporalTrajectoryValueHead；Variant B/discrete menu/tree search 仍不进入代码路径。
    cfgs_value_planning = args.get("value_planning", {}) or {}
    value_planning = ValuePlanningConfig(
        enabled=bool(cfgs_value_planning.get("enabled", False)),
        variant=str(cfgs_value_planning.get("variant", "a_method1")),
        prefix_steps=int(cfgs_value_planning.get("prefix_steps", 2)),
        gamma=float(cfgs_value_planning.get("gamma", 0.99)),
        lambda_return=float(cfgs_value_planning.get("lambda_return", 0.8)),
        bootstrap_horizon=int(cfgs_value_planning.get("bootstrap_horizon", 3)),
        progress_weight=float(cfgs_value_planning.get("progress_weight", 1.0)),
        comfort_weight=float(cfgs_value_planning.get("comfort_weight", 0.2)),
        value_loss_weight=float(cfgs_value_planning.get("value_loss_weight", 1.0)),
        td_loss_weight=float(cfgs_value_planning.get("td_loss_weight", 1.0)),
        safe_floor_weight=float(cfgs_value_planning.get("safe_floor_weight", 0.05)),
        episode_ranking_weight=float(cfgs_value_planning.get("episode_ranking_weight", 0.1)),
        episode_ranking_margin=float(cfgs_value_planning.get("episode_ranking_margin", 1.0)),
        validation_calibration_weight=float(cfgs_value_planning.get("validation_calibration_weight", 1.0)),
        validation_ranking_weight=float(cfgs_value_planning.get("validation_ranking_weight", 1.0)),
        srpo_shaping_weight=float(cfgs_value_planning.get("srpo_shaping_weight", 0.0)),
        srpo_rho_mode=str(cfgs_value_planning.get("srpo_rho_mode", "nearest_accident_latent")),
        srpo_potential_based=bool(cfgs_value_planning.get("srpo_potential_based", True)),
        pred_consistency_weight=float(cfgs_value_planning.get("pred_consistency_weight", 0.0)),
        target_tau=float(cfgs_value_planning.get("target_tau", 0.995)),
    )
    if value_planning.variant not in {"a_method1", "latent_guidance"}:
        if value_planning.variant.startswith("b_") or value_planning.variant in {"variant_b", "tree_search"}:
            raise ValueError(
                f"value_planning.variant must be one of ['a_method1', 'latent_guidance']; "
                f"got {value_planning.variant!r}. Variant B/discrete action menu/tree search is intentionally "
                "not implemented."
            )
        raise ValueError(
            f"value_planning.variant must be one of ['a_method1', 'latent_guidance']; "
            f"got {value_planning.variant!r}."
        )
    if value_planning.prefix_steps < 1:
        raise ValueError(f"value_planning.prefix_steps must be >= 1, got {value_planning.prefix_steps}")
    if value_planning.bootstrap_horizon < 1:
        raise ValueError(f"value_planning.bootstrap_horizon must be >= 1, got {value_planning.bootstrap_horizon}")
    if not 0.0 < value_planning.gamma <= 1.0:
        raise ValueError(f"value_planning.gamma must be in (0, 1], got {value_planning.gamma}")
    if not 0.0 <= value_planning.lambda_return <= 1.0:
        raise ValueError(f"value_planning.lambda_return must be in [0, 1], got {value_planning.lambda_return}")
    if not math.isfinite(value_planning.target_tau) or not 0.0 <= value_planning.target_tau < 1.0:
        raise ValueError(f"value_planning.target_tau must be finite and in [0, 1), got {value_planning.target_tau}")
    _vp_nonnegative = {
        "comfort_weight": value_planning.comfort_weight,
        "value_loss_weight": value_planning.value_loss_weight,
        "td_loss_weight": value_planning.td_loss_weight,
        "safe_floor_weight": value_planning.safe_floor_weight,
        "episode_ranking_weight": value_planning.episode_ranking_weight,
        "episode_ranking_margin": value_planning.episode_ranking_margin,
        "validation_calibration_weight": value_planning.validation_calibration_weight,
        "validation_ranking_weight": value_planning.validation_ranking_weight,
        "srpo_shaping_weight": value_planning.srpo_shaping_weight,
        "pred_consistency_weight": value_planning.pred_consistency_weight,
    }
    _vp_bad = [
        name for name, value in _vp_nonnegative.items() if not math.isfinite(float(value)) or float(value) < 0.0
    ]
    if _vp_bad:
        field_name = _vp_bad[0]
        value = getattr(value_planning, field_name)
        raise ValueError(f"value_planning.{field_name} must be finite and non-negative, got {value}")
    if value_planning.validation_calibration_weight == 0.0 and value_planning.validation_ranking_weight == 0.0:
        raise ValueError(
            "value_planning.validation_calibration_weight and value_planning.validation_ranking_weight "
            "cannot both be zero"
        )
    if value_planning.srpo_shaping_weight > 0.0:
        raise ValueError(
            "value_planning.srpo_shaping_weight > 0 is not wired into the training batch yet; "
            "keep it at 0.0 until rho construction is implemented"
        )
    if value_planning.pred_consistency_weight > 0.0:
        raise ValueError(
            "value_planning.pred_consistency_weight > 0 is not wired into the training batch yet; "
            "keep it at 0.0 until predictor consistency targets are implemented"
        )
    cfgs_value_guidance = args.get("value_guidance", {}) or {}
    value_guidance = ValueGuidanceConfig(
        enabled=bool(cfgs_value_guidance.get("enabled", False)),
        steps=int(cfgs_value_guidance.get("steps", 1)),
        step_size=float(cfgs_value_guidance.get("step_size", 0.05)),
        max_delta_norm=float(cfgs_value_guidance.get("max_delta_norm", 0.25)),
        objective=str(cfgs_value_guidance.get("objective", "last")),
        detach_output=bool(cfgs_value_guidance.get("detach_output", True)),
    )
    if value_guidance.steps < 1:
        raise ValueError(f"value_guidance.steps must be >= 1, got {value_guidance.steps}")
    if value_guidance.step_size < 0.0:
        raise ValueError(f"value_guidance.step_size must be >= 0, got {value_guidance.step_size}")
    if value_guidance.max_delta_norm <= 0.0:
        raise ValueError(f"value_guidance.max_delta_norm must be > 0, got {value_guidance.max_delta_norm}")
    if value_guidance.objective not in {"last", "mean", "discounted"}:
        raise ValueError(
            "value_guidance.objective must be one of ['last', 'mean', 'discounted']; "
            f"got {value_guidance.objective!r}"
        )
    if value_guidance.enabled:
        if not value_planning.enabled and not cvoi.enabled and not _allow_evaluation_value_guidance:
            raise ValueError("value_guidance.enabled=true requires value_planning.enabled=true")
        if value_planning.enabled and value_planning.variant != "latent_guidance":
            raise ValueError(
                "value_guidance.enabled=true requires value_planning.variant='latent_guidance'; "
                f"got {value_planning.variant!r}"
            )
    if (
        counterfactual_supervision.enabled
        and counterfactual_supervision.protocol_version == "cf_supervision_v2"
        and value_planning.enabled
        and not value_guidance.enabled
        and counterfactual_supervision.hazard_target_mode != "episode_ranking"
    ):
        raise ValueError(
            "cf_supervision_v2 trainable value planning requires "
            "counterfactual_supervision.hazard_target_mode='episode_ranking'"
        )

    cfgs_budget_controller = args.get("budget_controller", {}) or {}
    budget_controller = BudgetControllerConfig(
        enabled=bool(cfgs_budget_controller.get("enabled", False)),
        mode=str(cfgs_budget_controller.get("mode", "oracle_distillation")),
        policy_dist=str(cfgs_budget_controller.get("policy_dist", "beta")),
        lambda_compute=float(cfgs_budget_controller.get("lambda_compute", 0.0)),
        oracle_budget_grid=list(cfgs_budget_controller.get("oracle_budget_grid", [0.0, 0.5, 1.0]) or []),
        schedule=dict(cfgs_budget_controller.get("schedule", {}) or {}),
        hidden_dim=int(cfgs_budget_controller.get("hidden_dim", 128)),
        feature_dim=int(cfgs_budget_controller.get("feature_dim", 0)),
        min_concentration=float(cfgs_budget_controller.get("min_concentration", 1.0)),
        oracle_path=cfgs_budget_controller.get("oracle_path"),
        oracle_output_path=cfgs_budget_controller.get("oracle_output_path"),
        output_checkpoint=cfgs_budget_controller.get("output_checkpoint"),
        controller_checkpoint=cfgs_budget_controller.get("controller_checkpoint"),
        bc_mse_weight=float(cfgs_budget_controller.get("bc_mse_weight", 0.1)),
        grpo_num_samples_per_scene=int(cfgs_budget_controller.get("grpo_num_samples_per_scene", 4)),
        grpo_bc_weight=float(cfgs_budget_controller.get("grpo_bc_weight", 0.0)),
        grpo_reward_interp=str(cfgs_budget_controller.get("grpo_reward_interp", "linear")),
        online_reward_source=str(cfgs_budget_controller.get("online_reward_source", "world4drive_l2_avg")),
        online_resume_checkpoint=cfgs_budget_controller.get("online_resume_checkpoint"),
    )
    if budget_controller.mode not in {
        "oracle_collection",
        "oracle_distillation",
        "grpo",
        "bc_then_grpo",
        "online_grpo",
        "eval",
    }:
        raise ValueError(
            "budget_controller.mode must be one of "
            "['oracle_collection', 'oracle_distillation', 'grpo', 'bc_then_grpo', 'online_grpo', 'eval']; "
            f"got {budget_controller.mode!r}"
        )
    if budget_controller.policy_dist != "beta":
        raise ValueError(f"budget_controller.policy_dist must be 'beta', got {budget_controller.policy_dist!r}")
    if budget_controller.lambda_compute < 0.0:
        raise ValueError(f"budget_controller.lambda_compute must be >= 0, got {budget_controller.lambda_compute}")
    if not budget_controller.oracle_budget_grid:
        raise ValueError("budget_controller.oracle_budget_grid must contain at least one budget")
    _bad_budgets = [b for b in budget_controller.oracle_budget_grid if not 0.0 <= float(b) <= 1.0]
    if _bad_budgets:
        raise ValueError(f"budget_controller.oracle_budget_grid entries must be in [0, 1], got {_bad_budgets}")
    if budget_controller.oracle_budget_grid != sorted(budget_controller.oracle_budget_grid):
        raise ValueError("budget_controller.oracle_budget_grid must be sorted ascending")
    budget_schedule = BudgetSchedule.from_mapping(budget_controller.schedule)
    if budget_controller.hidden_dim <= 0:
        raise ValueError(f"budget_controller.hidden_dim must be > 0, got {budget_controller.hidden_dim}")
    if budget_controller.feature_dim < 0:
        raise ValueError(f"budget_controller.feature_dim must be >= 0, got {budget_controller.feature_dim}")
    if budget_controller.min_concentration <= 0.0:
        raise ValueError(f"budget_controller.min_concentration must be > 0, got {budget_controller.min_concentration}")
    if budget_controller.bc_mse_weight < 0.0:
        raise ValueError(f"budget_controller.bc_mse_weight must be >= 0, got {budget_controller.bc_mse_weight}")
    if budget_controller.grpo_num_samples_per_scene < 2:
        raise ValueError(
            "budget_controller.grpo_num_samples_per_scene must be >= 2, "
            f"got {budget_controller.grpo_num_samples_per_scene}"
        )
    if budget_controller.grpo_bc_weight < 0.0:
        raise ValueError(f"budget_controller.grpo_bc_weight must be >= 0, got {budget_controller.grpo_bc_weight}")
    if budget_controller.grpo_reward_interp != "linear":
        raise ValueError(
            "budget_controller.grpo_reward_interp must be 'linear', " f"got {budget_controller.grpo_reward_interp!r}"
        )
    if budget_controller.mode == "grpo":
        if not budget_controller.oracle_path:
            raise ValueError("budget_controller.oracle_path is required when budget_controller.mode='grpo'")
        if not budget_controller.controller_checkpoint:
            raise ValueError(
                "budget_controller.controller_checkpoint is required when budget_controller.mode='grpo' "
                "so GRPO can warm-start from Stage4C BC"
            )
    if budget_controller.mode == "bc_then_grpo" and not budget_controller.oracle_path:
        raise ValueError("budget_controller.oracle_path is required when budget_controller.mode='bc_then_grpo'")
    if budget_controller.mode == "online_grpo":
        if not budget_controller.controller_checkpoint:
            raise ValueError(
                "budget_controller.controller_checkpoint is required when "
                "budget_controller.mode='online_grpo' so training can warm-start from Stage4C BC"
            )
        if budget_controller.oracle_path:
            raise ValueError(
                "budget_controller.oracle_path must be unset when budget_controller.mode='online_grpo'; "
                "online rewards are produced by the frozen main policy"
            )
        if budget_controller.online_reward_source != "world4drive_l2_avg":
            raise ValueError(
                "budget_controller.online_reward_source must be 'world4drive_l2_avg' for online_grpo, "
                f"got {budget_controller.online_reward_source!r}"
            )
        if budget_controller.feature_dim != 0:
            raise ValueError("budget_controller.feature_dim must be 0 for online_grpo")
        if budget_controller.grpo_bc_weight != 0.0:
            raise ValueError(
                "budget_controller.grpo_bc_weight must be 0 for online_grpo because online batches "
                "do not contain oracle BC targets"
            )
        if int(data.batch_size) != 1:
            raise ValueError(
                "data.batch_size must be 1 for online_grpo because rollout_future_steps is batch-global, "
                f"got {data.batch_size}"
            )
        if str(train.predictor_type).lower() != "ac_transformer":
            raise ValueError(
                "online_grpo requires train.predictor_type='ac_transformer', " f"got {train.predictor_type!r}"
            )
        if bool(train.use_parallel_predictor):
            raise ValueError("online_grpo requires train.use_parallel_predictor=false")
        if not bool(train.predictor_inference_consistent):
            raise ValueError("online_grpo requires train.predictor_inference_consistent=true")
        if bool(dynamic_rollout.enabled):
            raise ValueError(
                "online_grpo requires predictor_dynamic_rollout.enabled=false because the sampled "
                "controller budget owns rollout length"
            )
        if not bool(planner.use_planner):
            raise ValueError("online_grpo requires planner.use_planner=true")
        if str(planner.planner_type).lower() != "diffusion":
            raise ValueError(
                "online_grpo requires planner.planner_type='diffusion' so all sampled budgets "
                "can share one controlled diffusion-noise seed"
            )
        if value_planning.enabled or value_guidance.enabled:
            raise ValueError("online_grpo requires value_planning.enabled=false and value_guidance.enabled=false")
        if budget_schedule.rollout_future_steps != (0, "full"):
            raise ValueError(
                "online_grpo requires budget_controller.schedule.rollout_future_steps=[0, 'full'], "
                f"got {list(budget_schedule.rollout_future_steps)!r}"
            )
        max_future_steps = int(data.num_target_frames) - int(train.num_observed_frames)
        if max_future_steps <= 0:
            raise ValueError(
                "online_grpo requires data.num_target_frames > train.num_observed_frames, "
                f"got {data.num_target_frames} <= {train.num_observed_frames}"
            )

    # 解析 Phase 3 轨迹优化器配置
    cfgs_traj_opt = args.get("traj_opt", {})
    _to_enabled = bool(cfgs_traj_opt.get("enabled", False))
    traj_opt = WMTrajOptConfig(
        enabled=_to_enabled,
        steps=int(cfgs_traj_opt.get("steps", 5)),
        lr=float(cfgs_traj_opt.get("lr", 0.1)),
        trust_radius_xy=float(cfgs_traj_opt.get("trust_radius_xy", 1.0)),
        trust_radius_yaw=float(cfgs_traj_opt.get("trust_radius_yaw", 0.2)),
        comfort_weight=float(cfgs_traj_opt.get("comfort_weight", 0.1)),
    )
    if _to_enabled:
        if traj_opt.steps < 1:
            raise ValueError(f"traj_opt.steps must be >= 1, got {traj_opt.steps}")
        if traj_opt.lr <= 0:
            raise ValueError(f"traj_opt.lr must be > 0, got {traj_opt.lr}")
        if traj_opt.trust_radius_xy <= 0 or traj_opt.trust_radius_yaw <= 0:
            raise ValueError(
                f"traj_opt trust radii must be > 0, got xy={traj_opt.trust_radius_xy}, "
                f"yaw={traj_opt.trust_radius_yaw}"
            )

    # 解析 optimization 配置
    cfgs_opt = args.get("optimization", {})
    optimization = OptimizationConfig(
        ipe=cfgs_opt.get("ipe"),
        weight_decay=float(cfgs_opt.get("weight_decay", 0.04)),
        final_weight_decay=float(cfgs_opt.get("final_weight_decay", 0.4)),
        epochs=cfgs_opt.get("epochs", 100),
        schedule_epochs=cfgs_opt.get("schedule_epochs"),
        anneal=cfgs_opt.get("anneal", 1),
        warmup=cfgs_opt.get("warmup", 10),
        start_lr=cfgs_opt.get("start_lr", 0.0001),
        lr=cfgs_opt.get("lr", 0.0001),
        final_lr=cfgs_opt.get("final_lr", 0.0),
        enc_lr_scale=cfgs_opt.get("enc_lr_scale", 1.0),
        predictor_lr_scale=cfgs_opt.get("predictor_lr_scale", 1.0),
        betas=cfgs_opt.get("betas", (0.9, 0.999)),
        eps=cfgs_opt.get("eps", 1e-8),
        grad_clip_norm=cfgs_opt.get("grad_clip_norm", 1.0),
        optimizer=cfgs_opt.get("optimizer", "adamw"),
        is_anneal=cfgs_opt.get("is_anneal", False),
        anneal_ckpt=cfgs_opt.get("anneal_ckpt", None),
        resume_anneal=cfgs_opt.get("resume_anneal", False),
        ipe_scale=cfgs_opt.get("ipe_scale", 1.0),
    )
    _validate_selection_checkpoint_epochs_bound(
        meta.selection_checkpoint_epochs,
        optimization_epochs=optimization.epochs,
    )

    # 解析 lewm 配置
    world_model = WorldModelConfig(
        enabled=bool(cfgs_world_model.get("enabled", False)),
        sigreg_weight=float(cfgs_world_model.get("sigreg_weight", 0.09)),
        sigreg_knots=int(cfgs_world_model.get("sigreg_knots", 17)),
        sigreg_num_proj=int(cfgs_world_model.get("sigreg_num_proj", 1024)),
        projector_hidden_dim=int(cfgs_world_model.get("projector_hidden_dim", 2048)),
        embed_dim=int(cfgs_world_model.get("embed_dim", 192)),
        num_subspaces=int(cfgs_world_model.get("num_subspaces", 1)),
        subspace_dim=(None if cfgs_world_model.get("subspace_dim") is None else int(cfgs_world_model["subspace_dim"])),
        init_mode=str(cfgs_world_model.get("init_mode", "orthogonal_frozen")),
        theta=float(cfgs_world_model.get("theta", 0.0)),
    )

    validate_cvoi_cross_section(
        cvoi,
        method=args.get("method", ""),
        model=model,
        value_guidance=value_guidance,
        value_planning=value_planning,
        budget_controller=budget_controller,
        train=train,
        planner=planner,
        data=data,
        world_model=world_model,
        data_aug=data_aug,
        dynamic_rollout=dynamic_rollout,
        validation_suite=validation_suite,
        ema=ema,
        segmentation=segmentation,
        loss=loss,
        multiview=multiview,
        token_ae=token_ae,
        predictor_dit=predictor_dit,
        proposal=proposal,
        counterfactual_supervision=counterfactual_supervision,
        reward=reward,
        reward_selector=reward_selector,
        wm_aux=wm_aux,
        traj_opt=traj_opt,
        rl=rl,
        meta=meta,
        optimization=optimization,
    )

    # 解析 Stage-2 / Stage-3 配置
    cfgs_refinement = args.get("refinement", {})
    cfgs_refinement_lambdas = cfgs_refinement.get("lambdas", {}) or {}
    refinement = RefinementConfig(
        num_modes=int(cfgs_refinement.get("num_modes", planner.num_modes)),
        inference_num_rounds=int(cfgs_refinement.get("inference_num_rounds", 1)),
        predictor_rollout_seconds=(
            None
            if cfgs_refinement.get("predictor_rollout_seconds") is None
            else float(cfgs_refinement.get("predictor_rollout_seconds"))
        ),
        detach_zfut=bool(cfgs_refinement.get("detach_zfut", True)),
        refine_stop_grad_to_proposer=bool(cfgs_refinement.get("refine_stop_grad_to_proposer", True)),
        refine_use_random_predictor_latent=bool(cfgs_refinement.get("refine_use_random_predictor_latent", False)),
        refine_keep_initial_actions=bool(cfgs_refinement.get("refine_keep_initial_actions", False)),
        predictor_finetune=bool(cfgs_refinement.get("predictor_finetune", False)),
        checkpoint=cfgs_refinement.get("checkpoint"),
        lambdas=StageLossWeightsConfig(
            prop=float(cfgs_refinement_lambdas.get("prop", 1.0)),
            refine=float(cfgs_refinement_lambdas.get("refine", 1.0)),
            anchor=float(cfgs_refinement_lambdas.get("anchor", 0.0)),
            div=float(cfgs_refinement_lambdas.get("div", 0.0)),
        ),
    )

    cfgs_refinement_gated = args.get("refinement_gated", {})
    stage3_num_rounds = int(cfgs_refinement_gated.get("num_rounds", 2))
    refinement_gated = RefinementGatedConfig(
        num_rounds=stage3_num_rounds,
        round_weights=list(cfgs_refinement_gated.get("round_weights", [1.0] * stage3_num_rounds)),
        predictor_rollout_seconds=(
            None
            if cfgs_refinement_gated.get("predictor_rollout_seconds") is None
            else float(cfgs_refinement_gated.get("predictor_rollout_seconds"))
        ),
        grad_checkpoint=bool(cfgs_refinement_gated.get("grad_checkpoint", True)),
        predictor_finetune=bool(cfgs_refinement_gated.get("predictor_finetune", False)),
        checkpoint=cfgs_refinement_gated.get("checkpoint"),
        refine_use_z_context=bool(cfgs_refinement_gated.get("refine_use_z_context", True)),
        refine_use_status_feature=bool(cfgs_refinement_gated.get("refine_use_status_feature", True)),
        refine_use_proposal_traj=bool(cfgs_refinement_gated.get("refine_use_proposal_traj", True)),
        refine_use_proposal_logits=bool(cfgs_refinement_gated.get("refine_use_proposal_logits", True)),
        refine_use_proposal_features=bool(cfgs_refinement_gated.get("refine_use_proposal_features", True)),
        refine_use_predictor_rollout=bool(cfgs_refinement_gated.get("refine_use_predictor_rollout", True)),
        refine_use_random_predictor_latent=bool(
            cfgs_refinement_gated.get("refine_use_random_predictor_latent", False)
        ),
        refine_keep_initial_actions=bool(cfgs_refinement_gated.get("refine_keep_initial_actions", False)),
        use_multimodal_final=bool(cfgs_refinement_gated.get("use_multimodal_final", False)),
    )

    config = TrainingConfig(
        method=str(args.get("method", args.get("training_method", args.get("predictor_method", "")))).lower(),
        meta=meta,
        model=model,
        train=train,
        multiview=multiview,
        predictor_dynamic_rollout=dynamic_rollout,
        validation_suite=validation_suite,
        predictor_dit=predictor_dit,
        ema=ema,
        segmentation=segmentation,
        planner=planner,
        proposal=proposal,
        counterfactual_supervision=counterfactual_supervision,
        data=data,
        data_aug=data_aug,
        token_ae=token_ae,
        loss=loss,
        world_model=world_model,
        refinement=refinement,
        refinement_gated=refinement_gated,
        reward=reward,
        reward_selector=reward_selector,
        wm_aux=wm_aux,
        value_planning=value_planning,
        value_guidance=value_guidance,
        budget_controller=budget_controller,
        cvoi=cvoi,
        traj_opt=traj_opt,
        rl=rl,
        optimization=optimization,
    )
    raw_predictor_lora_enabled = (args.get("predictor_lora", {}) or {}).get("enabled", False)
    if config.train.predictor_split_tf_ar_backward and type(raw_predictor_lora_enabled) is not bool:
        raise ValueError("predictor_lora.enabled must be a bool when " "train.predictor_split_tf_ar_backward=true")
    validate_predictor_split_tf_ar_backward_config(
        config,
        predictor_lora_enabled=raw_predictor_lora_enabled,
    )
    _validate_explicit_frozen_encoder_config(
        config,
        encoder_train_configured="encoder_train" in cfgs_train,
    )
    _validate_reuse_context_as_target_config(config)
    _validate_vjepa_parallel_predictor_config(config)
    _validate_lewm_normalize_reps_config(config)
    _validate_multiview_config(config)
    _validate_policy_output_source_config(config)
    if config.predictor_dynamic_rollout.enabled and config.planner.diff_train_prefix_conditioning:
        raise ValueError(
            "predictor_dynamic_rollout.enabled and planner.diff_train_prefix_conditioning cannot both be true: "
            "outer and planner-internal prefix sampling would sample twice"
        )
    _resolve_image_require_policies_in_place(config)
    return config
