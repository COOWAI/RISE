"""Real-geometry anchored Counterfactual Value-of-Imagination configuration."""

import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, List, Mapping, Optional

from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import (
    CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL,
    CvoiFormalV2NavSimE120AblationSignature,
    parse_cvoi_ablation_signature,
)
from app.vjepa_cowa_world_model.training.configs.data import (
    NAVSIM_DEFAULT_MAX_AGENTS,
    NavSimConfig,
    validate_navsim_cvoi_geometry_contract,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_full_state_warmstart import (
    validate_formal_v2_full_state_warmstart_paths,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_config import (
    build_formal_v2_navsim_e120_public_config,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_E120_DEFAULT_LAMBDA,
    FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
    FORMAL_V2_NAVSIM_MAX_AGENTS,
    FORMAL_V2_NAVSIM_MAX_HORIZON,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P0_POLICIES,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    build_formal_v2_navsim_root_catalog,
    validate_formal_v2_navsim_direct_task_projection,
)
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import (
    resolve_cvoi_manual_full_results_root_from_config,
    resolve_cvoi_manual_gate_branch,
    resolve_cvoi_manual_value_lineage,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import NAVTRAIN_GATE_TRAINING_BATCH_SIZE

CVOI_DUAL_VALUE_SCHEMA = "cvoi_dual_value_v1"
CVOI_PROTOCOL_LEGACY_V1 = "legacy_v1"
CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1 = CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL
CVOI_DUAL_VALUE_NAVSIM_E120_SCHEMA = "cvoi_dual_value_navsim_e120_v1"
CVOI_FULL_STATE_WARMSTART_CONFIG_SCHEMA = "cvoi_full_state_warmstart_config_v1"
_NAVSIM_E120_ROOT_ROLES = {
    "navsim_real_train": "real_train",
    "navsim_cf_train": "cf_train",
    "navsim_real_navtest": "real_navtest",
    "navsim_cf_val": "cf_val",
}
CVOI_FORMAL_V2_NAVSIM_E120_STAGES = frozenset(
    {
        "unguided_planner",
        "guided_planner",
        "field_warmup",
        "field_calibrated",
        "stop_calibrated",
        "gate_distillation",
    }
)
CVOI_FORMAL_V2_NAVSIM_E120_MODEL_STAGES = CVOI_FORMAL_V2_NAVSIM_E120_STAGES - {"gate_distillation"}
CVOI_FORMAL_V2_NAVSIM_E120_PLANNER_STAGES = frozenset({"unguided_planner", "guided_planner"})
CVOI_CONTROLLER_LINEAGES = frozenset({"value_guided", "p0_controller"})
CVOI_FIELD_WARMUP_DOMAINS = frozenset({"real", "real_cf"})
CVOI_NAVSIM_OFFLINE_ADAPTER_FACTORY = (
    "app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter:create_navsim_cvoi_offline_adapter"
)
CVOI_TRAINING_STAGES = CVOI_FORMAL_V2_NAVSIM_E120_STAGES


@dataclass(frozen=True)
class CvoiPinnedArtifactConfig:
    """Exact path to one warm-start source artifact."""

    path: str


@dataclass(frozen=True)
class CvoiFullStateWarmstartConfig:
    """Direct binding for the one allowed e120 full-state import."""

    schema: str
    import_mode: str
    source_checkpoint: CvoiPinnedArtifactConfig
    source_params_pretrain: CvoiPinnedArtifactConfig


@dataclass
class CVoIConfig:
    """Fixed protocol and artifact paths for real-geometry anchored CVoI."""

    enabled: bool = False
    protocol_version: str = CVOI_PROTOCOL_LEGACY_V1
    ablation_mode: str = "baseline"
    ablation_signature: Optional[CvoiFormalV2NavSimE120AblationSignature] = None
    schema: str = CVOI_DUAL_VALUE_SCHEMA
    stage: str = "field_warmup"
    evaluation_mode: str = "controller"
    field_warmup_domain: str = "real_cf"
    max_agents: int = NAVSIM_DEFAULT_MAX_AGENTS
    guidance_steps: int = 2
    guidance_objective: str = "last"
    max_horizon: int = 3
    rollout_horizons: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    controller_batch_size: int = 1
    gate_training_batch_size: Optional[int] = None
    controller_lineage: str = "value_guided"
    lambda_compute: float = 0.0
    lambda_grid: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    compute_costs: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0])
    seed_planner_checkpoint: Optional[str] = None
    unguided_planner_checkpoint: Optional[str] = None
    field_checkpoint: Optional[str] = None
    guided_planner_checkpoint: Optional[str] = None
    dual_value_checkpoint: Optional[str] = None
    oracle_path: Optional[str] = None
    gate_checkpoint: Optional[str] = None
    output_checkpoint: Optional[str] = None
    world_model_checkpoint: Optional[str] = None
    token_ae_checkpoint: Optional[str] = None
    offline_adapter_factory: Optional[str] = None
    offline_runtime_factory: Optional[str] = None
    value_hidden_dim: int = 512
    value_num_layers: int = 1
    value_dropout: float = 0.0
    value_updates_per_epoch: Optional[int] = None
    tokens_per_frame: Optional[int] = None
    field_calibration_num_perturbations: int = 4
    field_calibration_perturbation_scale: float = 0.05
    field_calibration_max_delta_norm: float = 0.25
    field_calibration_order_margin: float = 0.1
    full_state_warmstart: Optional[CvoiFullStateWarmstartConfig] = None


def _require_finite_nonnegative(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite non-negative real number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative real number, got {value!r}")
    return normalized


def _require_finite_positive(value: Any, *, field_name: str) -> float:
    normalized = _require_finite_nonnegative(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive real number, got {value!r}")
    return normalized


def _require_exact_mapping(value: object, expected_fields: frozenset[str], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")
    actual_fields = frozenset(value)
    if actual_fields != expected_fields:
        raise ValueError(
            f"{field_name} fields mismatch: "
            f"missing={sorted(expected_fields - actual_fields)}, unexpected={sorted(actual_fields - expected_fields)}"
        )
    return value


def _require_absolute_json_path(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty absolute JSON path, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.suffix.lower() != ".json":
        raise ValueError(f"{field_name} must be an absolute path ending in '.json', got {value!r}")
    return value


def _require_absolute_yaml_path(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty absolute YAML path, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.suffix.lower() != ".yaml":
        raise ValueError(f"{field_name} must be an absolute path ending in '.yaml', got {value!r}")
    return value


def _parse_pinned_artifact_config(
    value: object,
    *,
    field_name: str,
) -> CvoiPinnedArtifactConfig:
    raw = _require_exact_mapping(value, frozenset({"path"}), field_name=field_name)
    path = raw["path"]
    if type(path) is not str or not path.strip():
        raise ValueError(f"{field_name}.path must be a non-empty path string, got {path!r}")
    return CvoiPinnedArtifactConfig(path=path)


def _parse_full_state_warmstart_config(value: object) -> CvoiFullStateWarmstartConfig:
    field_name = "cvoi.full_state_warmstart"
    raw = _require_exact_mapping(
        value,
        frozenset({"schema", "import_mode", "source_checkpoint", "source_params_pretrain"}),
        field_name=field_name,
    )
    fixed_strings = {
        "schema": CVOI_FULL_STATE_WARMSTART_CONFIG_SCHEMA,
        "import_mode": "full_state_warmstart",
    }
    for key, expected in fixed_strings.items():
        actual = raw[key]
        if type(actual) is not str or actual != expected:
            raise ValueError(f"{field_name}.{key} must be exactly {expected!r}, got {actual!r}")
    source_checkpoint = _parse_pinned_artifact_config(
        raw["source_checkpoint"],
        field_name=f"{field_name}.source_checkpoint",
    )
    source_params_pretrain = _parse_pinned_artifact_config(
        raw["source_params_pretrain"],
        field_name=f"{field_name}.source_params_pretrain",
    )
    validate_formal_v2_full_state_warmstart_paths(
        source_checkpoint.path,
        source_params_pretrain.path,
    )
    return CvoiFullStateWarmstartConfig(
        schema=raw["schema"],
        import_mode=raw["import_mode"],
        source_checkpoint=source_checkpoint,
        source_params_pretrain=source_params_pretrain,
    )


def parse_cvoi_config(values: Any) -> CVoIConfig:
    """Parse the CVoI section without activating its cross-section contract."""

    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValueError(f"cvoi must be a mapping, got {type(values).__name__}")
    unexpected_fields = sorted(set(values) - set(CVoIConfig.__dataclass_fields__))
    if unexpected_fields:
        raise ValueError(f"cvoi contains unsupported field(s): {unexpected_fields}")
    protocol_version = values.get("protocol_version", CVOI_PROTOCOL_LEGACY_V1)
    raw_lambda_grid = values.get("lambda_grid", [0.0, 0.05, 0.1])
    if not isinstance(raw_lambda_grid, (list, tuple)):
        raise ValueError(f"cvoi.lambda_grid must be a list or tuple, got {raw_lambda_grid!r}")
    raw_rollout_horizons = values.get("rollout_horizons", [0, 1, 2, 3])
    if not isinstance(raw_rollout_horizons, (list, tuple)) or any(
        type(horizon) is not int for horizon in raw_rollout_horizons
    ):
        raise ValueError("cvoi.rollout_horizons must be a list or tuple of integers")
    rollout_horizons = list(raw_rollout_horizons)
    if not rollout_horizons or any(
        current >= following for current, following in zip(rollout_horizons, rollout_horizons[1:])
    ):
        raise ValueError("cvoi.rollout_horizons must be non-empty, unique, and strictly increasing")
    enabled = values.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"cvoi.enabled must be a boolean, got {enabled!r}")
    if not enabled:
        unsupported_fields = sorted(set(values) - {"enabled"})
        if unsupported_fields:
            raise ValueError(
                "disabled cvoi is reserved for evaluation-only consumers and accepts only enabled=false; "
                f"unexpected fields={unsupported_fields}"
            )
        return CVoIConfig()
    if protocol_version != CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1:
        raise ValueError(
            "enabled cvoi.protocol_version must be exactly "
            f"{CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1!r}, got {protocol_version!r}"
        )
    evaluation_mode = values.get("evaluation_mode", "controller")
    if evaluation_mode not in {"controller", "p0_forced", "p1_field_forced"} or not isinstance(evaluation_mode, str):
        raise ValueError("cvoi.evaluation_mode must be one of controller, p0_forced, or p1_field_forced")
    ablation_mode = values.get("ablation_mode", "baseline")
    if not isinstance(ablation_mode, str) or ablation_mode not in {"baseline", "manual_ablation"}:
        raise ValueError(
            "cvoi.ablation_mode must be one of ['baseline', 'manual_ablation'], " f"got {ablation_mode!r}"
        )
    raw_ablation_signature = values.get("ablation_signature")
    ablation_signature = (
        None if raw_ablation_signature is None else parse_cvoi_ablation_signature(raw_ablation_signature)
    )
    field_warmup_domain = values.get("field_warmup_domain", "real_cf")
    if not isinstance(field_warmup_domain, str) or field_warmup_domain not in CVOI_FIELD_WARMUP_DOMAINS:
        raise ValueError(
            "cvoi.field_warmup_domain must be one of "
            f"{sorted(CVOI_FIELD_WARMUP_DOMAINS)}, got {field_warmup_domain!r}"
        )
    controller_lineage = values.get("controller_lineage", "value_guided")
    if not isinstance(controller_lineage, str) or controller_lineage not in CVOI_CONTROLLER_LINEAGES:
        raise ValueError(
            "cvoi.controller_lineage must be one of " f"{sorted(CVOI_CONTROLLER_LINEAGES)}, got {controller_lineage!r}"
        )
    return CVoIConfig(
        enabled=enabled,
        protocol_version=protocol_version,
        ablation_mode=ablation_mode,
        ablation_signature=ablation_signature,
        schema=values.get("schema", CVOI_DUAL_VALUE_SCHEMA),
        stage=values.get("stage", "field_warmup"),
        evaluation_mode=evaluation_mode,
        field_warmup_domain=field_warmup_domain,
        max_agents=values.get("max_agents", NAVSIM_DEFAULT_MAX_AGENTS),
        guidance_steps=values.get("guidance_steps", 2),
        guidance_objective=values.get("guidance_objective", "last"),
        max_horizon=values.get("max_horizon", 3),
        rollout_horizons=rollout_horizons,
        controller_batch_size=values.get("controller_batch_size", 1),
        gate_training_batch_size=values.get("gate_training_batch_size"),
        controller_lineage=controller_lineage,
        lambda_compute=values.get("lambda_compute", 0.0),
        lambda_grid=list(raw_lambda_grid),
        compute_costs=list(values.get("compute_costs", [0.0, 1.0, 2.0, 3.0])),
        seed_planner_checkpoint=values.get("seed_planner_checkpoint"),
        unguided_planner_checkpoint=values.get("unguided_planner_checkpoint"),
        field_checkpoint=values.get("field_checkpoint"),
        guided_planner_checkpoint=values.get("guided_planner_checkpoint"),
        dual_value_checkpoint=values.get("dual_value_checkpoint"),
        oracle_path=values.get("oracle_path"),
        gate_checkpoint=values.get("gate_checkpoint"),
        output_checkpoint=values.get("output_checkpoint"),
        world_model_checkpoint=values.get("world_model_checkpoint"),
        token_ae_checkpoint=values.get("token_ae_checkpoint"),
        offline_adapter_factory=values.get("offline_adapter_factory"),
        offline_runtime_factory=values.get("offline_runtime_factory"),
        value_hidden_dim=values.get("value_hidden_dim", 512),
        value_num_layers=values.get("value_num_layers", 1),
        value_dropout=values.get("value_dropout", 0.0),
        value_updates_per_epoch=values.get("value_updates_per_epoch"),
        tokens_per_frame=values.get("tokens_per_frame"),
        field_calibration_num_perturbations=values.get("field_calibration_num_perturbations", 4),
        field_calibration_perturbation_scale=values.get("field_calibration_perturbation_scale", 0.05),
        field_calibration_max_delta_norm=values.get("field_calibration_max_delta_norm", 0.25),
        field_calibration_order_margin=values.get("field_calibration_order_margin", 0.1),
        full_state_warmstart=(
            None
            if values.get("full_state_warmstart") is None
            else _parse_full_state_warmstart_config(values["full_state_warmstart"])
        ),
    )


def is_cvoi_formal_v2_navsim_e120_profile(config: object) -> bool:
    """Return whether ``config`` selects the isolated NavSim e120 protocol."""

    return isinstance(config, CVoIConfig) and config.protocol_version == CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1


def cvoi_uses_managed_predictor_initialization(config: object) -> bool:
    """Predicate used by Planner lines to bypass their generic predictor-loader guard."""

    return is_cvoi_formal_v2_navsim_e120_profile(config) and isinstance(
        config.full_state_warmstart, CvoiFullStateWarmstartConfig
    )


def resolve_cvoi_formal_v2_navsim_e120_planner_stage(config: CVoIConfig) -> str:
    """Map the outer CVoI stage to the public P0/P1 schedule identity."""

    if not is_cvoi_formal_v2_navsim_e120_profile(config):
        raise ValueError("CVoI config is not the formal_v2_navsim_e120_h4_v3 profile")
    if config.stage == "unguided_planner":
        return "p0"
    if config.stage == "guided_planner":
        return "p1"
    raise ValueError(
        "NavSim e120 Planner profile supports only cvoi.stage='unguided_planner' or 'guided_planner', "
        f"got {config.stage!r}"
    )


def _require_absolute_artifact_path(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip() or not Path(value).expanduser().is_absolute():
        raise ValueError(f"{field_name} must be a non-empty absolute path, got {value!r}")
    return value


_NAVSIM_E120_PATH_FIELDS = frozenset(
    {
        "world_model_checkpoint",
        "seed_planner_checkpoint",
        "unguided_planner_checkpoint",
        "field_checkpoint",
        "guided_planner_checkpoint",
        "dual_value_checkpoint",
        "oracle_path",
        "gate_checkpoint",
        "output_checkpoint",
        "token_ae_checkpoint",
    }
)

_NAVSIM_E120_DIRECT_ARTIFACTS = {
    "unguided_planner": frozenset({"world_model_checkpoint", "seed_planner_checkpoint", "output_checkpoint"}),
    "field_warmup": frozenset(
        {
            "world_model_checkpoint",
            "seed_planner_checkpoint",
            "unguided_planner_checkpoint",
            "output_checkpoint",
        }
    ),
    "field_calibrated": frozenset(
        {
            "world_model_checkpoint",
            "seed_planner_checkpoint",
            "unguided_planner_checkpoint",
            "field_checkpoint",
            "output_checkpoint",
        }
    ),
    "guided_planner": frozenset(
        {
            "world_model_checkpoint",
            "seed_planner_checkpoint",
            "unguided_planner_checkpoint",
            "field_checkpoint",
            "output_checkpoint",
        }
    ),
    "stop_calibrated": frozenset(
        {
            "world_model_checkpoint",
            "seed_planner_checkpoint",
            "unguided_planner_checkpoint",
            "field_checkpoint",
            "guided_planner_checkpoint",
            "output_checkpoint",
        }
    ),
    "gate_distillation": frozenset({"oracle_path", "output_checkpoint"}),
    "evaluation": frozenset(
        {
            "world_model_checkpoint",
            "seed_planner_checkpoint",
            "unguided_planner_checkpoint",
            "field_checkpoint",
            "guided_planner_checkpoint",
            "dual_value_checkpoint",
        }
    ),
}


def _navsim_e120_stage_artifacts(config: CVoIConfig) -> tuple[frozenset[str], frozenset[str]]:
    required_fields = _NAVSIM_E120_DIRECT_ARTIFACTS.get(config.stage)
    if required_fields is None:
        raise ValueError(
            "NavSim e120 direct cvoi.stage must be one of "
            f"{sorted(_NAVSIM_E120_DIRECT_ARTIFACTS)}, got {config.stage!r}"
        )
    return required_fields, _NAVSIM_E120_PATH_FIELDS - required_fields


def _normalized_absolute_artifact_path(value: object, *, field_name: str) -> str:
    raw = _require_absolute_artifact_path(value, field_name=field_name)
    return str(Path(raw).expanduser().resolve(strict=False))


def _validate_navsim_e120_artifacts(config: CVoIConfig) -> None:
    required_fields, forbidden_fields = _navsim_e120_stage_artifacts(config)
    for field_name in sorted(required_fields):
        _require_absolute_artifact_path(getattr(config, field_name), field_name=f"cvoi.{field_name}")
    configured_forbidden = sorted(
        field_name for field_name in forbidden_fields if getattr(config, field_name) is not None
    )
    if configured_forbidden:
        raise ValueError(f"NavSim e120 stage={config.stage!r} forbids artifact path(s): {configured_forbidden}")

    path_roles: dict[str, list[str]] = {}

    def _record(field_name: str, value: object) -> None:
        if value is None:
            return
        normalized = _normalized_absolute_artifact_path(value, field_name=field_name)
        path_roles.setdefault(normalized, []).append(field_name)

    for field_name in sorted(_NAVSIM_E120_PATH_FIELDS):
        _record(f"cvoi.{field_name}", getattr(config, field_name))
    if config.full_state_warmstart is not None:
        _record(
            "cvoi.full_state_warmstart.source_checkpoint.path",
            config.full_state_warmstart.source_checkpoint.path,
        )
        _record(
            "cvoi.full_state_warmstart.source_params_pretrain.path",
            config.full_state_warmstart.source_params_pretrain.path,
        )

    allowed_source_roles = sorted(
        {
            "cvoi.world_model_checkpoint",
            "cvoi.seed_planner_checkpoint",
            "cvoi.full_state_warmstart.source_checkpoint.path",
        }
    )
    source_checkpoint_path = (
        None
        if config.full_state_warmstart is None
        else _normalized_absolute_artifact_path(
            config.full_state_warmstart.source_checkpoint.path,
            field_name="cvoi.full_state_warmstart.source_checkpoint.path",
        )
    )
    aliases = {
        path: sorted(roles)
        for path, roles in path_roles.items()
        if len(roles) > 1 and not (path == source_checkpoint_path and sorted(roles) == allowed_source_roles)
    }
    if aliases:
        raise ValueError(f"NavSim e120 artifact paths must be distinct; aliases={aliases}")

    source_roles = sorted(path_roles.get(source_checkpoint_path, [])) if source_checkpoint_path is not None else []
    if config.stage != "gate_distillation":
        if source_roles != allowed_source_roles:
            raise ValueError(
                "world_model_checkpoint and seed_planner_checkpoint must match the configured e120 source; "
                f"got roles={source_roles}"
            )
    elif source_roles:
        raise ValueError("NavSim e120 Gate artifacts must not point at an e120 warm-start source")


def _validate_navsim_e120_root(root: Mapping[str, object], *, field_name: str) -> str:
    domain = root.get("domain")
    if domain not in {"real", "counterfactual"}:
        raise ValueError(f"{field_name}.domain must be 'real' or 'counterfactual', got {domain!r}")
    name = root.get("name")
    role = _NAVSIM_E120_ROOT_ROLES.get(name) if type(name) is str else None
    if role is None:
        raise ValueError(f"{field_name}.name is not a committed NavSim root authority role: {name!r}")
    expected_domain = "counterfactual" if role in {"cf_train", "cf_val"} else "real"
    if domain != expected_domain:
        raise ValueError(
            f"{field_name}.domain must match root name {name!r}: expected={expected_domain!r}, got={domain!r}"
        )
    for path_key in ("data_path", "sensor_blobs_path"):
        if type(root.get(path_key)) is not str or not str(root[path_key]).strip():
            raise ValueError(f"{field_name}.{path_key} must be a non-empty path")
    if "load_agent_annotations" not in root:
        raise ValueError(f"{field_name}.load_agent_annotations must be explicitly declared")
    if root.get("max_scenes") is not None:
        raise ValueError(f"{field_name}.max_scenes must be null")
    if root.get("window_stride") != 4 or type(root.get("window_stride")) is not int:
        raise ValueError(f"{field_name}.window_stride must be exactly 4")
    if domain == "real":
        if "annotations_require_trajectory_match" in root:
            raise ValueError(f"{field_name} real root forbids annotations_require_trajectory_match")
        return domain
    if root.get("annotations_require_trajectory_match") is not True:
        raise ValueError(f"{field_name} counterfactual root requires annotations_require_trajectory_match=true")
    if type(root.get("annotations_path")) is not str or not str(root["annotations_path"]).strip():
        raise ValueError(f"{field_name} counterfactual root requires annotations_path")
    if root.get("annotations_drop_distorted") is not True:
        raise ValueError(f"{field_name} counterfactual root requires annotations_drop_distorted=true")
    if root.get("pose_overlay_required") is not True:
        raise ValueError(f"{field_name} counterfactual root requires pose_overlay_required=true")
    if type(root.get("pose_overlay_path")) is not str or not str(root["pose_overlay_path"]).strip():
        raise ValueError(f"{field_name} counterfactual root requires pose_overlay_path")
    if root.get("pose_overlay_coord_frame") != "opencv_first_frame":
        raise ValueError(f"{field_name}.pose_overlay_coord_frame must be exactly 'opencv_first_frame'")
    pose_start = root.get("pose_overlay_txt_start_seconds")
    if isinstance(pose_start, bool) or not isinstance(pose_start, Real) or float(pose_start) != 0.0:
        raise ValueError(f"{field_name}.pose_overlay_txt_start_seconds must be exactly 0.0")
    if type(root.get("trajectory_quality_path")) is not str or not str(root["trajectory_quality_path"]).strip():
        raise ValueError(f"{field_name} counterfactual root requires trajectory_quality_path")
    if root.get("window_start_policy") != "counterfactual_scene_start":
        raise ValueError(f"{field_name}.window_start_policy must be exactly 'counterfactual_scene_start'")
    if root.get("tail_seconds") is not None:
        raise ValueError(f"{field_name}.tail_seconds must be null")
    return domain


def _navsim_e120_projection_identity(
    config: CVoIConfig,
    *,
    full_results_root: Path,
) -> tuple[str, str]:
    signature = config.ablation_signature
    if not isinstance(signature, CvoiFormalV2NavSimE120AblationSignature):
        raise ValueError("NavSim e120 direct root projection requires the typed ablation signature")
    if config.stage == "unguided_planner":
        return "p0", signature.p0_prefix_mode
    lineage = resolve_cvoi_manual_value_lineage(
        signature,
        stage=config.stage,
        full_results_root=full_results_root,
    )
    branch = "strict_real_only" if lineage.name == "no_cf" and config.stage != "stop_calibrated" else lineage.name
    stage = {
        "field_warmup": "field",
        "field_calibrated": "calibration",
        "guided_planner": "p1",
        "stop_calibrated": "stop",
    }.get(config.stage)
    if stage is None:
        raise ValueError(f"NavSim e120 stage={config.stage!r} does not consume a direct NavSim root projection")
    return stage, branch


def _validate_navsim_e120_roots(config: CVoIConfig, navsim: Optional[NavSimConfig]) -> None:
    if config.stage == "gate_distillation":
        if navsim is not None and (navsim.enabled or navsim.train_roots or navsim.val_roots):
            raise ValueError(
                "NavSim e120 Gate must consume only its Oracle artifact and must not configure NavSim roots"
            )
        return
    full_results_root = resolve_cvoi_manual_full_results_root_from_config(config)
    if navsim is None or navsim.enabled is not True:
        raise ValueError(f"NavSim e120 stage={config.stage!r} requires data.navsim.enabled=true")
    if navsim.max_agents != FORMAL_V2_NAVSIM_MAX_AGENTS:
        raise ValueError(f"NavSim e120 requires data.navsim.max_agents={FORMAL_V2_NAVSIM_MAX_AGENTS}")
    if navsim.max_scenes is not None or navsim.max_val_scenes is not None:
        raise ValueError("NavSim e120 requires max_scenes and max_val_scenes to be null")
    if navsim.window_stride != 4 or navsim.val_window_stride != 4:
        raise ValueError("NavSim e120 requires train and validation window_stride=4")

    roots = [*navsim.train_roots, *navsim.val_roots]
    train_domains = [
        _validate_navsim_e120_root(root, field_name=f"data.navsim.train_roots[{index}]")
        for index, root in enumerate(navsim.train_roots)
    ]
    val_domains = [
        _validate_navsim_e120_root(root, field_name=f"data.navsim.val_roots[{index}]")
        for index, root in enumerate(navsim.val_roots)
    ]
    resolved_capacity = validate_navsim_cvoi_geometry_contract(
        roots,
        default_max_agents=navsim.max_agents,
        default_load_agent_annotations=navsim.load_agent_annotations,
    )
    if resolved_capacity != FORMAL_V2_NAVSIM_MAX_AGENTS:
        raise ValueError(f"NavSim e120 roots must resolve max_agents={FORMAL_V2_NAVSIM_MAX_AGENTS}")
    signature = config.ablation_signature
    strict_real_field = (
        config.stage == "field_warmup"
        and isinstance(signature, CvoiFormalV2NavSimE120AblationSignature)
        and (signature.cf_field_supervision, signature.field_calibration_mode) == ("none", "local_geometry")
    )
    expected_warmup_domain = "real" if strict_real_field else "real_cf"
    if config.field_warmup_domain != expected_warmup_domain:
        raise ValueError(
            f"NavSim e120 mechanism requires cvoi.field_warmup_domain={expected_warmup_domain!r}, "
            f"got {config.field_warmup_domain!r}"
        )
    cf_field = config.stage == "field_warmup" and not strict_real_field
    if cf_field:
        allowed_field_mechanisms = {
            ("hazard_quality", "local_geometry"),
            ("hazard_only", "local_geometry"),
            ("quality_only", "local_geometry"),
        }
        mechanism = (signature.cf_field_supervision, signature.field_calibration_mode)
        if mechanism not in allowed_field_mechanisms:
            raise ValueError(
                "NavSim e120 Field producers are limited to full/hazard_only/quality_only/strict_real_only; "
                f"got mechanism={mechanism!r}"
            )
        expected_train = ["real", "counterfactual"]
        expected_val = ["real", "counterfactual"]
        expected_balance = True
    else:
        expected_train = ["real"]
        expected_val = ["real"]
        expected_balance = False
    if train_domains != expected_train or val_domains != expected_val:
        projection_name = "real-only" if expected_train == ["real"] and expected_val == ["real"] else "root projection"
        raise ValueError(
            f"NavSim e120 stage={config.stage!r} {projection_name} mismatch: "
            f"train={train_domains}, val={val_domains}, expected_train={expected_train}, expected_val={expected_val}"
        )
    if navsim.balance_train_roots is not expected_balance:
        raise ValueError(
            f"NavSim e120 stage={config.stage!r} requires " f"data.navsim.balance_train_roots={expected_balance}"
        )
    if config.stage == "evaluation":
        return
    projection_stage, projection_branch = _navsim_e120_projection_identity(
        config,
        full_results_root=full_results_root,
    )
    validate_formal_v2_navsim_direct_task_projection(
        projection_stage,
        projection_branch,
        {
            "train_roots": navsim.train_roots,
            "val_roots": navsim.val_roots,
            "balance_train_roots": navsim.balance_train_roots,
        },
        build_formal_v2_navsim_root_catalog(),
        full_results_root / "preflight",
    )


def _validate_cvoi_common_safety_fields(
    config: CVoIConfig,
    *,
    navsim_e120: bool,
) -> None:
    """Validate scalar, path, and Field-calibration invariants shared by CVoI protocols."""

    config.lambda_compute = _require_finite_nonnegative(
        config.lambda_compute,
        field_name="cvoi.lambda_compute",
    )
    if not config.lambda_grid:
        raise ValueError("cvoi.lambda_grid must contain at least one compute penalty")
    normalized_grid = [
        _require_finite_nonnegative(value, field_name="cvoi.lambda_grid") for value in config.lambda_grid
    ]
    if any(current >= following for current, following in zip(normalized_grid, normalized_grid[1:])):
        raise ValueError(f"cvoi.lambda_grid must be unique and strictly increasing, got {normalized_grid}")
    config.lambda_grid = normalized_grid
    if navsim_e120:
        if tuple(config.lambda_grid) != FORMAL_V2_NAVSIM_E120_LAMBDA_GRID:
            raise ValueError(
                "NavSim e120 cvoi.lambda_grid must be exactly " f"{list(FORMAL_V2_NAVSIM_E120_LAMBDA_GRID)!r}"
            )
        if config.lambda_compute != FORMAL_V2_NAVSIM_E120_DEFAULT_LAMBDA:
            raise ValueError(
                "NavSim e120 cvoi.lambda_compute must be exactly "
                f"{FORMAL_V2_NAVSIM_E120_DEFAULT_LAMBDA}, "
                f"got {config.lambda_compute}"
            )

    if len(config.compute_costs) != config.max_horizon + 1:
        raise ValueError(
            f"cvoi.compute_costs must contain H+1={config.max_horizon + 1} entries, got {config.compute_costs!r}"
        )
    config.compute_costs = [
        _require_finite_nonnegative(value, field_name="cvoi.compute_costs") for value in config.compute_costs
    ]
    if any(right < left for left, right in zip(config.compute_costs, config.compute_costs[1:])):
        raise ValueError(f"cvoi.compute_costs must be non-decreasing, got {config.compute_costs}")

    path_or_factory_fields = [
        "seed_planner_checkpoint",
        "unguided_planner_checkpoint",
        "field_checkpoint",
        "guided_planner_checkpoint",
        "dual_value_checkpoint",
        "oracle_path",
        "gate_checkpoint",
        "output_checkpoint",
        "world_model_checkpoint",
        "token_ae_checkpoint",
        "offline_adapter_factory",
        "offline_runtime_factory",
    ]
    for field_name in path_or_factory_fields:
        value = getattr(config, field_name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"cvoi.{field_name} must be a non-empty path when set, got {value!r}")
    for field_name in ("value_hidden_dim", "value_num_layers"):
        value = getattr(config, field_name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"cvoi.{field_name} must be a positive integer, got {value!r}")
    if isinstance(config.value_dropout, bool) or not isinstance(config.value_dropout, Real):
        raise ValueError(f"cvoi.value_dropout must be in [0, 1), got {config.value_dropout!r}")
    if not 0.0 <= float(config.value_dropout) < 1.0:
        raise ValueError(f"cvoi.value_dropout must be in [0, 1), got {config.value_dropout!r}")
    config.value_dropout = float(config.value_dropout)
    if config.tokens_per_frame is not None and (
        type(config.tokens_per_frame) is not int or config.tokens_per_frame <= 0
    ):
        raise ValueError(f"cvoi.tokens_per_frame must be a positive integer when set, got {config.tokens_per_frame!r}")
    if config.value_updates_per_epoch is not None and (
        type(config.value_updates_per_epoch) is not int or config.value_updates_per_epoch <= 0
    ):
        raise ValueError(
            "cvoi.value_updates_per_epoch must be a positive integer when set, "
            f"got {config.value_updates_per_epoch!r}"
        )
    if navsim_e120 and config.value_updates_per_epoch is not None:
        raise ValueError("NavSim e120 cvoi.value_updates_per_epoch must be null so Field consumes a complete epoch")
    if type(config.field_calibration_num_perturbations) is not int or config.field_calibration_num_perturbations <= 0:
        raise ValueError(
            "cvoi.field_calibration_num_perturbations must be a positive integer, "
            f"got {config.field_calibration_num_perturbations!r}"
        )
    config.field_calibration_perturbation_scale = _require_finite_positive(
        config.field_calibration_perturbation_scale,
        field_name="cvoi.field_calibration_perturbation_scale",
    )
    config.field_calibration_max_delta_norm = _require_finite_positive(
        config.field_calibration_max_delta_norm,
        field_name="cvoi.field_calibration_max_delta_norm",
    )
    config.field_calibration_order_margin = _require_finite_nonnegative(
        config.field_calibration_order_margin,
        field_name="cvoi.field_calibration_order_margin",
    )
    if navsim_e120:
        if config.field_calibration_num_perturbations != 4:
            raise ValueError("NavSim e120 cvoi.field_calibration_num_perturbations must be exactly 4")
        if config.field_calibration_perturbation_scale != 0.05:
            raise ValueError("NavSim e120 cvoi.field_calibration_perturbation_scale must be exactly 0.05")
        if config.field_calibration_order_margin != 0.1:
            raise ValueError("NavSim e120 cvoi.field_calibration_order_margin must be exactly 0.1")
    if (
        config.offline_adapter_factory == CVOI_NAVSIM_OFFLINE_ADAPTER_FACTORY
        and config.offline_runtime_factory is None
    ):
        raise ValueError(
            "the built-in NavSim CVoI offline adapter requires " "cvoi.offline_runtime_factory='module:factory'"
        )


def _validate_navsim_e120_stage_identity(config: CVoIConfig) -> None:
    if config.stage != "evaluation" and config.evaluation_mode != "controller":
        raise ValueError(
            f"NavSim e120 stage={config.stage!r} requires cvoi.evaluation_mode='controller', "
            f"got {config.evaluation_mode!r}"
        )
    value_guided_only_stages = {
        "unguided_planner",
        "guided_planner",
        "field_warmup",
        "field_calibrated",
    }
    if config.stage in value_guided_only_stages and config.controller_lineage != "value_guided":
        raise ValueError(
            f"NavSim e120 stage={config.stage!r} requires cvoi.controller_lineage='value_guided', "
            f"got {config.controller_lineage!r}"
        )
    if config.stage == "evaluation":
        lineage_by_forced_mode = {
            "p0_forced": "p0_controller",
            "p1_field_forced": "value_guided",
        }
        expected_lineage = lineage_by_forced_mode.get(config.evaluation_mode)
        if expected_lineage is not None and config.controller_lineage != expected_lineage:
            raise ValueError(
                f"NavSim e120 cvoi.evaluation_mode={config.evaluation_mode!r} requires "
                f"cvoi.controller_lineage={expected_lineage!r}, got {config.controller_lineage!r}"
            )

    signature = config.ablation_signature
    if not isinstance(signature, CvoiFormalV2NavSimE120AblationSignature):
        return
    if config.stage == "gate_distillation":
        resolve_cvoi_manual_gate_branch(signature)
    gate_variant_stage = config.stage == "gate_distillation" or (
        config.stage == "evaluation" and config.evaluation_mode == "controller"
    )
    if not gate_variant_stage and signature.gate_feature_mode != "full":
        raise ValueError(
            f"NavSim e120 stage={config.stage!r} requires cvoi.ablation_signature.gate_feature_mode='full'"
        )


def _validate_formal_v2_navsim_e120_config(config: CVoIConfig, navsim: Optional[NavSimConfig]) -> None:
    if type(config.enabled) is not bool or config.enabled is not True:
        raise ValueError("NavSim e120 profile requires cvoi.enabled=true")
    if config.ablation_mode != "manual_ablation":
        raise ValueError("NavSim e120 profile requires cvoi.ablation_mode='manual_ablation'")
    if not isinstance(config.ablation_signature, CvoiFormalV2NavSimE120AblationSignature):
        raise ValueError("NavSim e120 profile requires a cvoi_formal_v2_navsim_e120_ablation_v1 signature")
    if config.ablation_signature.protocol_version != config.protocol_version:
        raise ValueError("NavSim e120 ablation signature protocol_version does not match cvoi.protocol_version")
    fixed_fields = {
        "schema": CVOI_DUAL_VALUE_NAVSIM_E120_SCHEMA,
        "max_agents": FORMAL_V2_NAVSIM_MAX_AGENTS,
        "guidance_steps": 2,
        "guidance_objective": "last",
        "max_horizon": FORMAL_V2_NAVSIM_MAX_HORIZON,
        "rollout_horizons": list(range(FORMAL_V2_NAVSIM_MAX_HORIZON + 1)),
        "controller_batch_size": 1,
        "gate_training_batch_size": (
            NAVTRAIN_GATE_TRAINING_BATCH_SIZE if config.stage == "gate_distillation" else None
        ),
    }
    for field_name, expected in fixed_fields.items():
        actual = getattr(config, field_name)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"NavSim e120 profile requires cvoi.{field_name}={expected!r}, got {actual!r}")
    if config.stage not in _NAVSIM_E120_DIRECT_ARTIFACTS:
        raise ValueError(
            "NavSim e120 direct cvoi.stage must be one of "
            f"{sorted(_NAVSIM_E120_DIRECT_ARTIFACTS)}, got {config.stage!r}"
        )
    _validate_navsim_e120_stage_identity(config)

    model_stage = config.stage != "gate_distillation"
    if model_stage:
        if not isinstance(config.full_state_warmstart, CvoiFullStateWarmstartConfig):
            raise ValueError(f"NavSim e120 stage={config.stage!r} requires cvoi.full_state_warmstart")
        source_checkpoint_path = config.full_state_warmstart.source_checkpoint.path
        for field_name in ("world_model_checkpoint", "seed_planner_checkpoint"):
            actual = getattr(config, field_name)
            if type(actual) is not str or actual != source_checkpoint_path:
                raise ValueError(
                    f"NavSim e120 stage={config.stage!r} requires "
                    f"cvoi.{field_name} to match cvoi.full_state_warmstart.source_checkpoint.path="
                    f"{source_checkpoint_path!r}, got {actual!r}"
                )
    elif config.full_state_warmstart is not None:
        raise ValueError("NavSim e120 Gate forbids cvoi.full_state_warmstart")

    _validate_cvoi_common_safety_fields(
        config,
        navsim_e120=True,
    )
    _validate_navsim_e120_artifacts(config)
    _validate_navsim_e120_roots(config, navsim)


def validate_cvoi_config(config: CVoIConfig, navsim: Optional[NavSimConfig]) -> None:
    """Accept only the retained NavSim e120 training profile or disabled evaluation state."""

    if config.ablation_mode == "manual_ablation" and config.ablation_signature is None:
        raise ValueError("cvoi.ablation_mode='manual_ablation' requires cvoi.ablation_signature")
    if config.ablation_mode == "baseline" and config.ablation_signature is not None:
        raise ValueError("cvoi.ablation_signature requires cvoi.ablation_mode='manual_ablation'")
    if is_cvoi_formal_v2_navsim_e120_profile(config):
        _validate_formal_v2_navsim_e120_config(config, navsim)
        return
    if config.enabled:
        raise ValueError(
            "enabled cvoi.protocol_version must be exactly "
            f"{CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1!r}, got {config.protocol_version!r}"
        )
    if config.full_state_warmstart is not None:
        raise ValueError(
            "cvoi.full_state_warmstart requires " f"protocol_version={CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1!r}"
        )


def _validate_training_attributes(
    owner: object,
    expected_fields: Mapping[str, object],
    *,
    prefix: str,
    skip: frozenset[str] = frozenset(),
    expected_overrides: Optional[Mapping[str, object]] = None,
) -> None:
    overrides = {} if expected_overrides is None else dict(expected_overrides)
    for field_name, signed_expected in expected_fields.items():
        if field_name in skip:
            continue
        expected = overrides.get(field_name, signed_expected)
        actual = getattr(owner, field_name, None)
        if prefix == "model" and field_name == "vjepa_resolution":
            expected = tuple(expected)
        elif prefix == "data" and field_name == "crop_size":
            expected = (expected, expected) if type(expected) is int else tuple(expected)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"NavSim e120 training contract requires {prefix}.{field_name}={expected!r}, got {actual!r}"
            )


def _validate_formal_v2_navsim_e120_training_contract(
    config: CVoIConfig,
    *,
    method: Any,
    model: Any,
    train: Any,
    planner: Any,
    data: Any,
    world_model: Any,
    data_aug: Any,
    validation_suite: Any,
    ema: Any,
    segmentation: Any,
    loss: Any,
    multiview: Any,
    token_ae: Any,
    rl: Any,
    meta: Any,
    optimization: Any,
) -> None:
    if config.stage == "gate_distillation":
        return
    public = build_formal_v2_navsim_e120_public_config()
    compatibility = public["compatibility"]
    parser_defaults = public["parser_defaults"]
    preserved = public["preserved"]
    if method != compatibility["method"] or type(method) is not type(compatibility["method"]):
        raise ValueError(
            "NavSim e120 training contract requires " f"method={compatibility['method']!r}, got {method!r}"
        )
    _validate_training_attributes(model, compatibility["model"], prefix="model")
    _validate_training_attributes(model, parser_defaults["model"], prefix="model")
    _validate_training_attributes(train, compatibility["predictor_inputs"], prefix="train")
    _validate_training_attributes(
        train,
        compatibility["training_state"],
        prefix="train",
        expected_overrides={"predictor_planner_finetune": config.stage in CVOI_FORMAL_V2_NAVSIM_E120_PLANNER_STAGES},
    )
    _validate_training_attributes(
        train,
        parser_defaults["train"],
        prefix="train",
        expected_overrides={
            "predictor_validation_enabled": config.stage in CVOI_FORMAL_V2_NAVSIM_E120_PLANNER_STAGES,
        },
    )
    _validate_training_attributes(planner, compatibility["planner"], prefix="planner")
    _validate_training_attributes(planner, parser_defaults["planner"], prefix="planner")
    _validate_training_attributes(
        planner,
        {
            "tf_d_model": 256,
            "tf_d_ffn": 1024,
            "tf_num_layers": 3,
            "tf_num_head": 8,
            "tf_dropout": 0.0,
            "enable_rl_actor_critic": False,
            "rl_action_dim": 2,
        },
        prefix="planner",
    )
    _validate_training_attributes(train, {"num_encoder_frames": 4}, prefix="train")

    signed_data = {key: value for key, value in compatibility["data"].items() if not key.startswith("navsim_")}
    _validate_training_attributes(data, signed_data, prefix="data")
    _validate_training_attributes(
        data,
        {
            "datasets": [],
            "val_datasets": None,
            "camera_frame": False,
            "camera_views": ["left_mp4_path"],
            "stereo_view": False,
            "bench2drive": None,
            "mongo_raw": None,
        },
        prefix="data",
    )
    navsim = getattr(data, "navsim", None)
    signed_navsim = {
        key.removeprefix("navsim_"): value for key, value in compatibility["data"].items() if key.startswith("navsim_")
    }
    _validate_training_attributes(navsim, signed_navsim, prefix="data.navsim")
    _validate_training_attributes(
        navsim,
        {
            "enabled": True,
            "data_path": "",
            "sensor_blobs_path": "",
            "val_data_path": None,
            "val_sensor_blobs_path": None,
            "val_domain": None,
            "val_annotation_selection": "all_valid",
            "camera_names": ["CAM_F0"],
            "num_history_frames": None,
            "max_scenes": None,
            "max_val_scenes": None,
            "index_cache": True,
            "window_stride": 4,
            "val_window_stride": 4,
            "tail_seconds": None,
            "val_tail_seconds": None,
            "counterfactual_tail_seconds": None,
            "max_agents": FORMAL_V2_NAVSIM_MAX_AGENTS,
            "load_agent_annotations": True,
            "scene_filter_yaml": None,
            "val_scene_filter_yaml": None,
            "pose_overlay_path": None,
            "val_pose_overlay_path": None,
            "pose_overlay_coord_frame": "opencv_first_frame",
            "pose_overlay_required": False,
            "val_annotations_path": None,
            "val_annotations_drop_distorted": None,
        },
        prefix="data.navsim",
    )

    _validate_training_attributes(
        world_model,
        {
            "enabled": True,
            "sigreg_weight": 0.09,
            "sigreg_knots": 17,
            "sigreg_num_proj": 1024,
            "projector_hidden_dim": 2048,
            "embed_dim": 192,
            "num_subspaces": 1,
            "subspace_dim": None,
            "init_mode": "orthogonal_frozen",
            "theta": 0.0,
        },
        prefix="world_model",
    )
    _validate_training_attributes(rl, {"enabled": False}, prefix="rl")

    runtime_meta = {
        key: compatibility["runtime"][key]
        for key in ("dtype", "seed", "use_sdpa", "context_encoder_key", "target_encoder_key")
    }
    _validate_training_attributes(meta, runtime_meta, prefix="meta")
    _validate_training_attributes(
        meta,
        parser_defaults["meta"],
        prefix="meta",
        skip=frozenset({"selection_checkpoint_epochs"}),
    )
    _validate_training_attributes(meta, preserved["checkpoint_cadence"], prefix="meta")
    _validate_training_attributes(ema, preserved["ema"], prefix="ema")
    _validate_training_attributes(loss, preserved["loss"], prefix="loss")
    _validate_training_attributes(loss, {"auto_steps": None}, prefix="loss")
    _validate_training_attributes(data_aug, preserved["data_aug"], prefix="data_aug")
    _validate_training_attributes(optimization, parser_defaults["optimization"], prefix="optimization")
    if config.stage in CVOI_FORMAL_V2_NAVSIM_E120_PLANNER_STAGES:
        _validate_training_attributes(
            optimization,
            public["optimization"]["shared_adamw"],
            prefix="optimization",
        )

    runtime = compatibility["runtime"]
    segmentation_expected = {
        "use_segmentation": runtime["segmentation_enabled"],
        "seg_loss_weight": runtime["segmentation_loss_weight"],
    }
    _validate_training_attributes(segmentation, segmentation_expected, prefix="segmentation")
    _validate_training_attributes(token_ae, {"enabled": runtime["token_ae_enabled"]}, prefix="token_ae")
    if bool(getattr(multiview, "enabled", False)):
        raise ValueError("NavSim e120 training contract requires multiview.enabled=false")
    expected_cf_validation = config.stage == "field_warmup" and config.ablation_signature.cf_field_supervision in {
        "hazard_quality",
        "hazard_only",
        "quality_only",
    }
    if getattr(validation_suite, "enabled", None) is not expected_cf_validation:
        raise ValueError(
            "NavSim e120 stage/ablation requires validation_suite.enabled="
            f"{expected_cf_validation!r}, got {getattr(validation_suite, 'enabled', None)!r}"
        )


def _validate_formal_v2_navsim_e120_cross_section(
    config: CVoIConfig,
    *,
    method: Any,
    model: Any,
    value_guidance: Any,
    value_planning: Any,
    budget_controller: Any,
    train: Any,
    planner: Any,
    data: Any,
    world_model: Any,
    data_aug: Any,
    dynamic_rollout: Any,
    validation_suite: Any,
    ema: Any,
    segmentation: Any,
    loss: Any,
    multiview: Any,
    token_ae: Any,
    predictor_dit: Any,
    proposal: Any,
    counterfactual_supervision: Any,
    reward: Any,
    reward_selector: Any,
    wm_aux: Any,
    traj_opt: Any,
    rl: Any,
    meta: Any,
    optimization: Any,
) -> None:
    signature = config.ablation_signature
    if not isinstance(signature, CvoiFormalV2NavSimE120AblationSignature):
        raise ValueError("NavSim e120 profile requires its typed ablation signature")

    if bool(getattr(value_planning, "enabled", False)):
        raise ValueError("NavSim e120 profile forbids legacy value_planning.enabled=true")
    if bool(getattr(budget_controller, "enabled", False)):
        raise ValueError("NavSim e120 profile forbids legacy budget_controller.enabled=true")
    disabled_components = {
        "predictor_dit.masked_inpainting_enabled": getattr(predictor_dit, "masked_inpainting_enabled", None),
        "predictor_dit.joint_action_enabled": getattr(predictor_dit, "joint_action_enabled", None),
        "proposal.enabled": getattr(proposal, "enabled", None),
        "counterfactual_supervision.enabled": getattr(counterfactual_supervision, "enabled", None),
        "reward.enabled": getattr(reward, "enabled", None),
        "reward_selector.enabled": getattr(reward_selector, "enabled", None),
        "traj_opt.enabled": getattr(traj_opt, "enabled", None),
        "rl.enabled": getattr(rl, "enabled", None),
    }
    for field_name, actual in disabled_components.items():
        if actual is not False:
            raise ValueError(f"NavSim e120 profile requires {field_name}=false, got {actual!r}")
    _validate_training_attributes(
        wm_aux,
        {
            "multistep_discount": None,
            "reward_head_weight": 0.0,
            "contrastive_weight": 0.0,
        },
        prefix="wm_aux",
    )

    _validate_formal_v2_navsim_e120_training_contract(
        config,
        method=method,
        model=model,
        train=train,
        planner=planner,
        data=data,
        world_model=world_model,
        data_aug=data_aug,
        validation_suite=validation_suite,
        ema=ema,
        segmentation=segmentation,
        loss=loss,
        multiview=multiview,
        token_ae=token_ae,
        rl=rl,
        meta=meta,
        optimization=optimization,
    )

    planner_stage = (
        resolve_cvoi_formal_v2_navsim_e120_planner_stage(config)
        if config.stage in CVOI_FORMAL_V2_NAVSIM_E120_PLANNER_STAGES
        else None
    )
    if config.stage == "gate_distillation":
        if bool(getattr(value_guidance, "enabled", False)):
            raise ValueError("NavSim e120 Gate requires value_guidance.enabled=false")
        if getattr(meta, "selection_checkpoint_epochs", None) != ():
            raise ValueError("NavSim e120 Gate requires meta.selection_checkpoint_epochs to be empty")
        return

    meta_fields = {
        "seed": 239,
        "dtype": "bfloat16",
        "pretrain_repo": None,
        "resume_checkpoint": None,
        "pretrain_checkpoint": None,
        "pretrain_checkpoint_full": None,
        "predictor_checkpoint": None,
        "value_checkpoint": None,
        "planner_value_checkpoint": None,
        "ae_checkpoint": None,
        "load_encoder": False,
        "load_predictor": False,
        "load_planner": False,
        "load_seg": False,
        "context_encoder_key": "encoder",
        "target_encoder_key": "target_encoder",
        "save_every_freq": 10,
        "use_sdpa": True,
        "val_freq": 5,
        "resume_model_only": False,
        "auto_resume_latest": False,
    }
    for field_name, expected in meta_fields.items():
        actual = getattr(meta, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"NavSim e120 Planner profile requires meta.{field_name}={expected!r}, got {actual!r}")

    train_fields = {
        "encoder_train": False,
        "seg_head": False,
        "encoder_ema": False,
        "perceiver_ema": False,
        "predictor_train": False,
        "predictor_planner_finetune": planner_stage is not None,
        "use_states_for_predictor": False,
        "action_dim": 3,
        "state_dim": 8,
        "use_drive_command": False,
        "predictor_inference_consistent": True,
        "predictor_aux_policy": "inference_consistent",
        "use_parallel_predictor": False,
        "predictor_supervision_mode": "tf_ar",
        "predictor_loss_scope": "future_only",
        "predictor_use_z_ar_supervision": True,
        "predictor_no_aux_input": False,
        "num_observed_frames": 4,
        "predictor_type": "ac_transformer",
    }
    for field_name, expected in train_fields.items():
        actual = getattr(train, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"NavSim e120 profile requires train.{field_name}={expected!r}, got {actual!r}")
    expected_predictor_validation = planner_stage is not None
    if getattr(train, "predictor_validation_enabled", None) is not expected_predictor_validation:
        raise ValueError(
            "NavSim e120 stage="
            f"{config.stage!r} requires train.predictor_validation_enabled={expected_predictor_validation}"
        )

    planner_fields = {
        "use_planner": True,
        "planner_loss_weight": 1,
        "planner_type": "diffusion",
        "policy_output_source": "planner",
        "planner_input_source": "z_ar",
        "z_ar_mode": "full",
        "use_z_context": False,
        "observed_token_mode": "concat_type_embed",
        "use_observed_tokens": True,
        "use_action_history_for_planner": True,
        "action_history_dim": 3,
        "use_status_for_planner": True,
        "use_states_for_planner": True,
        "status_dim": 8,
        "split_status_embedding": False,
        "use_drive_command": False,
        "diff_train_prefix_conditioning": False,
        "diff_dt": 0.5,
    }
    for field_name, expected in planner_fields.items():
        actual = getattr(planner, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"NavSim e120 Planner profile requires planner.{field_name}={expected!r}, got {actual!r}")

    if bool(getattr(multiview, "enabled", False)):
        raise ValueError("NavSim e120 Planner profile requires multiview.enabled=false")
    if bool(getattr(token_ae, "enabled", False)):
        raise ValueError("NavSim e120 Planner profile requires token_ae.enabled=false")
    data_fields = {
        "batch_size": 16,
        "crop_size": (256, 256),
        "dataset_fpcs": [12],
        "fps": 2,
        "num_target_frames": 12,
        "num_workers": 4,
        "patch_size": 16,
        "persistent_workers": True,
        "pin_mem": True,
        "tubelet_size": 2,
        "use_tubelet_repeat": True,
    }
    for field_name, expected in data_fields.items():
        actual = getattr(data, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"NavSim e120 Planner profile requires data.{field_name}={expected!r}, got {actual!r}")
    navsim = getattr(data, "navsim", None)
    navsim_fields = {
        "camera_name": "CAM_F0",
        "image_require_policy": "observed_only",
        "max_frame_gap": 1,
    }
    for field_name, expected in navsim_fields.items():
        actual = getattr(navsim, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"NavSim e120 Planner profile requires data.navsim.{field_name}={expected!r}, got {actual!r}"
            )

    if getattr(dynamic_rollout, "enabled", None) is not True:
        raise ValueError("NavSim e120 Planner profile requires predictor_dynamic_rollout.enabled=true")
    expected_probabilities = FORMAL_V2_NAVSIM_P0_POLICIES[signature.p0_prefix_mode]
    dynamic_fields = {
        "full_prefix_prob": expected_probabilities[-1],
        "min_prefix_steps": 0,
        "max_non_full_prefix_steps": FORMAL_V2_NAVSIM_MAX_HORIZON - 1,
        "max_horizon": FORMAL_V2_NAVSIM_MAX_HORIZON,
    }
    for field_name, expected in dynamic_fields.items():
        actual = getattr(dynamic_rollout, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                "NavSim e120 Planner profile requires "
                f"predictor_dynamic_rollout.{field_name}={expected!r}, got {actual!r}"
            )
    if getattr(dynamic_rollout, "horizon_probabilities", None) != expected_probabilities:
        raise ValueError(
            f"NavSim e120 p0_prefix_mode={signature.p0_prefix_mode!r} requires "
            f"predictor_dynamic_rollout.horizon_probabilities={expected_probabilities!r}"
        )

    selection_epochs = getattr(meta, "selection_checkpoint_epochs", None)
    epochs = getattr(optimization, "epochs", None)
    schedule_epochs = getattr(optimization, "schedule_epochs", None)
    if planner_stage is not None:
        optimizer_fields = {
            "ipe": None,
            "optimizer": "adamw",
            "eps": 1e-8,
            "lr": 2e-4,
            "start_lr": 2e-5,
            "final_lr": 0.0,
            "weight_decay": 0.04,
            "final_weight_decay": 0.04,
            "enc_lr_scale": 0.001,
            "predictor_lr_scale": 0.1,
            "warmup": 15,
            "anneal": 15,
        }
        for field_name, expected in optimizer_fields.items():
            actual = getattr(optimization, field_name, None)
            if actual != expected or type(actual) is not type(expected):
                raise ValueError(
                    f"NavSim e120 Planner profile requires optimization.{field_name}={expected!r}, got {actual!r}"
                )
        betas = getattr(optimization, "betas", None)
        if type(betas) not in (list, tuple) or tuple(betas) != (0.9, 0.999):
            raise ValueError("NavSim e120 Planner profile requires optimization.betas=[0.9, 0.999]")
    elif selection_epochs != ():
        raise ValueError(f"NavSim e120 stage={config.stage!r} requires selection_checkpoint_epochs to be empty")

    if planner_stage == "p0":
        if schedule_epochs != 50 or type(schedule_epochs) is not int:
            raise ValueError(f"NavSim e120 P0 requires optimization.schedule_epochs=50, got {schedule_epochs!r}")
        if signature.p0_prefix_mode == "uniform":
            if epochs != 50 or type(epochs) is not int:
                raise ValueError(f"NavSim e120 Uniform P0 requires optimization.epochs=50, got {epochs!r}")
            if selection_epochs != FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS:
                raise ValueError(
                    "NavSim e120 Uniform P0 selection_checkpoint_epochs must be exactly "
                    f"{FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS!r}, got {selection_epochs!r}"
                )
        else:
            if type(epochs) is not int or epochs not in FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS:
                raise ValueError(
                    "NavSim e120 non-Uniform P0 optimization.epochs must equal a selected P0 candidate, "
                    f"got {epochs!r}"
                )
            if selection_epochs != ():
                raise ValueError("NavSim e120 non-Uniform P0 selection_checkpoint_epochs must be empty")
    elif planner_stage == "p1":
        if signature.p0_prefix_mode != "uniform":
            raise ValueError("NavSim e120 P1 must bind the selected Uniform-P0 lineage")
        if epochs != 80 or type(epochs) is not int:
            raise ValueError(f"NavSim e120 P1 requires optimization.epochs=80, got {epochs!r}")
        if schedule_epochs != 80 or type(schedule_epochs) is not int:
            raise ValueError(f"NavSim e120 P1 requires optimization.schedule_epochs=80, got {schedule_epochs!r}")
        if selection_epochs != FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS:
            raise ValueError(
                "NavSim e120 P1 selection_checkpoint_epochs must be exactly "
                f"{FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS!r}, got {selection_epochs!r}"
            )

    guidance_expected = (
        config.stage in {"guided_planner", "stop_calibrated", "evaluation"}
        and config.controller_lineage == "value_guided"
    )
    guidance_fields = {
        "enabled": guidance_expected,
        "steps": 2,
        "step_size": 0.05,
        "max_delta_norm": 0.25,
        "objective": "last",
        "detach_output": True,
    }
    for field_name, expected in guidance_fields.items():
        actual = getattr(value_guidance, field_name, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"NavSim e120 stage={config.stage!r} requires "
                f"value_guidance.{field_name}={expected!r}, got {actual!r}"
            )
    if config.guidance_steps != value_guidance.steps:
        raise ValueError(
            "NavSim e120 requires cvoi.guidance_steps to match value_guidance.steps, "
            f"got {config.guidance_steps!r} and {value_guidance.steps!r}"
        )
    if config.guidance_objective != value_guidance.objective:
        raise ValueError(
            "NavSim e120 requires cvoi.guidance_objective to match value_guidance.objective, "
            f"got {config.guidance_objective!r} and {value_guidance.objective!r}"
        )


def validate_cvoi_cross_section(
    config: CVoIConfig,
    *,
    method: Any,
    model: Any,
    value_guidance: Any,
    value_planning: Any,
    budget_controller: Any,
    train: Any,
    planner: Any,
    data: Any,
    world_model: Any,
    data_aug: Any,
    dynamic_rollout: Any,
    validation_suite: Any,
    ema: Any,
    segmentation: Any,
    loss: Any,
    multiview: Any,
    token_ae: Any,
    predictor_dit: Any,
    proposal: Any,
    counterfactual_supervision: Any,
    reward: Any,
    reward_selector: Any,
    wm_aux: Any,
    traj_opt: Any,
    rl: Any,
    meta: Any,
    optimization: Any,
) -> None:
    """Validate the e120 training contract or allow a disabled evaluation-only base."""

    if is_cvoi_formal_v2_navsim_e120_profile(config):
        _validate_formal_v2_navsim_e120_cross_section(
            config,
            method=method,
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
        return
    if config.enabled:
        raise ValueError(
            "enabled cvoi.protocol_version must be exactly "
            f"{CVOI_PROTOCOL_FORMAL_V2_NAVSIM_E120_V1!r}, got {config.protocol_version!r}"
        )
