"""Contracts and aggregation for multi-domain rollout validation.

The full-rollout pass is always retained. Dynamic-rollout horizons are an
additional deterministic curve and never inherit probabilities from the
training prefix sampler.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.training.validation_rng import VALIDATION_RNG_CONTRACT_VERSION

VALIDATION_SUITE_VERSION = "dynamic_rollout_validation_v2"
PRIMARY_VALIDATION_SLICE = {"domain": "real", "cohort": "all", "protocol": "full"}
PREDICTOR_H0_EXCLUSION = "predictor loss has no future target"
_DIRECTIONS = {"lower", "higher"}
_COUNTERFACTUAL_COHORTS = {"all", "safe", "hazard"}
_VALIDATION_DATA_SEMANTIC_FIELDS = (
    "frames_per_clip",
    "fps",
    "num_observed_frames",
    "action_dim",
    "predictor_timeline",
    "planner_target_timeline",
    "metric_timestep_sec",
    "validation_transform",
    "proposal_transform",
    "validation_rng",
    "camera_name",
    "camera_names",
    "max_scenes",
    "window_stride",
    "max_frame_gap",
    "max_agents",
    "load_agent_annotations",
    "image_require_policy",
    "tail_seconds",
    "counterfactual_tail_seconds",
    "scene_filter_enabled",
    "pose_overlay_enabled",
    "pose_overlay_coord_frame",
    "pose_overlay_required",
)
_VALIDATION_DATA_SIGNATURE_FIELDS = (
    "frames_per_clip",
    "fps",
    "num_observed_frames",
    "action_dim",
    "predictor_timeline",
    "planner_target_timeline",
    "metric_timestep_sec",
    "validation_transform",
    "proposal_transform",
    "validation_rng",
)
_LOWER_IS_BETTER_BASE_METRICS = {
    "ade",
    "fde",
    "minade_k",
    "minfde_k",
    "l2_avg",
    "l2_point_avg",
    "collision_rate",
    "point_collision_rate",
}


def default_validation_metric_directions() -> Dict[str, str]:
    """Return the stable direction registry for every maintained open-loop scalar."""

    metric_names = set(_LOWER_IS_BETTER_BASE_METRICS)
    for second in (1, 2, 3, 4):
        metric_names.update(
            {
                f"l2_at_{second}s",
                f"l2_point_at_{second}s",
                f"collision_at_{second}s",
                f"point_collision_at_{second}s",
            }
        )
    directions = {name: "lower" for name in sorted(metric_names)}
    directions.update(
        {
            "longitudinal_progress_m": "higher",
            "forward_progress_m": "higher",
            "reverse_distance_m": "lower",
            "path_length_m": "lower",
            "progress_efficiency": "higher",
            "accel_mean_mps2": "lower",
            "accel_violation_rate": "lower",
            "jerk_mean_mps3": "lower",
            "jerk_violation_rate": "lower",
            "yaw_rate_mean_radps": "lower",
            "yaw_rate_violation_rate": "lower",
            "comfort_risk": "lower",
        }
    )
    return directions


@dataclass(frozen=True)
class RolloutValidationProtocol:
    """One full or cumulative-prefix validation pass."""

    name: str
    horizon: Optional[int]


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)):
        raise TypeError("validation horizons must be a sequence of integers")
    normalized = []
    for value in horizons:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"validation horizons must be integers, got {value!r}")
        horizon = int(value)
        if horizon < 0:
            raise ValueError(f"validation horizons must be non-negative, got {horizon}")
        normalized.append(horizon)
    if any(current >= following for current, following in zip(normalized, normalized[1:])):
        raise ValueError("validation horizons must be unique and strictly increasing, " f"got {normalized}")
    return tuple(normalized)


def enumerate_rollout_validation_protocols(horizons: Sequence[int]) -> tuple[RolloutValidationProtocol, ...]:
    """Return ``full`` followed by every configured cumulative prefix."""

    normalized = _validate_horizons(horizons)
    return (RolloutValidationProtocol(name="full", horizon=None),) + tuple(
        RolloutValidationProtocol(name=f"h{horizon}", horizon=horizon) for horizon in normalized
    )


def enumerate_predictor_validation_protocols(
    horizons: Sequence[int],
) -> tuple[RolloutValidationProtocol, ...]:
    """Enumerate predictor protocols while explicitly excluding target-free h0."""

    normalized = _validate_horizons(horizons)
    positive = tuple(horizon for horizon in normalized if horizon > 0)
    if not positive:
        raise ValueError("predictor validation suite requires at least one positive horizon")
    return (RolloutValidationProtocol(name="full", horizon=None),) + tuple(
        RolloutValidationProtocol(name=f"h{horizon}", horizon=horizon) for horizon in positive
    )


def resolve_validation_rollout_end_step(
    *,
    validation_horizon: Optional[int],
    observed_steps: int,
    total_steps: int,
    budget_rollout_end_step: Optional[int],
) -> Optional[int]:
    """Resolve the AC predictor rollout end for one deterministic suite protocol."""

    observed_steps = int(observed_steps)
    total_steps = int(total_steps)
    if observed_steps < 1 or total_steps <= observed_steps:
        raise ValueError(
            "validation rollout requires 0 < observed_steps < total_steps, "
            f"got observed_steps={observed_steps}, total_steps={total_steps}"
        )
    if validation_horizon is None:
        return budget_rollout_end_step
    if budget_rollout_end_step is not None:
        raise ValueError("budget rollout profile cannot be combined with a validation suite horizon")
    if isinstance(validation_horizon, bool) or not isinstance(validation_horizon, Integral):
        raise TypeError(f"validation horizon must be an integer or None, got {validation_horizon!r}")
    horizon = int(validation_horizon)
    max_future_steps = total_steps - observed_steps
    if horizon < 0:
        raise ValueError(f"validation horizon must be non-negative, got {horizon}")
    if horizon > max_future_steps:
        raise ValueError(f"validation horizon {horizon} exceeds available future steps {max_future_steps}")
    return observed_steps + horizon


def truncate_validation_future_tokens(
    future_tokens: torch.Tensor,
    *,
    validation_horizon: Optional[int],
    tokens_per_frame: int,
) -> torch.Tensor:
    """Cumulatively truncate parallel/latent predictor output for ``h`` validation."""

    if validation_horizon is None:
        return future_tokens
    if future_tokens.ndim != 3:
        raise ValueError(f"validation future tokens must have shape [B, N, D], got {future_tokens.shape}")
    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame < 1 or future_tokens.shape[1] % tokens_per_frame != 0:
        raise ValueError(
            "validation future tokens must be frame aligned, "
            f"tokens={future_tokens.shape[1]}, tokens_per_frame={tokens_per_frame}"
        )
    if isinstance(validation_horizon, bool) or not isinstance(validation_horizon, Integral):
        raise TypeError(f"validation horizon must be an integer or None, got {validation_horizon!r}")
    horizon = int(validation_horizon)
    available_steps = future_tokens.shape[1] // tokens_per_frame
    if horizon < 0:
        raise ValueError(f"validation horizon must be non-negative, got {horizon}")
    if horizon > available_steps:
        raise ValueError(f"validation horizon {horizon} exceeds available future steps {available_steps}")
    return future_tokens[:, : horizon * tokens_per_frame]


def flatten_validation_suite_result(suite_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep legacy primary fields while adding nested and stable slash diagnostics."""

    primary = suite_result.get("primary")
    if not isinstance(primary, Mapping) or not primary:
        raise ValueError("validation suite result requires non-empty primary metrics")
    output: Dict[str, Any] = {str(key): float(value) for key, value in primary.items()}
    output["validation_suite"] = suite_result

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(f"{prefix}/{key}", value[key])
            return
        if isinstance(value, Real) and not isinstance(value, bool):
            output[prefix] = float(value)

    for section in ("metrics", "aggregates"):
        if section in suite_result:
            visit(f"validation_suite/{section}", suite_result[section])
    return output


def require_checkpoint_validation_signature(
    checkpoint: Mapping[str, Any],
    *,
    key: str,
    expected: Mapping[str, Any],
    checkpoint_name: str,
    match_mode: str = "exact",
) -> None:
    """Reject checkpoint-selection history produced under different validation data/protocols."""

    stored = checkpoint.get(key)
    if stored is None:
        raise RuntimeError(f"checkpoint {checkpoint_name} is missing {key}")
    if match_mode == "exact":
        matches = stored == expected
    elif match_mode == "compatible":
        expected_projected = _project_checkpoint_validation_signature(
            expected,
            key=key,
            require_full_suite=False,
        )
        try:
            stored_projected = _project_checkpoint_validation_signature(
                stored,
                key=key,
                require_full_suite=True,
            )
        except (KeyError, TypeError, ValueError):
            matches = False
        else:
            matches = stored_projected == expected_projected
    else:
        raise ValueError(f"validation signature match_mode must be 'exact' or 'compatible', got {match_mode!r}")
    if not matches:
        raise RuntimeError(f"checkpoint {checkpoint_name} {key} mismatch")


def _project_checkpoint_validation_signature(
    signature: Any,
    *,
    key: str,
    require_full_suite: bool,
) -> Dict[str, Any]:
    """Project only the two approved cross-stage differences, then compare exactly."""

    if not isinstance(signature, Mapping):
        raise TypeError(f"{key} must be a mapping, got {type(signature).__name__}")
    projected = copy.deepcopy(signature)
    if key == "value_validation_signature" and "validation_data" in projected:
        projected["validation_data"] = build_validation_data_compatibility_signature(projected["validation_data"])
        return projected
    if key == "validation_suite_signature":
        suite_signature = projected
    else:
        suite_signature = projected.get("validation_suite_signature")
        if not isinstance(suite_signature, Mapping):
            raise ValueError(f"compatible {key} must contain validation_suite_signature")
    if require_full_suite:
        _require_full_compatibility_source_signature(suite_signature)
    suite_projected = build_validation_suite_compatibility_signature(suite_signature)
    if key == "validation_suite_signature":
        return suite_projected
    projected["validation_suite_signature"] = suite_projected
    return projected


def _require_full_compatibility_source_signature(signature: Mapping[str, Any]) -> None:
    roots = signature.get("val_roots")
    if isinstance(roots, (str, bytes)) or not isinstance(roots, Sequence) or not roots:
        raise ValueError("stored validation suite signature requires non-empty val_roots")
    for root in roots:
        if not isinstance(root, Mapping):
            raise ValueError(f"stored validation suite root must be a mapping, got {root!r}")
        sampling = root.get("sampling")
        input_semantics = root.get("input_semantics")
        if not isinstance(sampling, Mapping) or "window_stride" not in sampling:
            raise ValueError(f"stored validation suite root lacks sampling.window_stride: {root!r}")
        if not isinstance(input_semantics, Mapping) or "image_require_policy" not in input_semantics:
            raise ValueError(f"stored validation suite root lacks input_semantics.image_require_policy: {root!r}")


def _validate_expected_weights(
    horizons: tuple[int, ...],
    expected_weights: Mapping[int, float],
) -> Dict[int, float]:
    normalized: Dict[int, float] = {}
    for raw_horizon, raw_weight in expected_weights.items():
        if not isinstance(raw_horizon, Integral) or isinstance(raw_horizon, bool):
            raise TypeError(f"expected-weight horizon keys must be integers, got {raw_horizon!r}")
        horizon = int(raw_horizon)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise TypeError(f"expected horizon weights must be real numbers, got {raw_weight!r}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"expected horizon weights must be finite and non-negative, got {weight}")
        normalized[horizon] = weight
    if set(normalized) != set(horizons):
        raise ValueError(
            "expected horizon weight keys must exactly match metric horizons: "
            f"weights={sorted(normalized)}, horizons={list(horizons)}"
        )
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "expected horizon weights must sum to exactly 1; they are not implicitly renormalized, "
            f"got {sum(normalized.values())}"
        )
    return normalized


def condition_expected_weights_on_positive_horizons(
    horizons: Sequence[int],
    expected_weights: Mapping[int, float],
) -> Dict[int, float]:
    """Condition planner horizon probabilities on ``h > 0`` for predictor loss."""

    normalized_horizons = _validate_horizons(horizons)
    normalized = _validate_expected_weights(normalized_horizons, expected_weights)
    positive_mass = sum(weight for horizon, weight in normalized.items() if horizon > 0)
    if positive_mass <= 0.0:
        raise ValueError("predictor expected weights have zero positive-horizon mass")
    return {horizon: normalized[horizon] / positive_mass for horizon in normalized_horizons if horizon > 0}


def _validate_metric_table(
    metrics_by_horizon: Mapping[int, Mapping[str, float]],
    metric_directions: Mapping[str, str],
) -> tuple[tuple[int, ...], tuple[str, ...], Dict[int, Dict[str, float]], Dict[str, str]]:
    if not metrics_by_horizon:
        raise ValueError("horizon metric table must not be empty")
    horizons = _validate_horizons(list(metrics_by_horizon.keys()))
    first_keys: Optional[tuple[str, ...]] = None
    normalized_metrics: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        row = metrics_by_horizon[horizon]
        if not isinstance(row, Mapping) or not row:
            raise ValueError(f"horizon h={horizon} metrics must be a non-empty mapping")
        row_keys = tuple(sorted(str(key) for key in row))
        if first_keys is None:
            first_keys = row_keys
        elif row_keys != first_keys:
            raise ValueError(
                "every horizon must report the same metric keys, " f"expected={first_keys}, h={horizon} has={row_keys}"
            )
        normalized_row: Dict[str, float] = {}
        for key, raw_value in row.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(f"validation metric {key!r} at h={horizon} must be real, got {raw_value!r}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"validation metric {key!r} at h={horizon} is not finite: {value}")
            normalized_row[str(key)] = value
        normalized_metrics[horizon] = normalized_row
    metric_keys = first_keys or ()
    normalized_directions = {str(key): str(value) for key, value in metric_directions.items()}
    if set(normalized_directions) != set(metric_keys):
        raise ValueError(
            "metric direction keys must exactly match horizon metric keys: "
            f"directions={sorted(normalized_directions)}, metrics={list(metric_keys)}"
        )
    invalid_directions = {key: value for key, value in normalized_directions.items() if value not in _DIRECTIONS}
    if invalid_directions:
        raise ValueError(f"metric directions must be one of {_DIRECTIONS}, got {invalid_directions}")
    return horizons, metric_keys, normalized_metrics, normalized_directions


def aggregate_horizon_metrics(
    metrics_by_horizon: Mapping[int, Mapping[str, float]],
    *,
    expected_weights: Mapping[int, float],
    metric_directions: Mapping[str, str],
) -> Dict[str, Dict[str, float]]:
    """Report direction-aware expected, worst, and normalized trapezoid AUC."""

    horizons, metric_keys, metrics, directions = _validate_metric_table(
        metrics_by_horizon,
        metric_directions,
    )
    weights = _validate_expected_weights(horizons, expected_weights)
    result: Dict[str, Dict[str, float]] = {"expected": {}, "worst": {}, "auc": {}}
    for metric in metric_keys:
        values = [metrics[horizon][metric] for horizon in horizons]
        result["expected"][metric] = sum(weights[horizon] * metrics[horizon][metric] for horizon in horizons)
        result["worst"][metric] = max(values) if directions[metric] == "lower" else min(values)
        if len(horizons) == 1:
            result["auc"][metric] = values[0]
        else:
            area = sum(
                (right_horizon - left_horizon) * (left_value + right_value) * 0.5
                for left_horizon, right_horizon, left_value, right_value in zip(
                    horizons,
                    horizons[1:],
                    values,
                    values[1:],
                )
            )
            result["auc"][metric] = area / (horizons[-1] - horizons[0])
    return result


def run_rollout_validation_suite(
    run_protocol: Callable[
        [RolloutValidationProtocol],
        Mapping[str, Mapping[str, Mapping[str, float]]],
    ],
    *,
    horizons: Sequence[int],
    expected_weights: Mapping[int, float],
    metric_directions: Mapping[str, str],
) -> Dict[str, Any]:
    """Execute full + horizon protocols and organize authoritative/diagnostic slices."""

    protocols = enumerate_rollout_validation_protocols(horizons)
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    expected_slices: Optional[set[tuple[str, str]]] = None
    for protocol in protocols:
        protocol_result = run_protocol(protocol)
        if not isinstance(protocol_result, Mapping) or not protocol_result:
            raise ValueError(f"validation protocol {protocol.name!r} returned no domain metrics")
        observed_slices = {
            (str(domain), str(cohort)) for domain, cohorts in protocol_result.items() for cohort in cohorts
        }
        if expected_slices is None:
            expected_slices = observed_slices
        elif observed_slices != expected_slices:
            raise ValueError(
                "every validation protocol must report the same domain/cohort slices, "
                f"expected={sorted(expected_slices)}, protocol={protocol.name} has={sorted(observed_slices)}"
            )
        for domain, cohorts in protocol_result.items():
            if not isinstance(cohorts, Mapping) or not cohorts:
                raise ValueError(f"validation domain {domain!r} has no cohorts for protocol {protocol.name}")
            for cohort, raw_metrics in cohorts.items():
                if not isinstance(raw_metrics, Mapping) or not raw_metrics:
                    raise ValueError(f"validation slice {domain}/{cohort}/{protocol.name} must contain metrics")
                normalized = {}
                for key, raw_value in raw_metrics.items():
                    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                        raise TypeError(f"validation metric {domain}/{cohort}/{protocol.name}/{key} must be real")
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError(
                            f"validation metric {domain}/{cohort}/{protocol.name}/{key} is not finite: {value}"
                        )
                    normalized[str(key)] = value
                metrics.setdefault(str(domain), {}).setdefault(str(cohort), {})[protocol.name] = normalized

    if "counterfactual" in metrics:
        validate_counterfactual_diagnostics(metrics["counterfactual"])
    primary = select_primary_validation_metrics(metrics)

    aggregates: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    normalized_horizons = _validate_horizons(horizons)
    for domain, cohorts in metrics.items():
        for cohort, protocol_metrics in cohorts.items():
            horizon_metrics = {horizon: protocol_metrics[f"h{horizon}"] for horizon in normalized_horizons}
            metric_keys = set(next(iter(horizon_metrics.values())))
            missing_directions = sorted(metric_keys - set(metric_directions))
            if missing_directions:
                raise ValueError(
                    f"validation slice {domain}/{cohort} lacks metric directions for {missing_directions}"
                )
            slice_directions = {key: metric_directions[key] for key in metric_keys}
            aggregates.setdefault(domain, {})[cohort] = aggregate_horizon_metrics(
                horizon_metrics,
                expected_weights=expected_weights,
                metric_directions=slice_directions,
            )
    return {"metrics": metrics, "aggregates": aggregates, "primary": primary}


def select_primary_validation_metrics(
    suite_metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> Dict[str, float]:
    """Return the only checkpoint-authoritative slice: ``real/all/full``."""

    try:
        primary = suite_metrics["real"]["all"]["full"]
    except KeyError as exc:
        raise KeyError("validation suite is missing required primary slice real/all/full") from exc
    if not isinstance(primary, Mapping) or not primary:
        raise ValueError("validation suite primary slice real/all/full must contain metrics")
    return {str(key): float(value) for key, value in primary.items()}


def validate_counterfactual_diagnostics(
    diagnostics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    """Require all CF cohorts and reject collision metrics with invalid future geometry."""

    missing = sorted(_COUNTERFACTUAL_COHORTS - set(diagnostics))
    extra = sorted(set(diagnostics) - _COUNTERFACTUAL_COHORTS)
    if missing:
        raise KeyError(f"counterfactual diagnostics are missing cohorts: {missing}")
    if extra:
        raise KeyError(f"counterfactual diagnostics contain unknown cohorts: {extra}")
    for cohort, protocols in diagnostics.items():
        if "full" not in protocols:
            raise KeyError(f"counterfactual {cohort} diagnostics are missing full protocol")
        for protocol, metrics in protocols.items():
            forbidden_fragments = ("safety", "collision", "ttc", "offroad", "red_light")
            forbidden_keys = sorted(
                key for key in metrics if any(fragment in str(key).lower() for fragment in forbidden_fragments)
            )
            if forbidden_keys:
                raise ValueError(
                    "counterfactual diagnostics must not report unavailable safety metrics because future-agent "
                    f"geometry is invalid; cohort={cohort}, protocol={protocol}, keys={forbidden_keys}"
                )


def _normalize_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return int(value)


def _normalize_optional_nonnegative_float(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{field_name} must be a non-negative number or None, got {value!r}")
    return float(value)


def _normalize_positive_float(value: Any, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be a positive finite number, got {value!r}")
    return float(value)


def _normalize_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


def _normalize_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _normalize_stable_dataset_id(value: Any, *, field_name: str) -> str:
    dataset_id = _normalize_nonempty_string(value, field_name=field_name)
    if PurePosixPath(dataset_id).is_absolute() or PureWindowsPath(dataset_id).is_absolute():
        raise ValueError(f"stable {field_name} must not be an absolute machine path, got {dataset_id!r}")
    return dataset_id


def _normalize_strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool, got {value!r}")
    return value


def _normalize_camera_names(value: Any, *, field_name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a non-empty sequence of unique camera names, got {value!r}")
    names = [_normalize_nonempty_string(name, field_name=field_name) for name in value]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{field_name} must be a non-empty sequence of unique camera names, got {value!r}")
    return names


def _require_exact_mapping(value: Any, *, fields: Sequence[str], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")
    required = set(fields)
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise ValueError(f"{field_name} must contain exactly {sorted(required)}; missing={missing}, extra={extra}")
    return value


def _normalize_hw(value: Any, *, field_name: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-element height/width sequence, got {value!r}")
    return [
        _normalize_positive_integer(value[0], field_name=f"{field_name}[0]"),
        _normalize_positive_integer(value[1], field_name=f"{field_name}[1]"),
    ]


def _normalize_validation_transform(value: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")
    transform_type = _normalize_nonempty_string(value.get("type"), field_name=f"{field_name}.type")
    common_fields = ("type", "interpolation", "align_corners", "normalization")
    if transform_type == "vjepa":
        value = _require_exact_mapping(
            value,
            fields=common_fields + ("resolution", "crop_top_bottom"),
            field_name=field_name,
        )
        crop_top_bottom = _normalize_integer(value["crop_top_bottom"], field_name=f"{field_name}.crop_top_bottom")
        if crop_top_bottom < 0:
            raise ValueError(f"{field_name}.crop_top_bottom must be non-negative, got {crop_top_bottom}")
        normalized = {
            "type": transform_type,
            "resolution": _normalize_hw(value["resolution"], field_name=f"{field_name}.resolution"),
            "crop_top_bottom": crop_top_bottom,
        }
    elif transform_type == "vjepa_deterministic":
        value = _require_exact_mapping(
            value,
            fields=common_fields + ("crop_size", "resize_policy"),
            field_name=field_name,
        )
        normalized = {
            "type": transform_type,
            "crop_size": _normalize_hw(value["crop_size"], field_name=f"{field_name}.crop_size"),
            "resize_policy": _normalize_nonempty_string(
                value["resize_policy"], field_name=f"{field_name}.resize_policy"
            ),
        }
    elif transform_type == "dinov2":
        value = _require_exact_mapping(
            value,
            fields=common_fields
            + ("resolution", "resize_policy", "frame_selection", "frame_selection_stage", "frame_stride"),
            field_name=field_name,
        )
        normalized = {
            "type": transform_type,
            "resolution": _normalize_hw(value["resolution"], field_name=f"{field_name}.resolution"),
            "resize_policy": _normalize_nonempty_string(
                value["resize_policy"], field_name=f"{field_name}.resize_policy"
            ),
            "frame_selection": _normalize_nonempty_string(
                value["frame_selection"], field_name=f"{field_name}.frame_selection"
            ),
            "frame_selection_stage": _normalize_nonempty_string(
                value["frame_selection_stage"], field_name=f"{field_name}.frame_selection_stage"
            ),
            "frame_stride": _normalize_positive_integer(
                value["frame_stride"], field_name=f"{field_name}.frame_stride"
            ),
        }
    else:
        raise ValueError(f"{field_name}.type has unsupported value {transform_type!r}")
    normalized.update(
        {
            "interpolation": _normalize_nonempty_string(
                value["interpolation"], field_name=f"{field_name}.interpolation"
            ),
            "align_corners": _normalize_strict_bool(value["align_corners"], field_name=f"{field_name}.align_corners"),
            "normalization": _normalize_nonempty_string(
                value["normalization"], field_name=f"{field_name}.normalization"
            ),
        }
    )
    if transform_type == "dinov2":
        expected = {
            "resize_policy": "cover_then_center_crop_v1",
            "frame_selection": "end_of_chunk",
            "frame_selection_stage": "encoder_adapter",
            "interpolation": "bilinear",
            "align_corners": False,
            "normalization": "imagenet_rgb_255_v1",
        }
        actual = {key: normalized[key] for key in expected}
        if actual != expected:
            raise ValueError(f"{field_name} violates the DINOv2 transform contract: {actual} != {expected}")
    return normalized


def _normalize_proposal_transform(value: Any) -> Dict[str, Any]:
    field_name = "validation_data_semantics.proposal_transform"
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")
    enabled = _normalize_strict_bool(value.get("enabled"), field_name=f"{field_name}.enabled")
    expected_fields = ("enabled", "transform") if enabled else ("enabled",)
    value = _require_exact_mapping(value, fields=expected_fields, field_name=field_name)
    if not enabled:
        return {"enabled": False}
    transform = _normalize_validation_transform(value["transform"], field_name=f"{field_name}.transform")
    if transform["type"] != "vjepa":
        raise ValueError(f"{field_name}.transform must describe the separate V-JEPA proposal transform")
    return {"enabled": True, "transform": transform}


def _normalize_predictor_timeline(
    value: Any,
    *,
    frames_per_clip: int,
    num_observed_frames: int,
) -> Dict[str, int]:
    field_name = "validation_data_semantics.predictor_timeline"
    value = _require_exact_mapping(
        value,
        fields=("frame_stride", "total_steps", "observed_steps", "future_steps"),
        field_name=field_name,
    )
    normalized = {
        key: _normalize_positive_integer(value[key], field_name=f"{field_name}.{key}")
        for key in ("frame_stride", "total_steps", "observed_steps", "future_steps")
    }
    if normalized["total_steps"] != normalized["observed_steps"] + normalized["future_steps"]:
        raise ValueError(f"{field_name} total_steps must equal observed_steps + future_steps, got {normalized}")
    if normalized["total_steps"] * normalized["frame_stride"] != frames_per_clip:
        raise ValueError(f"{field_name} does not represent frames_per_clip={frames_per_clip}: {normalized}")
    if normalized["observed_steps"] * normalized["frame_stride"] != num_observed_frames:
        raise ValueError(f"{field_name} does not represent num_observed_frames={num_observed_frames}: {normalized}")
    return normalized


def _normalize_planner_target_timeline(
    value: Any,
    *,
    frames_per_clip: int,
    num_observed_frames: int,
) -> Dict[str, Any]:
    field_name = "validation_data_semantics.planner_target_timeline"
    value = _require_exact_mapping(
        value,
        fields=("predictor_inference_consistent", "future_start_index", "origin_index", "num_poses"),
        field_name=field_name,
    )
    inference_consistent = _normalize_strict_bool(
        value["predictor_inference_consistent"],
        field_name=f"{field_name}.predictor_inference_consistent",
    )
    future_start_index = _normalize_positive_integer(
        value["future_start_index"], field_name=f"{field_name}.future_start_index"
    )
    origin_index = _normalize_integer(value["origin_index"], field_name=f"{field_name}.origin_index")
    num_poses = _normalize_positive_integer(value["num_poses"], field_name=f"{field_name}.num_poses")
    expected_future_start = num_observed_frames if inference_consistent else 1
    expected = {
        "predictor_inference_consistent": inference_consistent,
        "future_start_index": expected_future_start,
        "origin_index": expected_future_start - 1,
        "num_poses": frames_per_clip - expected_future_start,
    }
    normalized = {
        "predictor_inference_consistent": inference_consistent,
        "future_start_index": future_start_index,
        "origin_index": origin_index,
        "num_poses": num_poses,
    }
    if normalized != expected:
        raise ValueError(f"{field_name} does not match the resolved planner GT timeline: {normalized} != {expected}")
    return normalized


def _normalize_validation_rng(value: Any) -> Dict[str, Any]:
    field_name = "validation_data_semantics.validation_rng"
    value = _require_exact_mapping(
        value,
        fields=("version", "base_seed", "stable_across_epochs"),
        field_name=field_name,
    )
    version = _normalize_nonempty_string(value["version"], field_name=f"{field_name}.version")
    if version != VALIDATION_RNG_CONTRACT_VERSION:
        raise ValueError(f"{field_name}.version must be {VALIDATION_RNG_CONTRACT_VERSION!r}, got {version!r}")
    stable_across_epochs = _normalize_strict_bool(
        value["stable_across_epochs"], field_name=f"{field_name}.stable_across_epochs"
    )
    if not stable_across_epochs:
        raise ValueError(f"{field_name}.stable_across_epochs must be true for the common-random validation contract")
    return {
        "version": version,
        "base_seed": _normalize_integer(value["base_seed"], field_name=f"{field_name}.base_seed"),
        "stable_across_epochs": True,
    }


def _normalize_validation_data_semantics(values: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"validation_data_semantics must be a mapping, got {type(values).__name__}")
    required = set(_VALIDATION_DATA_SEMANTIC_FIELDS)
    missing = sorted(required - set(values))
    extra = sorted(set(values) - required)
    if missing or extra:
        raise ValueError(
            "validation_data_semantics must contain exactly the stable validation-data fields; "
            f"missing={missing}, extra={extra}"
        )

    frames_per_clip = _normalize_positive_integer(
        values["frames_per_clip"], field_name="validation_data_semantics.frames_per_clip"
    )
    num_observed_frames = _normalize_positive_integer(
        values["num_observed_frames"], field_name="validation_data_semantics.num_observed_frames"
    )
    normalized = {
        "frames_per_clip": frames_per_clip,
        "fps": _normalize_positive_integer(values["fps"], field_name="validation_data_semantics.fps"),
        "num_observed_frames": num_observed_frames,
        "action_dim": _normalize_positive_integer(
            values["action_dim"], field_name="validation_data_semantics.action_dim"
        ),
        "predictor_timeline": _normalize_predictor_timeline(
            values["predictor_timeline"],
            frames_per_clip=frames_per_clip,
            num_observed_frames=num_observed_frames,
        ),
        "planner_target_timeline": _normalize_planner_target_timeline(
            values["planner_target_timeline"],
            frames_per_clip=frames_per_clip,
            num_observed_frames=num_observed_frames,
        ),
        "metric_timestep_sec": _normalize_positive_float(
            values["metric_timestep_sec"], field_name="validation_data_semantics.metric_timestep_sec"
        ),
        "validation_transform": _normalize_validation_transform(
            values["validation_transform"], field_name="validation_data_semantics.validation_transform"
        ),
        "proposal_transform": _normalize_proposal_transform(values["proposal_transform"]),
        "validation_rng": _normalize_validation_rng(values["validation_rng"]),
        "camera_name": _normalize_nonempty_string(
            values["camera_name"], field_name="validation_data_semantics.camera_name"
        ),
        "camera_names": _normalize_camera_names(
            values["camera_names"], field_name="validation_data_semantics.camera_names"
        ),
        "max_scenes": (
            None
            if values["max_scenes"] is None
            else _normalize_positive_integer(values["max_scenes"], field_name="validation_data_semantics.max_scenes")
        ),
        "window_stride": _normalize_positive_integer(
            values["window_stride"], field_name="validation_data_semantics.window_stride"
        ),
        "max_frame_gap": _normalize_positive_integer(
            values["max_frame_gap"], field_name="validation_data_semantics.max_frame_gap"
        ),
        "max_agents": _normalize_positive_integer(
            values["max_agents"], field_name="validation_data_semantics.max_agents"
        ),
        "load_agent_annotations": _normalize_strict_bool(
            values["load_agent_annotations"], field_name="validation_data_semantics.load_agent_annotations"
        ),
        "image_require_policy": _normalize_nonempty_string(
            values["image_require_policy"], field_name="validation_data_semantics.image_require_policy"
        ),
        "tail_seconds": _normalize_optional_nonnegative_float(
            values["tail_seconds"], field_name="validation_data_semantics.tail_seconds"
        ),
        "counterfactual_tail_seconds": _normalize_optional_nonnegative_float(
            values["counterfactual_tail_seconds"],
            field_name="validation_data_semantics.counterfactual_tail_seconds",
        ),
        "scene_filter_enabled": _normalize_strict_bool(
            values["scene_filter_enabled"], field_name="validation_data_semantics.scene_filter_enabled"
        ),
        "pose_overlay_enabled": _normalize_strict_bool(
            values["pose_overlay_enabled"], field_name="validation_data_semantics.pose_overlay_enabled"
        ),
        "pose_overlay_coord_frame": _normalize_nonempty_string(
            values["pose_overlay_coord_frame"],
            field_name="validation_data_semantics.pose_overlay_coord_frame",
        ),
        "pose_overlay_required": _normalize_strict_bool(
            values["pose_overlay_required"], field_name="validation_data_semantics.pose_overlay_required"
        ),
    }
    if normalized["num_observed_frames"] >= normalized["frames_per_clip"]:
        raise ValueError(
            "validation_data_semantics.num_observed_frames must be smaller than frames_per_clip, "
            f"got {normalized['num_observed_frames']} >= {normalized['frames_per_clip']}"
        )
    validation_transform = normalized["validation_transform"]
    if (
        validation_transform["type"] == "dinov2"
        and validation_transform["frame_stride"] != normalized["predictor_timeline"]["frame_stride"]
    ):
        raise ValueError(
            "validation_data_semantics.validation_transform.frame_stride="
            f"{validation_transform['frame_stride']} must match "
            "validation_data_semantics.predictor_timeline.frame_stride="
            f"{normalized['predictor_timeline']['frame_stride']}"
        )
    return normalized


def build_validation_suite_signature(
    *,
    horizons: Sequence[int],
    expected_weights: Mapping[int, float],
    metric_directions: Mapping[str, str],
    validation_data_semantics: Mapping[str, Any],
    val_roots: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a path-independent checkpoint guard for validation data and protocol semantics.

    ``validation_data_semantics`` contains resolved loader defaults. Overridable defaults are used only to
    derive each root's effective values; only clip-level, non-overridable fields remain at the top level.
    Artifact paths are reduced to enabled/disabled semantics, with ``dataset_id`` serving as their stable
    scene-filter, annotation, and pose-overlay version.
    """

    protocols = enumerate_rollout_validation_protocols(horizons)
    normalized_horizons = tuple(protocol.horizon for protocol in protocols if protocol.horizon is not None)
    weights = _validate_expected_weights(normalized_horizons, expected_weights)
    normalized_directions = {str(key): str(value) for key, value in sorted(metric_directions.items())}
    invalid_directions = {key: value for key, value in normalized_directions.items() if value not in _DIRECTIONS}
    if not normalized_directions or invalid_directions:
        raise ValueError(
            f"validation signature requires non-empty lower/higher metric directions, got {normalized_directions}"
        )
    normalized_data_semantics = _normalize_validation_data_semantics(validation_data_semantics)

    normalized_roots = []
    seen_names = set()
    seen_dataset_ids = set()
    for index, root in enumerate(val_roots):
        name = str(root.get("name", "")).strip()
        dataset_id = _normalize_stable_dataset_id(
            root.get("dataset_id", ""), field_name=f"validation root {name!r} dataset_id"
        )
        domain = str(root.get("domain", "")).strip()
        annotation_selection = str(root.get("annotation_selection", "all_valid")).strip()
        if not name or name in seen_names:
            raise ValueError(f"validation root {index} requires a unique non-empty name, got {name!r}")
        if not dataset_id or dataset_id in seen_dataset_ids:
            raise ValueError(f"validation root {name!r} requires a unique non-empty dataset_id, got {dataset_id!r}")
        if domain not in {"real", "counterfactual"}:
            raise ValueError(f"validation root {name!r} has invalid domain {domain!r}")
        if annotation_selection != "all_valid":
            raise ValueError(
                "validation suite roots must use annotation_selection='all_valid' so safe/hazard cohorts are "
                f"both observable, got root={name!r}, selection={annotation_selection!r}"
            )
        seen_names.add(name)
        seen_dataset_ids.add(dataset_id)

        root_tail_seconds = root.get("tail_seconds") if "tail_seconds" in root else None
        if "tail_seconds" not in root:
            root_tail_seconds = (
                normalized_data_semantics["counterfactual_tail_seconds"]
                if domain == "counterfactual"
                else normalized_data_semantics["tail_seconds"]
            )
        root_tail_seconds = _normalize_optional_nonnegative_float(
            root_tail_seconds, field_name=f"validation root {name!r}.tail_seconds"
        )
        root_max_scenes = root.get("max_scenes", normalized_data_semantics["max_scenes"])
        if root_max_scenes is not None:
            root_max_scenes = _normalize_positive_integer(
                root_max_scenes, field_name=f"validation root {name!r}.max_scenes"
            )
        root_window_stride = _normalize_positive_integer(
            root.get("window_stride", normalized_data_semantics["window_stride"]),
            field_name=f"validation root {name!r}.window_stride",
        )
        root_max_frame_gap = _normalize_positive_integer(
            root.get("max_frame_gap", normalized_data_semantics["max_frame_gap"]),
            field_name=f"validation root {name!r}.max_frame_gap",
        )
        scene_filter_enabled = (
            bool(root.get("scene_filter_yaml"))
            if "scene_filter_yaml" in root
            else normalized_data_semantics["scene_filter_enabled"]
        )
        root_camera_name = _normalize_nonempty_string(
            root.get("camera_name", normalized_data_semantics["camera_name"]),
            field_name=f"validation root {name!r}.camera_name",
        )
        root_camera_names = _normalize_camera_names(
            root.get("camera_names", normalized_data_semantics["camera_names"]),
            field_name=f"validation root {name!r}.camera_names",
        )
        root_max_agents = _normalize_positive_integer(
            root.get("max_agents", normalized_data_semantics["max_agents"]),
            field_name=f"validation root {name!r}.max_agents",
        )
        root_load_agent_annotations = _normalize_strict_bool(
            root.get("load_agent_annotations", normalized_data_semantics["load_agent_annotations"]),
            field_name=f"validation root {name!r}.load_agent_annotations",
        )
        root_image_require_policy = _normalize_nonempty_string(
            root.get("image_require_policy", normalized_data_semantics["image_require_policy"]),
            field_name=f"validation root {name!r}.image_require_policy",
        )
        annotations_enabled = bool(root.get("annotations_path"))
        annotations_drop_distorted = root.get("annotations_drop_distorted")
        if annotations_enabled and annotations_drop_distorted is not None:
            annotations_drop_distorted = _normalize_strict_bool(
                annotations_drop_distorted,
                field_name=f"validation root {name!r}.annotations_drop_distorted",
            )
        if not annotations_enabled:
            annotations_drop_distorted = None
        annotations_require_trajectory_match = root.get("annotations_require_trajectory_match", False)
        if annotations_enabled:
            annotations_require_trajectory_match = _normalize_strict_bool(
                annotations_require_trajectory_match,
                field_name=f"validation root {name!r}.annotations_require_trajectory_match",
            )
        else:
            annotations_require_trajectory_match = None
        pose_overlay_enabled = (
            bool(root.get("pose_overlay_path"))
            if "pose_overlay_path" in root
            else normalized_data_semantics["pose_overlay_enabled"]
        )
        if pose_overlay_enabled:
            pose_overlay_coord_frame = _normalize_nonempty_string(
                root.get("pose_overlay_coord_frame", normalized_data_semantics["pose_overlay_coord_frame"]),
                field_name=f"validation root {name!r}.pose_overlay_coord_frame",
            )
            pose_overlay_required = _normalize_strict_bool(
                root.get("pose_overlay_required", normalized_data_semantics["pose_overlay_required"]),
                field_name=f"validation root {name!r}.pose_overlay_required",
            )
        else:
            pose_overlay_coord_frame = None
            pose_overlay_required = None
        normalized_roots.append(
            {
                "name": name,
                "dataset_id": dataset_id,
                "domain": domain,
                "annotation_selection": annotation_selection,
                "sampling": {
                    "tail_seconds": root_tail_seconds,
                    "window_stride": root_window_stride,
                    "max_scenes": root_max_scenes,
                    "max_frame_gap": root_max_frame_gap,
                    "scene_filter": {
                        "enabled": scene_filter_enabled,
                        "dataset_version": dataset_id if scene_filter_enabled else None,
                    },
                },
                "input_semantics": {
                    "camera_name": root_camera_name,
                    "camera_names": root_camera_names,
                    "image_require_policy": root_image_require_policy,
                    "max_agents": root_max_agents,
                    "load_agent_annotations": root_load_agent_annotations,
                },
                "annotations": {
                    "enabled": annotations_enabled,
                    "dataset_version": dataset_id if annotations_enabled else None,
                    "drop_distorted": annotations_drop_distorted,
                    **({"require_trajectory_match": True} if annotations_require_trajectory_match else {}),
                },
                "pose_overlay": {
                    "enabled": pose_overlay_enabled,
                    "dataset_version": dataset_id if pose_overlay_enabled else None,
                    "coord_frame": pose_overlay_coord_frame,
                    "required": pose_overlay_required,
                },
            }
        )
    domains = {root["domain"] for root in normalized_roots}
    if domains != {"real", "counterfactual"}:
        raise ValueError(
            "validation suite requires both real and counterfactual roots, " f"got domains={sorted(domains)}"
        )

    return {
        "version": VALIDATION_SUITE_VERSION,
        "protocols": [{"name": protocol.name, "horizon": protocol.horizon} for protocol in protocols],
        "expected_horizon_weights": [
            {"horizon": horizon, "weight": weights[horizon]} for horizon in normalized_horizons
        ],
        "metric_directions": normalized_directions,
        "primary": dict(PRIMARY_VALIDATION_SLICE),
        "counterfactual_cohorts": sorted(_COUNTERFACTUAL_COHORTS),
        "counterfactual_safety": {
            "status": "unavailable",
            "reason": "future-agent geometry is not trustworthy",
        },
        "auc_definition": "normalized_trapezoid_over_cumulative_prefix_steps",
        "worst_definition": "direction_aware_extremum_over_configured_horizons",
        "validation_data_semantics": {
            key: normalized_data_semantics[key] for key in _VALIDATION_DATA_SIGNATURE_FIELDS
        },
        "val_roots": normalized_roots,
    }


def build_validation_data_signature(
    validation_data_semantics: Mapping[str, Any],
    val_roots: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the path-independent data portion shared by all validation protocols."""

    suite_signature = build_validation_suite_signature(
        horizons=[0],
        expected_weights={0: 1.0},
        metric_directions={"_data_contract": "lower"},
        validation_data_semantics=validation_data_semantics,
        val_roots=val_roots,
    )
    return {
        "validation_data_semantics": suite_signature["validation_data_semantics"],
        "val_roots": suite_signature["val_roots"],
    }


def build_validation_data_compatibility_signature(signature: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the two approved cross-stage validation-data sampling differences."""

    if set(signature) != {"validation_data_semantics", "val_roots"}:
        raise ValueError(
            "validation data signature fields mismatch: "
            f"expected=['val_roots', 'validation_data_semantics'], got={sorted(signature)}"
        )
    projected = copy.deepcopy(dict(signature))
    roots = projected["val_roots"]
    if isinstance(roots, (str, bytes)) or not isinstance(roots, Sequence) or not roots:
        raise ValueError("validation data signature requires non-empty val_roots")
    for root in roots:
        try:
            del root["sampling"]["window_stride"]
            del root["input_semantics"]["image_require_policy"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"validation data root lacks compatibility semantics: {root!r}") from exc
    return projected


def build_validation_suite_compatibility_signature(signature: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract invariant source-artifact provenance while allowing intentional stage sampling changes.

    Trackers and same-stage resume compare the full signature exactly. Cross-stage predictor/value consumers use
    this subset because Stage1/2 and Stage3 intentionally validate the same versioned datasets with different
    window strides and image requirements.
    """

    required_top_level = (
        "version",
        "protocols",
        "expected_horizon_weights",
        "metric_directions",
        "primary",
        "counterfactual_cohorts",
        "counterfactual_safety",
        "auc_definition",
        "worst_definition",
        "validation_data_semantics",
        "val_roots",
    )
    missing = [key for key in required_top_level if key not in signature]
    extra = sorted(set(signature) - set(required_top_level))
    if missing or extra:
        raise ValueError(f"validation suite signature compatibility fields mismatch: missing={missing}, extra={extra}")
    if signature["version"] != VALIDATION_SUITE_VERSION:
        raise ValueError(
            f"validation suite compatibility requires version {VALIDATION_SUITE_VERSION!r}, "
            f"got {signature['version']!r}"
        )

    roots = []
    for root in signature["val_roots"]:
        try:
            compatible_root = copy.deepcopy(root)
            has_window_stride = "window_stride" in compatible_root["sampling"]
            has_image_require_policy = "image_require_policy" in compatible_root["input_semantics"]
            if has_window_stride != has_image_require_policy:
                raise ValueError(
                    "validation compatibility roots must either include or omit both window_stride and "
                    "image_require_policy"
                )
            if has_window_stride:
                del compatible_root["sampling"]["window_stride"]
                del compatible_root["input_semantics"]["image_require_policy"]
            roots.append(compatible_root)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"validation root lacks compatibility semantics: {root!r}") from exc

    return {key: copy.deepcopy(signature[key]) for key in required_top_level if key != "val_roots"} | {
        "val_roots": roots
    }
