"""Factual-authoritative predictor validation across rollout horizons and CF cohorts."""

from __future__ import annotations

import copy
from numbers import Real
from typing import Any, Callable, Dict, Mapping, Sequence

from app.vjepa_cowa_world_model.training.validation_suite import (
    PREDICTOR_H0_EXCLUSION,
    RolloutValidationProtocol,
    aggregate_horizon_metrics,
    condition_expected_weights_on_positive_horizons,
    enumerate_predictor_validation_protocols,
)

PREDICTOR_VALIDATION_SUITE_VERSION = "predictor_cf_robustness_validation_v1"


def build_predictor_validation_suite_signature(
    *,
    horizons: Sequence[int],
    expected_weights: Mapping[int, float],
    validation_data_semantics: Mapping[str, Any],
    val_roots: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Snapshot the fixed factual selector and diagnostic-only CF protocol."""

    protocols = enumerate_predictor_validation_protocols(horizons)
    positive_weights = condition_expected_weights_on_positive_horizons(horizons, expected_weights)
    domains = {str(root.get("domain")) for root in val_roots}
    if domains != {"real", "counterfactual"}:
        raise ValueError(f"predictor validation suite requires real and counterfactual roots, got {sorted(domains)}")
    return {
        "version": PREDICTOR_VALIDATION_SUITE_VERSION,
        "selector": {
            "domain": "real",
            "cohort": "all",
            "protocol": "full",
            "metric": "predictor_loss",
        },
        "protocols": [{"name": protocol.name, "horizon": protocol.horizon} for protocol in protocols],
        "h0_exclusion": PREDICTOR_H0_EXCLUSION,
        "expected_weights_given_h_gt_0": dict(positive_weights),
        "cohorts": {"real": ["all"], "counterfactual": ["all", "safe", "hazard"]},
        "metric_definitions": {
            "predictor_loss": "model-specific factual validation objective; checkpoint-authoritative only at full",
            "predictor_rollout_mse": "per-sample deployment rollout latent MSE; CF diagnostic only",
        },
        "validation_data_semantics": copy.deepcopy(dict(validation_data_semantics)),
        "validation_roots": [
            {key: copy.deepcopy(root.get(key)) for key in ("name", "dataset_id", "domain", "annotation_selection")}
            for root in val_roots
        ],
    }


def run_predictor_validation_suite(
    run_domain_protocol: Callable[[str, RolloutValidationProtocol], Mapping[str, Mapping[str, float]]],
    *,
    horizons: Sequence[int],
    expected_weights: Mapping[int, float],
    metric_directions: Mapping[str, str],
) -> Dict[str, Any]:
    """Run full plus positive predictor horizons for factual and CF data."""

    protocols = enumerate_predictor_validation_protocols(horizons)
    positive_weights = condition_expected_weights_on_positive_horizons(horizons, expected_weights)
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        "real": {"all": {}},
        "counterfactual": {"all": {}, "safe": {}, "hazard": {}},
    }
    for protocol in protocols:
        for domain in ("real", "counterfactual"):
            rows = run_domain_protocol(domain, protocol)
            required = ("all",) if domain == "real" else ("all", "safe", "hazard")
            if set(rows) != set(required):
                raise ValueError(
                    f"predictor validation cohorts mismatch for {domain}/{protocol.name}: "
                    f"expected={sorted(required)}, got={sorted(rows)}"
                )
            for cohort in required:
                metrics[domain][cohort][protocol.name] = dict(rows[cohort])

    aggregates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    positive_horizons = tuple(protocol.horizon for protocol in protocols if protocol.horizon is not None)
    for domain, cohorts in metrics.items():
        aggregates[domain] = {}
        for cohort, protocol_rows in cohorts.items():
            by_horizon = {horizon: protocol_rows[f"h{horizon}"] for horizon in positive_horizons}
            cohort_metric_keys = tuple(next(iter(by_horizon.values())))
            cohort_directions = {}
            for key in cohort_metric_keys:
                if key in metric_directions:
                    cohort_directions[key] = metric_directions[key]
                elif key.startswith("predictor_"):
                    cohort_directions[key] = "lower"
                else:
                    raise ValueError(f"missing predictor metric direction for {key!r}")
            summary = aggregate_horizon_metrics(
                by_horizon,
                expected_weights=positive_weights,
                metric_directions=cohort_directions,
            )
            summary["expected_given_h_gt_0"] = summary.pop("expected")
            aggregates[domain][cohort] = summary

    robustness: Dict[str, Dict[str, float]] = {}
    for protocol in protocols:
        real = float(metrics["real"]["all"][protocol.name]["predictor_rollout_mse"])
        counterfactual = float(metrics["counterfactual"]["all"][protocol.name]["predictor_rollout_mse"])
        if real == 0.0:
            raise ValueError(f"factual predictor_rollout_mse is zero for {protocol.name}")
        robustness[protocol.name] = {
            "cf_minus_real_predictor_rollout_mse": counterfactual - real,
            "cf_to_real_predictor_rollout_mse_ratio": counterfactual / real,
        }

    return {
        "metrics": metrics,
        "aggregates": aggregates,
        "robustness": robustness,
        "primary": dict(metrics["real"]["all"]["full"]),
    }


def select_predictor_diagnostic_metrics(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Drop accounting fields while retaining loss/error diagnostics for horizon curves."""

    selected: Dict[str, Dict[str, float]] = {}
    for cohort, row in rows.items():
        selected_row = {
            str(key): float(value)
            for key, value in row.items()
            if str(key).startswith("predictor_")
            and "num_tokens" not in str(key)
            and "num_samples" not in str(key)
            and not str(key).endswith("_batches")
            and not str(key).endswith("_failed_batches")
        }
        if "predictor_rollout_mse" not in selected_row:
            raise ValueError(f"predictor validation cohort {cohort!r} is missing predictor_rollout_mse")
        selected[str(cohort)] = selected_row
    return selected


def flatten_predictor_validation_suite_result(suite_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose legacy primary scalars while retaining the structured diagnostic suite."""

    primary = suite_result.get("primary")
    if not isinstance(primary, Mapping) or not primary:
        raise ValueError("predictor validation suite requires non-empty real/all/full primary metrics")
    output: Dict[str, Any] = {str(key): float(value) for key, value in primary.items()}
    output["predictor_validation_suite"] = suite_result

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(f"{prefix}/{key}", value[key])
            return
        if isinstance(value, Real) and not isinstance(value, bool):
            output[prefix] = float(value)

    for section in ("metrics", "aggregates", "robustness"):
        visit(f"predictor_validation_suite/{section}", suite_result[section])
    return output
