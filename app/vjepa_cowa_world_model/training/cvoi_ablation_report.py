"""Descriptive paired statistics for direct NavSim CVoI EPDMS runs."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence
from uuid import uuid4

from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import CVOI_DIRECT_EPDMS_BRANCHES, DirectEpdmsSceneRecord

_EXPECTED_EVALUATION_SEED = 239
_CONTROLLER_BRANCHES = frozenset(
    {
        "full",
        "no_cf",
        "without_field",
        "without_stop",
        "without_value_summary",
    }
)
_FORCED_BRANCHES = frozenset({"hazard_only", "quality_only"})
_FORCED_BRANCH_ORDER = ("hazard_only", "quality_only")
_CONTROLLER_ABLATIONS = frozenset(
    {
        "no_cf",
        "without_field",
        "without_stop",
        "without_value_summary",
    }
)
_DIRECT_METRICS = frozenset({"epdms", "total_latency_ms", "final_horizon"})
_FORCED_METRICS = frozenset({"epdms", "total_latency_ms"})
_INTERVAL_METHOD = "paired_scenario_percentile"
_EVIDENCE_STATUS = "exploratory_single_trained_lineage"


def _require_exact_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class DirectEpdmsResultRun:
    """One validated result set from the retained direct EPDMS evaluator."""

    branch: str
    forced_horizon: int | None
    evaluation_seed: int
    records: tuple[DirectEpdmsSceneRecord, ...]

    def __post_init__(self) -> None:
        if self.branch not in CVOI_DIRECT_EPDMS_BRANCHES:
            raise ValueError(f"unknown direct EPDMS branch: {self.branch!r}")
        if type(self.evaluation_seed) is not int or self.evaluation_seed != _EXPECTED_EVALUATION_SEED:
            raise ValueError("direct EPDMS report requires evaluation_seed=239")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple")
        if not self.records:
            raise ValueError("direct EPDMS result run must contain records")
        if any(not isinstance(record, DirectEpdmsSceneRecord) for record in self.records):
            raise TypeError("records must contain only DirectEpdmsSceneRecord values")

        tokens = tuple(record.scenario_token for record in self.records)
        if tokens != tuple(sorted(tokens)) or len(tokens) != len(set(tokens)):
            raise ValueError("direct EPDMS scenario tokens must be sorted and unique")
        if any(record.branch != self.branch for record in self.records):
            raise ValueError("direct EPDMS record branch differs from its result binding")
        if any(record.evaluation_seed != self.evaluation_seed for record in self.records):
            raise ValueError("direct EPDMS record evaluation seed differs from its result binding")

        if self.branch in _CONTROLLER_BRANCHES:
            if self.forced_horizon is not None:
                raise ValueError("controller result forbids forced_horizon")
            return
        if type(self.forced_horizon) is not int or self.forced_horizon not in range(5):
            raise ValueError("forced result requires one forced_horizon in H0--H4")
        if any(record.final_horizon != self.forced_horizon for record in self.records):
            raise ValueError("forced result records differ from their directory horizon")


@dataclass(frozen=True)
class DirectEpdmsPairedComparison:
    """One same-scenario descriptive contrast, expressed as reference minus comparison."""

    reference_branch: str
    comparison_branch: str
    comparison_kind: str
    metric: str
    forced_horizon: int | None
    evaluation_seed: int
    scenario_count: int
    mean_delta: float
    std_delta: float
    ci_low: float
    ci_high: float
    confidence_level: float
    bootstrap_seed: int
    bootstrap_replicates: int
    interval_method: str = _INTERVAL_METHOD
    evidence_status: str = _EVIDENCE_STATUS

    def __post_init__(self) -> None:
        if self.reference_branch not in CVOI_DIRECT_EPDMS_BRANCHES:
            raise ValueError(f"unknown reference branch: {self.reference_branch!r}")
        if self.comparison_branch not in CVOI_DIRECT_EPDMS_BRANCHES:
            raise ValueError(f"unknown comparison branch: {self.comparison_branch!r}")
        if self.comparison_kind not in {
            "controller_full_minus_ablation",
            "forced_hazard_minus_quality_same_horizon",
        }:
            raise ValueError(f"unknown direct EPDMS comparison kind: {self.comparison_kind!r}")
        if self.metric not in _DIRECT_METRICS:
            raise ValueError(f"unsupported direct EPDMS metric: {self.metric!r}")
        if self.comparison_kind == "forced_hazard_minus_quality_same_horizon" and self.metric not in _FORCED_METRICS:
            raise ValueError("forced direct EPDMS comparisons support only EPDMS and total latency")
        if type(self.forced_horizon) is int:
            if self.forced_horizon not in range(5):
                raise ValueError("forced_horizon must be in H0--H4")
        elif self.forced_horizon is not None:
            raise TypeError("forced_horizon must be an integer or None")
        if type(self.evaluation_seed) is not int or self.evaluation_seed != _EXPECTED_EVALUATION_SEED:
            raise ValueError("direct EPDMS comparison requires evaluation_seed=239")
        _require_exact_int(self.scenario_count, field="scenario_count")
        if self.scenario_count < 1:
            raise ValueError("scenario_count must be positive")
        for field in ("mean_delta", "std_delta", "ci_low", "ci_high", "confidence_level"):
            object.__setattr__(
                self,
                field,
                _require_finite_number(getattr(self, field), field=field),
            )
        if self.std_delta < 0.0:
            raise ValueError("std_delta must be non-negative")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        _require_exact_int(self.bootstrap_seed, field="bootstrap_seed")
        _require_exact_int(self.bootstrap_replicates, field="bootstrap_replicates")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if self.interval_method != _INTERVAL_METHOD:
            raise ValueError(f"interval_method must be {_INTERVAL_METHOD!r}")
        if self.evidence_status != _EVIDENCE_STATUS:
            raise ValueError(f"evidence_status must be {_EVIDENCE_STATUS!r}")

    def as_dict(self) -> dict[str, object]:
        """Return the comparison as a JSON-safe mapping."""

        return asdict(self)


@dataclass(frozen=True)
class DirectEpdmsForcedHorizonSummary:
    """Absolute descriptive means for one forced branch and horizon."""

    branch: str
    horizon: int
    evaluation_seed: int
    scenario_count: int
    mean_epdms: float
    mean_total_latency_ms: float
    evidence_status: str = _EVIDENCE_STATUS

    def __post_init__(self) -> None:
        if self.branch not in _FORCED_BRANCHES:
            raise ValueError(f"forced curve branch must be one of {sorted(_FORCED_BRANCHES)}")
        _require_exact_int(self.horizon, field="horizon")
        if self.horizon not in range(5):
            raise ValueError("horizon must be in H0--H4")
        if type(self.evaluation_seed) is not int or self.evaluation_seed != _EXPECTED_EVALUATION_SEED:
            raise ValueError("forced curve requires evaluation_seed=239")
        _require_exact_int(self.scenario_count, field="scenario_count")
        if self.scenario_count < 1:
            raise ValueError("scenario_count must be positive")
        object.__setattr__(
            self,
            "mean_epdms",
            _require_finite_number(self.mean_epdms, field="mean_epdms"),
        )
        object.__setattr__(
            self,
            "mean_total_latency_ms",
            _require_finite_number(self.mean_total_latency_ms, field="mean_total_latency_ms"),
        )
        if not 0.0 <= self.mean_epdms <= 1.0:
            raise ValueError("mean_epdms must be in [0, 1]")
        if self.mean_total_latency_ms < 0.0:
            raise ValueError("mean_total_latency_ms must be non-negative")
        if self.evidence_status != _EVIDENCE_STATUS:
            raise ValueError(f"evidence_status must be {_EVIDENCE_STATUS!r}")

    def as_dict(self) -> dict[str, object]:
        """Return the curve row as a JSON-safe mapping."""

        return asdict(self)


def _comparison_identity(
    reference: DirectEpdmsResultRun,
    comparison: DirectEpdmsResultRun,
) -> tuple[str, int | None]:
    if (
        reference.branch == "full"
        and comparison.branch in _CONTROLLER_ABLATIONS
        and reference.forced_horizon is None
        and comparison.forced_horizon is None
    ):
        return "controller_full_minus_ablation", None
    if (
        reference.branch == "hazard_only"
        and comparison.branch == "quality_only"
        and reference.forced_horizon is not None
        and reference.forced_horizon == comparison.forced_horizon
    ):
        return "forced_hazard_minus_quality_same_horizon", reference.forced_horizon
    raise ValueError(
        "direct EPDMS comparison must be Full minus one controller ablation or "
        "hazard-only minus quality-only at the same forced horizon"
    )


def _validated_metrics(metrics: Sequence[str], *, comparison_kind: str) -> tuple[str, ...]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise TypeError("metrics must be a non-empty sequence")
    result = tuple(metrics)
    if not result:
        raise ValueError("metrics must be a non-empty sequence")
    if any(type(metric) is not str for metric in result):
        raise TypeError("metrics must contain only strings")
    if len(result) != len(set(result)):
        raise ValueError("metrics must be unique")
    unsupported = set(result) - _DIRECT_METRICS
    if unsupported:
        raise ValueError(f"unsupported direct EPDMS metrics: {sorted(unsupported)}")
    if comparison_kind == "forced_hazard_minus_quality_same_horizon" and set(result) - _FORCED_METRICS:
        raise ValueError("forced direct EPDMS comparisons support only EPDMS and total latency")
    return result


def _record_metric(record: DirectEpdmsSceneRecord, metric: str) -> float:
    if metric == "epdms":
        return record.epdms
    if metric == "total_latency_ms":
        return record.latency_ms
    if metric == "final_horizon":
        return float(record.final_horizon)
    raise ValueError(f"unsupported direct EPDMS metric: {metric!r}")


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compare_direct_epdms_runs(
    *,
    reference: DirectEpdmsResultRun,
    comparison: DirectEpdmsResultRun,
    metrics: Sequence[str],
    bootstrap_seed: int = 239,
    bootstrap_replicates: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[DirectEpdmsPairedComparison, ...]:
    """Compare two permitted direct runs using paired NavTest scenario resampling."""

    if not isinstance(reference, DirectEpdmsResultRun) or not isinstance(comparison, DirectEpdmsResultRun):
        raise TypeError("reference and comparison must be DirectEpdmsResultRun values")
    comparison_kind, forced_horizon = _comparison_identity(reference, comparison)
    selected_metrics = _validated_metrics(metrics, comparison_kind=comparison_kind)
    if reference.evaluation_seed != comparison.evaluation_seed:
        raise ValueError("paired direct EPDMS runs must use the same evaluation seed")
    _require_exact_int(bootstrap_seed, field="bootstrap_seed")
    _require_exact_int(bootstrap_replicates, field="bootstrap_replicates")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    confidence_level = _require_finite_number(confidence_level, field="confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    reference_by_token = {record.scenario_token: record for record in reference.records}
    comparison_by_token = {record.scenario_token: record for record in comparison.records}
    if set(reference_by_token) != set(comparison_by_token):
        raise ValueError("paired direct EPDMS runs must use the same scenario-token set")
    tokens = tuple(sorted(reference_by_token))
    scenario_count = len(tokens)

    rng = random.Random(bootstrap_seed)
    resample_indices = tuple(
        tuple(rng.randrange(scenario_count) for _ in range(scenario_count)) for _ in range(bootstrap_replicates)
    )
    alpha = (1.0 - confidence_level) / 2.0
    results = []
    for metric in selected_metrics:
        deltas = tuple(
            _record_metric(reference_by_token[token], metric) - _record_metric(comparison_by_token[token], metric)
            for token in tokens
        )
        bootstrap_means = tuple(fmean(deltas[index] for index in indices) for indices in resample_indices)
        results.append(
            DirectEpdmsPairedComparison(
                reference_branch=reference.branch,
                comparison_branch=comparison.branch,
                comparison_kind=comparison_kind,
                metric=metric,
                forced_horizon=forced_horizon,
                evaluation_seed=reference.evaluation_seed,
                scenario_count=scenario_count,
                mean_delta=fmean(deltas),
                std_delta=stdev(deltas) if scenario_count > 1 else 0.0,
                ci_low=_percentile(bootstrap_means, alpha),
                ci_high=_percentile(bootstrap_means, 1.0 - alpha),
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            )
        )
    return tuple(results)


def summarize_forced_horizon_curves(
    runs_by_branch: Mapping[str, Sequence[DirectEpdmsResultRun]],
) -> tuple[DirectEpdmsForcedHorizonSummary, ...]:
    """Summarize exactly two complete forced H0--H4 curves in public order."""

    if not isinstance(runs_by_branch, Mapping):
        raise TypeError("runs_by_branch must be a mapping")
    if set(runs_by_branch) != _FORCED_BRANCHES:
        raise ValueError(f"forced curves require exactly {list(_FORCED_BRANCH_ORDER)}")

    summaries = []
    for branch in _FORCED_BRANCH_ORDER:
        runs = runs_by_branch[branch]
        if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
            raise TypeError(f"forced curve {branch!r} must be a sequence")
        run_tuple = tuple(runs)
        if len(run_tuple) != 5:
            raise ValueError(f"forced curve {branch!r} must contain H0--H4")
        if tuple(run.forced_horizon for run in run_tuple) != tuple(range(5)):
            raise ValueError(f"forced curve {branch!r} must be ordered H0--H4")
        for run in run_tuple:
            if not isinstance(run, DirectEpdmsResultRun) or run.branch != branch:
                raise ValueError(f"forced curve {branch!r} contains the wrong result branch")
            summaries.append(
                DirectEpdmsForcedHorizonSummary(
                    branch=branch,
                    horizon=run.forced_horizon,
                    evaluation_seed=run.evaluation_seed,
                    scenario_count=len(run.records),
                    mean_epdms=fmean(record.epdms for record in run.records),
                    mean_total_latency_ms=fmean(record.latency_ms for record in run.records),
                )
            )
    return tuple(summaries)


def write_cvoi_ablation_report(
    *,
    output_dir: str | Path,
    report: Mapping[str, object],
    paired_csv: str,
    paper_markdown: str,
) -> tuple[Path, ...]:
    """Exclusively publish the three direct EPDMS report artifacts."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    if not isinstance(paired_csv, str):
        raise TypeError("paired_csv must be a string")
    if not isinstance(paper_markdown, str):
        raise TypeError("paper_markdown must be a string")
    payloads = {
        "report.json": json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        "paired_comparisons.csv": paired_csv,
        "paper_tables.md": paper_markdown,
    }

    directory = Path(output_dir)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"CVoI ablation report output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, payload in payloads.items():
            path = staging / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        staging.rename(directory)
    except Exception:
        try:
            for path in staging.iterdir():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            staging.rmdir()
        except OSError:
            pass
        raise
    return tuple(directory / name for name in payloads)
