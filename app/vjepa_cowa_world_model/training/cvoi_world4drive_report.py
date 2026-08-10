"""Paper report construction from validated CVoI World4Drive cache records."""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import CVOI_WORLD4DRIVE_LAMBDA_COMPUTE
from app.vjepa_cowa_world_model.training.cvoi_horizon_cache import CvoiHorizonCacheRecord
from app.vjepa_cowa_world_model.training.cvoi_policy_replay import (
    replay_controller_stop,
    replay_fixed_horizon,
    replay_uniform_random_stop,
)


@dataclass(frozen=True)
class World4DriveRecordSummary:
    """Aggregate canonical metrics for one report row."""

    sample_count: int
    scene_count: int
    l2_at_1s: float
    l2_at_2s: float
    l2_at_3s: float
    l2_avg: float
    collision_at_1s: float
    collision_at_2s: float
    collision_at_3s: float
    collision_rate: float
    average_rollout_count: float
    rollout_histogram: Mapping[int, int]


@dataclass(frozen=True)
class MainAblationRow(World4DriveRecordSummary):
    """One approved Controller/Value/CF main-ablation row."""

    controller: bool
    value_head: bool
    cf_data: bool
    lineage: str


@dataclass(frozen=True)
class StoppingStrategyRow(World4DriveRecordSummary):
    """One random-versus-learned stopping comparison row."""

    method: str


@dataclass(frozen=True)
class GuidanceStepsRow(World4DriveRecordSummary):
    """One evaluation-only Value-guidance step-count row."""

    guidance_steps: int


@dataclass(frozen=True)
class OracleMatrixReport:
    """Oracle grouping counts and separate forced-horizon metric matrices."""

    sample_counts: tuple[int, int, int, int]
    scene_counts: tuple[int, int, int, int]
    l2_matrix: tuple[tuple[float | None, ...], ...]
    collision_matrix: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """Paired scene-cluster bootstrap difference interval (left minus right)."""

    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    sample_count: int
    scene_count: int
    repetitions: int
    seed: int


def _mean(records: Sequence[CvoiHorizonCacheRecord], field: str) -> float:
    return float(np.mean([float(getattr(record, field)) for record in records]))


def summarize_world4drive_records(
    records: Sequence[CvoiHorizonCacheRecord],
) -> World4DriveRecordSummary:
    """Aggregate selected cache records without recomputing model outputs."""

    if not records:
        raise ValueError("cannot summarize an empty World4Drive record selection")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("World4Drive report selection contains duplicate sample IDs")
    histogram = {horizon: 0 for horizon in range(4)}
    for record in records:
        histogram[record.horizon] += 1
    return World4DriveRecordSummary(
        sample_count=len(records),
        scene_count=len({record.source_scene_id for record in records}),
        l2_at_1s=_mean(records, "l2_at_1s"),
        l2_at_2s=_mean(records, "l2_at_2s"),
        l2_at_3s=_mean(records, "l2_at_3s"),
        l2_avg=_mean(records, "l2_avg"),
        collision_at_1s=_mean(records, "collision_at_1s"),
        collision_at_2s=_mean(records, "collision_at_2s"),
        collision_at_3s=_mean(records, "collision_at_3s"),
        collision_rate=_mean(records, "collision_rate"),
        average_rollout_count=_mean(records, "horizon"),
        rollout_histogram=MappingProxyType(histogram),
    )


def _summary_kwargs(summary: World4DriveRecordSummary) -> dict[str, object]:
    return {
        "sample_count": summary.sample_count,
        "scene_count": summary.scene_count,
        "l2_at_1s": summary.l2_at_1s,
        "l2_at_2s": summary.l2_at_2s,
        "l2_at_3s": summary.l2_at_3s,
        "l2_avg": summary.l2_avg,
        "collision_at_1s": summary.collision_at_1s,
        "collision_at_2s": summary.collision_at_2s,
        "collision_at_3s": summary.collision_at_3s,
        "collision_rate": summary.collision_rate,
        "average_rollout_count": summary.average_rollout_count,
        "rollout_histogram": summary.rollout_histogram,
    }


def build_main_ablation_rows(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    controller_horizons: Mapping[str, Mapping[str, int]],
) -> tuple[MainAblationRow, ...]:
    """Build the approved four-row Controller/Value/CF ablation table."""

    expected_lineages = {"p0_controller", "real_only_value", "real_cf_value"}
    if set(controller_horizons) != expected_lineages:
        raise ValueError(
            f"controller_horizons must contain exactly {sorted(expected_lineages)}, "
            f"got {sorted(controller_horizons)}"
        )
    selections = (
        (
            False,
            True,
            False,
            "real_only_value",
            replay_fixed_horizon(
                records,
                lineage="real_only_value",
                horizon=3,
                guidance_steps=2,
            ),
        ),
        (
            True,
            False,
            False,
            "p0_controller",
            replay_controller_stop(
                records,
                lineage="p0_controller",
                selected_horizons=controller_horizons["p0_controller"],
                guidance_steps=0,
            ),
        ),
        (
            True,
            True,
            False,
            "real_only_value",
            replay_controller_stop(
                records,
                lineage="real_only_value",
                selected_horizons=controller_horizons["real_only_value"],
                guidance_steps=2,
            ),
        ),
        (
            True,
            True,
            True,
            "real_cf_value",
            replay_controller_stop(
                records,
                lineage="real_cf_value",
                selected_horizons=controller_horizons["real_cf_value"],
                guidance_steps=2,
            ),
        ),
    )
    rows = []
    for controller, value_head, cf_data, lineage, selected in selections:
        summary = summarize_world4drive_records(selected)
        rows.append(
            MainAblationRow(
                **_summary_kwargs(summary),
                controller=controller,
                value_head=value_head,
                cf_data=cf_data,
                lineage=lineage,
            )
        )
    return tuple(rows)


def build_stopping_strategy_rows(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    controller_horizons: Mapping[str, int],
    random_stop_seed: int,
) -> tuple[StoppingStrategyRow, ...]:
    """Build uniform-random versus learned-Controller rows on Real+CF P1."""

    selections = (
        (
            "uniform random stop",
            replay_uniform_random_stop(
                records,
                lineage="real_cf_value",
                random_stop_seed=random_stop_seed,
                guidance_steps=2,
            ),
        ),
        (
            "learned Controller",
            replay_controller_stop(
                records,
                lineage="real_cf_value",
                selected_horizons=controller_horizons,
                guidance_steps=2,
            ),
        ),
    )
    return tuple(
        StoppingStrategyRow(
            **_summary_kwargs(summarize_world4drive_records(selected)),
            method=method,
        )
        for method, selected in selections
    )


def build_guidance_steps_rows(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    controller_horizons: Mapping[str, int],
) -> tuple[GuidanceStepsRow, ...]:
    """Build K=1..4 rows while keeping Controller horizons fixed."""

    rows = []
    for guidance_steps in (1, 2, 3, 4):
        selected = replay_controller_stop(
            records,
            lineage="real_cf_value",
            selected_horizons=controller_horizons,
            guidance_steps=guidance_steps,
        )
        rows.append(
            GuidanceStepsRow(
                **_summary_kwargs(summarize_world4drive_records(selected)),
                guidance_steps=guidance_steps,
            )
        )
    rollout_signatures = {(row.average_rollout_count, tuple(row.rollout_histogram.items())) for row in rows}
    if len(rollout_signatures) != 1:
        raise ValueError("guidance-step study changed Controller-selected horizons")
    return tuple(rows)


def _default_real_cf_by_horizon(
    records: Sequence[CvoiHorizonCacheRecord],
) -> dict[int, dict[str, CvoiHorizonCacheRecord]]:
    by_horizon = {}
    expected_samples: set[str] | None = None
    for horizon in range(4):
        selected = replay_fixed_horizon(
            records,
            lineage="real_cf_value",
            horizon=horizon,
            guidance_steps=0 if horizon == 0 else 2,
        )
        current = {record.sample_id: record for record in selected}
        if expected_samples is None:
            expected_samples = set(current)
        elif set(current) != expected_samples:
            raise ValueError("Oracle fixed-horizon selections have different sample sets")
        by_horizon[horizon] = current
    return by_horizon


def build_oracle_matrices(records: Sequence[CvoiHorizonCacheRecord]) -> OracleMatrixReport:
    """Group by penalized-utility Oracle and build separate 4x4 matrices."""

    by_horizon = _default_real_cf_by_horizon(records)
    sample_ids = tuple(sorted(by_horizon[0]))
    groups: dict[int, list[str]] = {horizon: [] for horizon in range(4)}
    for sample_id in sample_ids:
        utilities = {
            horizon: by_horizon[horizon][sample_id].task_score
            - CVOI_WORLD4DRIVE_LAMBDA_COMPUTE * by_horizon[horizon][sample_id].compute_cost
            for horizon in range(4)
        }
        maximum = max(utilities.values())
        oracle_horizon = min(
            horizon
            for horizon, utility in utilities.items()
            if math.isclose(utility, maximum, rel_tol=0.0, abs_tol=1e-12)
        )
        groups[oracle_horizon].append(sample_id)
    sample_counts = tuple(len(groups[horizon]) for horizon in range(4))
    scene_counts = tuple(
        len({by_horizon[0][sample_id].source_scene_id for sample_id in groups[horizon]}) for horizon in range(4)
    )
    l2_rows = []
    collision_rows = []
    for oracle_horizon in range(4):
        group = groups[oracle_horizon]
        if not group:
            l2_rows.append((None, None, None, None))
            collision_rows.append((None, None, None, None))
            continue
        l2_rows.append(
            tuple(float(np.mean([by_horizon[forced][sample_id].l2_avg for sample_id in group])) for forced in range(4))
        )
        collision_rows.append(
            tuple(
                float(np.mean([by_horizon[forced][sample_id].collision_rate for sample_id in group]))
                for forced in range(4)
            )
        )
    return OracleMatrixReport(
        sample_counts=sample_counts,
        scene_counts=scene_counts,
        l2_matrix=tuple(l2_rows),
        collision_matrix=tuple(collision_rows),
    )


def paired_scene_bootstrap(
    left: Sequence[CvoiHorizonCacheRecord],
    right: Sequence[CvoiHorizonCacheRecord],
    *,
    metric: str,
    seed: int,
    repetitions: int,
) -> PairedBootstrapInterval:
    """Compute a paired percentile interval by resampling unique scenes."""

    allowed_metrics = {
        "l2_at_1s",
        "l2_at_2s",
        "l2_at_3s",
        "l2_avg",
        "collision_at_1s",
        "collision_at_2s",
        "collision_at_3s",
        "collision_rate",
    }
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}, got {metric!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError(f"repetitions must be a positive integer, got {repetitions!r}")
    left_by_id = {(record.sample_id, record.source_scene_id): record for record in left}
    right_by_id = {(record.sample_id, record.source_scene_id): record for record in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right) or set(left_by_id) != set(right_by_id):
        raise ValueError("left and right must contain identical unique paired sample identities")
    if not left_by_id:
        raise ValueError("paired bootstrap inputs must not be empty")
    differences_by_scene: dict[str, list[float]] = {}
    for identity in sorted(left_by_id):
        left_record = left_by_id[identity]
        right_record = right_by_id[identity]
        differences_by_scene.setdefault(identity[1], []).append(
            float(getattr(left_record, metric)) - float(getattr(right_record, metric))
        )
    scenes = tuple(sorted(differences_by_scene))
    all_differences = [value for scene in scenes for value in differences_by_scene[scene]]
    estimate = float(np.mean(all_differences))
    rng = np.random.default_rng(seed)
    bootstrap_values = []
    for _ in range(repetitions):
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        sampled_differences = [value for scene in sampled_scenes for value in differences_by_scene[str(scene)]]
        bootstrap_values.append(float(np.mean(sampled_differences)))
    ci_low, ci_high = np.percentile(np.asarray(bootstrap_values), [2.5, 97.5])
    return PairedBootstrapInterval(
        metric=metric,
        estimate=estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        sample_count=len(left_by_id),
        scene_count=len(scenes),
        repetitions=repetitions,
        seed=seed,
    )
