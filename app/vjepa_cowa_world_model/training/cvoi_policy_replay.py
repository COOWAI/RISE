"""Deterministic stopping-policy replay over CVoI horizon cache records."""

import hashlib
from typing import Mapping, Sequence

from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import CVOI_WORLD4DRIVE_LINEAGES
from app.vjepa_cowa_world_model.training.cvoi_horizon_cache import CvoiHorizonCacheRecord


def uniform_random_horizon(sample_id: str, global_seed: int) -> int:
    """Choose a process- and host-stable horizon uniformly from ``0..3``."""

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"sample_id must be a non-empty string, got {sample_id!r}")
    if type(global_seed) is not int or global_seed < 0:
        raise ValueError(f"global_seed must be a non-negative integer, got {global_seed!r}")
    payload = f"{sample_id}\0{global_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 4


def _lineage_records(
    records: Sequence[CvoiHorizonCacheRecord],
    lineage: str,
) -> tuple[CvoiHorizonCacheRecord, ...]:
    if lineage not in CVOI_WORLD4DRIVE_LINEAGES:
        raise ValueError(f"lineage must be one of {sorted(CVOI_WORLD4DRIVE_LINEAGES)}, got {lineage!r}")
    selected = tuple(record for record in records if record.lineage == lineage)
    if not selected:
        raise ValueError(f"missing cache record for lineage={lineage!r}")
    keys = [record.cache_key for record in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate cache record identity in lineage={lineage!r}")
    sample_scenes: dict[str, str] = {}
    for record in selected:
        previous = sample_scenes.setdefault(record.sample_id, record.source_scene_id)
        if previous != record.source_scene_id:
            raise ValueError(
                f"sample_id={record.sample_id!r} maps to multiple source scenes: "
                f"{previous!r}, {record.source_scene_id!r}"
            )
    return selected


def _record_index(
    records: Sequence[CvoiHorizonCacheRecord],
    lineage: str,
) -> tuple[dict[tuple[str, int, int], CvoiHorizonCacheRecord], tuple[str, ...]]:
    lineage_records = _lineage_records(records, lineage)
    index = {(record.sample_id, record.horizon, record.guidance_steps): record for record in lineage_records}
    sample_ids = tuple(sorted({record.sample_id for record in lineage_records}))
    return index, sample_ids


def _select_records(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    lineage: str,
    selected_horizons: Mapping[str, int],
    guidance_steps: int,
) -> tuple[CvoiHorizonCacheRecord, ...]:
    index, sample_ids = _record_index(records, lineage)
    if set(selected_horizons) != set(sample_ids):
        raise ValueError(
            "selected_horizons sample set must exactly match the cache lineage sample set: "
            f"expected={list(sample_ids)!r}, got={sorted(selected_horizons)!r}"
        )
    if type(guidance_steps) is not int or guidance_steps < 0:
        raise ValueError(f"guidance_steps must be a non-negative integer, got {guidance_steps!r}")
    if lineage == "p0_controller" and guidance_steps != 0:
        raise ValueError("p0_controller replay must use guidance_steps=0")
    selected = []
    for sample_id in sample_ids:
        horizon = selected_horizons[sample_id]
        if type(horizon) is not int or horizon not in {0, 1, 2, 3}:
            raise ValueError(f"selected horizon for sample_id={sample_id!r} must be in [0,3], got {horizon!r}")
        effective_guidance = 0 if horizon == 0 or lineage == "p0_controller" else guidance_steps
        key = (sample_id, horizon, effective_guidance)
        record = index.get(key)
        if record is None:
            raise ValueError(
                "missing cache record for "
                f"sample_id={sample_id!r}, lineage={lineage!r}, horizon={horizon}, "
                f"guidance_steps={effective_guidance}"
            )
        selected.append(record)
    return tuple(selected)


def replay_fixed_horizon(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    lineage: str,
    horizon: int,
    guidance_steps: int,
) -> tuple[CvoiHorizonCacheRecord, ...]:
    """Select one forced horizon for every sample in a lineage."""

    _, sample_ids = _record_index(records, lineage)
    return _select_records(
        records,
        lineage=lineage,
        selected_horizons={sample_id: horizon for sample_id in sample_ids},
        guidance_steps=guidance_steps,
    )


def replay_uniform_random_stop(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    lineage: str,
    random_stop_seed: int,
    guidance_steps: int = 2,
) -> tuple[CvoiHorizonCacheRecord, ...]:
    """Replay stable per-sample uniform stopping without running a Gate."""

    _, sample_ids = _record_index(records, lineage)
    return _select_records(
        records,
        lineage=lineage,
        selected_horizons={sample_id: uniform_random_horizon(sample_id, random_stop_seed) for sample_id in sample_ids},
        guidance_steps=guidance_steps,
    )


def replay_controller_stop(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    lineage: str,
    selected_horizons: Mapping[str, int],
    guidance_steps: int = 2,
) -> tuple[CvoiHorizonCacheRecord, ...]:
    """Replay externally recorded Controller decisions over cached outcomes."""

    return _select_records(
        records,
        lineage=lineage,
        selected_horizons=selected_horizons,
        guidance_steps=guidance_steps,
    )
