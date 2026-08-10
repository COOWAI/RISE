"""Collect all retained CVoI lineages over the Real-only World4Drive cohort."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from app.vjepa_cowa_world_model.training.config import TrainingConfig
from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import (
    CVOI_WORLD4DRIVE_LINEAGE_ORDER,
    CvoiWorld4DriveDirectConfig,
    parse_cvoi_world4drive_direct_config,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import seed_cvoi_process
from app.vjepa_cowa_world_model.training.cvoi_horizon_cache import (
    CvoiExpectedCacheSample,
    CvoiHorizonCacheRecord,
    CvoiHorizonStudyPoint,
    validate_cvoi_horizon_cache,
    write_cvoi_horizon_cache,
)
from app.vjepa_cowa_world_model.training.cvoi_runtime import parse_world4drive_base_model_config
from app.vjepa_cowa_world_model.training.cvoi_world4drive_identity import (
    CvoiWorld4DriveCacheProvenance,
    build_world4drive_cache_provenance,
    cvoi_world4drive_evaluation_dataset_signature,
    load_world4drive_audit_manifest,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    CvoiWorld4DriveRuntimeBinding,
    build_cvoi_world4drive_runtime_binding,
    load_cvoi_world4drive_gate,
    preflight_world4drive_direct_semantics,
    require_world4drive_outputs_absent,
    validate_world4drive_base_data_root,
    validate_world4drive_direct_inputs,
)
from app.vjepa_cowa_world_model.training.data import create_val_dataloader
from app.vjepa_cowa_world_model.training.geometry_outcome import PlanningOutcomeEvaluator
from app.vjepa_cowa_world_model.training.navsim_cvoi_evaluation import evaluate_navsim_real_planning_outcome
from app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime import create_navsim_cvoi_world4drive_runtime
from app.vjepa_cowa_world_model.training.navsim_cvoi_world4drive import (
    collect_navsim_world4drive_controller_horizons_from_runtime,
    collect_navsim_world4drive_lineage,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _study_points(lineage: str) -> tuple[CvoiHorizonStudyPoint, ...]:
    if lineage == "p0_controller":
        return tuple(CvoiHorizonStudyPoint(lineage, horizon, 0) for horizon in range(4))
    if lineage == "real_only_value":
        return tuple(CvoiHorizonStudyPoint(lineage, horizon, 0 if horizon == 0 else 2) for horizon in range(4))
    if lineage == "real_cf_value":
        return (CvoiHorizonStudyPoint(lineage, 0, 0),) + tuple(
            CvoiHorizonStudyPoint(lineage, horizon, guidance_steps)
            for horizon in (1, 2, 3)
            for guidance_steps in (1, 2, 3, 4)
        )
    raise ValueError(f"unsupported World4Drive lineage: {lineage!r}")


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: object) -> Path:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_cvoi_world4drive_collection(
    output_dir: str | Path,
    *,
    lineage: str,
    records: Sequence[CvoiHorizonCacheRecord],
    controller_horizons: Mapping[str, int],
    provenance: CvoiWorld4DriveCacheProvenance,
) -> Mapping[str, Path]:
    """Validate and write one complete lineage without replacing existing output."""

    if not isinstance(provenance, CvoiWorld4DriveCacheProvenance):
        raise TypeError("provenance must be CvoiWorld4DriveCacheProvenance")
    records = tuple(records)
    identities = sorted({(record.sample_id, record.source_scene_id, record.seed) for record in records})
    expected_samples = tuple(CvoiExpectedCacheSample(*identity) for identity in identities)
    records = validate_cvoi_horizon_cache(
        records,
        expected_samples=expected_samples,
        expected_study_points=_study_points(lineage),
    )
    expected_provenance = {
        "checkpoint_signature": provenance.checkpoint_signature,
        "dataset_signature": provenance.dataset_signature,
        "code_signature": provenance.code_signature,
        "converter_signature": provenance.converter_signature,
        "evaluator_signature": provenance.evaluator_signature,
    }
    for record in records:
        if any(getattr(record, name) != value for name, value in expected_provenance.items()):
            raise ValueError("World4Drive collection record provenance does not match the declared lineage")
    sample_ids = {sample.sample_id for sample in expected_samples}
    if not isinstance(controller_horizons, Mapping) or set(controller_horizons) != sample_ids:
        raise ValueError("World4Drive Controller horizons must match the cache sample IDs")
    if any(type(horizon) is not int or horizon not in {0, 1, 2, 3} for horizon in controller_horizons.values()):
        raise ValueError("World4Drive Controller horizons must be integers in {0,1,2,3}")

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"World4Drive collection output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "records.jsonl": write_cvoi_horizon_cache(output / "records.jsonl", records),
        "controller_horizons.json": _write_json_exclusive(
            output / "controller_horizons.json",
            {
                "schema": "cvoi_world4drive_controller_horizons_v1",
                "lineage": lineage,
                "horizons": dict(sorted(controller_horizons.items())),
            },
        ),
    }
    manifest = {
        "schema": "cvoi_world4drive_collection_v1",
        "dataset_domain": "real",
        "lineage": lineage,
        "sample_count": len(expected_samples),
        "scene_count": len({sample.source_scene_id for sample in expected_samples}),
        "record_count": len(records),
        "study_points": [
            {"horizon": point.horizon, "guidance_steps": point.guidance_steps} for point in _study_points(lineage)
        ],
        "provenance": expected_provenance,
        "artifacts": {name: _artifact_sha256(path) for name, path in sorted(paths.items())},
    }
    paths["manifest.json"] = _write_json_exclusive(output / "manifest.json", manifest)
    return paths


def _collect_and_write_world4drive_quality(
    *,
    base_config: TrainingConfig,
    direct_config: CvoiWorld4DriveDirectConfig,
    binding: CvoiWorld4DriveRuntimeBinding,
    runtime: object,
    output_dir: Path,
) -> Mapping[str, Path]:
    """Run the characterized exhaustive-quality and Controller passes for one lineage."""

    runtime_config = getattr(runtime, "config", None)
    if runtime_config is None:
        raise TypeError("World4Drive runtime must expose its private evaluation config")
    if str(runtime_config.cvoi.stage) != "evaluation" or int(runtime_config.cvoi.max_horizon) != 3:
        raise ValueError("World4Drive runtime must bind the evaluation stage with H=3")

    lineage = binding.lineage
    device = torch.device("cuda")
    evaluator = PlanningOutcomeEvaluator(timestep_sec=1.0 / float(base_config.data.fps))
    evaluation_audit = load_world4drive_audit_manifest(
        direct_config.dataset_audit_manifest_path,
        expected_real_root=direct_config.real_val_root,
    )
    evaluation_dataset_signature = cvoi_world4drive_evaluation_dataset_signature(
        evaluation_audit,
        real_val_root=direct_config.real_val_root,
        config=base_config,
    )
    provenance = build_world4drive_cache_provenance(
        binding,
        evaluator=evaluator,
        source_root=Path(__file__).resolve().parents[4],
        evaluation_audit=evaluation_audit,
        evaluation_dataset_signature=evaluation_dataset_signature,
    )
    gate = load_cvoi_world4drive_gate(binding, device=device)

    try:
        fixed_loader, _ = create_val_dataloader(
            base_config,
            rank=0,
            world_size=1,
            validation_domain="real",
        )
        if fixed_loader is None:
            raise ValueError("collect_cvoi_world4drive requires a Real validation root")

        def task_score_provider(navsim_batch, result) -> float:
            outcome = evaluate_navsim_real_planning_outcome(
                navsim_batch,
                result.pred_trajs,
                result.confidences,
                config=runtime_config,
                evaluator=evaluator,
            )
            return float(outcome.task_score[0].item())

        try:
            records = collect_navsim_world4drive_lineage(
                fixed_loader,
                runtime=runtime,
                config=runtime_config,
                device=device,
                lineage=lineage,
                study_points=_study_points(lineage),
                provenance=provenance,
                task_score_provider=task_score_provider,
            )
        finally:
            del fixed_loader
            gc.collect()
            torch.cuda.empty_cache()

        controller_loader, _ = create_val_dataloader(
            base_config,
            rank=0,
            world_size=1,
            validation_domain="real",
        )
        if controller_loader is None:
            raise ValueError("collect_cvoi_world4drive requires a second Real validation pass")
        try:
            horizons = collect_navsim_world4drive_controller_horizons_from_runtime(
                controller_loader,
                runtime=runtime,
                gate=gate,
                config=runtime_config,
                device=device,
                lambda_compute=direct_config.lambda_compute,
            )
        finally:
            del controller_loader
    finally:
        del gate
        gc.collect()
        torch.cuda.empty_cache()

    paths = write_cvoi_world4drive_collection(
        output_dir,
        lineage=lineage,
        records=records,
        controller_horizons=horizons,
        provenance=provenance,
    )
    logger.info(
        "[cvoi_world4drive_collection] lineage=%s samples=%d records=%d output=%s",
        lineage,
        len(horizons),
        len(records),
        output_dir,
    )
    return paths


def collect_world4drive_quality_lineage(
    *,
    base_config: TrainingConfig,
    direct_config: CvoiWorld4DriveDirectConfig,
    lineage: str,
    output_dir: Path,
) -> Mapping[str, Path]:
    """Create, execute, and release one read-only lineage runtime."""

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("collect_cvoi_world4drive requires world_size=1")
    if int(base_config.data.batch_size) != 1:
        raise ValueError("collect_cvoi_world4drive requires data.batch_size=1")
    if not torch.cuda.is_available():
        raise RuntimeError("collect_cvoi_world4drive requires CUDA")

    binding = build_cvoi_world4drive_runtime_binding(direct_config, lineage=lineage)
    runtime = create_navsim_cvoi_world4drive_runtime(
        config=base_config,
        binding=binding,
        device=torch.device("cuda"),
    )
    try:
        return _collect_and_write_world4drive_quality(
            base_config=base_config,
            direct_config=direct_config,
            binding=binding,
            runtime=runtime,
            output_dir=output_dir,
        )
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()


def run_world4drive_collect(
    args: Mapping[str, object],
    *,
    collect_lineage: Callable[..., Mapping[str, Path]] = collect_world4drive_quality_lineage,
) -> Mapping[str, Mapping[str, Path]]:
    """Preflight once, then collect the three authoritative lineages in fixed order."""

    if not isinstance(args, Mapping):
        raise TypeError("collect_cvoi_world4drive args must be a mapping")
    direct = parse_cvoi_world4drive_direct_config(args.get("cvoi_world4drive"))
    if direct.job != "collect":
        raise ValueError("collect_cvoi_world4drive requires cvoi_world4drive.job='collect'")
    base = parse_world4drive_base_model_config(args)
    validate_world4drive_base_data_root(base, direct)
    outputs = {lineage: Path(direct.output_root) / "quality" / lineage for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER}
    validate_world4drive_direct_inputs(direct, require_model_artifacts=True)
    require_world4drive_outputs_absent(outputs.values())
    preflight_world4drive_direct_semantics(direct)

    results = {}
    for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER:
        seed_cvoi_process(base)
        results[lineage] = collect_lineage(
            base_config=base,
            direct_config=direct,
            lineage=lineage,
            output_dir=outputs[lineage],
        )
    return results


def main(args, resume_preempt: bool = False):
    """Run the standalone no-resume three-lineage collect command."""

    if resume_preempt:
        raise ValueError("collect_cvoi_world4drive does not support resume_preempt")
    return run_world4drive_collect(args)
