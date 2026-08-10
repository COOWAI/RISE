"""Collect exact online latency for all retained World4Drive lineages."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import torch

from app.vjepa_cowa_world_model.training.config import TrainingConfig
from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import (
    CVOI_WORLD4DRIVE_LINEAGE_ORDER,
    CvoiWorld4DriveDirectConfig,
    parse_cvoi_world4drive_direct_config,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import seed_cvoi_process
from app.vjepa_cowa_world_model.training.cvoi_runtime import parse_world4drive_base_model_config
from app.vjepa_cowa_world_model.training.cvoi_world4drive_latency_collection import (
    collect_cvoi_world4drive_latency_reports,
    validate_world4drive_latency_lineage_identity,
    write_cvoi_world4drive_latency_collection,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    World4DriveRuntimeDeployment,
    build_cvoi_world4drive_runtime_binding,
    load_cvoi_world4drive_gate,
    preflight_world4drive_direct_semantics,
    require_world4drive_outputs_absent,
    validate_world4drive_base_data_root,
    validate_world4drive_direct_inputs,
)
from app.vjepa_cowa_world_model.training.data import create_val_dataloader
from app.vjepa_cowa_world_model.training.navsim_cvoi_evaluation import build_cvoi_evaluation_runtime_environment
from app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime import create_navsim_cvoi_world4drive_runtime
from app.vjepa_cowa_world_model.training.navsim_cvoi_world4drive import _metadata_entry, _require_real_batch
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_world4drive_latency_runtime_environment(
    base_config: TrainingConfig,
    direct_config: CvoiWorld4DriveDirectConfig,
    *,
    device: torch.device,
) -> Mapping[str, object]:
    """Capture one shared hardware/runtime identity before any lineage is loaded."""

    base = build_cvoi_evaluation_runtime_environment(base_config, device=device).to_manifest()
    hardware = base["hardware"]
    runtime = {
        "software": base["software"],
        "precision_contract": base["precision_contract"],
        "batch_size": 1,
        "diffusion_inference_steps": int(base_config.planner.diff_inference_steps),
        "latency_warmup": direct_config.latency_warmup,
        "latency_repetitions": direct_config.latency_repetitions,
        "timed_region": "encoder_to_ready_trajectory_and_confidence",
        "cuda_synchronize_before_after": True,
    }
    return {
        "schema": "cvoi_world4drive_latency_environment_v1",
        "hardware_signature": _canonical_sha256(hardware),
        "runtime_signature": _canonical_sha256(runtime),
        "hardware": hardware,
        "runtime": runtime,
    }


def _stable_sample_id(metadata: Mapping[str, object]) -> str:
    value = _metadata_entry(metadata, "stable_sample_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("World4Drive stable_sample_id must contain one non-empty string")
    return value


def _validate_latency_batch_contract(base_config: TrainingConfig) -> None:
    batch_size = getattr(getattr(base_config, "data", None), "batch_size", None)
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError("collect_cvoi_world4drive_latency requires data.batch_size=1")


def _validate_latency_world_contract() -> None:
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("collect_cvoi_world4drive_latency requires world_size=1")


def _validate_latency_execution_contract(base_config: TrainingConfig) -> None:
    _validate_latency_batch_contract(base_config)
    _validate_latency_world_contract()


def collect_world4drive_latency_lineage(
    *,
    base_config: TrainingConfig,
    direct_config: CvoiWorld4DriveDirectConfig,
    lineage: str,
    output_dir: Path,
    runtime_environment: Mapping[str, object],
) -> Mapping[str, Path]:
    """Build, measure, write, and release exactly one read-only latency lineage."""

    _validate_latency_execution_contract(base_config)
    if not torch.cuda.is_available():
        raise RuntimeError("collect_cvoi_world4drive_latency requires CUDA")
    if not isinstance(runtime_environment, Mapping):
        raise TypeError("World4Drive latency runtime_environment must be a mapping")
    hardware_signature = runtime_environment.get("hardware_signature")
    runtime_signature = runtime_environment.get("runtime_signature")
    if not isinstance(hardware_signature, str) or not hardware_signature:
        raise ValueError("World4Drive latency runtime environment requires a hardware signature")
    if not isinstance(runtime_signature, str) or not runtime_signature:
        raise ValueError("World4Drive latency runtime environment requires a runtime signature")

    device = torch.device("cuda")
    runtime = None
    gate = None
    loader = None
    deployment = None
    prepared_samples = None
    raw_batch = None
    navsim_batch = None
    metadata = None
    reports = None
    paths = None
    report_count = 0
    sample_count = 0
    try:
        binding = build_cvoi_world4drive_runtime_binding(direct_config, lineage=lineage)
        runtime = create_navsim_cvoi_world4drive_runtime(
            config=base_config,
            binding=binding,
            device=device,
        )
        gate = load_cvoi_world4drive_gate(binding, device=device)
        loader, _ = create_val_dataloader(
            base_config,
            rank=0,
            world_size=1,
            validation_domain="real",
        )
        if loader is None:
            raise ValueError("collect_cvoi_world4drive_latency requires a Real validation root")
        deployment = World4DriveRuntimeDeployment(runtime, binding=binding, gate=gate)
        required_samples = direct_config.latency_warmup + direct_config.latency_repetitions
        prepared_samples = []
        seen_sample_ids = set()
        for raw_batch in loader:
            navsim_batch, metadata = _require_real_batch(raw_batch, config=runtime.config)
            sample_id = _stable_sample_id(metadata)
            if sample_id in seen_sample_ids:
                raise ValueError(f"World4Drive latency sample IDs must be unique, got {sample_id!r}")
            seen_sample_ids.add(sample_id)
            prepared_samples.append((sample_id, deployment.prepare(navsim_batch)))
            if len(prepared_samples) == required_samples:
                break
        if len(prepared_samples) != required_samples:
            raise ValueError(
                "World4Drive latency requires " f"{required_samples} matched Real samples, got {len(prepared_samples)}"
            )
        sample_ids = tuple(sample_id for sample_id, _prepared in prepared_samples)
        reports = collect_cvoi_world4drive_latency_reports(
            lineage=lineage,
            prepared_samples=prepared_samples,
            deployment=deployment,
            random_stop_seed=direct_config.random_stop_seed,
            lambda_compute=direct_config.lambda_compute,
            warmup=direct_config.latency_warmup,
            repetitions=direct_config.latency_repetitions,
            hardware_signature=hardware_signature,
            runtime_signature=runtime_signature,
            device=device,
        )
        paths = write_cvoi_world4drive_latency_collection(
            output_dir,
            lineage=lineage,
            reports=reports,
            sample_ids=sample_ids,
            runtime_environment=runtime_environment,
        )
        report_count = len(reports)
        sample_count = len(prepared_samples)
    finally:
        if prepared_samples is not None:
            prepared_samples.clear()
        prepared_samples = None
        raw_batch = None
        navsim_batch = None
        metadata = None
        reports = None
        deployment = None
        loader = None
        gate = None
        runtime = None
        gc.collect()
        torch.cuda.empty_cache()

    if paths is None:
        raise RuntimeError("World4Drive latency lineage did not produce artifacts")
    logger.info(
        "[cvoi_world4drive_latency] lineage=%s rows=%d samples=%d output=%s",
        lineage,
        report_count,
        sample_count,
        output_dir,
    )
    return paths


def run_world4drive_latency(
    args: Mapping[str, object],
    *,
    collect_lineage: Callable[..., Mapping[str, Path]] = collect_world4drive_latency_lineage,
) -> Mapping[str, Mapping[str, Path]]:
    """Preflight once, then collect the three latency subsets in fixed order."""

    if not isinstance(args, Mapping):
        raise TypeError("collect_cvoi_world4drive_latency args must be a mapping")
    direct = parse_cvoi_world4drive_direct_config(args.get("cvoi_world4drive"))
    if direct.job != "latency":
        raise ValueError("collect_cvoi_world4drive_latency requires cvoi_world4drive.job='latency'")
    base = parse_world4drive_base_model_config(args)
    validate_world4drive_base_data_root(base, direct)
    _validate_latency_batch_contract(base)
    outputs = {lineage: Path(direct.output_root) / "latency" / lineage for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER}
    validate_world4drive_direct_inputs(direct, require_model_artifacts=True)
    require_world4drive_outputs_absent(outputs.values())
    preflight_world4drive_direct_semantics(direct)
    _validate_latency_world_contract()

    if not torch.cuda.is_available():
        raise RuntimeError("collect_cvoi_world4drive_latency requires CUDA")
    device = torch.device("cuda")
    environment = build_world4drive_latency_runtime_environment(base, direct, device=device)
    results = {}
    reference_identity = None
    for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER:
        seed_cvoi_process(base)
        results[lineage] = collect_lineage(
            base_config=base,
            direct_config=direct,
            lineage=lineage,
            output_dir=outputs[lineage],
            runtime_environment=environment,
        )
        reference_identity = validate_world4drive_latency_lineage_identity(
            results[lineage],
            lineage=lineage,
            expected=reference_identity,
        )
    return results


def main(args, resume_preempt: bool = False):
    """Run the standalone no-resume three-lineage latency command."""

    if resume_preempt:
        raise ValueError("collect_cvoi_world4drive_latency does not support resume_preempt")
    return run_world4drive_latency(args)
