"""Fixed-policy budget oracle collection for the continuous controller."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from app.vjepa_cowa_world_model.training.budget_control import BudgetProfile, BudgetSchedule
from app.vjepa_cowa_world_model.training.configs.common import (
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_max_future_steps_from_config(config: Any) -> int:
    data_cfg = getattr(config, "data", None)
    num_raw_frames = int(getattr(data_cfg, "num_target_frames", 0))
    num_total = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_raw_frames)
    num_observed = resolve_main_encoder_num_observed_steps(config)
    max_future_steps = num_total - num_observed
    if max_future_steps <= 0:
        raise ValueError(
            "budget rollout schedule requires predictor timeline total_steps > observed_steps, "
            f"got {num_total} <= {num_observed}"
        )
    return max_future_steps


def _as_batch_list(value: Any, batch_size: int, *, name: str) -> Optional[List[Any]]:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return [tensor.item()] * batch_size
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"metadata[{name!r}] first dim must be batch_size={batch_size}, got {tuple(tensor.shape)}"
            )
        return [item.item() if torch.is_tensor(item) and item.ndim == 0 else item.tolist() for item in tensor]
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, Sequence):
        if len(value) != batch_size:
            raise ValueError(f"metadata[{name!r}] must have length batch_size={batch_size}, got {len(value)}")
        return list(value)
    if hasattr(value, "tolist"):
        as_list = value.tolist()
        if not isinstance(as_list, list):
            return [as_list] * batch_size
        if len(as_list) != batch_size:
            raise ValueError(f"metadata[{name!r}] must have length batch_size={batch_size}, got {len(as_list)}")
        return as_list
    return [value] * batch_size


def _stringify_scene_component(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if not text:
        raise ValueError("scene id components must not be empty")
    return text.replace("/", "_")


def resolve_oracle_scene_ids(
    metadata: Mapping[str, Any],
    *,
    batch_size: int,
    batch_idx: int,
    rank: int,
) -> List[str]:
    """Resolve stable per-sample scene ids for grouping oracle budget sweeps."""
    del batch_idx, rank
    if not isinstance(metadata, Mapping):
        raise ValueError("budget oracle collection requires dict-like batch metadata")

    stable_ids = _as_batch_list(metadata.get("stable_sample_id"), batch_size, name="stable_sample_id")
    if stable_ids is not None:
        return [_stringify_scene_component(value) for value in stable_ids]

    for key in ("scene_id", "token", "sample_token"):
        values = _as_batch_list(metadata.get(key), batch_size, name=key)
        if values is not None:
            return [_stringify_scene_component(value) for value in values]

    scene_names = _as_batch_list(metadata.get("scene_name"), batch_size, name="scene_name")
    window_starts = _as_batch_list(metadata.get("window_start_pos"), batch_size, name="window_start_pos")
    dataset_roots = _as_batch_list(metadata.get("dataset_root_name"), batch_size, name="dataset_root_name")
    if scene_names is not None:
        if window_starts is None:
            raise ValueError("metadata.scene_name is present but metadata.window_start_pos is missing")
        scene_ids = []
        for index, scene_name in enumerate(scene_names):
            root_value = None if dataset_roots is None else str(dataset_roots[index]).strip()
            prefix = "" if not root_value else f"{_stringify_scene_component(root_value)}:"
            scene_ids.append(
                f"{prefix}{_stringify_scene_component(scene_name)}:"
                f"{_stringify_scene_component(window_starts[index])}"
            )
        return scene_ids

    path_source = metadata.get("pkl_path")
    if path_source is None:
        path_source = metadata.get("ann_file")
    path_values = _as_batch_list(path_source, batch_size, name="pkl_path")
    if path_values is not None:
        if window_starts is None:
            raise ValueError("metadata.pkl_path/ann_file is present but metadata.window_start_pos is missing")
        return [
            f"{Path(_stringify_scene_component(path_value)).stem}:"
            f"{_stringify_scene_component(window_starts[index])}"
            for index, path_value in enumerate(path_values)
        ]

    raise ValueError(
        "budget oracle collection requires one of metadata.scene_id/token/sample_token, "
        "or metadata.scene_name + metadata.window_start_pos"
    )


class BudgetOracleBatchRecorder:
    """Append per-sample fixed-policy budget sweep records as JSONL."""

    def __init__(
        self,
        *,
        output_path: Path,
        budget: float,
        profile: BudgetProfile,
        compute_cost: float,
        lambda_compute: float,
        rank: int,
        world_size: int,
        append: bool = False,
    ):
        self.output_path = Path(output_path)
        self.budget = float(budget)
        self.profile = profile
        self.compute_cost = float(compute_cost)
        self.lambda_compute = float(lambda_compute)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0.0 <= self.budget <= 1.0:
            raise ValueError(f"budget must be in [0, 1], got {budget}")
        if self.compute_cost < 0.0:
            raise ValueError(f"compute_cost must be >= 0, got {compute_cost}")
        if self.lambda_compute < 0.0:
            raise ValueError(f"lambda_compute must be >= 0, got {lambda_compute}")
        if self.world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.shard_path = (
            self.output_path if self.world_size == 1 else Path(str(self.output_path) + f".rank{self.rank}.jsonl")
        )
        self._handle = self.shard_path.open("a" if append else "w", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def record_batch(
        self,
        *,
        metadata: Mapping[str, Any],
        pooled_latent: torch.Tensor,
        task_score: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if pooled_latent.ndim != 2:
            raise ValueError(f"pooled_latent must be [B, D], got {tuple(pooled_latent.shape)}")
        if task_score.ndim != 1 or task_score.shape[0] != pooled_latent.shape[0]:
            raise ValueError(
                f"task_score must be [B] matching pooled_latent, got {tuple(task_score.shape)} "
                f"for pooled_latent {tuple(pooled_latent.shape)}"
            )
        if not torch.isfinite(pooled_latent).all():
            raise ValueError("pooled_latent contains NaN/Inf")
        if not torch.isfinite(task_score).all():
            raise ValueError("task_score contains NaN/Inf")

        batch_size = int(pooled_latent.shape[0])
        scene_ids = resolve_oracle_scene_ids(metadata, batch_size=batch_size, batch_idx=batch_idx, rank=self.rank)
        pooled_latent_cpu = pooled_latent.detach().float().cpu()
        task_score_cpu = task_score.detach().float().cpu()
        profile_dict = asdict(self.profile)
        for local_idx, scene_id in enumerate(scene_ids):
            score = float(task_score_cpu[local_idx].item())
            utility = score - self.lambda_compute * self.compute_cost
            row = {
                "scene_id": scene_id,
                "budget": self.budget,
                "profile": profile_dict,
                "task_score": score,
                "compute_cost": self.compute_cost,
                "utility": utility,
                "pooled_latent": [float(value) for value in pooled_latent_cpu[local_idx].tolist()],
                "rank": self.rank,
                "batch_idx": int(batch_idx),
                "local_sample_index": int(local_idx),
            }
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()


def estimate_budget_compute_cost(profile: BudgetProfile, *, max_future_steps: int) -> float:
    """Return normalized rollout work in ``[0, 1]`` for an applied profile."""
    max_future_steps = int(max_future_steps)
    if max_future_steps <= 0:
        raise ValueError(f"max_future_steps must be > 0, got {max_future_steps}")
    if int(profile.rollout_future_steps) < 0:
        raise ValueError(f"rollout_future_steps must be >= 0, got {profile.rollout_future_steps}")
    if int(profile.rollout_future_steps) > max_future_steps:
        raise ValueError(
            "rollout_future_steps cannot exceed max_future_steps, "
            f"got {profile.rollout_future_steps} > {max_future_steps}"
        )
    cost = float(profile.rollout_future_steps) / float(max_future_steps)
    return max(0.0, min(1.0, cost))


def merge_budget_oracle_shards(output_path: Path, *, world_size: int) -> None:
    """Merge rank-local JSONL shards into the final oracle JSONL file."""
    output_path = Path(output_path)
    if int(world_size) <= 1:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = [Path(str(output_path) + f".rank{rank}.jsonl") for rank in range(int(world_size))]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing budget oracle shard(s): {missing}")
    with output_path.open("w", encoding="utf-8") as merged:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as shard:
                for line in shard:
                    merged.write(line)


def run_budget_oracle_collection(
    *,
    encoder: Any,
    predictor: Any,
    planner: Any,
    val_loader: Any,
    val_sampler: Any,
    config: Any,
    rank: int,
    world_size: int,
    token_ae: Any = None,
    runtime_normalize_reps: Optional[bool] = None,
    multiview_fusion: Any = None,
    value_head: Any = None,
) -> Dict[str, Dict[str, float]]:
    """Run validation once per budget grid value and write oracle records."""
    from app.vjepa_cowa_world_model.val_command import run_validation

    budget_cfg = getattr(config, "budget_controller", None)
    if budget_cfg is None or not bool(getattr(budget_cfg, "enabled", False)):
        raise ValueError("budget oracle collection requires budget_controller.enabled=true")
    if getattr(budget_cfg, "mode", None) != "oracle_collection":
        raise ValueError(
            "run_budget_oracle_collection requires budget_controller.mode='oracle_collection', "
            f"got {getattr(budget_cfg, 'mode', None)!r}"
        )
    if int(getattr(budget_cfg, "feature_dim", 0)) != 0:
        raise ValueError("Stage3B oracle collection currently supports budget_controller.feature_dim=0 only")
    if val_loader is None or val_sampler is None:
        raise ValueError("Stage3B oracle collection requires a validation dataloader and sampler")
    output_path_raw = getattr(budget_cfg, "oracle_output_path", None)
    if not output_path_raw:
        raise ValueError("budget_controller.oracle_output_path is required for oracle_collection")
    predictor_type = getattr(getattr(config, "train", None), "predictor_type", None)
    if predictor_type != "ac_transformer":
        raise ValueError(
            "rollout budget oracle collection currently supports train.predictor_type='ac_transformer' only, "
            f"got {predictor_type!r}"
        )

    schedule = BudgetSchedule.from_mapping(getattr(budget_cfg, "schedule", None))
    max_future_steps = _resolve_max_future_steps_from_config(config)

    output_path = Path(output_path_raw)
    metrics_by_budget: Dict[str, Dict[str, float]] = {}
    budget_grid = [float(budget) for budget in getattr(budget_cfg, "oracle_budget_grid")]
    for budget_index, budget in enumerate(budget_grid):
        profile = schedule.profile(budget, max_future_steps=max_future_steps)
        compute_cost = estimate_budget_compute_cost(profile, max_future_steps=max_future_steps)
        recorder = BudgetOracleBatchRecorder(
            output_path=output_path,
            budget=budget,
            profile=profile,
            compute_cost=compute_cost,
            lambda_compute=float(getattr(budget_cfg, "lambda_compute", 0.0)),
            rank=rank,
            world_size=world_size,
            append=budget_index > 0,
        )
        logger.info(
            "[budget_oracle] budget=%.3f profile=%s compute_cost=%.4f shard=%s",
            budget,
            profile,
            compute_cost,
            recorder.shard_path,
        )
        try:
            metrics_by_budget[f"{budget:.6g}"] = run_validation(
                encoder=encoder,
                predictor=predictor,
                planner=planner,
                val_loader=val_loader,
                val_sampler=val_sampler,
                config=config,
                epoch=0,
                rank=rank,
                world_size=world_size,
                use_tubelet_repeat=config.data.use_tubelet_repeat,
                vis_output_dir=None,
                token_ae=token_ae,
                runtime_normalize_reps=runtime_normalize_reps,
                multiview_fusion=multiview_fusion,
                value_head=value_head,
                budget_oracle_recorder=recorder,
                budget_oracle_profile=profile,
            )
        finally:
            recorder.close()

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        merge_budget_oracle_shards(output_path, world_size=world_size)
        logger.info("[budget_oracle] wrote oracle records to %s", output_path)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return metrics_by_budget
