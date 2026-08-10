"""Online contextual-bandit GRPO helpers for the rollout budget controller."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_cowa_world_model.training.budget_control import (
    BudgetController,
    BudgetProfile,
    BudgetSchedule,
    budget_controller_grpo_loss,
    compute_budget_utility,
    load_budget_controller_from_checkpoint,
    sample_beta_budget,
)
from app.vjepa_cowa_world_model.training.budget_oracle_collection import estimate_budget_compute_cost
from app.vjepa_cowa_world_model.training.runtimes.budgeted_rollout_runtime import (
    prepare_budgeted_rollout_batch,
    score_prepared_budget_profile,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import resolve_main_timeline
from app.vjepa_cowa_world_model.utils.eval_determinism import deterministic_eval_rng, normalize_seed_value
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module
from src.utils.logging import get_logger

logger = get_logger(__name__)

ProfileScore = Union[float, torch.Tensor]
ProfileScoreFn = Callable[[BudgetProfile, int], ProfileScore]
PrepareBatchFn = Callable[..., tuple[Any, torch.Tensor, Mapping[str, Any]]]
PreparedProfileScoreFn = Callable[..., ProfileScore]


def make_online_group_seed(
    *,
    base_seed: int,
    epoch: int,
    batch_idx: int,
    metadata: Mapping[str, Any] | None,
) -> int:
    """Build a stable per-scene, per-epoch seed shared by all budgets in one GRPO group."""
    identity = normalize_seed_value(dict(metadata)) if metadata else {"batch_idx": int(batch_idx)}
    payload = (int(base_seed), int(epoch), int(batch_idx), identity)
    digest = hashlib.blake2b(repr(payload).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _as_scalar_score(value: ProfileScore, *, device: torch.device) -> torch.Tensor:
    score = value.detach() if torch.is_tensor(value) else torch.tensor(float(value))
    if score.numel() != 1:
        raise ValueError(f"score_profile_fn must return one scalar score, got shape {tuple(score.shape)}")
    score = score.reshape(()).to(device=device, dtype=torch.float32)
    if not torch.isfinite(score):
        raise ValueError("score_profile_fn returned NaN/Inf")
    return score


def evaluate_online_budget_group(
    *,
    sampled_budget: torch.Tensor,
    log_prob: torch.Tensor,
    schedule: BudgetSchedule,
    max_future_steps: int,
    lambda_compute: float,
    group_seed: int,
    score_profile_fn: ProfileScoreFn,
) -> Dict[str, torch.Tensor]:
    """Evaluate sampled budgets online and return a one-scene GRPO loss and diagnostics."""
    if sampled_budget.ndim != 1 or log_prob.ndim != 1 or sampled_budget.shape != log_prob.shape:
        raise ValueError(
            "sampled_budget and log_prob must be matching [K] tensors, got "
            f"{tuple(sampled_budget.shape)} and {tuple(log_prob.shape)}"
        )
    if sampled_budget.numel() < 2:
        raise ValueError(f"online GRPO requires at least 2 samples per scene, got {sampled_budget.numel()}")
    if not torch.isfinite(sampled_budget).all() or not torch.isfinite(log_prob).all():
        raise ValueError("sampled_budget and log_prob must contain only finite values")
    if not torch.all((sampled_budget >= 0.0) & (sampled_budget <= 1.0)):
        raise ValueError("sampled_budget entries must be in [0, 1]")

    profiles = [
        schedule.profile(float(value), max_future_steps=max_future_steps)
        for value in sampled_budget.detach().float().cpu().tolist()
    ]
    score_cache: Dict[int, torch.Tensor] = {}
    for profile in profiles:
        steps = int(profile.rollout_future_steps)
        if steps not in score_cache:
            score_cache[steps] = _as_scalar_score(
                score_profile_fn(profile, int(group_seed)),
                device=log_prob.device,
            )

    task_score = torch.stack([score_cache[int(profile.rollout_future_steps)] for profile in profiles])
    compute_cost = torch.tensor(
        [estimate_budget_compute_cost(profile, max_future_steps=max_future_steps) for profile in profiles],
        device=log_prob.device,
        dtype=torch.float32,
    )
    utility = compute_budget_utility(task_score, compute_cost, lambda_compute=lambda_compute)
    loss_dict = budget_controller_grpo_loss(log_prob, utility)
    return {
        **loss_dict,
        "task_score": task_score,
        "compute_cost": compute_cost,
        "utility": utility,
        "rollout_steps": torch.tensor(
            [int(profile.rollout_future_steps) for profile in profiles],
            device=log_prob.device,
            dtype=torch.long,
        ),
        "unique_profile_count": torch.tensor(len(score_cache), device=log_prob.device, dtype=torch.long),
    }


def online_grpo_controller_step(
    *,
    controller: Any,
    pooled_latent: torch.Tensor,
    num_samples: int,
    schedule: BudgetSchedule,
    max_future_steps: int,
    lambda_compute: float,
    group_seed: int,
    score_profile_fn: ProfileScoreFn,
) -> Dict[str, torch.Tensor]:
    """Sample one scene-level budget group through ``controller.forward`` and build its GRPO loss."""
    if pooled_latent.ndim != 2 or int(pooled_latent.shape[0]) != 1:
        raise ValueError(f"pooled_latent must be [1, D] for online GRPO, got {tuple(pooled_latent.shape)}")
    num_samples = int(num_samples)
    if num_samples < 2:
        raise ValueError(f"num_samples must be >= 2, got {num_samples}")
    repeated = pooled_latent.detach().repeat(num_samples, 1)
    alpha, beta = controller(repeated)
    output = sample_beta_budget(alpha, beta, deterministic=False)
    result = evaluate_online_budget_group(
        sampled_budget=output.budget,
        log_prob=output.log_prob,
        schedule=schedule,
        max_future_steps=max_future_steps,
        lambda_compute=lambda_compute,
        group_seed=group_seed,
        score_profile_fn=score_profile_fn,
    )
    result["sampled_budget"] = output.budget.detach()
    result["alpha"] = output.alpha.detach()
    result["beta"] = output.beta.detach()
    return result


def _freeze_main_policy(*modules: Any) -> None:
    for module in modules:
        if module is None:
            continue
        core = unwrap_module(module)
        core.eval()
        for parameter in core.parameters():
            parameter.requires_grad_(False)


def resolve_online_max_future_steps(config: Any, encoder: Any) -> int:
    """Resolve the budget horizon in predictor steps, accounting for encoder frame stride."""
    timeline = resolve_main_timeline(
        config,
        encoder=unwrap_module(encoder),
        num_raw_frames=int(config.data.num_target_frames),
    )
    max_future_steps = int(timeline.num_future_steps)
    if max_future_steps <= 0:
        raise ValueError(
            "online_grpo requires at least one future predictor step, got "
            f"raw_frames={timeline.raw_num_frames}, frame_stride={timeline.frame_stride}, "
            f"observed_steps={timeline.num_observed_steps}, total_steps={timeline.num_time_steps}"
        )
    return max_future_steps


def _online_output_paths(config: Any) -> tuple[Path, Path]:
    configured = config.budget_controller.output_checkpoint
    final_path = Path(configured or os.path.join(config.meta.folder, "budget_controller.pt"))
    latest_path = final_path.with_name(f"{final_path.stem}_latest{final_path.suffix}")
    return latest_path, final_path


def _require_epoch_latest_resume_path(config: Any, path: str | Path) -> Path:
    expected_path, _ = _online_output_paths(config)
    configured_path = Path(path)
    if configured_path.absolute() != expected_path.absolute():
        raise ValueError(
            "online_grpo resume checkpoint must be the run's epoch-latest checkpoint: "
            f"expected {expected_path}, got {configured_path}"
        )
    return expected_path


def configure_online_grpo_preempt_resume(config: Any, *, resume_preempt: bool) -> None:
    """Point a preempted online run at its dedicated epoch-latest controller checkpoint."""
    if not bool(resume_preempt):
        return
    if config.budget_controller.online_resume_checkpoint:
        latest_path = _require_epoch_latest_resume_path(
            config,
            config.budget_controller.online_resume_checkpoint,
        )
        if not latest_path.exists():
            raise FileNotFoundError(f"online_grpo epoch-latest checkpoint does not exist: {latest_path}")
        logger.info(
            "[budget_controller][online_grpo] keeping explicit resume checkpoint %s",
            latest_path,
        )
        return
    latest_path, _ = _online_output_paths(config)
    if not latest_path.exists():
        raise FileNotFoundError(
            "online_grpo resume_preempt=True but the controller latest checkpoint does not exist: " f"{latest_path}"
        )
    config.budget_controller.online_resume_checkpoint = str(latest_path)
    logger.info(
        "[budget_controller][online_grpo] resume_preempt=True; resuming controller from %s",
        latest_path,
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _controller_checkpoint_payload(
    *,
    controller: BudgetController,
    optimizer: torch.optim.Optimizer,
    config: Any,
    completed_epoch: int,
    global_step: int,
    world_size: int,
    max_future_steps: int,
    metrics: Mapping[str, float],
    complete: bool,
) -> Dict[str, Any]:
    budget_config = config.budget_controller
    return {
        "controller": controller.state_dict(),
        "optimizer": optimizer.state_dict(),
        "latent_dim": int(controller.latent_dim),
        "feature_dim": int(controller.feature_dim),
        "hidden_dim": int(controller.hidden_dim),
        "min_concentration": float(controller.min_concentration),
        "mode": "online_grpo",
        "complete": bool(complete),
        "completed_epoch": int(completed_epoch),
        "global_step": int(global_step),
        "init_checkpoint": str(budget_config.controller_checkpoint),
        "predictor_checkpoint": str(config.meta.predictor_checkpoint),
        "planner_checkpoint": str(config.meta.pretrain_checkpoint_full),
        "reward_source": str(budget_config.online_reward_source),
        "lambda_compute": float(budget_config.lambda_compute),
        "schedule": dict(budget_config.schedule),
        "grpo_num_samples_per_scene": int(budget_config.grpo_num_samples_per_scene),
        "max_future_steps": int(max_future_steps),
        "seed": int(config.meta.seed),
        "optimizer_config": {
            "lr": float(config.optimization.lr),
            "betas": tuple(float(value) for value in config.optimization.betas),
            "eps": float(config.optimization.eps),
            "weight_decay": float(config.optimization.weight_decay),
            "grad_clip_norm": float(config.optimization.grad_clip_norm),
            "ipe": None if config.optimization.ipe is None else int(config.optimization.ipe),
        },
        "world_size": int(world_size),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def _load_online_resume_checkpoint(
    *,
    path: str,
    controller: BudgetController,
    optimizer: torch.optim.Optimizer,
    config: Any,
    world_size: int,
    max_future_steps: int,
    device: torch.device,
) -> tuple[int, int, Dict[str, float]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"budget_controller.online_resume_checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"online GRPO checkpoint must be a mapping: {checkpoint_path}")
    expected = {
        "mode": "online_grpo",
        "init_checkpoint": str(config.budget_controller.controller_checkpoint),
        "predictor_checkpoint": str(config.meta.predictor_checkpoint),
        "planner_checkpoint": str(config.meta.pretrain_checkpoint_full),
        "reward_source": str(config.budget_controller.online_reward_source),
        "lambda_compute": float(config.budget_controller.lambda_compute),
        "schedule": dict(config.budget_controller.schedule),
        "grpo_num_samples_per_scene": int(config.budget_controller.grpo_num_samples_per_scene),
        "max_future_steps": int(max_future_steps),
        "seed": int(config.meta.seed),
        "optimizer_config": {
            "lr": float(config.optimization.lr),
            "betas": tuple(float(value) for value in config.optimization.betas),
            "eps": float(config.optimization.eps),
            "weight_decay": float(config.optimization.weight_decay),
            "grad_clip_norm": float(config.optimization.grad_clip_norm),
            "ipe": None if config.optimization.ipe is None else int(config.optimization.ipe),
        },
        "world_size": int(world_size),
    }
    mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise ValueError(f"online GRPO resume checkpoint metadata mismatch: {mismatches}")
    for key in ("controller", "optimizer", "completed_epoch", "global_step"):
        if key not in payload:
            raise RuntimeError(f"online GRPO resume checkpoint missing key {key!r}: {checkpoint_path}")
    controller.load_state_dict(payload["controller"])
    optimizer.load_state_dict(payload["optimizer"])
    metrics = {key: float(value) for key, value in dict(payload.get("metrics", {})).items()}
    return int(payload["completed_epoch"]) + 1, int(payload["global_step"]), metrics


def _reduce_epoch_metrics(
    sums: Mapping[str, float],
    *,
    count: int,
    device: torch.device,
    world_size: int,
) -> Dict[str, float]:
    keys = tuple(sorted(sums))
    values = [float(sums[key]) for key in keys] + [float(count)]
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if int(world_size) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = float(tensor[-1].item())
    if total_count <= 0.0:
        raise RuntimeError("online GRPO epoch produced no training groups")
    return {key: float(tensor[index].item() / total_count) for index, key in enumerate(keys)}


def train_budget_controller_online_grpo(
    *,
    config: Any,
    encoder: Any,
    predictor: Any,
    planner: Any,
    train_loader: Any,
    train_sampler: Any,
    device: torch.device,
    rank: int,
    world_size: int,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    token_ae: Any = None,
    multiview_fusion: Any = None,
    prepare_batch_fn: PrepareBatchFn = prepare_budgeted_rollout_batch,
    profile_score_fn: PreparedProfileScoreFn = score_prepared_budget_profile,
) -> Dict[str, float]:
    """Fine-tune a BC controller with rewards from live frozen AC predictor/planner rollouts."""
    budget_config = config.budget_controller
    if not budget_config.enabled or budget_config.mode != "online_grpo":
        raise ValueError(
            "train_budget_controller_online_grpo requires budget_controller.enabled=true and mode='online_grpo'"
        )
    if int(world_size) > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("online_grpo world_size > 1 requires an initialized torch.distributed process group")
        if dist.get_world_size() != int(world_size) or dist.get_rank() != int(rank):
            raise RuntimeError(
                "online_grpo distributed context mismatch: "
                f"arguments rank/world_size={rank}/{world_size}, "
                f"process group={dist.get_rank()}/{dist.get_world_size()}"
            )

    max_future_steps = resolve_online_max_future_steps(config, encoder)
    _freeze_main_policy(encoder, predictor, planner, token_ae, multiview_fusion)
    controller_core = load_budget_controller_from_checkpoint(
        budget_config.controller_checkpoint,
        device=device,
        expected_mode="oracle_distillation",
    )
    if int(controller_core.feature_dim) != 0:
        raise ValueError(f"online_grpo controller checkpoint feature_dim must be 0, got {controller_core.feature_dim}")
    for parameter in controller_core.parameters():
        parameter.requires_grad_(True)
    controller_core.train()
    controller: Any = controller_core
    if int(world_size) > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        controller = DistributedDataParallel(
            controller_core,
            device_ids=device_ids,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(config.optimization.lr),
        betas=config.optimization.betas,
        eps=float(config.optimization.eps),
        weight_decay=float(config.optimization.weight_decay),
    )
    start_epoch = 0
    global_step = 0
    metrics: Dict[str, float] = {}
    if budget_config.online_resume_checkpoint:
        resume_checkpoint = _require_epoch_latest_resume_path(
            config,
            budget_config.online_resume_checkpoint,
        )
        start_epoch, global_step, metrics = _load_online_resume_checkpoint(
            path=str(resume_checkpoint),
            controller=controller_core,
            optimizer=optimizer,
            config=config,
            world_size=world_size,
            max_future_steps=max_future_steps,
            device=device,
        )
        logger.info(
            "[budget_controller][online_grpo] resumed from %s at epoch=%d global_step=%d",
            resume_checkpoint,
            start_epoch,
            global_step,
        )

    epochs = int(config.optimization.epochs)
    if epochs <= 0:
        raise ValueError(f"optimization.epochs must be > 0, got {epochs}")
    if start_epoch >= epochs:
        raise ValueError(
            f"online GRPO resume already completed configured epochs: start_epoch={start_epoch}, epochs={epochs}"
        )
    configured_ipe = config.optimization.ipe
    if configured_ipe is not None and int(configured_ipe) <= 0:
        raise ValueError(f"optimization.ipe must be > 0 when set, got {configured_ipe}")

    schedule = BudgetSchedule.from_mapping(budget_config.schedule)
    num_samples = int(budget_config.grpo_num_samples_per_scene)
    latest_path, final_path = _online_output_paths(config)
    completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        controller.train()
        sums = {
            "compute_cost": 0.0,
            "loss": 0.0,
            "mean_budget": 0.0,
            "rollout_steps": 0.0,
            "task_score": 0.0,
            "unique_profiles": 0.0,
            "utility": 0.0,
            "zero_advantage_group": 0.0,
        }
        group_count = 0
        for batch_idx, sample in enumerate(train_loader):
            if configured_ipe is not None and batch_idx >= int(configured_ipe):
                break
            prepared, pooled_latent, metadata = prepare_batch_fn(
                sample,
                encoder=encoder,
                predictor=predictor,
                planner=planner,
                config=config,
                device=device,
                tokens_per_frame=tokens_per_frame,
                runtime_normalize_reps=runtime_normalize_reps,
                token_ae=token_ae,
                multiview_fusion=multiview_fusion,
            )
            pooled_latent = pooled_latent.detach().float().to(device)
            if int(pooled_latent.shape[-1]) != int(controller_core.latent_dim):
                raise ValueError(
                    "online_grpo observed latent dimension does not match the controller checkpoint: "
                    f"latent={pooled_latent.shape[-1]}, controller={controller_core.latent_dim}"
                )
            group_seed = make_online_group_seed(
                base_seed=int(config.meta.seed),
                epoch=epoch,
                batch_idx=batch_idx,
                metadata=metadata,
            )

            def score_profile(profile: BudgetProfile, seed: int) -> ProfileScore:
                return profile_score_fn(prepared, profile, seed=seed)

            with deterministic_eval_rng(group_seed ^ 0x5DEECE66D, device):
                result = online_grpo_controller_step(
                    controller=controller,
                    pooled_latent=pooled_latent,
                    num_samples=num_samples,
                    schedule=schedule,
                    max_future_steps=max_future_steps,
                    lambda_compute=float(budget_config.lambda_compute),
                    group_seed=group_seed,
                    score_profile_fn=score_profile,
                )
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"online GRPO loss became non-finite at epoch={epoch}, batch={batch_idx}: {loss.item()}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(config.optimization.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    controller.parameters(),
                    max_norm=float(config.optimization.grad_clip_norm),
                )
            optimizer.step()

            sums["loss"] += float(loss.detach().cpu())
            sums["task_score"] += float(result["task_score"].mean().detach().cpu())
            sums["compute_cost"] += float(result["compute_cost"].mean().detach().cpu())
            sums["utility"] += float(result["utility"].mean().detach().cpu())
            sums["mean_budget"] += float(result["sampled_budget"].mean().detach().cpu())
            sums["rollout_steps"] += float(result["rollout_steps"].float().mean().detach().cpu())
            sums["unique_profiles"] += float(result["unique_profile_count"].detach().cpu())
            sums["zero_advantage_group"] += float(bool(torch.all(result["advantage"].detach().abs() <= 1e-12).item()))
            group_count += 1
            global_step += 1

        metrics = _reduce_epoch_metrics(
            sums,
            count=group_count,
            device=device,
            world_size=world_size,
        )
        completed_epoch = epoch
        if rank == 0:
            logger.info(
                "[budget_controller][online_grpo][epoch %d/%d] loss=%.6f task_score=%.6f "
                "utility=%.6f cost=%.4f budget=%.4f rollout=%.3f unique=%.3f zero_adv=%.3f",
                epoch + 1,
                epochs,
                metrics["loss"],
                metrics["task_score"],
                metrics["utility"],
                metrics["compute_cost"],
                metrics["mean_budget"],
                metrics["rollout_steps"],
                metrics["unique_profiles"],
                metrics["zero_advantage_group"],
            )
            _atomic_torch_save(
                _controller_checkpoint_payload(
                    controller=controller_core,
                    optimizer=optimizer,
                    config=config,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    world_size=world_size,
                    max_future_steps=max_future_steps,
                    metrics=metrics,
                    complete=False,
                ),
                latest_path,
            )
        if int(world_size) > 1:
            dist.barrier()

    if rank == 0:
        _atomic_torch_save(
            _controller_checkpoint_payload(
                controller=controller_core,
                optimizer=optimizer,
                config=config,
                completed_epoch=completed_epoch,
                global_step=global_step,
                world_size=world_size,
                max_future_steps=max_future_steps,
                metrics=metrics,
                complete=True,
            ),
            final_path,
        )
        logger.info("[budget_controller][online_grpo] saved final checkpoint to %s", final_path)
    if int(world_size) > 1:
        dist.barrier()
    return metrics
