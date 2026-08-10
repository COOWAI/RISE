"""Offline training utilities for the continuous budget controller."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from app.vjepa_cowa_world_model.training.budget_control import (
    BudgetController,
    budget_controller_bc_loss,
    budget_controller_grpo_loss,
    load_budget_controller_from_checkpoint,
)
from app.vjepa_cowa_world_model.training.config import TrainingConfig
from app.vjepa_cowa_world_model.training.loop import make_dataloader_generator
from src.utils.logging import AverageMeter, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BudgetOracleExamples:
    """Distilled oracle targets for ``Controller(z_obs) -> budget``."""

    scene_ids: List[str]
    pooled_latent: torch.Tensor
    cheap_features: Optional[torch.Tensor]
    target_budget: torch.Tensor
    target_utility: torch.Tensor


@dataclass(frozen=True)
class BudgetOracleSweeps:
    """Per-scene oracle utility curves for offline contextual-bandit GRPO."""

    scene_ids: List[str]
    pooled_latent: torch.Tensor
    cheap_features: Optional[torch.Tensor]
    budget_grid: torch.Tensor
    utility_grid: torch.Tensor
    target_budget: torch.Tensor
    target_utility: torch.Tensor


class BudgetOracleDataset(Dataset):
    """Tiny tensor dataset for offline budget-controller distillation."""

    def __init__(self, examples: BudgetOracleExamples):
        self.examples = examples

    def __len__(self) -> int:
        return int(self.examples.target_budget.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "pooled_latent": self.examples.pooled_latent[index],
            "target_budget": self.examples.target_budget[index],
            "target_utility": self.examples.target_utility[index],
        }
        if self.examples.cheap_features is not None:
            item["cheap_features"] = self.examples.cheap_features[index]
        return item


class BudgetOracleSweepDataset(Dataset):
    """Tiny tensor dataset preserving the full oracle sweep per scene."""

    def __init__(self, sweeps: BudgetOracleSweeps):
        self.sweeps = sweeps

    def __len__(self) -> int:
        return int(self.sweeps.budget_grid.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "pooled_latent": self.sweeps.pooled_latent[index],
            "budget_grid": self.sweeps.budget_grid[index],
            "utility_grid": self.sweeps.utility_grid[index],
            "target_budget": self.sweeps.target_budget[index],
            "target_utility": self.sweeps.target_utility[index],
        }
        if self.sweeps.cheap_features is not None:
            item["cheap_features"] = self.sweeps.cheap_features[index]
        return item


def _as_float_list(record: Dict[str, Any], key: str, *, line_number: int) -> List[float]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"budget oracle line {line_number}: {key!r} must be a non-empty list")
    values = [float(item) for item in value]
    if not all(torch.isfinite(torch.tensor(values)).tolist()):
        raise ValueError(f"budget oracle line {line_number}: {key!r} contains non-finite values")
    return values


def _record_budget(record: Dict[str, Any], *, line_number: int) -> float:
    if "target_budget" in record:
        budget = float(record["target_budget"])
    elif "budget" in record:
        budget = float(record["budget"])
    else:
        raise ValueError(f"budget oracle line {line_number}: expected 'target_budget' or sweep 'budget'")
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget oracle line {line_number}: budget must be in [0, 1], got {budget}")
    return budget


def _record_utility(record: Dict[str, Any], *, lambda_compute: float, line_number: int) -> float:
    if "target_budget" in record and "utility" not in record:
        return 0.0
    if "utility" in record:
        utility = float(record["utility"])
    elif "task_score" in record and "compute_cost" in record:
        utility = float(record["task_score"]) - float(lambda_compute) * float(record["compute_cost"])
    else:
        raise ValueError(
            f"budget oracle line {line_number}: sweep rows require 'utility' or both 'task_score' and 'compute_cost'"
        )
    if not torch.isfinite(torch.tensor(utility)):
        raise ValueError(f"budget oracle line {line_number}: utility must be finite, got {utility}")
    return utility


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"budget_controller.oracle_path does not exist: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"budget oracle line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"budget oracle line {line_number}: expected a JSON object")
            row["_line_number"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"budget oracle file is empty: {path}")
    return rows


def load_budget_oracle_examples(
    oracle_path: str | Path,
    *,
    feature_dim: int,
    lambda_compute: float,
) -> BudgetOracleExamples:
    """Load JSONL oracle sweeps and choose the best budget target per scene.

    Each JSON line must include ``scene_id``, ``pooled_latent`` and either a
    direct ``target_budget`` or one sweep ``budget`` with ``utility``. Sweep rows
    are grouped by ``scene_id`` and the highest-utility budget becomes the
    supervised target.
    """
    feature_dim = int(feature_dim)
    if feature_dim < 0:
        raise ValueError(f"feature_dim must be >= 0, got {feature_dim}")
    lambda_compute = float(lambda_compute)
    if lambda_compute < 0.0:
        raise ValueError(f"lambda_compute must be >= 0, got {lambda_compute}")

    best_by_scene: Dict[str, Dict[str, Any]] = {}
    for row in _load_jsonl_rows(Path(oracle_path)):
        line_number = int(row["_line_number"])
        scene_id_raw = row.get("scene_id")
        if scene_id_raw is None:
            raise ValueError(f"budget oracle line {line_number}: missing 'scene_id'")
        scene_id = str(scene_id_raw)
        pooled_latent = _as_float_list(row, "pooled_latent", line_number=line_number)
        budget = _record_budget(row, line_number=line_number)
        utility = _record_utility(row, lambda_compute=lambda_compute, line_number=line_number)
        cheap_features: Optional[List[float]] = None
        if feature_dim > 0:
            cheap_features = _as_float_list(row, "cheap_features", line_number=line_number)
            if len(cheap_features) != feature_dim:
                raise ValueError(
                    f"budget oracle line {line_number}: cheap_features length must be {feature_dim}, "
                    f"got {len(cheap_features)}"
                )
        elif "cheap_features" in row and row["cheap_features"]:
            raise ValueError("budget oracle contains cheap_features but budget_controller.feature_dim=0")

        candidate = {
            "scene_id": scene_id,
            "pooled_latent": pooled_latent,
            "cheap_features": cheap_features,
            "target_budget": budget,
            "target_utility": utility,
        }
        current = best_by_scene.get(scene_id)
        if current is None or utility > float(current["target_utility"]):
            best_by_scene[scene_id] = candidate

    scene_ids = sorted(best_by_scene)
    latent_rows = [best_by_scene[scene_id]["pooled_latent"] for scene_id in scene_ids]
    latent_dim = len(latent_rows[0])
    if any(len(row) != latent_dim for row in latent_rows):
        raise ValueError("budget oracle pooled_latent length must be identical for all scenes")

    pooled_latent = torch.tensor(latent_rows, dtype=torch.float32)
    target_budget = torch.tensor(
        [best_by_scene[scene_id]["target_budget"] for scene_id in scene_ids], dtype=torch.float32
    )
    target_utility = torch.tensor(
        [best_by_scene[scene_id]["target_utility"] for scene_id in scene_ids], dtype=torch.float32
    )
    cheap_features_tensor: Optional[torch.Tensor] = None
    if feature_dim > 0:
        cheap_rows = [best_by_scene[scene_id]["cheap_features"] for scene_id in scene_ids]
        cheap_features_tensor = torch.tensor(cheap_rows, dtype=torch.float32)

    return BudgetOracleExamples(
        scene_ids=scene_ids,
        pooled_latent=pooled_latent,
        cheap_features=cheap_features_tensor,
        target_budget=target_budget,
        target_utility=target_utility,
    )


def load_budget_oracle_sweeps(
    oracle_path: str | Path,
    *,
    feature_dim: int,
    lambda_compute: float,
) -> BudgetOracleSweeps:
    """Load per-scene budget/utility curves from a Stage3C oracle JSONL.

    Rows are grouped by ``scene_id`` and sorted by ``budget``. Each scene must
    contain at least two strictly increasing budget points, because GRPO rewards
    for sampled continuous budgets are obtained by linear interpolation.
    """
    feature_dim = int(feature_dim)
    if feature_dim < 0:
        raise ValueError(f"feature_dim must be >= 0, got {feature_dim}")
    lambda_compute = float(lambda_compute)
    if lambda_compute < 0.0:
        raise ValueError(f"lambda_compute must be >= 0, got {lambda_compute}")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in _load_jsonl_rows(Path(oracle_path)):
        line_number = int(row["_line_number"])
        scene_id_raw = row.get("scene_id")
        if scene_id_raw is None:
            raise ValueError(f"budget oracle line {line_number}: missing 'scene_id'")
        scene_id = str(scene_id_raw)
        pooled_latent = _as_float_list(row, "pooled_latent", line_number=line_number)
        budget = _record_budget(row, line_number=line_number)
        utility = _record_utility(row, lambda_compute=lambda_compute, line_number=line_number)
        cheap_features: Optional[List[float]] = None
        if feature_dim > 0:
            cheap_features = _as_float_list(row, "cheap_features", line_number=line_number)
            if len(cheap_features) != feature_dim:
                raise ValueError(
                    f"budget oracle line {line_number}: cheap_features length must be {feature_dim}, "
                    f"got {len(cheap_features)}"
                )
        elif "cheap_features" in row and row["cheap_features"]:
            raise ValueError("budget oracle contains cheap_features but budget_controller.feature_dim=0")
        grouped.setdefault(scene_id, []).append(
            {
                "line_number": line_number,
                "pooled_latent": pooled_latent,
                "cheap_features": cheap_features,
                "budget": budget,
                "utility": utility,
            }
        )

    scene_ids = sorted(grouped)
    latent_rows: List[List[float]] = []
    cheap_rows: List[Optional[List[float]]] = []
    budget_rows: List[List[float]] = []
    utility_rows: List[List[float]] = []
    target_budget_rows: List[float] = []
    target_utility_rows: List[float] = []
    expected_grid_size: Optional[int] = None
    for scene_id in scene_ids:
        entries = sorted(grouped[scene_id], key=lambda item: float(item["budget"]))
        if len(entries) < 2:
            raise ValueError(f"budget oracle scene {scene_id!r} must contain at least 2 budget points")
        budgets = [float(entry["budget"]) for entry in entries]
        if any(budgets[index] <= budgets[index - 1] for index in range(1, len(budgets))):
            raise ValueError(f"budget oracle scene {scene_id!r} budget grid must be strictly increasing")
        if expected_grid_size is None:
            expected_grid_size = len(entries)
        elif len(entries) != expected_grid_size:
            raise ValueError(
                "budget oracle sweeps must have the same number of budget points per scene; "
                f"expected {expected_grid_size}, got {len(entries)} for scene {scene_id!r}"
            )

        pooled_latent = list(entries[0]["pooled_latent"])
        if any(list(entry["pooled_latent"]) != pooled_latent for entry in entries[1:]):
            raise ValueError(f"budget oracle scene {scene_id!r} pooled_latent must be identical across budget rows")
        cheap_features = entries[0]["cheap_features"]
        if feature_dim > 0 and any(entry["cheap_features"] != cheap_features for entry in entries[1:]):
            raise ValueError(f"budget oracle scene {scene_id!r} cheap_features must be identical across budget rows")

        utilities = [float(entry["utility"]) for entry in entries]
        best_index = max(range(len(utilities)), key=lambda index: utilities[index])
        latent_rows.append(pooled_latent)
        cheap_rows.append(cheap_features)
        budget_rows.append(budgets)
        utility_rows.append(utilities)
        target_budget_rows.append(budgets[best_index])
        target_utility_rows.append(utilities[best_index])

    latent_dim = len(latent_rows[0])
    if any(len(row) != latent_dim for row in latent_rows):
        raise ValueError("budget oracle pooled_latent length must be identical for all scenes")
    pooled_latent = torch.tensor(latent_rows, dtype=torch.float32)
    budget_grid = torch.tensor(budget_rows, dtype=torch.float32)
    utility_grid = torch.tensor(utility_rows, dtype=torch.float32)
    target_budget = torch.tensor(target_budget_rows, dtype=torch.float32)
    target_utility = torch.tensor(target_utility_rows, dtype=torch.float32)
    cheap_features_tensor: Optional[torch.Tensor] = None
    if feature_dim > 0:
        cheap_features_tensor = torch.tensor(cheap_rows, dtype=torch.float32)

    return BudgetOracleSweeps(
        scene_ids=scene_ids,
        pooled_latent=pooled_latent,
        cheap_features=cheap_features_tensor,
        budget_grid=budget_grid,
        utility_grid=utility_grid,
        target_budget=target_budget,
        target_utility=target_utility,
    )


def interpolate_budget_utilities(
    budget_grid: torch.Tensor,
    utility_grid: torch.Tensor,
    sampled_budget: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate oracle utility at sampled continuous budgets.

    Parameters
    ----------
    budget_grid:
        ``[B, G]`` strictly increasing budget points.
    utility_grid:
        ``[B, G]`` utility values aligned with ``budget_grid``.
    sampled_budget:
        ``[B]`` or ``[B, S]`` sampled budgets.
    """
    if budget_grid.ndim != 2:
        raise ValueError(f"budget_grid must be [B, G], got {tuple(budget_grid.shape)}")
    if utility_grid.shape != budget_grid.shape:
        raise ValueError(
            f"utility_grid shape must match budget_grid {tuple(budget_grid.shape)}, got {tuple(utility_grid.shape)}"
        )
    if int(budget_grid.shape[1]) < 2:
        raise ValueError("budget_grid must contain at least 2 budget points")
    squeezed = False
    if sampled_budget.ndim == 1:
        sampled_budget = sampled_budget.unsqueeze(1)
        squeezed = True
    if sampled_budget.ndim != 2 or int(sampled_budget.shape[0]) != int(budget_grid.shape[0]):
        raise ValueError(
            f"sampled_budget must be [B] or [B, S] with B={budget_grid.shape[0]}, "
            f"got {tuple(sampled_budget.shape)}"
        )
    if not torch.all(torch.isfinite(utility_grid)):
        raise ValueError("utility_grid must be finite")
    if not torch.all(torch.isfinite(sampled_budget)):
        raise ValueError("sampled_budget must be finite")
    if not torch.all((budget_grid[:, 1:] - budget_grid[:, :-1]) > 0.0):
        raise ValueError("budget_grid rows must be strictly increasing")

    budget_grid = budget_grid.to(device=sampled_budget.device, dtype=sampled_budget.dtype)
    utility_grid = utility_grid.to(device=sampled_budget.device, dtype=sampled_budget.dtype)
    rewards = []
    for row_index in range(int(budget_grid.shape[0])):
        row_budget = budget_grid[row_index]
        row_utility = utility_grid[row_index]
        sample = sampled_budget[row_index].clamp(min=float(row_budget[0]), max=float(row_budget[-1]))
        upper = torch.searchsorted(row_budget.contiguous(), sample.contiguous(), right=False)
        upper = upper.clamp(min=1, max=int(row_budget.shape[0]) - 1)
        lower = upper - 1
        left_budget = row_budget[lower]
        right_budget = row_budget[upper]
        left_utility = row_utility[lower]
        right_utility = row_utility[upper]
        alpha = (sample - left_budget) / (right_budget - left_budget).clamp_min(1e-12)
        rewards.append(left_utility + alpha * (right_utility - left_utility))
    interpolated = torch.stack(rewards, dim=0)
    if squeezed:
        return interpolated[:, 0]
    return interpolated


def _evaluate_bc_loss(
    controller: BudgetController,
    loader: DataLoader,
    *,
    device: torch.device,
    mse_weight: float,
) -> float:
    meter = AverageMeter()
    controller.eval()
    with torch.no_grad():
        for batch in loader:
            pooled_latent = batch["pooled_latent"].to(device)
            target_budget = batch["target_budget"].to(device)
            cheap_features = batch.get("cheap_features")
            if cheap_features is not None:
                cheap_features = cheap_features.to(device)
            loss_dict = budget_controller_bc_loss(
                controller,
                pooled_latent,
                target_budget,
                cheap_features,
                mse_weight=mse_weight,
            )
            meter.update(float(loss_dict["loss"].detach().cpu()), n=int(pooled_latent.shape[0]))
    return float(meter.avg)


def _evaluate_expected_utility(
    controller: BudgetController,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    meter = AverageMeter()
    controller.eval()
    with torch.no_grad():
        for batch in loader:
            pooled_latent = batch["pooled_latent"].to(device)
            cheap_features = batch.get("cheap_features")
            if cheap_features is not None:
                cheap_features = cheap_features.to(device)
            output = controller.sample_budget(pooled_latent, cheap_features, deterministic=True)
            reward = interpolate_budget_utilities(
                batch["budget_grid"].to(device),
                batch["utility_grid"].to(device),
                output.budget,
            )
            meter.update(float(reward.detach().mean().cpu()), n=int(pooled_latent.shape[0]))
    return float(meter.avg)


def train_budget_controller_from_oracle(
    config: TrainingConfig, *, device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Train ``BudgetController`` from an oracle JSONL file and save a standalone checkpoint."""
    budget_config = config.budget_controller
    if not budget_config.enabled:
        raise ValueError("budget_controller.enabled must be true for controller training")
    if budget_config.mode != "oracle_distillation":
        raise ValueError(
            "train_budget_controller_from_oracle supports budget_controller.mode='oracle_distillation', "
            f"got {budget_config.mode!r}"
        )
    if not budget_config.oracle_path:
        raise ValueError("budget_controller.oracle_path is required for oracle_distillation")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("train_budget_controller currently requires single-process launch; set tasks_per_node=1")

    torch.manual_seed(int(config.meta.seed))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_budget_oracle_examples(
        budget_config.oracle_path,
        feature_dim=budget_config.feature_dim,
        lambda_compute=budget_config.lambda_compute,
    )
    latent_dim = int(examples.pooled_latent.shape[1])
    controller = BudgetController(
        latent_dim=latent_dim,
        feature_dim=budget_config.feature_dim,
        hidden_dim=budget_config.hidden_dim,
        min_concentration=budget_config.min_concentration,
    ).to(device)

    batch_size = int(config.data.batch_size)
    if batch_size <= 0:
        raise ValueError(f"data.batch_size must be > 0, got {batch_size}")
    epochs = int(config.optimization.epochs)
    if epochs <= 0:
        raise ValueError(f"optimization.epochs must be > 0, got {epochs}")

    dataset = BudgetOracleDataset(examples)
    generator = torch.Generator()
    generator.manual_seed(int(config.meta.seed))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        generator=make_dataloader_generator(rank=0, stream="budget_controller/bc_eval"),
    )
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(config.optimization.lr),
        betas=config.optimization.betas,
        eps=float(config.optimization.eps),
        weight_decay=float(config.optimization.weight_decay),
    )

    initial_loss = _evaluate_bc_loss(
        controller,
        eval_loader,
        device=device,
        mse_weight=budget_config.bc_mse_weight,
    )
    final_loss = initial_loss
    for epoch in range(epochs):
        meter = AverageMeter()
        controller.train()
        for batch in loader:
            pooled_latent = batch["pooled_latent"].to(device)
            target_budget = batch["target_budget"].to(device)
            cheap_features = batch.get("cheap_features")
            if cheap_features is not None:
                cheap_features = cheap_features.to(device)
            loss_dict = budget_controller_bc_loss(
                controller,
                pooled_latent,
                target_budget,
                cheap_features,
                mse_weight=budget_config.bc_mse_weight,
            )
            loss = loss_dict["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"budget controller loss became non-finite at epoch {epoch}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.optimization.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), float(config.optimization.grad_clip_norm))
            optimizer.step()
            meter.update(float(loss.detach().cpu()), n=int(pooled_latent.shape[0]))
        final_loss = _evaluate_bc_loss(
            controller,
            eval_loader,
            device=device,
            mse_weight=budget_config.bc_mse_weight,
        )
        logger.info(
            "[budget_controller][epoch %d/%d] train_loss=%.6f eval_loss=%.6f",
            epoch + 1,
            epochs,
            meter.avg,
            final_loss,
        )

    output_checkpoint = budget_config.output_checkpoint or os.path.join(config.meta.folder, "budget_controller.pt")
    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "num_examples": float(len(dataset)),
    }
    torch.save(
        {
            "controller": controller.state_dict(),
            "latent_dim": latent_dim,
            "feature_dim": int(budget_config.feature_dim),
            "hidden_dim": int(budget_config.hidden_dim),
            "min_concentration": float(budget_config.min_concentration),
            "oracle_path": str(budget_config.oracle_path),
            "mode": "oracle_distillation",
            "metrics": metrics,
        },
        output_path,
    )
    logger.info("[budget_controller] saved checkpoint to %s", output_path)
    return metrics


def train_budget_controller_grpo_from_oracle(
    config: TrainingConfig, *, device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Fine-tune a BC-initialized controller with offline oracle-backed GRPO."""
    budget_config = config.budget_controller
    if not budget_config.enabled:
        raise ValueError("budget_controller.enabled must be true for GRPO controller training")
    if budget_config.mode != "grpo":
        raise ValueError(
            "train_budget_controller_grpo_from_oracle supports budget_controller.mode='grpo', "
            f"got {budget_config.mode!r}"
        )
    if not budget_config.oracle_path:
        raise ValueError("budget_controller.oracle_path is required for grpo")
    if not budget_config.controller_checkpoint:
        raise ValueError("budget_controller.controller_checkpoint is required for grpo warm start")
    if budget_config.grpo_reward_interp != "linear":
        raise ValueError("budget_controller.grpo_reward_interp must be 'linear'")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "train_budget_controller GRPO currently requires single-process launch; set tasks_per_node=1"
        )

    num_samples = int(budget_config.grpo_num_samples_per_scene)
    if num_samples < 2:
        raise ValueError(f"budget_controller.grpo_num_samples_per_scene must be >= 2, got {num_samples}")
    torch.manual_seed(int(config.meta.seed))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweeps = load_budget_oracle_sweeps(
        budget_config.oracle_path,
        feature_dim=budget_config.feature_dim,
        lambda_compute=budget_config.lambda_compute,
    )
    latent_dim = int(sweeps.pooled_latent.shape[1])
    controller = load_budget_controller_from_checkpoint(budget_config.controller_checkpoint, device=device)
    if int(controller.latent_dim) != latent_dim:
        raise ValueError(
            f"controller checkpoint latent_dim={controller.latent_dim} does not match oracle latent_dim={latent_dim}"
        )
    if int(controller.feature_dim) != int(budget_config.feature_dim):
        raise ValueError(
            f"controller checkpoint feature_dim={controller.feature_dim} does not match "
            f"budget_controller.feature_dim={budget_config.feature_dim}"
        )
    for parameter in controller.parameters():
        parameter.requires_grad_(True)

    batch_size = int(config.data.batch_size)
    if batch_size <= 0:
        raise ValueError(f"data.batch_size must be > 0, got {batch_size}")
    epochs = int(config.optimization.epochs)
    if epochs <= 0:
        raise ValueError(f"optimization.epochs must be > 0, got {epochs}")

    dataset = BudgetOracleSweepDataset(sweeps)
    generator = torch.Generator()
    generator.manual_seed(int(config.meta.seed))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        generator=make_dataloader_generator(rank=0, stream="budget_controller/grpo_eval"),
    )
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(config.optimization.lr),
        betas=config.optimization.betas,
        eps=float(config.optimization.eps),
        weight_decay=float(config.optimization.weight_decay),
    )

    initial_expected_utility = _evaluate_expected_utility(controller, eval_loader, device=device)
    final_expected_utility = initial_expected_utility
    final_loss = 0.0
    for epoch in range(epochs):
        meter = AverageMeter()
        reward_meter = AverageMeter()
        controller.train()
        for batch in loader:
            pooled_latent = batch["pooled_latent"].to(device)
            cheap_features = batch.get("cheap_features")
            if cheap_features is not None:
                cheap_features = cheap_features.to(device)
            batch_size_actual = int(pooled_latent.shape[0])
            pooled_repeated = pooled_latent.repeat_interleave(num_samples, dim=0)
            cheap_repeated = None
            if cheap_features is not None:
                cheap_repeated = cheap_features.repeat_interleave(num_samples, dim=0)
            output = controller.sample_budget(pooled_repeated, cheap_repeated, deterministic=False)
            sampled_budget = output.budget.view(batch_size_actual, num_samples)
            reward = interpolate_budget_utilities(
                batch["budget_grid"].to(device),
                batch["utility_grid"].to(device),
                sampled_budget,
            ).reshape(-1)
            group_ids = torch.arange(batch_size_actual, device=device).repeat_interleave(num_samples)
            loss_dict = budget_controller_grpo_loss(output.log_prob, reward, group_ids=group_ids)
            loss = loss_dict["loss"]
            bc_weight = float(budget_config.grpo_bc_weight)
            if bc_weight > 0.0:
                bc_loss_dict = budget_controller_bc_loss(
                    controller,
                    pooled_latent,
                    batch["target_budget"].to(device),
                    cheap_features,
                    mse_weight=budget_config.bc_mse_weight,
                )
                loss = loss + bc_weight * bc_loss_dict["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"budget controller GRPO loss became non-finite at epoch {epoch}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.optimization.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), float(config.optimization.grad_clip_norm))
            optimizer.step()
            meter.update(float(loss.detach().cpu()), n=int(reward.shape[0]))
            reward_meter.update(float(reward.detach().mean().cpu()), n=batch_size_actual)
        final_loss = float(meter.avg)
        final_expected_utility = _evaluate_expected_utility(controller, eval_loader, device=device)
        logger.info(
            "[budget_controller][grpo epoch %d/%d] train_loss=%.6f sampled_reward=%.6f expected_utility=%.6f",
            epoch + 1,
            epochs,
            meter.avg,
            reward_meter.avg,
            final_expected_utility,
        )

    output_checkpoint = budget_config.output_checkpoint or os.path.join(config.meta.folder, "budget_controller.pt")
    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "initial_expected_utility": float(initial_expected_utility),
        "final_expected_utility": float(final_expected_utility),
        "final_loss": float(final_loss),
        "num_examples": float(len(dataset)),
    }
    torch.save(
        {
            "controller": controller.state_dict(),
            "latent_dim": int(controller.latent_dim),
            "feature_dim": int(controller.feature_dim),
            "hidden_dim": int(controller.hidden_dim),
            "min_concentration": float(controller.min_concentration),
            "oracle_path": str(budget_config.oracle_path),
            "init_checkpoint": str(budget_config.controller_checkpoint),
            "grpo_num_samples_per_scene": num_samples,
            "grpo_bc_weight": float(budget_config.grpo_bc_weight),
            "grpo_reward_interp": str(budget_config.grpo_reward_interp),
            "mode": "grpo",
            "metrics": metrics,
        },
        output_path,
    )
    logger.info("[budget_controller] saved GRPO checkpoint to %s", output_path)
    return metrics


def train_budget_controller_from_config(
    config: TrainingConfig, *, device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Dispatch offline controller training by ``budget_controller.mode``."""
    mode = str(config.budget_controller.mode)
    if mode == "oracle_distillation":
        return train_budget_controller_from_oracle(config, device=device)
    if mode == "grpo":
        return train_budget_controller_grpo_from_oracle(config, device=device)
    if mode != "bc_then_grpo":
        raise ValueError(
            "train_budget_controller_from_config supports mode='oracle_distillation', 'grpo', or 'bc_then_grpo', "
            f"got {mode!r}"
        )

    original_mode = config.budget_controller.mode
    original_controller_checkpoint = config.budget_controller.controller_checkpoint
    output_checkpoint = config.budget_controller.output_checkpoint or os.path.join(
        config.meta.folder, "budget_controller.pt"
    )
    try:
        config.budget_controller.mode = "oracle_distillation"
        bc_metrics = train_budget_controller_from_oracle(config, device=device)
        config.budget_controller.mode = "grpo"
        config.budget_controller.controller_checkpoint = output_checkpoint
        grpo_metrics = train_budget_controller_grpo_from_oracle(config, device=device)
    finally:
        config.budget_controller.mode = original_mode
        config.budget_controller.controller_checkpoint = original_controller_checkpoint

    merged: Dict[str, float] = {}
    merged.update({f"bc_{key}": float(value) for key, value in bc_metrics.items()})
    merged.update({f"grpo_{key}": float(value) for key, value in grpo_metrics.items()})
    return merged
