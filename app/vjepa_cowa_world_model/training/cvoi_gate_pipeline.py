"""SQLite-only distillation pipeline for the manual NavSim CVoI Gate."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import (
    MANUAL_ORACLE_STORE_SCHEMA_V2,
    NAVTRAIN_GATE_PROTOCOL_ID,
    NAVTRAIN_GATE_TRAINING_BATCH_SIZE,
)
from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
    CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES,
    SequentialRolloutGate,
    apply_cvoi_formal_v2_navsim_e120_gate_feature_mask,
)
from app.vjepa_cowa_world_model.training.sequential_gate_training import (
    SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3,
    save_sequential_gate_checkpoint_replace,
    train_sequential_gate_epoch,
    validate_formal_v2_lambda_grid,
)

CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION = "offline_navsim_e120_official_epdms_gate_distillation_v1"
CVOI_NAVSIM_E120_OFFICIAL_GATE_SPLIT_POLICY = "sha256_utf8_log_name_mod10_bucket0_dev_buckets1_9_train_v1"
_NAVTRAIN_GATE_ORACLE_FEATURE_SCHEMA = "sequential_cvoi_gate_features_lambda_independent_h4_v1"
_GATE_SCALAR_FEATURES = 7


@dataclass(frozen=True)
class CvoiGateValidationReport:
    """Held-out marginal-utility prediction metrics."""

    sign_accuracy: float
    mae: float
    roll_rate: float
    num_examples: int


@dataclass(frozen=True)
class CvoiGateTrainingReport:
    """Artifacts and metrics produced by one offline distillation run."""

    checkpoint_path: Path
    latent_dim: int
    feature_dim: int
    train_metrics: tuple[Dict[str, float], ...]
    dev: CvoiGateValidationReport
    provenance: Dict[str, str]
    gate_feature_schema: str | None = None
    gate_feature_mode: str | None = None


def _normalize_lambda_grid(lambda_grid: Sequence[float]) -> tuple[float, ...]:
    if isinstance(lambda_grid, (str, bytes)):
        raise ValueError("lambda_grid must be a sequence of compute penalties")
    values = tuple(float(value) for value in lambda_grid)
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("lambda_grid must contain finite, non-negative values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("lambda_grid must be unique and strictly increasing")
    return values


def _infer_latent_dim(feature_dim: int) -> int:
    feature_dim = int(feature_dim)
    latent_features = feature_dim - _GATE_SCALAR_FEATURES
    if latent_features <= 0 or latent_features % 2 != 0:
        raise ValueError(
            f"CVoI Gate requires feature_dim=2*latent_dim+7 with positive latent_dim; got feature_dim={feature_dim}"
        )
    return latent_features // 2


class NavTrainGateOracleStoreDataset(Dataset):
    """Index a file-backed Oracle while materializing only the current batch."""

    def __init__(self, store: Any, *, split: str) -> None:
        super().__init__()
        if split not in {"train", "dev"}:
            raise ValueError("official-navtrain Gate split must be exactly 'train' or 'dev'")
        record_ids = tuple(store.iter_split_record_ids(split))
        if not record_ids:
            raise ValueError(f"official-navtrain Gate Oracle contains no records for split {split!r}")
        lambda_grid = tuple(store.metadata.lambda_grid)
        if not lambda_grid:
            raise ValueError("official-navtrain Gate Oracle lambda_grid must be non-empty")
        if type(store.feature_dim) is not int or store.feature_dim <= 0:
            raise ValueError("official-navtrain Gate Oracle feature dimension must be positive")
        self._store = store
        self._record_ids = record_ids
        self._lambda_grid = lambda_grid
        self._examples_per_record = len(lambda_grid) * 4
        self._feature_dim = store.feature_dim + 1

    def __len__(self) -> int:
        return len(self._record_ids) * self._examples_per_record

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _decode_index(self, index: int) -> tuple[int, float, int]:
        if type(index) is not int or index < 0 or index >= len(self):
            raise IndexError(f"official-navtrain Gate dataset index out of range: {index!r}")
        split_index, within_record = divmod(index, self._examples_per_record)
        lambda_index, horizon = divmod(within_record, 4)
        return self._record_ids[split_index], self._lambda_grid[lambda_index], horizon

    @staticmethod
    def _batch_to_mapping(batch: Sequence[Any]) -> Dict[str, object]:
        return {
            "features": torch.tensor([row.features for row in batch], dtype=torch.float32),
            "target_delta": torch.tensor([row.target_delta for row in batch], dtype=torch.float32),
            "continue_target": torch.tensor([row.continue_target for row in batch], dtype=torch.bool),
            "lambda_compute": torch.tensor([row.lambda_compute for row in batch], dtype=torch.float32),
            "horizon": torch.tensor([row.horizon for row in batch], dtype=torch.long),
            "token": [row.token for row in batch],
            "log_name": [row.log_name for row in batch],
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        record_id, lambda_compute, horizon = self._decode_index(index)
        batch = self._store.read_training_batch(
            record_ids=[record_id],
            horizons=[horizon],
            lambda_computes=[lambda_compute],
        )
        return {
            key: value[0] if isinstance(value, torch.Tensor) else value[0]
            for key, value in self._batch_to_mapping(batch).items()
        }

    def collate_indices(self, indices: Sequence[int]) -> Dict[str, object]:
        if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence) or not indices:
            raise ValueError("official-navtrain Gate collate requires a non-empty index sequence")
        decoded = [self._decode_index(index) for index in indices]
        batch = self._store.read_training_batch(
            record_ids=[row[0] for row in decoded],
            lambda_computes=[row[1] for row in decoded],
            horizons=[row[2] for row in decoded],
        )
        return self._batch_to_mapping(batch)


class DeterministicAffinePermutationSampler(Sampler[int]):
    """Shuffle an integer range with O(1) resident memory per epoch."""

    def __init__(self, size: int, *, seed: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("affine sampler size must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("affine sampler seed must be a non-negative integer")
        self._size = size
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[int]:
        payload = hashlib.sha256(f"{self._seed}:{self._epoch}:{self._size}".encode("ascii")).digest()
        multiplier = int.from_bytes(payload[:16], "big") % self._size
        while math.gcd(multiplier, self._size) != 1:
            multiplier = (multiplier + 1) % self._size
        offset = int.from_bytes(payload[16:], "big") % self._size
        self._epoch += 1
        return iter((multiplier * index + offset) % self._size for index in range(self._size))


def _official_navtrain_gate_store_checkpoint_provenance(
    _oracle_path: Path,
    store: Any,
    *,
    gate_feature_mode: str,
) -> Dict[str, str]:
    """Describe the semantic contract of one strict embedded manual Oracle."""

    gate_feature_mode = _normalize_formal_v2_navsim_e120_gate_feature_mode(gate_feature_mode)
    metadata = store.metadata
    if metadata.protocol_id != NAVTRAIN_GATE_PROTOCOL_ID:
        raise ValueError("official-navtrain Gate store protocol differs")
    oracle_sha256 = getattr(store, "sha256", None)
    if (
        type(oracle_sha256) is not str
        or len(oracle_sha256) != 64
        or any(character not in "0123456789abcdef" for character in oracle_sha256)
    ):
        raise ValueError("official-navtrain Gate store requires a lowercase Oracle SHA-256 identity")
    return {
        "gate_pipeline": CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION,
        "oracle_storage_schema": MANUAL_ORACLE_STORE_SCHEMA_V2,
        "oracle_protocol": NAVTRAIN_GATE_PROTOCOL_ID,
        "oracle_sha256": oracle_sha256,
        "oracle_lineage": metadata.lineage,
        "oracle_feature_schema": _NAVTRAIN_GATE_ORACLE_FEATURE_SCHEMA,
        "gate_feature_schema": CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
        "gate_feature_mode": gate_feature_mode,
        "gate_training_batch_size": str(NAVTRAIN_GATE_TRAINING_BATCH_SIZE),
        "split_policy": CVOI_NAVSIM_E120_OFFICIAL_GATE_SPLIT_POLICY,
        "shuffle_protocol": "sha256_seed_epoch_affine_permutation_o1_memory_v1",
    }


def build_navtrain_gate_checkpoint_provenance(
    oracle_path: str | Path,
    artifact: Any,
    *,
    gate_feature_mode: str,
) -> Dict[str, str]:
    """Build the dedicated official-navtrain Gate checkpoint identity."""

    source = Path(oracle_path)
    return _official_navtrain_gate_store_checkpoint_provenance(
        source,
        artifact,
        gate_feature_mode=gate_feature_mode,
    )


def _normalize_formal_v2_navsim_e120_gate_feature_mode(mode: str) -> str:
    if mode not in CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES:
        raise ValueError(
            "CVoI Formal v2 NavSim-e120 Gate feature mode must be one of "
            f"{sorted(CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES)}, got {mode!r}"
        )
    return mode


def _mask_formal_v2_navsim_e120_gate_batches(
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    latent_dim: int,
    mode: str,
) -> Iterator[Mapping[str, torch.Tensor]]:
    """Apply a NavSim-e120 mask after expanding the shared Oracle over lambda."""

    for batch in batches:
        if "features" not in batch:
            yield batch
            continue
        masked = dict(batch)
        masked["features"] = apply_cvoi_formal_v2_navsim_e120_gate_feature_mask(
            batch["features"],
            latent_dim=latent_dim,
            mode=mode,
        )
        yield masked


def _validate_batch(batch: Mapping[str, torch.Tensor], *, feature_dim: int) -> int:
    missing = sorted({"features", "target_delta", "continue_target"} - set(batch))
    if missing:
        raise ValueError(f"CVoI Gate validation batch is missing required keys: {missing}")
    features = batch["features"]
    target_delta = batch["target_delta"]
    continue_target = batch["continue_target"]
    if features.ndim != 2 or int(features.shape[0]) < 1 or int(features.shape[1]) != feature_dim:
        raise ValueError(
            f"CVoI Gate validation requires non-empty features [B, {feature_dim}], got {tuple(features.shape)}"
        )
    batch_size = int(features.shape[0])
    if target_delta.shape != (batch_size,) or continue_target.shape != (batch_size,):
        raise ValueError("target_delta and continue_target must have shape [B]")
    if continue_target.dtype != torch.bool:
        raise ValueError("continue_target must be a bool tensor")
    if not torch.isfinite(features).all() or not torch.isfinite(target_delta).all():
        raise ValueError("CVoI Gate validation tensors must contain only finite values")
    if not torch.equal(continue_target, target_delta > 0.0):
        raise ValueError("continue_target must use strict target_delta > 0 semantics so ties are STOP")
    return batch_size


def evaluate_cvoi_gate(
    gate: torch.nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
) -> CvoiGateValidationReport:
    """Evaluate sign decisions and marginal-utility error with ties mapped to STOP."""

    feature_dim = int(getattr(gate, "feature_dim", 0))
    _infer_latent_dim(feature_dim)
    device = torch.device(device)
    gate.to(device=device)
    gate.eval()
    correct = 0
    absolute_error = 0.0
    roll_count = 0
    num_examples = 0
    with torch.no_grad():
        for batch in batches:
            batch_size = _validate_batch(batch, feature_dim=feature_dim)
            features = batch["features"].to(device=device, dtype=torch.float32)
            target_delta = batch["target_delta"].to(device=device, dtype=torch.float32)
            continue_target = batch["continue_target"].to(device=device, dtype=torch.bool)
            predicted_delta = gate(features)
            if predicted_delta.shape != (batch_size,) or not torch.isfinite(predicted_delta).all():
                raise ValueError(f"CVoI Gate must return one finite delta per row, got {tuple(predicted_delta.shape)}")

            predicted_roll = predicted_delta > 0.0
            correct += int((predicted_roll == continue_target).sum().item())
            absolute_error += float(torch.abs(predicted_delta - target_delta).sum().item())
            roll_count += int(predicted_roll.sum().item())
            num_examples += batch_size
    if num_examples == 0:
        raise ValueError("CVoI Gate validation requires at least one example")
    return CvoiGateValidationReport(
        sign_accuracy=correct / num_examples,
        mae=absolute_error / num_examples,
        roll_rate=roll_count / num_examples,
        num_examples=num_examples,
    )


def _build_gate_optimizer(
    gate: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        gate.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=tuple(float(value) for value in betas),
        eps=float(eps),
    )


def _validate_fixed_gate_checkpoint_target(checkpoint_path: str | Path) -> Path:
    raw_path = os.fspath(checkpoint_path)
    normalized_path = os.path.normpath(raw_path)
    if not os.path.isabs(raw_path) or raw_path != normalized_path or raw_path.startswith("//"):
        raise ValueError("checkpoint_path must be absolute and lexical-normalized for the official-navtrain Gate")
    checkpoint = Path(raw_path)
    try:
        parent_metadata = checkpoint.parent.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"checkpoint parent directory does not exist: {checkpoint.parent}") from error
    try:
        resolved_parent = checkpoint.parent.resolve(strict=True)
    except OSError as error:
        raise NotADirectoryError(
            f"checkpoint parent must be a canonical non-symlink directory: {checkpoint.parent}"
        ) from error
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or resolved_parent != checkpoint.parent
    ):
        raise NotADirectoryError(f"checkpoint parent must be a non-symlink directory: {checkpoint.parent}")
    try:
        target_metadata = checkpoint.lstat()
    except FileNotFoundError:
        return checkpoint
    if not stat.S_ISREG(target_metadata.st_mode):
        raise ValueError(f"checkpoint target must be absent or a regular file: {checkpoint}")
    return checkpoint


def _train_cvoi_navsim_e120_official_gate_from_open_store(
    source: Path,
    artifact: Any,
    checkpoint_path: str | Path,
    *,
    gate_feature_mode: str,
    lambda_grid: Sequence[float],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
    batch_size: int,
    hidden_dim: int,
    temperature: float,
    regression_weight: float,
    seed: int,
    device: torch.device,
) -> CvoiGateTrainingReport:
    """Train while the already-verified SQLite Oracle handle is pinned open."""

    checkpoint = _validate_fixed_gate_checkpoint_target(checkpoint_path)
    if type(batch_size) is not int or batch_size != NAVTRAIN_GATE_TRAINING_BATCH_SIZE:
        raise ValueError(
            "official-navtrain CVoI Gate batch_size must be exactly "
            f"{NAVTRAIN_GATE_TRAINING_BATCH_SIZE}, got {batch_size!r}"
        )
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    normalized_betas = tuple(float(value) for value in betas)
    eps = float(eps)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be finite and positive, got {learning_rate}")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(f"weight_decay must be finite and non-negative, got {weight_decay}")
    if len(normalized_betas) != 2 or any(
        not math.isfinite(value) or not 0.0 <= value < 1.0 for value in normalized_betas
    ):
        raise ValueError(f"betas must contain two finite values in [0, 1), got {normalized_betas}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    gate_feature_mode = _normalize_formal_v2_navsim_e120_gate_feature_mode(gate_feature_mode)
    normalized_lambda_grid = _normalize_lambda_grid(lambda_grid)
    if normalized_lambda_grid != tuple(artifact.metadata.lambda_grid):
        raise ValueError(
            "lambda_grid must equal the Oracle artifact lambda_grid exactly: "
            f"expected {list(artifact.metadata.lambda_grid)}, got {list(normalized_lambda_grid)}"
        )
    normalized_lambda_grid = tuple(validate_formal_v2_lambda_grid(normalized_lambda_grid))
    provenance = build_navtrain_gate_checkpoint_provenance(
        source,
        artifact,
        gate_feature_mode=gate_feature_mode,
    )
    train_dataset = NavTrainGateOracleStoreDataset(artifact, split="train")
    dev_dataset = NavTrainGateOracleStoreDataset(artifact, split="dev")
    train_feature_dim = train_dataset.feature_dim
    dev_feature_dim = dev_dataset.feature_dim
    if dev_feature_dim != train_feature_dim:
        raise ValueError(
            "official-navtrain train/dev Gate feature dimensions differ: " f"{train_feature_dim}/{dev_feature_dim}"
        )
    latent_dim = _infer_latent_dim(train_feature_dim)

    train_loader = DataLoader(
        range(len(train_dataset)),
        batch_size=batch_size,
        sampler=DeterministicAffinePermutationSampler(len(train_dataset), seed=seed),
        num_workers=0,
        drop_last=False,
        collate_fn=train_dataset.collate_indices,
    )
    dev_loader = DataLoader(
        range(len(dev_dataset)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=dev_dataset.collate_indices,
    )
    device = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gate = SequentialRolloutGate(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device=device)
    optimizer = _build_gate_optimizer(
        gate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=normalized_betas,
        eps=eps,
    )
    train_metrics = []
    for _ in range(epochs):
        train_metrics.append(
            train_sequential_gate_epoch(
                gate,
                _mask_formal_v2_navsim_e120_gate_batches(
                    train_loader,
                    latent_dim=latent_dim,
                    mode=gate_feature_mode,
                ),
                optimizer=optimizer,
                device=device,
                temperature=temperature,
                regression_weight=regression_weight,
            )
        )
    dev_report = evaluate_cvoi_gate(
        gate,
        _mask_formal_v2_navsim_e120_gate_batches(
            dev_loader,
            latent_dim=latent_dim,
            mode=gate_feature_mode,
        ),
        device=device,
    )
    save_sequential_gate_checkpoint_replace(
        checkpoint,
        gate,
        lambda_grid=normalized_lambda_grid,
        provenance=provenance,
        protocol_version=SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3,
    )
    report = CvoiGateTrainingReport(
        checkpoint_path=checkpoint,
        latent_dim=latent_dim,
        feature_dim=train_feature_dim,
        train_metrics=tuple(train_metrics),
        dev=dev_report,
        provenance=provenance,
        gate_feature_schema=CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
        gate_feature_mode=gate_feature_mode,
    )
    return report
