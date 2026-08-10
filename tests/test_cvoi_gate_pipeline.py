import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from app.vjepa_cowa_world_model.training import cvoi_gate_pipeline as gate_pipeline_module
from app.vjepa_cowa_world_model.training import cvoi_navsim_navtrain_gate_store as gate_store_module
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
from app.vjepa_cowa_world_model.training.cvoi_gate_pipeline import (
    CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION,
    DeterministicAffinePermutationSampler,
    NavTrainGateOracleStoreDataset,
    _build_gate_optimizer,
    _train_cvoi_navsim_e120_official_gate_from_open_store,
    build_navtrain_gate_checkpoint_provenance,
    evaluate_cvoi_gate,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import (
    MANUAL_ORACLE_STORE_SCHEMA_V2,
    NAVTRAIN_GATE_PROTOCOL_ID,
    NAVTRAIN_GATE_TRAINING_BATCH_SIZE,
    FeatureRow,
    FeatureStoreMetadata,
    OracleStoreMetadata,
    ScoreRow,
    ScoreStoreMetadata,
    create_embedded_oracle_store_v2,
    create_feature_store,
    create_score_store,
    open_embedded_oracle_store_v2,
)
from app.vjepa_cowa_world_model.training.cvoi_runtime import build_cvoi_gate_provenance, load_cvoi_gate_for_evaluation
from app.vjepa_cowa_world_model.training.sequential_gate_training import load_sequential_gate_checkpoint


def test_gate_pipeline_surface_is_sqlite_manual_oracle_only() -> None:
    removed_names = {
        "NavTrainGateOracleDataset",
        "train_cvoi_gate_from_oracle",
        "train_cvoi_formal_v2_gate_from_oracle",
        "train_cvoi_navsim_e120_official_gate_from_oracle",
        "train_cvoi_navsim_e120_quality_gate_from_oracle",
    }

    assert not (removed_names & set(vars(gate_pipeline_module)))
    assert not ({"create_oracle_store", "open_oracle_store"} & set(vars(gate_store_module)))
    assert gate_store_module.NAVTRAIN_GATE_PROTOCOL_ID == "epdms_v2_one_stage_navtrain_gate_label_v1"
    assert gate_store_module.NAVTRAIN_GATE_TRAINING_BATCH_SIZE == 4096


def _train_gate_from_oracle(
    oracle_path: Path,
    checkpoint_path: Path,
    **kwargs: object,
):
    kwargs.setdefault("batch_size", NAVTRAIN_GATE_TRAINING_BATCH_SIZE)
    kwargs.setdefault("hidden_dim", 128)
    kwargs.setdefault("temperature", 0.05)
    kwargs.setdefault("regression_weight", 0.5)
    kwargs.setdefault("seed", 239)
    with open_embedded_oracle_store_v2(oracle_path) as oracle:
        return _train_cvoi_navsim_e120_official_gate_from_open_store(
            oracle_path,
            oracle,
            checkpoint_path,
            **kwargs,
        )


def test_deterministic_sampler_is_a_permutation_with_epoch_progression() -> None:
    sampler = DeterministicAffinePermutationSampler(17, seed=239)

    first = tuple(sampler)
    second = tuple(sampler)

    assert sorted(first) == list(range(17))
    assert sorted(second) == list(range(17))
    assert first != second


def test_checkpoint_provenance_accepts_only_the_embedded_sqlite_store(tmp_path: Path) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    _official_navtrain_oracle_store(oracle_path)

    with open_embedded_oracle_store_v2(oracle_path) as oracle:
        provenance = build_navtrain_gate_checkpoint_provenance(
            oracle_path,
            oracle,
            gate_feature_mode="full",
        )

    assert provenance["oracle_storage_schema"] == MANUAL_ORACLE_STORE_SCHEMA_V2
    assert provenance["oracle_protocol"] == NAVTRAIN_GATE_PROTOCOL_ID
    assert provenance["oracle_sha256"] == hashlib.sha256(oracle_path.read_bytes()).hexdigest()


def _optimizer_kwargs() -> dict[str, object]:
    return {"weight_decay": 0.04, "betas": (0.8, 0.9), "eps": 1e-6}


def _official_navtrain_oracle_store(
    path: Path,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, str], dict[int, str]]:
    """Build the minimal strict SQLite Oracle graph used by the formal Gate path."""

    scenario_manifest_sha256 = "e" * 64
    metric_cache_inventory_sha256 = "f" * 64
    score_store_paths: dict[int, Path] = {}
    feature_store_paths: dict[int, Path] = {}
    score_store_sha256s: dict[int, str] = {}
    feature_store_sha256s: dict[int, str] = {}
    identities = (
        (0, "token-dev", "a" * 64, "c" * 64, "log-b0-2", (0.50, 0.70, 0.80, 0.60, 0.90)),
        (1, "token-train", "b" * 64, "d" * 64, "log-b1-4", (0.51, 0.70, 0.80, 0.60, 0.90)),
    )
    for horizon in range(5):
        score_source = path.parent / f"official-score-h{horizon}.csv"
        score_source.write_text(f"official navtrain H{horizon}\n", encoding="utf-8")
        score_path = path.parent / f"score-h{horizon}.sqlite3"
        feature_path = path.parent / f"feature-h{horizon}.sqlite3"
        score_receipt = create_score_store(
            score_path,
            ScoreStoreMetadata(
                protocol_id=NAVTRAIN_GATE_PROTOCOL_ID,
                policy_id=f"policy-h{horizon}",
                lineage="p1_full",
                horizon=horizon,
                scenario_manifest_sha256=scenario_manifest_sha256,
                metric_cache_inventory_sha256=metric_cache_inventory_sha256,
                source_path=score_source,
                source_sha256=hashlib.sha256(score_source.read_bytes()).hexdigest(),
                score_semantics="official_v2_one_stage_ordinary_row_score",
            ),
            (
                ScoreRow(row_index, token, observation_key, log_name, scores[horizon])
                for row_index, token, observation_key, _observed_sha256, log_name, scores in identities
            ),
            aggregate_score=0.7,
        )
        feature_receipt = create_feature_store(
            feature_path,
            FeatureStoreMetadata(
                protocol_id=NAVTRAIN_GATE_PROTOCOL_ID,
                policy_id=f"policy-h{horizon}",
                lineage="p1_full",
                horizon=horizon,
                scenario_manifest_sha256=scenario_manifest_sha256,
                metric_cache_inventory_sha256=metric_cache_inventory_sha256,
                feature_schema="sequential_cvoi_gate_features_lambda_independent_h4_v1",
                feature_sources=("pooled_observed", "pooled_prefix", "field_value"),
                common_random_seed=239,
            ),
            (
                FeatureRow(
                    row_index,
                    token,
                    observation_key,
                    observed_sha256,
                    tuple(float(horizon + column) / 10.0 for column in range(10)),
                )
                for row_index, token, observation_key, observed_sha256, _log_name, _scores in identities
            ),
        )
        score_store_paths[horizon] = score_path
        feature_store_paths[horizon] = feature_path
        score_store_sha256s[horizon] = score_receipt.sha256
        feature_store_sha256s[horizon] = feature_receipt.sha256
    create_embedded_oracle_store_v2(
        path,
        OracleStoreMetadata(
            protocol_id=NAVTRAIN_GATE_PROTOCOL_ID,
            lineage="p1_full",
            scenario_manifest_sha256=scenario_manifest_sha256,
            metric_cache_inventory_sha256=metric_cache_inventory_sha256,
            lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        ),
        score_store_paths=score_store_paths,
        feature_store_paths=feature_store_paths,
    )
    return score_store_paths, feature_store_paths, score_store_sha256s, feature_store_sha256s


def test_official_navtrain_pipeline_trains_from_self_contained_v2_with_content_identity(
    tmp_path: Path,
) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    checkpoint_path = tmp_path / "official-gate.pt"
    score_paths, feature_paths, _, _ = _official_navtrain_oracle_store(oracle_path)
    for intermediate in (*score_paths.values(), *feature_paths.values()):
        intermediate.unlink()

    assert not hasattr(gate_pipeline_module, "_oracle_sha256")
    assert not hasattr(gate_store_module, "open_oracle_store")

    with patch.object(NavTrainGateOracleStoreDataset, "__getitem__", side_effect=AssertionError("scalar decode used")):
        report = _train_gate_from_oracle(
            oracle_path.resolve(),
            checkpoint_path,
            gate_feature_mode="full",
            lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
            epochs=1,
            learning_rate=1e-2,
            batch_size=NAVTRAIN_GATE_TRAINING_BATCH_SIZE,
            **_optimizer_kwargs(),
            hidden_dim=8,
            device=torch.device("cpu"),
        )

    assert report.provenance["gate_pipeline"] == CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION
    assert report.provenance["oracle_storage_schema"] == MANUAL_ORACLE_STORE_SCHEMA_V2
    assert report.provenance["oracle_sha256"] == hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    assert set(report.provenance) == {
        "gate_pipeline",
        "oracle_storage_schema",
        "oracle_protocol",
        "oracle_sha256",
        "oracle_lineage",
        "oracle_feature_schema",
        "gate_feature_schema",
        "gate_feature_mode",
        "gate_training_batch_size",
        "split_policy",
        "shuffle_protocol",
    }
    assert not any(
        forbidden_fragment in key
        for key in report.provenance
        for forbidden_fragment in ("receipt", "artifact", "path")
    )
    assert report.dev.num_examples == len(FORMAL_V2_NAVSIM_E120_LAMBDA_GRID) * 4
    assert report.feature_dim == 11
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "sequential_cvoi_gate_navsim_e120_v1"
    assert payload["provenance"] == report.provenance
    load_sequential_gate_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_provenance=report.provenance,
        expected_protocol_version="formal_v2_navsim_e120_h4_v3",
    )


def test_official_navtrain_e120_value_summary_pipeline_masks_exact_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    checkpoint_path = tmp_path / "official-gate.pt"
    _official_navtrain_oracle_store(oracle_path)
    captured_features: list[torch.Tensor] = []

    def capture_epoch(gate, batches, **kwargs):
        del gate, kwargs
        captured_features.extend(batch["features"].detach().clone() for batch in batches)
        return {"loss": 0.0, "classification": 0.0, "regression": 0.0}

    monkeypatch.setattr(gate_pipeline_module, "train_sequential_gate_epoch", capture_epoch)
    report = _train_gate_from_oracle(
        oracle_path.resolve(),
        checkpoint_path.resolve(),
        gate_feature_mode="without_value_summary",
        lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4096,
        **_optimizer_kwargs(),
        hidden_dim=8,
        device=torch.device("cpu"),
    )

    masked = torch.cat(captured_features, dim=0)
    original_rows = torch.tensor(
        [
            [float(horizon + index) / 10.0 for index in range(10)] + [lambda_compute]
            for horizon in range(4)
            for lambda_compute in FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
        ],
        dtype=torch.float32,
    )
    assert torch.equal(masked[:, 4:7], torch.zeros_like(masked[:, 4:7]))
    for column in (0, 1, 2, 3, 7, 8, 9, 10):
        assert sorted(masked[:, column].tolist()) == pytest.approx(sorted(original_rows[:, column].tolist()))
    assert report.gate_feature_mode == "without_value_summary"
    assert report.provenance["gate_feature_mode"] == "without_value_summary"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["provenance"]["gate_feature_mode"] == "without_value_summary"


def test_official_navtrain_gate_checkpoint_replaces_the_fixed_regular_target(tmp_path: Path) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    checkpoint_path = tmp_path / "official-gate.pt"
    _official_navtrain_oracle_store(oracle_path)
    first = _train_gate_from_oracle(
        oracle_path.resolve(),
        checkpoint_path.resolve(),
        gate_feature_mode="full",
        lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4096,
        **_optimizer_kwargs(),
        hidden_dim=8,
        seed=239,
        device=torch.device("cpu"),
    )
    first_bytes = checkpoint_path.read_bytes()

    second = _train_gate_from_oracle(
        oracle_path.resolve(),
        checkpoint_path.resolve(),
        gate_feature_mode="full",
        lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4096,
        **_optimizer_kwargs(),
        hidden_dim=8,
        seed=240,
        device=torch.device("cpu"),
    )

    assert checkpoint_path.read_bytes() != first_bytes
    load_sequential_gate_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_provenance=second.provenance,
        expected_protocol_version="formal_v2_navsim_e120_h4_v3",
    )
    assert first.provenance == second.provenance


def test_official_navtrain_gate_failures_preserve_the_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    checkpoint_path = tmp_path / "official-gate.pt"
    _official_navtrain_oracle_store(oracle_path)
    checkpoint_path.write_bytes(b"previous fixed Gate")
    previous = checkpoint_path.read_bytes()

    with pytest.raises(ValueError, match="lambda_grid must equal the Oracle artifact"):
        _train_gate_from_oracle(
            oracle_path.resolve(),
            checkpoint_path.resolve(),
            gate_feature_mode="full",
            lambda_grid=[0.0],
            epochs=1,
            learning_rate=1e-2,
            batch_size=4096,
            **_optimizer_kwargs(),
            hidden_dim=8,
            device=torch.device("cpu"),
        )
    assert checkpoint_path.read_bytes() == previous

    monkeypatch.setattr(
        gate_pipeline_module,
        "train_sequential_gate_epoch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected training failure")),
    )
    with pytest.raises(RuntimeError, match="injected training failure"):
        _train_gate_from_oracle(
            oracle_path.resolve(),
            checkpoint_path.resolve(),
            gate_feature_mode="full",
            lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
            epochs=1,
            learning_rate=1e-2,
            batch_size=4096,
            **_optimizer_kwargs(),
            hidden_dim=8,
            device=torch.device("cpu"),
        )
    assert checkpoint_path.read_bytes() == previous


def test_official_navtrain_gate_rejects_invalid_output_types_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    _official_navtrain_oracle_store(oracle_path)
    victim = tmp_path / "victim.pt"
    victim.write_bytes(b"unchanged")
    symlink_target = tmp_path / "symlink-gate.pt"
    symlink_target.symlink_to(victim)
    directory_target = tmp_path / "directory-gate.pt"
    directory_target.mkdir()
    parent_symlink = tmp_path / "parent-link"
    parent_symlink.symlink_to(tmp_path, target_is_directory=True)
    real_ancestor = tmp_path / "real-ancestor"
    nested_parent = real_ancestor / "nested"
    nested_parent.mkdir(parents=True)
    ancestor_symlink = tmp_path / "ancestor-link"
    ancestor_symlink.symlink_to(real_ancestor, target_is_directory=True)
    invalid_targets = (
        (symlink_target, ValueError, "regular file"),
        (directory_target, ValueError, "regular file"),
        (tmp_path / "missing-parent" / "gate.pt", FileNotFoundError, "parent directory"),
        (parent_symlink / "gate.pt", NotADirectoryError, "non-symlink directory"),
        (ancestor_symlink / "nested/gate.pt", NotADirectoryError, "non-symlink directory"),
    )
    training = patch.object(
        gate_pipeline_module,
        "train_sequential_gate_epoch",
        side_effect=AssertionError("training started before output validation"),
    )
    with training as train_epoch:
        for checkpoint_path, error_type, message in invalid_targets:
            with pytest.raises(error_type, match=message):
                _train_gate_from_oracle(
                    oracle_path.resolve(),
                    checkpoint_path,
                    gate_feature_mode="full",
                    lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
                    epochs=1,
                    learning_rate=1e-2,
                    batch_size=4096,
                    **_optimizer_kwargs(),
                    hidden_dim=8,
                    device=torch.device("cpu"),
                )
    train_epoch.assert_not_called()
    assert victim.read_bytes() == b"unchanged"


def test_schema_v2_oracle_checkpoint_is_accepted_by_production_evaluation_and_rejects_drift(
    tmp_path: Path,
) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    checkpoint_path = tmp_path / "official-gate.pt"
    _official_navtrain_oracle_store(oracle_path)
    report = _train_gate_from_oracle(
        oracle_path.resolve(),
        checkpoint_path.resolve(),
        gate_feature_mode="full",
        lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4096,
        **_optimizer_kwargs(),
        hidden_dim=8,
        device=torch.device("cpu"),
    )
    config = SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            stage="evaluation",
            evaluation_mode="controller",
            protocol_version="formal_v2_navsim_e120_h4_v3",
            oracle_path=str(oracle_path.resolve()),
            gate_checkpoint=str(checkpoint_path.resolve()),
            ablation_signature=SimpleNamespace(gate_feature_mode="full"),
        )
    )

    assert build_cvoi_gate_provenance(config) == report.provenance
    loaded = load_cvoi_gate_for_evaluation(config, device=torch.device("cpu"))
    assert loaded is not None
    assert loaded.latent_dim == report.latent_dim

    config.cvoi.ablation_signature.gate_feature_mode = "without_field"
    with pytest.raises(ValueError, match="provenance"):
        load_cvoi_gate_for_evaluation(config, device=torch.device("cpu"))


def test_official_navtrain_pipeline_rejects_lambda_grid_different_from_artifact(tmp_path: Path) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    _official_navtrain_oracle_store(oracle_path)

    with pytest.raises(ValueError, match="lambda_grid must equal the Oracle artifact"):
        _train_gate_from_oracle(
            oracle_path.resolve(),
            tmp_path / "gate.pt",
            gate_feature_mode="full",
            lambda_grid=[0.0],
            epochs=1,
            learning_rate=1e-2,
            **_optimizer_kwargs(),
            device=torch.device("cpu"),
        )


def test_official_navtrain_pipeline_requires_absolute_checkpoint_path(tmp_path: Path) -> None:
    oracle_path = tmp_path / "official-oracle.sqlite3"
    _official_navtrain_oracle_store(oracle_path)

    with pytest.raises(ValueError, match="checkpoint_path must be absolute"):
        _train_gate_from_oracle(
            oracle_path.resolve(),
            Path("relative-gate.pt"),
            gate_feature_mode="full",
            lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
            epochs=1,
            learning_rate=1e-2,
            **_optimizer_kwargs(),
            device=torch.device("cpu"),
        )


def test_gate_optimizer_uses_the_declared_adamw_policy() -> None:
    gate = torch.nn.Linear(3, 1)

    optimizer = _build_gate_optimizer(gate, learning_rate=1e-3, **_optimizer_kwargs())

    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.04
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)
    assert optimizer.param_groups[0]["eps"] == 1e-6


class _TieGate(torch.nn.Module):
    feature_dim = 11

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], device=features.device)


def test_validation_counts_prediction_ties_as_stop():
    batches = [
        {
            "features": torch.zeros(1, 11),
            "target_delta": torch.tensor([0.2]),
            "continue_target": torch.tensor([True]),
        },
        {
            "features": torch.zeros(1, 11),
            "target_delta": torch.tensor([0.0]),
            "continue_target": torch.tensor([False]),
        },
    ]

    report = evaluate_cvoi_gate(_TieGate(), batches, device=torch.device("cpu"))

    assert report.num_examples == 2
    assert report.sign_accuracy == 0.5
    assert report.mae == pytest.approx(0.1)
    assert report.roll_rate == 0.0
