import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage as cvoi_manual_lineage_module
from app.vjepa_cowa_world_model.training import cvoi_offline as cvoi_offline_module
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import (
    FeatureRow,
    FeatureStoreMetadata,
    OracleStoreMetadata,
    ScoreRow,
    ScoreStoreMetadata,
    create_embedded_oracle_store_v2,
    create_feature_store,
    create_score_store,
)
from app.vjepa_cowa_world_model.training.cvoi_offline import (
    CvoiOfflineAdapterRequiredError,
    CvoiValueTrainingReport,
    _build_value_optimizer,
    _validate_field_warmup_pair_coverage,
    load_cvoi_offline_adapter,
)
from app.vjepa_cowa_world_model.training.cvoi_offline import run_cvoi_offline_stage as _run_cvoi_offline_stage
from app.vjepa_cowa_world_model.training.cvoi_value import build_cvoi_navsim_e120_direct_value_checkpoint


def run_cvoi_offline_stage(config, *, device, adapter=None):
    return _run_cvoi_offline_stage(
        config,
        device=device,
        adapter=adapter,
        _allow_cpu_for_tests=True,
    )


def _config(tmp_path: Path, *, stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            stage=stage,
            oracle_path=str(tmp_path / "oracle.jsonl"),
            lambda_grid=[0.0, 0.1],
            controller_batch_size=1,
            gate_training_batch_size=None,
            output_checkpoint=str(tmp_path / "output.pt"),
            world_model_checkpoint=str(tmp_path / "world.pt"),
            seed_planner_checkpoint=str(tmp_path / "seed_planner.pt"),
            unguided_planner_checkpoint=str(tmp_path / "p0.pt"),
            field_checkpoint=str(tmp_path / "field.pt"),
            guided_planner_checkpoint=str(tmp_path / "p1.pt"),
            value_hidden_dim=8,
            value_num_layers=1,
            value_dropout=0.0,
            field_warmup_domain="real_cf",
            tokens_per_frame=None,
            max_horizon=3,
            compute_costs=[0.0, 1.0, 2.0, 3.0],
        ),
        optimization=SimpleNamespace(
            epochs=1,
            lr=1e-2,
            weight_decay=0.04,
            betas=(0.8, 0.9),
            eps=1e-6,
        ),
        meta=SimpleNamespace(seed=19, folder=str(tmp_path)),
        value_guidance=SimpleNamespace(
            steps=2,
            objective="last",
            step_size=0.05,
            max_delta_norm=0.25,
            detach_output=True,
        ),
        data=SimpleNamespace(
            fps=2,
            navsim=SimpleNamespace(
                train_roots=(
                    [{"name": "real", "domain": "real"}, {"name": "cf", "domain": "counterfactual"}]
                    if stage == "field_warmup"
                    else [{"name": "real", "domain": "real"}]
                ),
                val_roots=[],
            ),
        ),
    )


def test_offline_core_rejects_cpu_artifact_production(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        _run_cvoi_offline_stage(_config(tmp_path, stage="gate_distillation"), device=torch.device("cpu"))


def test_value_optimizer_uses_the_effective_adamw_config(tmp_path: Path) -> None:
    config = _config(tmp_path, stage="field_warmup")
    model = torch.nn.Linear(3, 2)

    optimizer = _build_value_optimizer(model, config.optimization)

    assert optimizer.param_groups[0]["lr"] == config.optimization.lr
    assert optimizer.param_groups[0]["weight_decay"] == config.optimization.weight_decay
    assert optimizer.param_groups[0]["betas"] == config.optimization.betas
    assert optimizer.param_groups[0]["eps"] == config.optimization.eps


def _write_embedded_gate_oracle(path: Path, *, lineage: str) -> None:
    fixture_root = path.parent / f"_oracle_fixture_{lineage}"
    fixture_root.mkdir(parents=True)
    scenario_sha = "a" * 64
    inventory_sha = "b" * 64

    def split(log_name: str) -> str:
        bucket = int(hashlib.sha256(log_name.encode("utf-8")).hexdigest(), 16) % 10
        return "dev" if bucket == 0 else "train"

    train_log = next(name for index in range(100) if split(name := f"log-{index}") == "train")
    dev_log = next(name for index in range(100) if split(name := f"log-{index}") == "dev")
    logs = (train_log, dev_log)
    score_paths: dict[int, Path] = {}
    feature_paths: dict[int, Path] = {}
    for horizon in range(5):
        source = fixture_root / f"source-h{horizon}.txt"
        source.write_text(f"score h{horizon}", encoding="utf-8")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        score_path = fixture_root / f"score-h{horizon}.sqlite3"
        feature_path = fixture_root / f"feature-h{horizon}.sqlite3"
        create_score_store(
            score_path,
            ScoreStoreMetadata(
                protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
                policy_id=f"forced-h{horizon}",
                lineage=lineage,
                horizon=horizon,
                scenario_manifest_sha256=scenario_sha,
                metric_cache_inventory_sha256=inventory_sha,
                source_path=source,
                source_sha256=source_sha,
                score_semantics="official_v2_one_stage_ordinary_row_score",
            ),
            [
                ScoreRow(0, "token-a", "1" * 64, logs[0], 0.1 + 0.05 * horizon),
                ScoreRow(1, "token-b", "2" * 64, logs[1], 0.2 + 0.05 * horizon),
            ],
            aggregate_score=0.3 + 0.05 * horizon,
        )
        create_feature_store(
            feature_path,
            FeatureStoreMetadata(
                protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
                policy_id=f"forced-h{horizon}",
                lineage=lineage,
                horizon=horizon,
                scenario_manifest_sha256=scenario_sha,
                metric_cache_inventory_sha256=inventory_sha,
                feature_schema="sequential_cvoi_gate_features_lambda_independent_h4_v1",
                feature_sources=("pooled_observed", "pooled_prefix", "field_value"),
                common_random_seed=239,
            ),
            [
                FeatureRow(0, "token-a", "1" * 64, "c" * 64, (float(horizon), 1.0, 2.0)),
                FeatureRow(1, "token-b", "2" * 64, "d" * 64, (float(horizon), 3.0, 4.0)),
            ],
        )
        score_paths[horizon] = score_path
        feature_paths[horizon] = feature_path
    create_embedded_oracle_store_v2(
        path,
        OracleStoreMetadata(
            protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
            lineage=lineage,
            scenario_manifest_sha256=scenario_sha,
            metric_cache_inventory_sha256=inventory_sha,
            lambda_grid=(0.0, 0.1),
        ),
        score_store_paths=score_paths,
        feature_store_paths=feature_paths,
    )


@pytest.mark.parametrize(
    ("path_field", "bad_path"),
    (
        ("oracle_path", "relative/oracle_full.sqlite3"),
        ("output_checkpoint", "relative/gate.pt"),
        ("oracle_path", "{tmp_path}/drift/oracle_full.sqlite3"),
        ("output_checkpoint", "{tmp_path}/drift/gate.pt"),
        ("oracle_path", "{handoff_root}/../handoff/oracle_full.sqlite3"),
        ("output_checkpoint", "{handoff_root}/./gate.pt"),
    ),
)
def test_navsim_e120_gate_distillation_rejects_handoff_path_drift_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_field: str,
    bad_path: str,
) -> None:
    handoff_root = tmp_path / "handoff"
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        tmp_path,
    )
    config = _config(tmp_path, stage="gate_distillation")
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.ablation_signature = SimpleNamespace(
        experiment_role="main",
        branch_id="full",
        cf_field_supervision="hazard_quality",
        field_calibration_mode="local_geometry",
        p0_prefix_mode="uniform",
        gate_feature_mode="full",
    )
    config.cvoi.oracle_path = str(handoff_root / "oracle_full.sqlite3")
    config.cvoi.output_checkpoint = str(handoff_root / "gate.pt")
    setattr(
        config.cvoi,
        path_field,
        bad_path.format(tmp_path=tmp_path, handoff_root=handoff_root),
    )

    with patch.object(
        cvoi_offline_module,
        "_train_cvoi_navsim_e120_official_gate_from_open_store",
        side_effect=AssertionError("fixed handoff drift must fail before Gate training"),
    ) as official_navtrain_train:
        with pytest.raises(ValueError, match=rf"cvoi\.{path_field}.*fixed handoff path"):
            run_cvoi_offline_stage(config, device=torch.device("cpu"))

    official_navtrain_train.assert_not_called()


def test_navsim_e120_gate_distillation_dispatches_to_official_navtrain_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = tmp_path / "handoff"
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        tmp_path,
    )
    config = _config(tmp_path, stage="gate_distillation")
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.ablation_signature = SimpleNamespace(
        experiment_role="main",
        branch_id="full",
        cf_field_supervision="hazard_quality",
        field_calibration_mode="local_geometry",
        p0_prefix_mode="uniform",
        gate_feature_mode="full",
    )
    config.cvoi.gate_training_batch_size = 4096
    config.cvoi.oracle_path = str(handoff_root / "oracle_full.sqlite3")
    config.cvoi.output_checkpoint = str(handoff_root / "gate.pt")
    oracle_path = config.cvoi.oracle_path
    output_checkpoint = config.cvoi.output_checkpoint
    sentinel = object()
    oracle_store = SimpleNamespace(metadata=SimpleNamespace(lineage="p1_full"))

    with (
        patch(
            "app.vjepa_cowa_world_model.training.cvoi_offline."
            "_train_cvoi_navsim_e120_official_gate_from_open_store",
            return_value=sentinel,
        ) as official_navtrain_train,
        patch.object(
            cvoi_offline_module,
            "open_embedded_oracle_store_v2",
            return_value=nullcontext(oracle_store),
        ),
    ):
        report = run_cvoi_offline_stage(config, device=torch.device("cpu"))

    assert report is sentinel
    assert official_navtrain_train.call_args.args == (Path(oracle_path), oracle_store, output_checkpoint)
    assert official_navtrain_train.call_args.kwargs["gate_feature_mode"] == "full"
    assert official_navtrain_train.call_args.kwargs["batch_size"] == 4096
    assert config.cvoi.controller_batch_size == 1


def _direct_navsim_e120_gate_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch_id: str,
    feature_mode: str,
    oracle_value_lineage: str,
) -> SimpleNamespace:
    _set_direct_navsim_e120_results_roots(tmp_path, monkeypatch)
    output_root = tmp_path if branch_id == "full" else tmp_path / "ablation" / branch_id
    oracle_root = tmp_path if oracle_value_lineage == "full" else tmp_path / "ablation" / oracle_value_lineage
    (output_root / "handoff").mkdir(parents=True, exist_ok=True)
    (oracle_root / "handoff").mkdir(parents=True, exist_ok=True)
    config = _config(tmp_path, stage="gate_distillation")
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.ablation_signature = SimpleNamespace(
        experiment_role="main" if branch_id == "full" else "ablation",
        branch_id=branch_id,
        cf_field_supervision="none" if branch_id == "no_cf" else "hazard_quality",
        field_calibration_mode="local_geometry",
        p0_prefix_mode="uniform",
        gate_feature_mode=feature_mode,
    )
    config.cvoi.oracle_path = str(oracle_root / "handoff/oracle_full.sqlite3")
    config.cvoi.output_checkpoint = str(output_root / "handoff/gate.pt")
    return config


@pytest.mark.parametrize(
    ("branch_id", "feature_mode", "oracle_value_lineage"),
    [
        ("full", "full", "full"),
        ("no_cf", "full", "no_cf"),
        ("without_field", "without_field", "full"),
        ("without_stop", "without_stop", "full"),
        ("without_value_summary", "without_value_summary", "full"),
    ],
)
def test_direct_navsim_e120_gate_uses_branch_output_and_exact_oracle_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch_id: str,
    feature_mode: str,
    oracle_value_lineage: str,
) -> None:
    config = _direct_navsim_e120_gate_config(
        tmp_path,
        monkeypatch,
        branch_id=branch_id,
        feature_mode=feature_mode,
        oracle_value_lineage=oracle_value_lineage,
    )
    _write_embedded_gate_oracle(
        Path(config.cvoi.oracle_path),
        lineage=f"p1_{oracle_value_lineage}",
    )
    sentinel = object()

    with patch.object(
        cvoi_offline_module,
        "_train_cvoi_navsim_e120_official_gate_from_open_store",
        return_value=sentinel,
    ) as train_gate:
        result = run_cvoi_offline_stage(config, device=torch.device("cpu"))

    assert result is sentinel
    assert train_gate.call_args.args[0] == Path(config.cvoi.oracle_path)
    assert train_gate.call_args.args[1].metadata.lineage == f"p1_{oracle_value_lineage}"
    assert train_gate.call_args.args[2] == config.cvoi.output_checkpoint
    assert train_gate.call_args.kwargs["gate_feature_mode"] == feature_mode


@pytest.mark.parametrize(
    ("branch_id", "feature_mode"),
    [
        ("without_field", "without_field"),
        ("without_stop", "without_stop"),
        ("without_value_summary", "without_value_summary"),
    ],
)
def test_gate_ablation_derives_full_oracle_and_ablation_output_roots_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch_id: str,
    feature_mode: str,
) -> None:
    config = _direct_navsim_e120_gate_config(
        tmp_path,
        monkeypatch,
        branch_id=branch_id,
        feature_mode=feature_mode,
        oracle_value_lineage="full",
    )
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        Path("/must/not/select/module-full-default"),
    )
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        Path("/must/not/select/module-ablation-default"),
    )

    oracle_path, output_checkpoint, gate_branch = cvoi_offline_module._preflight_direct_navsim_e120_gate_handoffs(
        config.cvoi
    )

    assert oracle_path == config.cvoi.oracle_path
    assert output_checkpoint == config.cvoi.output_checkpoint
    assert gate_branch.result_root == tmp_path / "ablation" / branch_id


@pytest.mark.parametrize(
    ("branch_id", "feature_mode", "expected_oracle_lineage", "stored_oracle_lineage"),
    [
        ("full", "full", "full", "no_cf"),
        ("no_cf", "full", "no_cf", "full"),
    ],
)
def test_direct_navsim_e120_gate_rejects_oracle_metadata_from_other_lineage_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch_id: str,
    feature_mode: str,
    expected_oracle_lineage: str,
    stored_oracle_lineage: str,
) -> None:
    config = _direct_navsim_e120_gate_config(
        tmp_path,
        monkeypatch,
        branch_id=branch_id,
        feature_mode=feature_mode,
        oracle_value_lineage=expected_oracle_lineage,
    )
    _write_embedded_gate_oracle(
        Path(config.cvoi.oracle_path),
        lineage=f"p1_{stored_oracle_lineage}",
    )

    with patch.object(
        cvoi_offline_module,
        "_train_cvoi_navsim_e120_official_gate_from_open_store",
        side_effect=AssertionError("Oracle lineage mismatch must fail before Gate training"),
    ) as train_gate:
        with pytest.raises(ValueError, match="manual Gate Oracle lineage mismatch"):
            run_cvoi_offline_stage(config, device=torch.device("cpu"))

    train_gate.assert_not_called()


@pytest.mark.parametrize(
    ("stage", "required_schema"),
    [
        ("field_warmup", "navsim_e120_quality_field_batch_v1"),
        ("field_calibrated", "navsim_e120_local_quality_calibration_v1"),
        ("stop_calibrated", "navsim_e120_stop_quality_v1"),
    ],
)
def test_navsim_e120_unbundled_stage_names_its_typed_adapter_contract(
    tmp_path: Path,
    stage: str,
    required_schema: str,
) -> None:
    config = _config(tmp_path, stage=stage)
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"

    with pytest.raises(CvoiOfflineAdapterRequiredError, match=required_schema):
        run_cvoi_offline_stage(config, device=torch.device("cpu"))


def test_planner_stage_rejects_offline_entry(tmp_path: Path) -> None:
    config = _config(tmp_path, stage="guided_planner")
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"

    with pytest.raises(ValueError, match="train_predictor_rollout_planner"):
        run_cvoi_offline_stage(config, device=torch.device("cpu"))


def test_training_line_parses_config_and_dispatches_gate(tmp_path: Path) -> None:
    from app.vjepa_cowa_world_model.training.lines import cvoi_offline

    config = _config(tmp_path, stage="gate_distillation")
    sentinel = object()
    with (
        patch.object(cvoi_offline, "parse_training_config", return_value=config) as parse,
        patch.object(cvoi_offline, "seed_cvoi_process") as seed,
        patch.object(cvoi_offline, "run_cvoi_offline_stage", return_value=sentinel) as dispatch,
        patch.object(torch.cuda, "is_available", return_value=True),
    ):
        result = cvoi_offline.main({"cvoi": {"enabled": True}}, resume_preempt=False)

    assert result is sentinel
    parse.assert_called_once_with({"cvoi": {"enabled": True}})
    seed.assert_called_once_with(config)
    dispatch.assert_called_once_with(config, device=torch.device("cuda"))


def test_training_line_requires_cuda_instead_of_silent_cpu_fallback(tmp_path: Path) -> None:
    from app.vjepa_cowa_world_model.training.lines import cvoi_offline

    config = _config(tmp_path, stage="gate_distillation")
    with (
        patch.object(cvoi_offline, "parse_training_config", return_value=config),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            cvoi_offline.main({}, resume_preempt=False)


def test_training_line_rejects_unsupported_preemption_resume(tmp_path: Path) -> None:
    from app.vjepa_cowa_world_model.training.lines import cvoi_offline

    config = _config(tmp_path, stage="gate_distillation")
    with patch.object(cvoi_offline, "parse_training_config", return_value=config):
        with pytest.raises(ValueError, match="resume_preempt"):
            cvoi_offline.main({}, resume_preempt=True)


def test_training_line_rejects_multi_process_checkpoint_writers(tmp_path: Path) -> None:
    from app.vjepa_cowa_world_model.training.lines import cvoi_offline

    config = _config(tmp_path, stage="gate_distillation")
    with (
        patch.object(cvoi_offline, "parse_training_config", return_value=config),
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
    ):
        with pytest.raises(ValueError, match="world_size=1"):
            cvoi_offline.main({}, resume_preempt=False)


def test_train_script_shim_exports_offline_main() -> None:
    from app.vjepa_cowa_world_model import train_cvoi_offline
    from app.vjepa_cowa_world_model.training.lines.cvoi_offline import main

    assert train_cvoi_offline.main is main


def test_direct_navsim_e120_field_validates_before_publishing_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(tmp_path, stage="field_warmup", handoff_root=handoff_root)
    Path(config.cvoi.world_model_checkpoint).write_bytes(b"world")
    Path(config.cvoi.unguided_planner_checkpoint).write_bytes(b"p0")
    adapter = _DirectNavSimE120LifecycleAdapter()

    report = run_cvoi_offline_stage(config, device=torch.device("cpu"), adapter=adapter)

    assert report.field_validation_metrics["real"]["sample_count"] == 2
    assert "provenance" not in report.__dataclass_fields__
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["phase"] == "field_warmup"
    assert payload["branch_id"] == "field_full"


def test_offline_module_exposes_no_legacy_navsim_e120_field_receipt_surface() -> None:
    forbidden = {
        "_navsim_field_validation_identity",
        "build_formal_v2_navsim_field_validation_receipt",
        "write_formal_v2_navsim_field_validation_receipt",
    }

    assert forbidden.isdisjoint(vars(cvoi_offline_module))
    assert "field_validation_receipt_path" not in CvoiValueTrainingReport.__dataclass_fields__


def test_direct_navsim_e120_field_validation_failure_preserves_previous_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(tmp_path, stage="field_warmup", handoff_root=handoff_root)
    Path(config.cvoi.world_model_checkpoint).write_bytes(b"world")
    Path(config.cvoi.unguided_planner_checkpoint).write_bytes(b"p0")
    Path(config.cvoi.output_checkpoint).write_bytes(b"previous-valid-field")
    adapter = _DirectNavSimE120LifecycleAdapter()

    with (
        patch(
            "app.vjepa_cowa_world_model.training.cvoi_offline.validate_cvoi_field_epoch",
            side_effect=RuntimeError("injected validation failure"),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.cvoi_offline.atomic_torch_save_replace",
            side_effect=AssertionError("failed validation must not publish a checkpoint"),
        ),
    ):
        with pytest.raises(RuntimeError, match="injected validation failure"):
            run_cvoi_offline_stage(config, device=torch.device("cpu"), adapter=adapter)

    assert Path(config.cvoi.output_checkpoint).read_bytes() == b"previous-valid-field"


@pytest.mark.parametrize(
    ("mode", "metrics"),
    [
        ("none", {"cf_hazard_pairs": 0.0, "cf_quality_pairs": 0.0}),
        ("hazard_only", {"cf_hazard_pairs": 1.0, "cf_quality_pairs": 0.0}),
        ("quality_only", {"cf_hazard_pairs": 0.0, "cf_quality_pairs": 1.0}),
        ("hazard_quality", {"cf_hazard_pairs": 1.0, "cf_quality_pairs": 1.0}),
    ],
)
def test_field_warmup_pair_coverage_only_requires_enabled_cf_labels(mode: str, metrics: dict[str, float]) -> None:
    _validate_field_warmup_pair_coverage(metrics, epoch=0, cf_field_supervision=mode)


@pytest.mark.parametrize(
    ("mode", "metrics", "missing"),
    [
        ("hazard_only", {"cf_hazard_pairs": 0.0, "cf_quality_pairs": 4.0}, "cf_hazard_pairs"),
        ("quality_only", {"cf_hazard_pairs": 4.0, "cf_quality_pairs": 0.0}, "cf_quality_pairs"),
        ("hazard_quality", {"cf_hazard_pairs": 0.0, "cf_quality_pairs": 0.0}, "cf_hazard_pairs"),
    ],
)
def test_field_warmup_pair_coverage_rejects_missing_enabled_pairs(
    mode: str,
    metrics: dict[str, float],
    missing: str,
) -> None:
    with pytest.raises(ValueError, match=missing):
        _validate_field_warmup_pair_coverage(metrics, epoch=0, cf_field_supervision=mode)


class _DirectNavSimE120LifecycleAdapter:
    embed_dim = 4

    @staticmethod
    def _field_batch(*, provenance):
        assert provenance is None
        return {
            "z_observed": torch.randn(4, 2, 3, 4),
            "z_future": torch.randn(4, 4, 3, 4),
            "dataset_domains": ["real", "counterfactual", "real", "counterfactual"],
            "real_quality_targets": torch.tensor(
                [
                    [0.2, 0.4, 0.6, 0.7],
                    [float("nan")] * 4,
                    [0.5, 0.7, 0.8, 0.9],
                    [float("nan")] * 4,
                ]
            ),
            "real_group_ids": ["scene-a", "", "scene-b", ""],
            "stable_sample_ids": ["real:a", "cf:a", "real:b", "cf:b"],
            "cf_hazard": torch.tensor([False, True, False, True]),
            "cf_hazard_types": ["", "非自车行为引起", "", "自车行为引起"],
            "cf_hazard_pair_real_indices": torch.tensor([0, 2], dtype=torch.long),
            "cf_hazard_pair_counterfactual_indices": torch.tensor([1, 3], dtype=torch.long),
            "cf_hazard_pair_keys": [("scene-a", 0), ("scene-b", 0)],
            "cf_quality": torch.tensor([float("nan"), 0.9, float("nan"), 0.1]),
            "adapter_schema": "navsim_e120_quality_field_batch_v1",
        }

    def value_batches(self, stage: str, epoch: int, *, provenance):
        assert epoch == 0
        if stage == "field_warmup":
            return [self._field_batch(provenance=provenance)]
        assert provenance is None
        common = {
            "z_observed": torch.randn(2, 2, 3, 4),
            "z_future": torch.randn(2, 4, 3, 4),
            "dataset_domains": ["real", "real"],
            "stable_sample_ids": ["real:a", "real:b"],
        }
        if stage == "field_calibrated":
            return [
                {
                    **common,
                    "adapter_schema": "navsim_e120_local_quality_calibration_v1",
                    "real_quality_targets": torch.tensor([[0.2, 0.4, 0.6, 0.7], [0.5, 0.7, 0.8, 0.9]]),
                    "real_group_ids": ["scene-a", "scene-a"],
                }
            ]
        if stage == "stop_calibrated":
            return [
                {
                    **common,
                    "adapter_schema": "navsim_e120_stop_quality_v1",
                    "stop_quality_targets": torch.tensor([[0.1, 0.2, 0.4, 0.6, 0.7], [0.2, 0.3, 0.5, 0.7, 0.8]]),
                }
            ]
        raise AssertionError(stage)

    def field_validation_batches(self, *, domain: str, provenance):
        assert provenance is None
        field = self._field_batch(provenance=None)
        if domain == "real":
            real_rows = torch.tensor([True, False, True, False])
            return [
                {
                    "z_observed": field["z_observed"][real_rows],
                    "z_future": field["z_future"][real_rows],
                    "dataset_domains": ["real", "real"],
                    "real_quality_targets": field["real_quality_targets"][real_rows],
                    "real_group_ids": ["scene-a", "scene-b"],
                    "stable_sample_ids": ["real:a", "real:b"],
                    "adapter_schema": field["adapter_schema"],
                }
            ]
        if domain == "matched_real_counterfactual":
            return [field]
        raise AssertionError(domain)


class _NoCfDirectNavSimE120LifecycleAdapter:
    embed_dim = 4

    def __init__(self) -> None:
        self.training_batches_consumed = 0
        self.real_validation_requests = 0
        self.counterfactual_requests = 0

    @staticmethod
    def _real_field_batch() -> dict[str, object]:
        return {
            "z_observed": torch.randn(2, 2, 3, 4),
            "z_future": torch.randn(2, 4, 3, 4),
            "dataset_domains": ["real", "real"],
            "real_quality_targets": torch.tensor([[0.2, 0.4, 0.6, 0.7], [0.5, 0.7, 0.8, 0.9]]),
            "real_group_ids": ["scene-a", "scene-b"],
            "stable_sample_ids": ["real:a", "real:b"],
            "adapter_schema": "navsim_e120_quality_field_batch_v1",
        }

    def value_batches(self, stage: str, epoch: int, *, provenance):
        assert stage == "field_warmup"
        assert epoch == 0
        assert provenance is None
        self.training_batches_consumed += 1
        yield self._real_field_batch()

    def field_validation_batches(self, *, domain: str, provenance):
        assert provenance is None
        if domain == "matched_real_counterfactual":
            self.counterfactual_requests += 1
            raise AssertionError("no-CF Field must not request counterfactual validation data")
        assert domain == "real"
        self.real_validation_requests += 1
        return [self._real_field_batch()]


def _set_direct_navsim_e120_results_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        tmp_path / "ablation",
    )


def _direct_navsim_e120_handoff_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    _set_direct_navsim_e120_results_roots(tmp_path, monkeypatch)
    return handoff_root


def _direct_navsim_e120_value_config(
    tmp_path: Path,
    *,
    stage: str,
    handoff_root: Path,
    lineage: str = "full",
) -> SimpleNamespace:
    supervision = {
        "full": "hazard_quality",
        "no_cf": "none",
        "hazard_only": "hazard_only",
        "quality_only": "quality_only",
    }[lineage]
    lineage_handoff_root = (
        handoff_root if lineage == "full" else handoff_root.parent / "ablation" / lineage / "handoff"
    )
    lineage_handoff_root.mkdir(parents=True, exist_ok=True)
    config = _config(tmp_path, stage=stage)
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.schema = "cvoi_dual_value_navsim_e120_v1"
    config.cvoi.ablation_signature = SimpleNamespace(
        experiment_role="main" if lineage == "full" else "ablation",
        branch_id=lineage,
        cf_field_supervision=supervision,
        field_calibration_mode="local_geometry",
        p0_prefix_mode="uniform",
        gate_feature_mode="full",
    )
    config.cvoi.value_updates_per_epoch = None
    config.cvoi.max_horizon = 4
    config.cvoi.compute_costs = [0.0, 1.0, 2.0, 3.0, 4.0]
    config.cvoi.controller_lineage = "value_guided"
    config.cvoi.unguided_planner_checkpoint = str(handoff_root / "p0_selected.pt")
    config.cvoi.field_warmup_domain = "real" if lineage == "no_cf" else "real_cf"
    if stage == "field_warmup":
        config.cvoi.output_checkpoint = str(lineage_handoff_root / "field.pt")
    elif stage == "field_calibrated":
        config.cvoi.field_checkpoint = str(lineage_handoff_root / "field.pt")
        config.cvoi.output_checkpoint = str(lineage_handoff_root / "calibration.pt")
    elif stage == "stop_calibrated":
        config.cvoi.field_checkpoint = str(lineage_handoff_root / "calibration.pt")
        config.cvoi.guided_planner_checkpoint = str(lineage_handoff_root / "p1_selected.pt")
        config.cvoi.output_checkpoint = str(lineage_handoff_root / "stop.pt")
    else:
        raise AssertionError(stage)
    return config


@pytest.mark.parametrize(
    ("lineage", "stage", "expected_branch"),
    [
        ("full", "field_warmup", "field_full"),
        ("no_cf", "field_warmup", "field_no_cf"),
        ("hazard_only", "field_warmup", "field_hazard_only"),
        ("quality_only", "field_warmup", "field_quality_only"),
        ("full", "field_calibrated", "calibration_full"),
        ("no_cf", "field_calibrated", "calibration_no_cf"),
        ("hazard_only", "field_calibrated", "calibration_hazard_only"),
        ("quality_only", "field_calibrated", "calibration_quality_only"),
        ("full", "stop_calibrated", "stop_full"),
        ("no_cf", "stop_calibrated", "stop_no_cf"),
    ],
)
def test_direct_navsim_e120_value_uses_branch_local_handoffs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
    stage: str,
    expected_branch: str,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage=stage,
        handoff_root=handoff_root,
        lineage=lineage,
    )

    cvoi_offline_module._preflight_direct_navsim_e120_value_handoffs(config)

    assert Path(config.cvoi.unguided_planner_checkpoint) == handoff_root / "p0_selected.pt"
    expected_lineage_root = tmp_path if lineage == "full" else tmp_path / "ablation" / lineage
    assert Path(config.cvoi.output_checkpoint).is_relative_to(expected_lineage_root / "handoff")
    assert (
        cvoi_manual_lineage_module.resolve_cvoi_manual_value_lineage(
            config.cvoi.ablation_signature,
            stage=stage,
        ).checkpoint_branch_id(stage)
        == expected_branch
    )


def test_ablation_value_preflight_derives_both_roots_from_configured_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_warmup",
        handoff_root=handoff_root,
        lineage="no_cf",
    )
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        Path("/must/not/select/module-ablation-default"),
    )

    lineage = cvoi_offline_module._preflight_direct_navsim_e120_value_handoffs(config)

    assert lineage.p0_result_root == tmp_path
    assert lineage.result_root == tmp_path / "ablation/no_cf"


@pytest.mark.parametrize(
    ("lineage", "cf_mode", "warmup_domain", "metrics"),
    [
        (
            "hazard_only",
            "hazard_only",
            "real_cf",
            {"cf_hazard_pairs": 2.0, "cf_quality_pairs": 0.0},
        ),
        (
            "quality_only",
            "quality_only",
            "real_cf",
            {"cf_hazard_pairs": 0.0, "cf_quality_pairs": 2.0},
        ),
    ],
)
def test_direct_navsim_e120_field_routes_branch_supervision_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
    cf_mode: str,
    warmup_domain: str,
    metrics: dict[str, float],
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_warmup",
        handoff_root=handoff_root,
        lineage=lineage,
    )
    training_metrics = {
        **metrics,
        "eligibility_schedule_signature": "a" * 64,
    }

    with (
        patch.object(
            cvoi_offline_module,
            "train_cvoi_value_epoch",
            return_value=training_metrics,
        ) as train_epoch,
        patch.object(
            cvoi_offline_module,
            "validate_cvoi_field_epoch",
            return_value={"validated": True},
        ) as validate_epoch,
    ):
        report = run_cvoi_offline_stage(
            config,
            device=torch.device("cpu"),
            adapter=_DirectNavSimE120LifecycleAdapter(),
        )

    assert train_epoch.call_args.kwargs["cf_field_supervision"] == cf_mode
    assert train_epoch.call_args.kwargs["field_warmup_domain"] == warmup_domain
    assert validate_epoch.call_args.kwargs["cf_field_supervision"] == cf_mode
    assert validate_epoch.call_args.kwargs["counterfactual_batches"] is not None
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["branch_id"] == f"field_{lineage}"


def test_direct_navsim_e120_no_cf_field_consumes_real_only_and_never_requests_cf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_warmup",
        handoff_root=handoff_root,
        lineage="no_cf",
    )
    adapter = _NoCfDirectNavSimE120LifecycleAdapter()
    real_train = cvoi_offline_module.train_cvoi_value_epoch
    real_validate = cvoi_offline_module.validate_cvoi_field_epoch

    with (
        patch.object(cvoi_offline_module, "train_cvoi_value_epoch", wraps=real_train) as train_epoch,
        patch.object(cvoi_offline_module, "validate_cvoi_field_epoch", wraps=real_validate) as validate_epoch,
    ):
        report = run_cvoi_offline_stage(
            config,
            device=torch.device("cpu"),
            adapter=adapter,
        )

    assert adapter.training_batches_consumed == 1
    assert adapter.real_validation_requests == 1
    assert adapter.counterfactual_requests == 0
    assert train_epoch.call_args.kwargs["cf_field_supervision"] == "none"
    assert train_epoch.call_args.kwargs["field_warmup_domain"] == "real"
    assert validate_epoch.call_args.kwargs["counterfactual_batches"] is None
    assert validate_epoch.call_args.kwargs["cf_field_supervision"] == "none"
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["branch_id"] == "field_no_cf"


@pytest.mark.parametrize(
    ("lineage", "phase", "parent_phase", "parent_branch", "output_branch"),
    [
        ("full", "field_calibrated", "field_warmup", "field_full", "calibration_full"),
        ("no_cf", "field_calibrated", "field_warmup", "field_no_cf", "calibration_no_cf"),
        (
            "hazard_only",
            "field_calibrated",
            "field_warmup",
            "field_hazard_only",
            "calibration_hazard_only",
        ),
        (
            "quality_only",
            "field_calibrated",
            "field_warmup",
            "field_quality_only",
            "calibration_quality_only",
        ),
        ("full", "stop_calibrated", "field_calibrated", "calibration_full", "stop_full"),
        ("no_cf", "stop_calibrated", "field_calibrated", "calibration_no_cf", "stop_no_cf"),
    ],
)
def test_direct_navsim_e120_value_reads_and_publishes_matching_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
    phase: str,
    parent_phase: str,
    parent_branch: str,
    output_branch: str,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage=phase,
        handoff_root=handoff_root,
        lineage=lineage,
    )
    reads: list[tuple[Path, str, str]] = []

    def read_parent(path: str | Path, **kwargs: object) -> dict[str, object]:
        reads.append((Path(path), str(kwargs["required_phase"]), str(kwargs["required_branch_id"])))
        return {}

    metrics = {"eligibility_schedule_signature": "b" * 64}
    with (
        patch.object(
            cvoi_offline_module,
            "read_cvoi_navsim_e120_direct_value_checkpoint",
            side_effect=read_parent,
        ),
        patch.object(
            cvoi_offline_module,
            "train_cvoi_value_epoch",
            return_value=metrics,
        ),
    ):
        report = run_cvoi_offline_stage(
            config,
            device=torch.device("cpu"),
            adapter=_DirectNavSimE120LifecycleAdapter(),
        )

    assert reads == [(Path(config.cvoi.field_checkpoint), parent_phase, parent_branch)]
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["branch_id"] == output_branch
    expected_lineage = cvoi_manual_lineage_module.resolve_cvoi_manual_value_lineage(
        config.cvoi.ablation_signature,
        stage=phase,
    )
    assert payload["parents"] == cvoi_manual_lineage_module.build_cvoi_manual_value_parents(
        expected_lineage,
        phase,
    )


def test_direct_navsim_e120_value_rejects_cross_lineage_parent_path_before_model_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_calibrated",
        handoff_root=handoff_root,
        lineage="no_cf",
    )
    config.cvoi.field_checkpoint = str(tmp_path / "handoff/field.pt")

    with (
        patch.object(
            cvoi_offline_module,
            "_new_value_model",
            side_effect=AssertionError("cross-lineage path must fail before model construction"),
        ) as new_model,
        patch.object(
            cvoi_offline_module,
            "read_cvoi_navsim_e120_direct_value_checkpoint",
            side_effect=AssertionError("cross-lineage path must fail before checkpoint reading"),
        ) as read_checkpoint,
    ):
        with pytest.raises(ValueError, match="fixed handoff path"):
            run_cvoi_offline_stage(
                config,
                device=torch.device("cpu"),
                adapter=_DirectNavSimE120LifecycleAdapter(),
            )

    new_model.assert_not_called()
    read_checkpoint.assert_not_called()


@pytest.mark.parametrize(
    ("stage", "path_field"),
    (
        ("field_warmup", "unguided_planner_checkpoint"),
        ("field_warmup", "output_checkpoint"),
        ("field_calibrated", "unguided_planner_checkpoint"),
        ("field_calibrated", "field_checkpoint"),
        ("field_calibrated", "output_checkpoint"),
        ("stop_calibrated", "unguided_planner_checkpoint"),
        ("stop_calibrated", "field_checkpoint"),
        ("stop_calibrated", "guided_planner_checkpoint"),
        ("stop_calibrated", "output_checkpoint"),
    ),
)
def test_direct_navsim_e120_value_rejects_fixed_handoff_drift_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    path_field: str,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(tmp_path, stage=stage, handoff_root=handoff_root)
    setattr(config.cvoi, path_field, str(tmp_path / f"drift-{path_field}.pt"))

    with (
        patch.object(
            cvoi_offline_module,
            "_new_value_model",
            side_effect=AssertionError("fixed handoff drift must fail before Value model construction"),
        ) as new_model,
        patch.object(
            cvoi_offline_module,
            "_load_direct_parent_value_model",
            side_effect=AssertionError("fixed handoff drift must fail before parent Value restore"),
        ) as load_parent,
    ):
        with pytest.raises(ValueError, match=rf"cvoi\.{path_field}.*fixed handoff path"):
            run_cvoi_offline_stage(
                config,
                device=torch.device("cpu"),
                adapter=_DirectNavSimE120LifecycleAdapter(),
            )

    new_model.assert_not_called()
    load_parent.assert_not_called()


def test_direct_navsim_e120_value_preflights_handoffs_before_adapter_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_warmup",
        handoff_root=handoff_root,
    )
    config.cvoi.output_checkpoint = str(tmp_path / "wrong-field.pt")
    config.cvoi.offline_adapter_factory = "poison.module:create"

    with patch.object(
        cvoi_offline_module.importlib,
        "import_module",
        side_effect=AssertionError("handoff preflight must run before adapter import"),
    ) as import_module:
        with pytest.raises(ValueError, match=r"cvoi\.output_checkpoint.*fixed handoff path"):
            load_cvoi_offline_adapter(config, device=torch.device("cpu"))

    import_module.assert_not_called()


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing_parent", "output parent directory does not exist"),
        ("symlink_parent", "output parent directory must not be a symlink"),
        ("symlink_target", "output target must be absent or a regular non-symlink file"),
        ("directory_target", "output target must be absent or a regular non-symlink file"),
    ),
)
def test_direct_navsim_e120_value_preflights_output_filesystem_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    handoff_root = tmp_path / "handoff"
    if case == "symlink_parent":
        real_root = tmp_path / "real-handoff"
        real_root.mkdir()
        handoff_root.symlink_to(real_root, target_is_directory=True)
    elif case != "missing_parent":
        handoff_root.mkdir()
    _set_direct_navsim_e120_results_roots(tmp_path, monkeypatch)
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_warmup",
        handoff_root=handoff_root,
    )
    if case == "missing_parent":
        handoff_root.rmdir()
    output_path = Path(config.cvoi.output_checkpoint)
    if case == "symlink_target":
        victim = tmp_path / "victim.pt"
        victim.write_bytes(b"victim")
        output_path.symlink_to(victim)
    elif case == "directory_target":
        output_path.mkdir()

    with patch.object(
        cvoi_offline_module,
        "_new_value_model",
        side_effect=AssertionError("output preflight must fail before Value model construction"),
    ) as new_model:
        with pytest.raises((FileNotFoundError, ValueError), match=message):
            run_cvoi_offline_stage(
                config,
                device=torch.device("cpu"),
                adapter=_DirectNavSimE120LifecycleAdapter(),
            )

    new_model.assert_not_called()


@pytest.mark.parametrize(
    "architecture_override",
    (
        {"embed_dim": 5},
        {"hidden_dim": 9},
        {"num_layers": 2},
        {"dropout": 0.25},
    ),
)
def test_direct_navsim_e120_parent_architecture_drift_fails_before_batches_and_preserves_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture_override: dict[str, object],
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    architecture = {
        "embed_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
        **architecture_override,
    }
    with torch.random.fork_rng(devices=[]):
        parent_model = PrefixDualValueModel(**architecture)
    parent_payload = build_cvoi_navsim_e120_direct_value_checkpoint(
        parent_model,
        phase="field_warmup",
        branch_id="field_full",
        epoch=1,
        parents={
            "unguided_planner": {
                "stage": "p0",
                "branch_id": "p0_uniform",
            },
        },
    )
    torch.save(parent_payload, handoff_root / "field.pt")
    config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_calibrated",
        handoff_root=handoff_root,
    )

    class _PoisonBatchAdapter:
        embed_dim = 4

        @staticmethod
        def value_batches(stage, epoch, *, provenance):
            del stage, epoch, provenance
            raise AssertionError("architecture drift must fail before adapter batches")

    torch.manual_seed(87123)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="architecture mismatch"):
        run_cvoi_offline_stage(
            config,
            device=torch.device("cpu"),
            adapter=_PoisonBatchAdapter(),
        )
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_direct_navsim_e120_value_chain_uses_only_structural_handoffs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _direct_navsim_e120_handoff_root(tmp_path, monkeypatch)
    Path(tmp_path / "world.pt").write_bytes(b"world")
    Path(handoff_root / "p0_selected.pt").write_bytes(b"p0")
    Path(handoff_root / "p1_selected.pt").write_bytes(b"p1")
    adapter = _DirectNavSimE120LifecycleAdapter()
    field_config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_warmup",
        handoff_root=handoff_root,
    )
    calibration_config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="field_calibrated",
        handoff_root=handoff_root,
    )
    stop_config = _direct_navsim_e120_value_config(
        tmp_path,
        stage="stop_calibrated",
        handoff_root=handoff_root,
    )

    forbidden = (
        "_verified_adapter_audit",
        "_value_batch_provenance",
        "_load_parent_value_model",
        "_warmup_data_from_checkpoint",
        "_sha256_file",
        "save_prefix_dual_value_checkpoint",
    )
    assert set(forbidden).isdisjoint(vars(cvoi_offline_module))
    field_report = run_cvoi_offline_stage(field_config, device=torch.device("cpu"), adapter=adapter)
    calibration_report = run_cvoi_offline_stage(
        calibration_config,
        device=torch.device("cpu"),
        adapter=adapter,
    )
    stop_report = run_cvoi_offline_stage(stop_config, device=torch.device("cpu"), adapter=adapter)

    expected = (
        (field_report, "field_warmup", "field_full", {"unguided_planner"}),
        (
            calibration_report,
            "field_calibrated",
            "calibration_full",
            {"unguided_planner", "field"},
        ),
        (
            stop_report,
            "stop_calibrated",
            "stop_full",
            {"unguided_planner", "calibration", "guided_planner"},
        ),
    )
    for report, phase, branch_id, parent_roles in expected:
        payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=False)
        assert set(payload) == {
            "schema",
            "phase",
            "protocol_version",
            "branch_id",
            "epoch",
            "architecture",
            "roles",
            "parents",
            "state_dict",
        }
        assert payload["phase"] == phase
        assert payload["branch_id"] == branch_id
        assert set(payload["parents"]) == parent_roles
        assert "provenance" not in report.__dataclass_fields__
    assert field_report.field_validation_metrics is not None
    assert calibration_report.field_validation_metrics is None
    assert stop_report.field_validation_metrics is None
