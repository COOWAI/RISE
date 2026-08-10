import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage as cvoi_manual_lineage_module
from app.vjepa_cowa_world_model.training import cvoi_runtime as cvoi_runtime_module
from app.vjepa_cowa_world_model.training import cvoi_value as cvoi_value_module
from app.vjepa_cowa_world_model.training.configs.planner import PlannerConfig, TokenAEConfig
from app.vjepa_cowa_world_model.training.cvoi_runtime import (
    apply_cvoi_planner_guidance,
    build_cvoi_gate_provenance,
    load_cvoi_dual_value_model,
    load_cvoi_gate_for_evaluation,
    require_cvoi_planner_stage,
    resolve_cvoi_planner_checkpoint_paths,
    resolve_cvoi_training_rollout_horizon,
    resolve_cvoi_validation_rollout_horizon,
    validate_cvoi_sequential_runtime_config,
)
from app.vjepa_cowa_world_model.training.cvoi_value import build_cvoi_navsim_e120_direct_value_checkpoint


def test_cvoi_training_rollout_full_prefix_is_controller_horizon_not_planner_pose_count() -> None:
    config = SimpleNamespace(cvoi=SimpleNamespace(enabled=True, max_horizon=4))

    assert resolve_cvoi_training_rollout_horizon(config, total_future_steps=6) == 4


def test_non_cvoi_training_rollout_keeps_the_full_future_horizon() -> None:
    config = SimpleNamespace(cvoi=SimpleNamespace(enabled=False, max_horizon=4))

    assert resolve_cvoi_training_rollout_horizon(config, total_future_steps=6) == 6


def test_cvoi_training_rollout_rejects_a_checkpoint_shorter_than_controller_horizon() -> None:
    config = SimpleNamespace(cvoi=SimpleNamespace(enabled=True, max_horizon=4))

    with pytest.raises(ValueError, match="at least H=4"):
        resolve_cvoi_training_rollout_horizon(config, total_future_steps=3)


def test_cvoi_planner_checkpoint_selection_uses_controller_full_horizon() -> None:
    config = SimpleNamespace(cvoi=SimpleNamespace(enabled=True, stage="unguided_planner", max_horizon=4))

    assert resolve_cvoi_validation_rollout_horizon(config) == 4
    config.cvoi.stage = "guided_planner"
    assert resolve_cvoi_validation_rollout_horizon(config) == 4
    config.cvoi.stage = "field_warmup"
    assert resolve_cvoi_validation_rollout_horizon(config) is None
    config.cvoi.enabled = False
    assert resolve_cvoi_validation_rollout_horizon(config) is None


def _audit_manifest(root: Path) -> dict:
    converter_path = root / "converter.py"
    split_path = root / "split.yaml"
    converter_path.write_text("# converter\n", encoding="utf-8")
    split_path.write_text("roots: {}\n", encoding="utf-8")
    converter_fingerprint = hashlib.sha256(converter_path.read_bytes()).hexdigest()
    split_fingerprint = hashlib.sha256(split_path.read_bytes()).hexdigest()
    root_reports = []
    for name, domain in (("real", "real"), ("counterfactual", "counterfactual")):
        data_path = root / name / "logs"
        data_path.mkdir(parents=True, exist_ok=True)
        scene_path = data_path / f"{name}-scene.pkl"
        scene_path.write_bytes(f"{name}-scene".encode("ascii"))
        root_reports.append(
            {
                "name": name,
                "domain": domain,
                "data_path": str(data_path.resolve()),
                "max_agents": 256,
                "sidecars": {},
                "scenes": [
                    {
                        "scene": scene_path.stem,
                        "relative_path": scene_path.name,
                        "file_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    manifest = {
        "manifest_version": 1,
        "contract": "navsim_cvoi_geometry",
        "passed": True,
        "ready_for_labels": True,
        "fingerprint_algorithm": "sha256",
        "converter_fingerprint": converter_fingerprint,
        "converter": {"path": str(converter_path.resolve()), "fingerprint": converter_fingerprint},
        "split_manifest": {
            "passed": True,
            "path": str(split_path.resolve()),
            "fingerprint": split_fingerprint,
        },
        "roots": root_reports,
    }
    canonical = json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {**manifest, "fingerprint": fingerprint, "dataset_fingerprint": fingerprint}


def _config(stage: str, *, field: Path, dual: Path) -> SimpleNamespace:
    guidance_enabled = stage in {"guided_planner", "stop_calibrated", "evaluation"}
    audit_path = field.parent / "audit.json"
    audit_path.write_text(json.dumps(_audit_manifest(field.parent)), encoding="utf-8")
    world_model_path = field.parent / "world_model.pt"
    if not world_model_path.exists():
        world_model_path.write_bytes(b"world-model")
    return SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            schema="cvoi_dual_value_v1",
            stage=stage,
            guidance_steps=2,
            guidance_objective="last",
            max_horizon=4,
            field_checkpoint=str(field),
            dual_value_checkpoint=str(dual),
            gate_checkpoint=None,
            unguided_planner_checkpoint=str(field.parent / "p0.pt"),
            guided_planner_checkpoint=str(dual.with_name(f"{dual.stem}_planner.pt")),
            audit_manifest_path=str(audit_path),
            audit_path_mode="exact",
            audit_verification_mode="live",
            world_model_checkpoint=str(world_model_path),
            token_ae_checkpoint=None,
            output_checkpoint=str(field.parent / "output.pt"),
            oracle_path=str(field.parent / "oracle.jsonl"),
        ),
        meta=SimpleNamespace(
            resume_checkpoint=None,
            predictor_runtime_normalize_reps=None,
            predictor_checkpoint=None,
            ae_checkpoint=None,
            dtype="bfloat16",
            seed=19,
            use_sdpa=True,
            deterministic=False,
        ),
        model=SimpleNamespace(
            backbone="vjepa2",
            vjepa_resolution=(256, 512),
            vjepa_crop_top_bottom=28,
            vjepa_num_frames=2,
            vjepa_checkpoint_key="target_encoder",
            vjepa_use_grid_mask=False,
            vjepa_use_causal_attention=True,
            patch_size=16,
            pred_depth=12,
            pred_num_heads=12,
            pred_embed_dim=384,
            pred_is_frame_causal=True,
            uniform_power=True,
            use_rope=True,
            use_silu=False,
            use_pred_silu=False,
            wide_silu=True,
            use_extrinsics=False,
            use_mask_tokens=False,
            zero_init_mask_tokens=True,
        ),
        train=SimpleNamespace(
            predictor_type="ac_transformer",
            num_observed_frames=2,
            num_encoder_frames=2,
            use_parallel_predictor=False,
            predictor_inference_consistent=True,
            predictor_aux_policy="inference_consistent",
            predictor_no_aux_input=False,
            use_states_for_predictor=False,
            action_dim=3,
            state_dim=8,
            command_dim=0,
            use_drive_command=False,
        ),
        loss=SimpleNamespace(normalize_reps=True),
        multiview=SimpleNamespace(enabled=False, freeze_fusion=False),
        planner=PlannerConfig(
            planner_type="diffusion",
            diff_inference_steps=20,
            diff_num_samples=6,
            diff_num_modes=6,
            diff_traj_dim=4,
            diff_dt=0.5,
            diff_trajectory_token_mode="per_pose_token",
            diff_use_anchor_frame=True,
            use_action_history_for_planner=True,
            action_history_dim=3,
            z_ar_mode="full",
            use_z_context=False,
        ),
        token_ae=TokenAEConfig(enabled=False),
        data=SimpleNamespace(
            fps=2,
            num_target_frames=6,
            tokens_per_frame=2,
            crop_size=(16, 16),
            patch_size=16,
            tubelet_size=2,
            use_tubelet_repeat=False,
            navsim=SimpleNamespace(
                train_roots=[
                    {
                        "name": "real",
                        "domain": "real",
                        "data_path": str(field.parent / "real" / "logs"),
                        "max_agents": 256,
                    },
                    {
                        "name": "counterfactual",
                        "domain": "counterfactual",
                        "data_path": str(field.parent / "counterfactual" / "logs"),
                        "max_agents": 256,
                    },
                ],
                val_roots=[],
            ),
        ),
        value_guidance=SimpleNamespace(
            enabled=guidance_enabled,
            steps=2,
            objective="last",
            step_size=0.05,
            max_delta_norm=0.25,
            detach_output=True,
        ),
    )


def _configure_h4v3_manual_runtime(
    config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    results_root: Path,
    lineage: str = "full",
) -> None:
    supervision = {
        "full": "hazard_quality",
        "no_cf": "none",
        "hazard_only": "hazard_only",
        "quality_only": "quality_only",
    }[lineage]
    stage = str(config.cvoi.stage)
    branch_id = f"p1_{lineage}" if stage in {"guided_planner", "evaluation"} else lineage
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.ablation_signature = SimpleNamespace(
        experiment_role="main" if lineage == "full" else "ablation",
        branch_id=branch_id,
        cf_field_supervision=supervision,
        field_calibration_mode="local_geometry",
        p0_prefix_mode="uniform",
        gate_feature_mode="full",
    )
    config.cvoi.unguided_planner_checkpoint = str(results_root / "handoff/p0_selected.pt")
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        results_root,
    )
    monkeypatch.setattr(
        cvoi_manual_lineage_module,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        results_root / "ablation",
    )


def test_sequential_runtime_allows_longer_planning_horizon_than_rollout_budget(tmp_path: Path) -> None:
    config = _config("evaluation", field=tmp_path / "field.pt", dual=tmp_path / "dual.pt")
    config.cvoi.max_horizon = 4
    config.data.num_target_frames = 8
    config.train.num_observed_frames = 2
    config.train.use_parallel_predictor = False
    config.train.predictor_inference_consistent = True
    config.planner = SimpleNamespace(
        z_ar_mode="full",
        use_planner=True,
        planner_type="diffusion",
        use_z_context=False,
        use_observed_tokens=False,
        use_action_history_for_planner=True,
    )

    validate_cvoi_sequential_runtime_config(config)


def _write_direct_value(
    path: Path,
    *,
    phase: str,
    lineage: str = "full",
    embed_dim: int = 4,
) -> dict[str, object]:
    branch_prefix = {
        "field_calibrated": "calibration",
        "stop_calibrated": "stop",
    }[phase]
    branch = f"{branch_prefix}_{lineage}"
    value_lineage = cvoi_manual_lineage_module.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase=phase,
        branch_id=branch,
    )
    model = PrefixDualValueModel(embed_dim=embed_dim, hidden_dim=6)
    payload = build_cvoi_navsim_e120_direct_value_checkpoint(
        model,
        phase=phase,
        branch_id=branch,
        epoch=7,
        parents=cvoi_manual_lineage_module.build_cvoi_manual_value_parents(value_lineage, phase),
    )
    torch.save(payload, path)
    return payload


@pytest.mark.parametrize(
    ("stage", "evaluation_mode", "path_field", "phase"),
    [
        ("guided_planner", None, "field_checkpoint", "field_calibrated"),
        ("stop_calibrated", None, "field_checkpoint", "field_calibrated"),
        ("evaluation", "controller", "dual_value_checkpoint", "stop_calibrated"),
        ("evaluation", "p1_field_forced", "field_checkpoint", "field_calibrated"),
    ],
)
def test_h4v3_direct_value_loader_uses_authoritative_reader_without_legacy_proofs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    evaluation_mode: str | None,
    path_field: str,
    phase: str,
) -> None:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    field = handoff_root / "calibration.pt"
    stop = handoff_root / "stop.pt"
    payload = _write_direct_value(field if phase == "field_calibrated" else stop, phase=phase)
    config = _config(stage, field=field, dual=stop)
    _configure_h4v3_manual_runtime(config, monkeypatch, results_root=tmp_path)
    if evaluation_mode is not None:
        config.cvoi.evaluation_mode = evaluation_mode
    direct_reads: list[tuple[Path, str, str]] = []
    direct_reader = cvoi_value_module.read_cvoi_navsim_e120_direct_value_checkpoint

    assert {
        "_load_configured_cvoi_audit",
        "_read_value_checkpoint_payload",
        "_sha256_file",
        "load_prefix_dual_value_checkpoint",
    }.isdisjoint(vars(cvoi_runtime_module))

    def read_direct(path: str | Path, **kwargs: object) -> dict[str, object]:
        direct_reads.append(
            (
                Path(path),
                str(kwargs["required_phase"]),
                str(kwargs["required_branch_id"]),
            )
        )
        return direct_reader(path, **kwargs)

    monkeypatch.setattr(cvoi_value_module, "read_cvoi_navsim_e120_direct_value_checkpoint", read_direct)
    before_rng = torch.random.get_rng_state().clone()
    loaded = load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))

    assert loaded is not None
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert direct_reads == [
        (
            Path(getattr(config.cvoi, path_field)),
            phase,
            "calibration_full" if phase == "field_calibrated" else "stop_full",
        )
    ]
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert all(torch.equal(loaded.state_dict()[key], value) for key, value in payload["state_dict"].items())


@pytest.mark.parametrize(
    ("lineage", "stage", "path_field", "phase", "branch_id"),
    [
        ("full", "guided_planner", "field_checkpoint", "field_calibrated", "calibration_full"),
        ("no_cf", "guided_planner", "field_checkpoint", "field_calibrated", "calibration_no_cf"),
        (
            "hazard_only",
            "guided_planner",
            "field_checkpoint",
            "field_calibrated",
            "calibration_hazard_only",
        ),
        (
            "quality_only",
            "guided_planner",
            "field_checkpoint",
            "field_calibrated",
            "calibration_quality_only",
        ),
    ],
)
def test_navsim_h4_direct_value_loader_uses_matching_manual_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
    stage: str,
    path_field: str,
    phase: str,
    branch_id: str,
) -> None:
    result_root = tmp_path if lineage == "full" else tmp_path / "ablation" / lineage
    handoff_root = result_root / "handoff"
    handoff_root.mkdir(parents=True)
    field = handoff_root / "calibration.pt"
    stop = handoff_root / "stop.pt"
    checkpoint_path = field if phase == "field_calibrated" else stop
    _write_direct_value(checkpoint_path, phase=phase, lineage=lineage)
    config = _config(stage, field=field, dual=stop)
    _configure_h4v3_manual_runtime(
        config,
        monkeypatch,
        results_root=tmp_path,
        lineage=lineage,
    )
    reads: list[tuple[Path, str, str]] = []
    direct_reader = cvoi_value_module.read_cvoi_navsim_e120_direct_value_checkpoint

    def read_direct(path: str | Path, **kwargs: object) -> dict[str, object]:
        reads.append((Path(path), str(kwargs["required_phase"]), str(kwargs["required_branch_id"])))
        return direct_reader(path, **kwargs)

    monkeypatch.setattr(cvoi_value_module, "read_cvoi_navsim_e120_direct_value_checkpoint", read_direct)

    loaded = load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))

    assert loaded is not None
    assert reads == [(Path(getattr(config.cvoi, path_field)), phase, branch_id)]


def test_navsim_h4_ablation_loader_derives_both_roots_from_configured_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = tmp_path / "custom-full"
    ablation_root = tmp_path / "custom-ablation"
    field = ablation_root / "no_cf/handoff/calibration.pt"
    field.parent.mkdir(parents=True)
    _write_direct_value(field, phase="field_calibrated", lineage="no_cf")
    config = _config("guided_planner", field=field, dual=ablation_root / "no_cf/handoff/stop.pt")
    _configure_h4v3_manual_runtime(
        config,
        monkeypatch,
        results_root=full_root,
        lineage="no_cf",
    )
    config.cvoi.output_checkpoint = str(ablation_root / "no_cf/p1/p1_planner_checkpoint.pt")
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

    loaded = load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))

    assert loaded is not None


def test_navsim_h4_direct_value_loader_rejects_cross_lineage_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_cf_root = tmp_path / "ablation/no_cf/handoff"
    no_cf_root.mkdir(parents=True)
    field = no_cf_root / "calibration.pt"
    _write_direct_value(field, phase="field_calibrated", lineage="full")
    config = _config("guided_planner", field=field, dual=no_cf_root / "stop.pt")
    _configure_h4v3_manual_runtime(
        config,
        monkeypatch,
        results_root=tmp_path,
        lineage="no_cf",
    )

    with pytest.raises(ValueError, match="branch"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


def test_navsim_h4_direct_value_loader_rejects_cross_lineage_path_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "handoff").mkdir()
    full_field = tmp_path / "handoff/calibration.pt"
    config = _config("guided_planner", field=full_field, dual=tmp_path / "handoff/stop.pt")
    _configure_h4v3_manual_runtime(
        config,
        monkeypatch,
        results_root=tmp_path,
        lineage="no_cf",
    )
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("cross-lineage path must fail before checkpoint reading"),
    )

    with pytest.raises(ValueError, match="fixed handoff path"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


def test_h4v3_direct_value_loader_p0_forced_reads_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "handoff").mkdir()
    field = tmp_path / "handoff/missing-calibration.pt"
    stop = tmp_path / "handoff/missing-stop.pt"
    config = _config("evaluation", field=field, dual=stop)
    _configure_h4v3_manual_runtime(config, monkeypatch, results_root=tmp_path)
    config.cvoi.evaluation_mode = "p0_forced"
    config.cvoi.ablation_signature = None
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("p0_forced must not read a Value artifact"),
    )

    assert load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu")) is None


@pytest.mark.parametrize(
    "stage",
    [
        "unguided_planner",
        "field_warmup",
        "field_calibrated",
        "gate_distillation",
    ],
)
def test_h4v3_direct_value_loader_explicit_no_value_stages_read_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    config = _config(
        stage,
        field=tmp_path / "missing-calibration.pt",
        dual=tmp_path / "missing-stop.pt",
    )
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(f"stage={stage!r} must not read a Value artifact"),
    )

    assert load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu")) is None


@pytest.mark.parametrize("stage", ["", "unknown_stage", None])
def test_h4v3_direct_value_loader_rejects_unknown_stage_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: object,
) -> None:
    config = _config(
        "field_warmup",
        field=tmp_path / "missing-calibration.pt",
        dual=tmp_path / "missing-stop.pt",
    )
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.stage = stage
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("unknown stage must fail before direct checkpoint reading"),
    )

    with pytest.raises(ValueError, match="h4v3.*stage"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


@pytest.mark.parametrize("evaluation_mode", ["", "unknown_mode", None])
def test_h4v3_direct_value_loader_rejects_unknown_evaluation_mode_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_mode: object,
) -> None:
    config = _config(
        "evaluation",
        field=tmp_path / "missing-calibration.pt",
        dual=tmp_path / "missing-stop.pt",
    )
    _configure_h4v3_manual_runtime(config, monkeypatch, results_root=tmp_path)
    config.cvoi.evaluation_mode = evaluation_mode
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("unknown mode must fail before direct checkpoint reading"),
    )

    with pytest.raises(ValueError, match="evaluation_mode"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


def test_h4v3_direct_value_loader_rejects_forced_mode_outside_evaluation_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        "guided_planner",
        field=tmp_path / "missing-calibration.pt",
        dual=tmp_path / "missing-stop.pt",
    )
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.evaluation_mode = "p0_forced"
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("non-evaluation mode drift must fail before direct checkpoint reading"),
    )

    with pytest.raises(ValueError, match="evaluation_mode='controller'"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("stage", "path_field", "expected_name"),
    [
        ("guided_planner", "field_checkpoint", "calibration.pt"),
        ("evaluation", "dual_value_checkpoint", "stop.pt"),
    ],
)
def test_h4v3_direct_value_loader_rejects_noncanonical_handoff_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    path_field: str,
    expected_name: str,
) -> None:
    handoff_root = tmp_path / "handoff"
    config = _config(
        stage,
        field=tmp_path / "wrong-calibration.pt",
        dual=tmp_path / "wrong-stop.pt",
    )
    _configure_h4v3_manual_runtime(config, monkeypatch, results_root=tmp_path)
    monkeypatch.setattr(
        cvoi_value_module,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("wrong fixed handoff path must fail before direct checkpoint reading"),
    )

    with pytest.raises(ValueError, match=rf"fixed handoff path.*{expected_name}"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))

    assert getattr(config.cvoi, path_field) != str(handoff_root / expected_name)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(
                phase="field_warmup",
                branch_id="field_full",
                parents={"unguided_planner": {"stage": "p0", "branch_id": "p0_uniform"}},
            ),
            "required phase",
        ),
        (lambda payload: payload.update(branch_id="stop_full"), "branch"),
    ],
)
def test_h4v3_direct_value_loader_rejects_phase_or_branch_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    field = handoff_root / "calibration.pt"
    stop = handoff_root / "unused-stop.pt"
    payload = _write_direct_value(field, phase="field_calibrated")
    mutation(payload)
    torch.save(payload, field)
    config = _config("guided_planner", field=field, dual=stop)
    _configure_h4v3_manual_runtime(config, monkeypatch, results_root=tmp_path)

    with pytest.raises(ValueError, match=message):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


def test_h4v3_direct_value_loader_rejects_embed_dim_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    field = handoff_root / "calibration.pt"
    stop = handoff_root / "unused-stop.pt"
    _write_direct_value(field, phase="field_calibrated")
    config = _config("guided_planner", field=field, dual=stop)
    _configure_h4v3_manual_runtime(config, monkeypatch, results_root=tmp_path)

    with pytest.raises(ValueError, match="embed_dim"):
        load_cvoi_dual_value_model(config, embed_dim=8, device=torch.device("cpu"))

    target = tmp_path / "calibration-target.pt"
    field.rename(target)
    field.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))


def test_unguided_runtime_preserves_exact_future_tensor(tmp_path: Path) -> None:
    config = _config("unguided_planner", field=tmp_path / "unused", dual=tmp_path / "unused2")
    observed = torch.randn(1, 4, 4)
    future = torch.randn(1, 6, 4)

    output, diagnostics = apply_cvoi_planner_guidance(
        observed,
        future,
        None,
        tokens_per_frame=2,
        config=config,
    )

    assert output is future
    assert diagnostics["guidance_steps"] == 0.0


def test_guided_runtime_uses_two_steps_and_h0_skips(tmp_path: Path) -> None:
    field = tmp_path / "field.pt"
    dual = tmp_path / "dual.pt"
    config = _config("guided_planner", field=field, dual=dual)
    model = PrefixDualValueModel(embed_dim=4, hidden_dim=6).eval()
    observed = torch.randn(1, 4, 4)

    guided, diagnostics = apply_cvoi_planner_guidance(
        observed,
        torch.randn(1, 6, 4),
        model,
        tokens_per_frame=2,
        config=config,
    )
    empty, h0 = apply_cvoi_planner_guidance(
        observed,
        torch.empty(1, 0, 4),
        model,
        tokens_per_frame=2,
        config=config,
    )

    assert guided.shape == (1, 6, 4)
    assert diagnostics["guidance_steps"] == 2.0
    assert empty.shape == (1, 0, 4)
    assert h0["guidance_steps"] == 0.0
    assert h0["guidance_skipped_h0"] == 1.0


def test_guided_runtime_caps_six_pose_world_model_rollout_at_controller_h4(tmp_path: Path) -> None:
    field = tmp_path / "field.pt"
    dual = tmp_path / "dual.pt"
    config = _config("guided_planner", field=field, dual=dual)
    model = PrefixDualValueModel(embed_dim=4, hidden_dim=6).eval()

    guided, diagnostics = apply_cvoi_planner_guidance(
        torch.randn(1, 8, 4),
        torch.randn(1, 12, 4),
        model,
        tokens_per_frame=2,
        config=config,
    )

    assert guided.shape == (1, 8, 4)
    assert diagnostics["guidance_steps"] == 2.0


def test_planner_entry_rejects_offline_or_evaluation_stages(tmp_path: Path) -> None:
    field = tmp_path / "field.pt"
    dual = tmp_path / "dual.pt"
    require_cvoi_planner_stage(_config("guided_planner", field=field, dual=dual))
    with pytest.raises(ValueError, match="cannot execute"):
        require_cvoi_planner_stage(_config("evaluation", field=field, dual=dual))


def test_cvoi_output_checkpoint_is_the_actual_latest_and_auto_resume_path(tmp_path: Path) -> None:
    config = _config("unguided_planner", field=tmp_path / "field.pt", dual=tmp_path / "dual.pt")
    latest, resume = resolve_cvoi_planner_checkpoint_paths(
        config,
        legacy_latest_path=str(tmp_path / "legacy-latest.pt"),
        legacy_resume_path=None,
    )
    assert latest == config.cvoi.output_checkpoint
    assert resume is None

    Path(latest).write_bytes(b"checkpoint")
    _, resume = resolve_cvoi_planner_checkpoint_paths(
        config,
        legacy_latest_path=str(tmp_path / "legacy-latest.pt"),
        legacy_resume_path=None,
    )
    assert resume == latest


def test_navsim_h4_gate_provenance_comes_only_from_the_embedded_manual_oracle_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("evaluation", field=tmp_path / "field.pt", dual=tmp_path / "dual.pt")
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.ablation_signature = SimpleNamespace(gate_feature_mode="without_field")
    oracle_path = tmp_path / "oracle.sqlite3"
    config.cvoi.oracle_path = str(oracle_path)
    oracle_path.write_bytes(b"official-navtrain-oracle")
    oracle = object()
    expected = {
        "oracle_protocol": "epdms_v2_one_stage_navtrain_gate_label_v1",
        "oracle_schema": "cvoi_navsim_v2_navtrain_gate_oracle_curve_v2",
        "gate_pipeline": "offline_navsim_e120_official_epdms_gate_distillation_v1",
        "gate_feature_mode": "without_field",
    }

    monkeypatch.setattr(
        cvoi_runtime_module,
        "open_embedded_oracle_store_v2",
        lambda path: nullcontext(oracle),
        raising=False,
    )
    assert not hasattr(cvoi_runtime_module, "open_oracle_store")

    def build_expected(path: Path, artifact: object, *, gate_feature_mode: str) -> dict[str, str]:
        assert path == oracle_path
        assert artifact is oracle
        assert gate_feature_mode == "without_field"
        return expected

    monkeypatch.setattr(cvoi_runtime_module, "build_navtrain_gate_checkpoint_provenance", build_expected)
    assert "build_cvoi_oracle_provenance" not in vars(cvoi_runtime_module)

    assert build_cvoi_gate_provenance(config) == expected


@pytest.mark.parametrize("evaluation_mode", ["p0_forced", "p1_field_forced"])
def test_fixed_only_evaluation_never_loads_a_gate(evaluation_mode: str) -> None:
    config = SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            stage="evaluation",
            evaluation_mode=evaluation_mode,
            gate_checkpoint=None,
        )
    )

    assert load_cvoi_gate_for_evaluation(config, device=torch.device("cpu")) is None


def test_retained_runtime_has_no_generic_planner_runtime_dependency() -> None:
    source = Path(cvoi_runtime_module.__file__).read_text(encoding="utf-8")

    assert "training.cvoi_planner_runtime" not in source
    assert "prepare_formal_v2_planner_training" not in source
    assert "validate_formal_v2_planner_resume_for_config" not in source
