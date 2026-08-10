"""Production frozen-model runtime for the NavSim CVoI offline adapter."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel, PrefixValueOutput
from app.vjepa_cowa_world_model.training import navsim_cvoi_model_runtime as runtime_module
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    build_formal_v2_navsim_e120_direct_checkpoint,
    build_formal_v2_navsim_e120_direct_lineage,
    read_formal_v2_navsim_e120_direct_checkpoint,
    resolve_formal_v2_navsim_e120_selected_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import CvoiWorld4DriveRuntimeBinding
from app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime import (
    NavSimCvoiPlannerContext,
    NavSimCvoiProductionModelRuntime,
    create_navsim_cvoi_model_runtime,
    create_navsim_cvoi_world4drive_runtime,
)
from app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter import NavSimCvoiModelBatch
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import PredictorTimelineInputs


class _FrozenModule(torch.nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))


class _TwoStateModule(torch.nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))
        self.offset = torch.nn.Parameter(torch.tensor(float(value)))


class _VectorModule(torch.nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2,), float(value)))


class _Planner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.use_anchor_frame = True
        self.traj_dim = 6
        self.num_modes = 2
        self.num_samples = 2
        self.num_poses = 6
        self.calls: list[dict[str, object]] = []

    def forward(self, z_future, status_feature, **kwargs):
        self.calls.append(
            {
                "z_future": z_future.detach().clone(),
                "status_feature": status_feature.detach().clone(),
                "kwargs": dict(kwargs),
            }
        )
        batch_size = z_future.shape[0]
        noise = kwargs["inference_noise"][..., :3]
        return {
            "trajectories": noise + self.bias,
            "confidences": torch.tensor([[0.2, 0.8]], device=z_future.device).expand(batch_size, -1),
        }


class _DualValue(torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.future_frame_counts: list[int] = []

    def forward(self, z_observed, z_future, *, tokens_per_frame):
        del z_observed
        if z_future.ndim == 3:
            future = z_future.reshape(z_future.shape[0], -1, tokens_per_frame, z_future.shape[-1])
        else:
            future = z_future
        frame_count = int(future.shape[1])
        self.future_frame_counts.append(frame_count)
        field = future.mean(dim=(2, 3)) * self.scale
        stop_h0 = field.new_zeros((field.shape[0], 1))
        return PrefixValueOutput(field_values=field, stop_values=torch.cat([stop_h0, field], dim=1))


def test_incremental_predictor_path_has_no_per_step_host_scalar_synchronization() -> None:
    source = "\n".join(
        (
            inspect.getsource(NavSimCvoiProductionModelRuntime._validate_online_prefix),
            inspect.getsource(NavSimCvoiProductionModelRuntime.rollout_online_step),
        )
    )
    assert ".item(" not in source


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        meta=SimpleNamespace(
            ae_checkpoint=None,
            load_encoder=True,
            load_predictor=False,
            load_planner=False,
            load_seg=False,
            pretrain_checkpoint=None,
            pretrain_checkpoint_full=None,
            predictor_checkpoint=None,
        ),
        model=SimpleNamespace(backbone="vjepa_img_encoder"),
        cvoi=SimpleNamespace(
            enabled=True,
            stage="evaluation",
            max_horizon=3,
            unguided_planner_checkpoint=str(tmp_path / "p0.pt"),
            guided_planner_checkpoint=str(tmp_path / "p1.pt"),
        ),
        data=SimpleNamespace(num_target_frames=10, fps=2),
        train=SimpleNamespace(
            num_observed_frames=4,
            predictor_type="ac_transformer",
            predictor_inference_consistent=True,
            predictor_no_aux_input=False,
            state_dim=8,
        ),
        planner=SimpleNamespace(
            use_planner=True,
            planner_type="diffusion",
            z_ar_mode="full",
            use_z_context=False,
            use_observed_tokens=True,
            use_action_history_for_planner=True,
            action_history_dim=3,
            diff_dt=0.5,
            status_dim=8,
            use_drive_command=False,
        ),
        value_guidance=SimpleNamespace(
            enabled=True,
            steps=2,
            objective="last",
            step_size=0.05,
            max_delta_norm=0.25,
            detach_output=True,
        ),
        token_ae=SimpleNamespace(enabled=True, num_latent_tokens=128),
        multiview=SimpleNamespace(enabled=False),
        mixed_precision=False,
        dtype=torch.float32,
    )


def _world4drive_binding(tmp_path: Path, *, lineage: str = "real_only_value") -> CvoiWorld4DriveRuntimeBinding:
    p0 = lineage == "p0_controller"
    return CvoiWorld4DriveRuntimeBinding(
        protocol_version="world4drive_evaluation_v1",
        lineage=lineage,
        stage="evaluation",
        max_horizon=3,
        tokens_per_frame=128,
        compute_costs=(0.0, 1.0, 2.0, 3.0),
        controller_batch_size=1,
        controller_lineage="p0_controller" if p0 else "value_guided",
        guidance_steps=2,
        guidance_objective="last",
        timestep_sec=0.5,
        world_model_checkpoint=str(tmp_path / "world.pt"),
        token_ae_checkpoint=str(tmp_path / "token-ae.pt"),
        unguided_planner_checkpoint=str(tmp_path / "p0.pt"),
        field_checkpoint=None if p0 else str(tmp_path / f"{lineage}-field.pt"),
        guided_planner_checkpoint=None if p0 else str(tmp_path / f"{lineage}-p1.pt"),
        dual_value_checkpoint=str(tmp_path / f"{lineage}-stop.pt"),
        oracle_path=str(tmp_path / f"{lineage}-oracle.jsonl"),
        gate_checkpoint=str(tmp_path / f"{lineage}-gate.pt"),
        checkpoint_audit_manifest_path=str(tmp_path / "checkpoint-audit.json"),
    )


def _retained_world4drive_planner_signature(
    binding: CvoiWorld4DriveRuntimeBinding,
    *,
    p0: bool,
) -> dict[str, object]:
    return {
        "schema": "cvoi_dual_value_v1",
        "stage": "unguided_planner" if p0 else "guided_planner",
        "guidance_steps": 2,
        "guidance_objective": "last",
        "guidance_step_size": 0.05,
        "guidance_max_delta_norm": 0.25,
        "guidance_detach_output": True,
        "audit_signature": {"audit": "checkpoint"},
        "predictor_type": "ac_transformer",
        "runtime_normalize_reps": False,
        "tokens_per_frame": 128,
        "num_observed_frames": 4,
        "num_target_frames": 10,
        "timestep_sec": 0.5,
        "multiview_signature": {"enabled": False},
        "planner_signature": {"schema": "planner-v1"},
        "world_execution_signature": {"schema": "world-v1"},
        "execution_dtype_signature": {"schema": "dtype-v1"},
        "inference_rng_signature": {"schema": "rng-v1"},
        "world_model_sha256": "a" * 64,
        "token_ae_sha256": "a" * 64,
        "parent_planner_sha256": "b" * 64 if p0 else "a" * 64,
        "dual_value_sha256": None if p0 else "a" * 64,
        "gate_sha256": None,
        "ablation_signature": runtime_module._expected_world4drive_planner_ablation(binding, p0=p0),
        "p0_protocol": "fixed_final_epoch_v1",
        "p0_prefix_distribution": {str(horizon): 0.25 for horizon in range(4)},
    }


def _save_planner(path: str, bias: float) -> None:
    torch.save({"planner": {"bias": torch.tensor(bias)}, "cvoi_runtime_signature": {}}, path)


def _navsim_ablation_signature(*, branch_id: str, cf_field_supervision: str = "hazard_quality") -> dict[str, object]:
    return {
        "schema": "cvoi_formal_v2_navsim_e120_ablation_v1",
        "protocol_version": "formal_v2_navsim_e120_h4_v3",
        "experiment_role": "main" if branch_id in {"p0_uniform", "full", "p1_full"} else "ablation",
        "branch_id": branch_id,
        "shared_cohort_id": "navsim_e120_s239_stride4",
        "initialization_mode": "full_state_warmstart",
        "cf_field_supervision": cf_field_supervision,
        "field_calibration_mode": "local_geometry",
        "p0_prefix_mode": "uniform",
        "gate_feature_mode": "full",
        "train_seed": 239,
        "evaluation_seed": 239,
        "training_stride": 4,
    }


def _set_parameter(module: torch.nn.Module, name: str, value: float) -> None:
    with torch.no_grad():
        getattr(module, name).fill_(value)


def _direct_policy_modules(
    *,
    encoder_value: float,
    predictor_value: float,
    planner_value: float,
    predictor: torch.nn.Module | None = None,
) -> dict[str, torch.nn.Module]:
    encoder = _FrozenModule(encoder_value)
    predictor = _FrozenModule(predictor_value) if predictor is None else predictor
    planner = _Planner()
    _set_parameter(planner, "bias", planner_value)
    return {
        "encoder": encoder,
        "predictor": predictor,
        "planner": planner,
    }


def _direct_policy_payload(
    *,
    stage: str,
    encoder_value: float = 1.0,
    predictor_value: float = 3.0,
    planner_value: float = 7.0,
    epoch: int | None = None,
    predictor: torch.nn.Module | None = None,
) -> dict[str, object]:
    is_p0 = stage == "p0"
    return build_formal_v2_navsim_e120_direct_checkpoint(
        modules=_direct_policy_modules(
            encoder_value=encoder_value,
            predictor_value=predictor_value,
            planner_value=planner_value,
            predictor=predictor,
        ),
        optimizer={"state": 1},
        scaler={},
        scheduler={"state": 2},
        wd_scheduler={"state": 3},
        run_id=f"{stage}_runtime_test",
        stage=stage,
        epoch=(35 if is_p0 else 25) if epoch is None else epoch,
        training_stop_epoch=50 if is_p0 else 80,
        schedule_epochs=50 if is_p0 else 80,
        selection_checkpoint_epochs=(
            FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS if is_p0 else FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
        ),
        cumulative_horizon_histogram={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        lineage=build_formal_v2_navsim_e120_direct_lineage(
            stage=stage,
            branch_id="p0_uniform" if is_p0 else "p1_full",
        ),
    )


def _write_direct_policy(
    root: Path,
    *,
    stage: str,
    payload: dict[str, object] | None = None,
    symlink: bool = False,
) -> Path:
    handoff = root / "handoff" / f"{stage}_selected.pt"
    handoff.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = handoff
    if symlink:
        checkpoint = root / stage / "run" / f"{stage}_selected.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_direct_policy_payload(stage=stage) if payload is None else payload, checkpoint)
    if symlink:
        handoff.symlink_to(checkpoint)
    return handoff


def _model_batch(*, metadata=None) -> NavSimCvoiModelBatch:
    return NavSimCvoiModelBatch(
        context_frames=torch.ones(1, 3, 4, 8, 8),
        actions=torch.ones(1, 3, 3),
        states=torch.ones(1, 4, 7),
        extrinsics=torch.ones(1, 4, 7),
        driving_command=torch.ones(1, 4, 4),
        ego_dynamics=torch.ones(1, 4, 4),
        proposal_context_frames=None,
        metadata={} if metadata is None else metadata,
    )


def _build_runtime(tmp_path: Path, *, dual_value=None, configure=None):
    config = _config(tmp_path)
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.max_horizon = 4
    config.cvoi.controller_lineage = "value_guided"
    config.cvoi.ablation_signature = _navsim_ablation_signature(branch_id="full")
    config.cvoi.full_state_warmstart = SimpleNamespace(
        source_checkpoint=SimpleNamespace(path="/locked/e120.pt"),
        source_params_pretrain=SimpleNamespace(path="/locked/params-pretrain.yaml"),
    )
    if configure is not None:
        configure(config)
    results_root = tmp_path / "cvoi_manual_full"
    config.cvoi.unguided_planner_checkpoint = str(
        _write_direct_policy(
            results_root,
            stage="p0",
            payload=_direct_policy_payload(
                stage="p0",
                encoder_value=1.0,
                predictor_value=3.0,
                planner_value=1.0,
            ),
        )
    )
    if config.cvoi.guided_planner_checkpoint is not None:
        config.cvoi.guided_planner_checkpoint = str(
            _write_direct_policy(
                results_root,
                stage="p1",
                payload=_direct_policy_payload(
                    stage="p1",
                    encoder_value=1.0,
                    predictor_value=5.0,
                    planner_value=2.0,
                ),
            )
        )
    encoder = _FrozenModule()
    predictor = _FrozenModule()
    predictor_p1 = _FrozenModule()
    token_ae = _FrozenModule()
    planners = [_Planner(), _Planner()]
    dual_value = dual_value or _DualValue(embed_dim=8)
    handoff_calls: list[str] = []

    def apply_warmstart_direct(checkpoint_path, params_pretrain_path, modules):
        assert checkpoint_path == "/locked/e120.pt"
        assert params_pretrain_path == "/locked/params-pretrain.yaml"
        _set_parameter(modules["encoder"], "weight", 1.0)
        _set_parameter(modules["predictor"], "weight", 2.0)
        _set_parameter(modules["planner"], "bias", 4.0)

    def read_direct(path):
        payload = read_formal_v2_navsim_e120_direct_checkpoint(path)
        handoff_calls.append(str(payload["stage"]))
        return payload

    patches = (
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.init_encoder",
            side_effect=AssertionError("retained runtime must not initialize the generic encoder path"),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.init_encoder_for_full_state_warmstart",
            return_value=(encoder, _FrozenModule()),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.get_encoder_embed_dim",
            return_value=8,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.resolve_main_predictor_runtime_overrides",
            return_value=(None, None),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.init_predictor_runtime_with_token_ae",
            side_effect=[(predictor, token_ae, 128, False), (predictor_p1, None, 128, False)],
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.init_planner",
            side_effect=planners,
        ),
        patch.object(
            runtime_module,
            "apply_formal_v2_full_state_warmstart_direct",
            side_effect=apply_warmstart_direct,
        ),
        patch.object(
            runtime_module,
            "read_formal_v2_navsim_e120_direct_checkpoint",
            side_effect=read_direct,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.load_cvoi_dual_value_model",
            return_value=dual_value,
        ),
    )
    for active_patch in patches:
        active_patch.start()
    try:
        runtime = create_navsim_cvoi_model_runtime(
            config=config,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()
    return config, runtime, encoder, predictor, token_ae, planners, dual_value, handoff_calls


def test_factory_rejects_cpu_without_explicit_test_hook(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        create_navsim_cvoi_model_runtime(config=config, device=torch.device("cpu"))


def test_world4drive_factory_rejects_cpu_and_enabled_training_cvoi_before_model_init(tmp_path: Path) -> None:
    binding = _world4drive_binding(tmp_path)
    base = _config(tmp_path)
    base.cvoi.enabled = False
    with pytest.raises(RuntimeError, match="requires CUDA"):
        create_navsim_cvoi_world4drive_runtime(config=base, binding=binding, device=torch.device("cpu"))

    base.cvoi.enabled = True
    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        pytest.raises(ValueError, match="must not enable.*training CVoI"),
    ):
        create_navsim_cvoi_world4drive_runtime(
            config=base,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


def test_world4drive_private_adapter_has_no_generic_ablation_parser_dependency() -> None:
    source = Path(runtime_module.__file__).read_text(encoding="utf-8")

    assert "training.configs.cvoi_ablation" not in source
    assert "parse_cvoi_ablation_signature" not in source
    assert "training.cvoi_planner_runtime" not in source
    assert "validate_formal_v2_planner_checkpoint" not in source
    assert "validate_formal_v2_planner_lineage_for_config" not in source


def test_world4drive_factory_rejects_token_ae_path_drift_before_model_init(tmp_path: Path) -> None:
    binding = _world4drive_binding(tmp_path)
    base = _config(tmp_path)
    base.cvoi.enabled = False
    base.meta.ae_checkpoint = str(tmp_path / "other-token-ae.pt")

    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        pytest.raises(ValueError, match="ae_checkpoint.*bound TokenAE"),
    ):
        create_navsim_cvoi_world4drive_runtime(
            config=base,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", "legacy_v1"),
        ("stage", "guided_planner"),
        ("max_horizon", 4),
        ("tokens_per_frame", 256),
        ("compute_costs", (0.0, 1.0, 2.0)),
        ("controller_batch_size", 2),
        ("guidance_steps", 1),
        ("guidance_objective", "mean"),
        ("timestep_sec", 0.25),
    ],
)
def test_world4drive_factory_rejects_binding_semantic_drift_before_model_init(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    binding = _world4drive_binding(tmp_path)
    binding = replace(binding, **{field: value})
    base = _config(tmp_path)
    base.cvoi.enabled = False
    base.meta.ae_checkpoint = binding.token_ae_checkpoint

    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        pytest.raises(ValueError, match="binding semantics mismatch"),
    ):
        create_navsim_cvoi_world4drive_runtime(
            config=base,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_size", 0.1),
        ("max_delta_norm", 0.5),
        ("detach_output", False),
    ],
)
def test_world4drive_factory_rejects_guidance_policy_drift_before_model_init(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    binding = _world4drive_binding(tmp_path)
    base = _config(tmp_path)
    base.cvoi.enabled = False
    base.meta.ae_checkpoint = binding.token_ae_checkpoint
    setattr(base.value_guidance, field, value)

    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        pytest.raises(ValueError, match="Guidance semantics mismatch"),
    ):
        create_navsim_cvoi_world4drive_runtime(
            config=base,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "backbone", "vjepa2"),
        ("meta", "load_encoder", False),
        ("meta", "load_predictor", True),
        ("meta", "load_planner", True),
        ("meta", "load_seg", True),
    ],
)
def test_world4drive_factory_rejects_base_model_load_policy_drift_before_init(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    binding = _world4drive_binding(tmp_path)
    base = _config(tmp_path)
    base.cvoi.enabled = False
    base.meta.ae_checkpoint = binding.token_ae_checkpoint
    setattr(getattr(base, section), field, value)

    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        pytest.raises(ValueError, match="backbone|load policy"),
    ):
        create_navsim_cvoi_world4drive_runtime(
            config=base,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "runtime_normalize_reps",
        "multiview_signature",
        "planner_signature",
        "world_execution_signature",
        "execution_dtype_signature",
        "inference_rng_signature",
    ],
)
def test_world4drive_planner_rejects_each_retained_runtime_signature_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    binding = _world4drive_binding(tmp_path, lineage="p0_controller")
    config = _config(tmp_path)
    signature = _retained_world4drive_planner_signature(binding, p0=True)
    signature[field] = {"mutated": field} if field != "runtime_normalize_reps" else True
    payload = {"planner": {"bias": torch.tensor(0.0)}, "epoch": 20, "cvoi_runtime_signature": signature}
    audit = SimpleNamespace(to_dict=lambda: {"audit": "checkpoint"})

    with (
        patch.object(runtime_module, "load_cvoi_audit_manifest", return_value=audit),
        patch.object(runtime_module, "_world4drive_sha256", return_value="a" * 64),
        patch.object(runtime_module, "resolve_predictor_runtime_normalize_reps", return_value=False),
        patch.object(runtime_module, "_cvoi_multiview_signature", return_value={"enabled": False}),
        patch.object(runtime_module, "_cvoi_planner_signature", return_value={"schema": "planner-v1"}),
        patch.object(runtime_module, "_cvoi_world_execution_signature", return_value={"schema": "world-v1"}),
        patch.object(runtime_module, "cvoi_execution_dtype_signature", return_value={"schema": "dtype-v1"}),
        patch.object(runtime_module, "cvoi_inference_rng_signature", return_value={"schema": "rng-v1"}),
        pytest.raises(ValueError, match="Planner runtime lineage mismatch"),
    ):
        runtime_module._validate_world4drive_planner_payload(payload, binding=binding, p0=True, config=config)


@pytest.mark.parametrize("field", ["ablation_signature", "p0_prefix_distribution"])
def test_world4drive_planner_rejects_nested_metadata_type_drift(tmp_path: Path, field: str) -> None:
    binding = _world4drive_binding(tmp_path, lineage="p0_controller")
    config = _config(tmp_path)
    signature = _retained_world4drive_planner_signature(binding, p0=True)
    if field == "ablation_signature":
        signature[field]["train_seed"] = 239.0
        message = "ablation"
    else:
        signature[field]["0"] = 0.25 + 0j
        message = "prefix distribution"
    payload = {"planner": {"bias": torch.tensor(0.0)}, "epoch": 20, "cvoi_runtime_signature": signature}
    audit = SimpleNamespace(to_dict=lambda: {"audit": "checkpoint"})

    with (
        patch.object(runtime_module, "load_cvoi_audit_manifest", return_value=audit),
        patch.object(runtime_module, "_world4drive_sha256", return_value="a" * 64),
        patch.object(runtime_module, "resolve_predictor_runtime_normalize_reps", return_value=False),
        patch.object(runtime_module, "_cvoi_multiview_signature", return_value={"enabled": False}),
        patch.object(runtime_module, "_cvoi_planner_signature", return_value={"schema": "planner-v1"}),
        patch.object(runtime_module, "_cvoi_world_execution_signature", return_value={"schema": "world-v1"}),
        patch.object(runtime_module, "cvoi_execution_dtype_signature", return_value={"schema": "dtype-v1"}),
        patch.object(runtime_module, "cvoi_inference_rng_signature", return_value={"schema": "rng-v1"}),
        pytest.raises(ValueError, match=message),
    ):
        runtime_module._validate_world4drive_planner_payload(payload, binding=binding, p0=True, config=config)


def test_world4drive_restore_rejects_unexpected_planner_metadata_before_any_state_load(tmp_path: Path) -> None:
    binding = _world4drive_binding(tmp_path, lineage="p0_controller")
    config = _config(tmp_path)
    signature = _retained_world4drive_planner_signature(binding, p0=True)
    signature["unexpected"] = "forbidden"
    checkpoints = {
        "world model": {
            "encoder": {"weight": torch.tensor(1.0)},
            "predictor": {"weight": torch.tensor(2.0)},
        },
        "TokenAE": {"token_ae": {"weight": torch.tensor(3.0)}},
        "P0 Planner": {
            "planner": {"bias": torch.tensor(4.0)},
            "epoch": 20,
            "cvoi_runtime_signature": signature,
        },
    }

    def read_checkpoint(_path, *, name):
        return checkpoints[name]

    with (
        patch.object(runtime_module, "_read_world4drive_checkpoint", side_effect=read_checkpoint),
        patch.object(
            runtime_module,
            "validate_cvoi_planner_lineage",
            side_effect=lambda payload, **_kwargs: payload["cvoi_runtime_signature"],
        ),
        patch.object(
            torch.nn.Module,
            "load_state_dict",
            side_effect=AssertionError("planner metadata must reject before any state mutation"),
        ) as load_state,
        pytest.raises(ValueError, match="signature fields mismatch.*unexpected"),
    ):
        runtime_module._restore_world4drive_read_only_runtime(
            config=config,
            binding=binding,
            encoder=_FrozenModule(),
            predictor=_FrozenModule(),
            token_ae=_FrozenModule(),
            planner_p0=_Planner(),
            planner_p1=None,
            multiview_fusion=None,
            embed_dim=8,
            device=torch.device("cpu"),
        )

    load_state.assert_not_called()


@pytest.mark.parametrize("lineage", ["p0_controller", "real_only_value", "real_cf_value"])
def test_world4drive_factory_uses_private_legacy_adapter_and_read_only_restore_only(
    tmp_path: Path,
    lineage: str,
) -> None:
    binding = _world4drive_binding(tmp_path, lineage=lineage)
    config = _config(tmp_path)
    config.cvoi.enabled = False
    config.meta = SimpleNamespace(
        ae_checkpoint=binding.token_ae_checkpoint,
        load_encoder=True,
        load_predictor=False,
        load_planner=False,
        load_seg=False,
        pretrain_checkpoint="public-pretrain.pt",
        pretrain_checkpoint_full="public-full-pretrain.pt",
        predictor_checkpoint="public-predictor.pt",
    )
    encoder = _FrozenModule()
    predictor = _FrozenModule()
    token_ae = _FrozenModule()
    planner_p0 = _Planner()
    planner_p1 = None if lineage == "p0_controller" else _Planner()
    planners = [planner_p0] if planner_p1 is None else [planner_p0, planner_p1]
    dual_value = _DualValue(embed_dim=8)
    restore_calls = []

    def restore_read_only(**kwargs):
        restore_calls.append(kwargs)
        assert kwargs["binding"] is binding
        assert kwargs["config"].cvoi.protocol_version == "legacy_v1"
        assert kwargs["config"].cvoi.stage == "evaluation"
        assert kwargs["config"].cvoi.controller_lineage == binding.controller_lineage
        assert kwargs["config"].meta.pretrain_checkpoint is None
        assert kwargs["config"].meta.pretrain_checkpoint_full is None
        assert kwargs["config"].meta.predictor_checkpoint is None
        return dual_value

    with (
        patch.object(runtime_module, "init_encoder", return_value=(encoder, _FrozenModule())),
        patch.object(runtime_module, "get_encoder_embed_dim", return_value=8),
        patch.object(runtime_module, "resolve_main_predictor_runtime_overrides", return_value=(None, None)),
        patch.object(
            runtime_module,
            "init_predictor_runtime_with_token_ae",
            return_value=(predictor, token_ae, 128, False),
        ) as init_predictor,
        patch.object(runtime_module, "resolve_main_encoder_raw_tokens_per_frame", return_value=256),
        patch.object(runtime_module, "_init_multiview_fusion", return_value=None),
        patch.object(runtime_module, "init_planner", side_effect=planners),
        patch.object(runtime_module, "validate_cvoi_world4drive_gate", return_value=None) as validate_gate,
        patch.object(runtime_module, "_restore_world4drive_read_only_runtime", side_effect=restore_read_only),
    ):
        runtime = create_navsim_cvoi_world4drive_runtime(
            config=config,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )

    assert config.cvoi.enabled is False
    assert config.meta.pretrain_checkpoint_full == "public-full-pretrain.pt"
    assert runtime.config.cvoi.protocol_version == "legacy_v1"
    assert runtime.config._world4drive_runtime_binding is binding
    assert runtime.max_horizon == 3
    assert runtime.tokens_per_frame == 128
    assert runtime.planner_p1 is planner_p1
    assert runtime.dual_value_model is dual_value
    assert len(restore_calls) == 1
    validate_gate.assert_called_once_with(binding)
    assert init_predictor.call_args.kwargs["_checkpoint_weights_only"] is True
    assert init_predictor.call_args.kwargs["_defer_token_ae_state_load"] is True


def test_world4drive_factory_rejects_bad_gate_before_initializing_models(tmp_path: Path) -> None:
    binding = _world4drive_binding(tmp_path)
    config = _config(tmp_path)
    config.cvoi.enabled = False
    config.meta.ae_checkpoint = binding.token_ae_checkpoint

    with (
        patch.object(
            runtime_module,
            "validate_cvoi_world4drive_gate",
            side_effect=ValueError("invalid direct Gate checkpoint"),
        ) as validate_gate,
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        patch.object(
            torch.nn.Module,
            "load_state_dict",
            side_effect=AssertionError("bad Gate must reject before any main-module state load"),
        ) as load_state,
        pytest.raises(ValueError, match="invalid direct Gate checkpoint"),
    ):
        create_navsim_cvoi_world4drive_runtime(
            config=config,
            binding=binding,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )

    validate_gate.assert_called_once_with(binding)
    load_state.assert_not_called()


def test_navsim_e120_gate_factory_rejects_before_initializing_any_model_pair(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.cvoi.stage = "gate_distillation"
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"

    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("Gate must not initialize models")),
        pytest.raises(ValueError, match="Gate.*Oracle.*must not load"),
    ):
        create_navsim_cvoi_model_runtime(
            config=config,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


def test_factory_strictly_loads_and_freezes_token_ae128_and_both_planners(tmp_path: Path) -> None:
    config, runtime, encoder, predictor, token_ae, planners, dual_value, handoff_calls = _build_runtime(tmp_path)

    assert isinstance(runtime, NavSimCvoiProductionModelRuntime)
    assert runtime.embed_dim == 8
    assert runtime.tokens_per_frame == 128
    assert runtime.max_horizon == 4
    assert runtime.num_planner_poses == 6
    assert handoff_calls == ["p0", "p1"]
    assert planners[0].bias.item() == pytest.approx(1.0)
    assert planners[1].bias.item() == pytest.approx(2.0)
    for module in (encoder, predictor, token_ae, planners[0], planners[1], dual_value):
        assert module.training is False
        assert all(parameter.requires_grad is False for parameter in module.parameters())
    assert config.data.num_target_frames - config.train.num_observed_frames == 6


def test_navsim_e120_runtime_exposes_no_generic_planner_only_loader() -> None:
    assert not hasattr(runtime_module, "_restore_planner_checkpoint")


def test_navsim_e120_model_runtime_exposes_no_automatic_dag_proof_surface() -> None:
    forbidden = {
        "CVOI_NAVSIM_E120_H4V3_MANUAL_RESULTS_ROOT",
        "_restore_navsim_e120_policy_checkpoint",
        "_selected_uniform_p0_receipt_identity",
        "formal_v2_navsim_epdms_selection_receipt_sha256",
        "read_formal_v2_navsim_e120_checkpoint",
        "read_formal_v2_navsim_epdms_selection_receipt",
        "role_state_sha256",
    }
    forbidden_methods = {"evaluate_oracle_horizon", "evaluate_p0_oracle_horizon"}

    assert forbidden.isdisjoint(vars(runtime_module))
    assert forbidden_methods.isdisjoint(vars(NavSimCvoiProductionModelRuntime))


@pytest.mark.parametrize(
    ("stage", "controller_lineage", "expects_p1"),
    (
        ("field_warmup", "value_guided", False),
        ("field_calibrated", "value_guided", False),
        ("stop_calibrated", "value_guided", True),
        ("evaluation", "value_guided", True),
    ),
)
def test_navsim_e120_factory_uses_only_direct_warmstart_and_fixed_policy_handoffs(
    tmp_path: Path,
    stage: str,
    controller_lineage: str,
    expects_p1: bool,
) -> None:
    results_root = tmp_path / "cvoi_manual_full"
    p0_path = _write_direct_policy(
        results_root,
        stage="p0",
        payload=_direct_policy_payload(
            stage="p0",
            encoder_value=1.0,
            predictor_value=3.0,
            planner_value=7.0,
        ),
    )
    p1_path = _write_direct_policy(
        results_root,
        stage="p1",
        payload=_direct_policy_payload(
            stage="p1",
            encoder_value=1.0,
            predictor_value=5.0,
            planner_value=9.0,
        ),
    )
    config = _config(tmp_path)
    config.cvoi.stage = stage
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.max_horizon = 4
    config.data.num_target_frames = 12
    config.cvoi.controller_lineage = controller_lineage
    config.cvoi.ablation_signature = _navsim_ablation_signature(branch_id="full")
    config.cvoi.unguided_planner_checkpoint = str(p0_path)
    config.cvoi.guided_planner_checkpoint = str(p1_path)
    config.cvoi.full_state_warmstart = SimpleNamespace(
        source_checkpoint=SimpleNamespace(path="/locked/e120.pt"),
        source_params_pretrain=SimpleNamespace(path="/locked/params-pretrain.yaml"),
    )
    config.token_ae.enabled = False
    encoder = _FrozenModule()
    predictors = [_FrozenModule(), _FrozenModule()]
    planners = [_Planner(), _Planner()]
    dual_value = _DualValue(embed_dim=8)
    events: list[str] = []

    def apply_warmstart_direct(checkpoint_path, params_pretrain_path, modules):
        assert checkpoint_path == "/locked/e120.pt"
        assert params_pretrain_path == "/locked/params-pretrain.yaml"
        assert set(modules) == {"encoder", "predictor", "planner"}
        _set_parameter(modules["encoder"], "weight", 1.0)
        _set_parameter(modules["predictor"], "weight", 2.0)
        _set_parameter(modules["planner"], "bias", 4.0)
        events.append("warmstart")

    def resolve_direct(path, *, results_root, stage):
        events.append(f"resolve_{stage}")
        return resolve_formal_v2_navsim_e120_selected_checkpoint(
            path,
            results_root=results_root,
            stage=stage,
        )

    def read_direct(path):
        payload = read_formal_v2_navsim_e120_direct_checkpoint(path)
        events.append(f"read_{payload['stage']}")
        return payload

    def init_encoder_with_rng_probe(*_args, **_kwargs):
        torch.rand(5)
        return encoder, _FrozenModule()

    before_rng = torch.random.get_rng_state().clone()
    with (
        patch.object(
            runtime_module,
            "init_encoder",
            side_effect=AssertionError("NavSim e120 must not use the legacy encoder checkpoint loader"),
        ),
        patch.object(
            runtime_module,
            "init_encoder_for_full_state_warmstart",
            side_effect=init_encoder_with_rng_probe,
        ),
        patch.object(runtime_module, "get_encoder_embed_dim", return_value=8),
        patch.object(runtime_module, "resolve_main_predictor_runtime_overrides", return_value=(None, None)),
        patch.object(
            runtime_module,
            "init_predictor_runtime_with_token_ae",
            side_effect=[(predictors[0], None, 128, False), (predictors[1], None, 128, False)],
        ),
        patch.object(runtime_module, "init_planner", side_effect=planners),
        patch.object(
            runtime_module,
            "apply_formal_v2_full_state_warmstart_direct",
            side_effect=apply_warmstart_direct,
            create=True,
        ),
        patch.object(
            runtime_module,
            "resolve_formal_v2_navsim_e120_selected_checkpoint",
            side_effect=resolve_direct,
            create=True,
        ),
        patch.object(
            runtime_module,
            "read_formal_v2_navsim_e120_direct_checkpoint",
            side_effect=read_direct,
            create=True,
        ),
        patch.object(
            runtime_module,
            "load_cvoi_dual_value_model",
            return_value=dual_value if stage in {"stop_calibrated", "evaluation"} else None,
        ),
    ):
        runtime = create_navsim_cvoi_model_runtime(
            config=config,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )

    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert isinstance(runtime, NavSimCvoiProductionModelRuntime)
    assert runtime.predictor_p0 is predictors[0]
    assert runtime.predictor_p1 is (predictors[1] if expects_p1 else None)
    assert encoder.weight.item() == pytest.approx(1.0)
    assert predictors[0].weight.item() == pytest.approx(3.0)
    assert planners[0].bias.item() == pytest.approx(7.0)
    if expects_p1:
        assert predictors[1].weight.item() == pytest.approx(5.0)
        assert planners[1].bias.item() == pytest.approx(9.0)
    assert events == [
        "warmstart",
        "resolve_p0",
        "read_p0",
        *(["resolve_p1", "read_p1"] if expects_p1 else []),
    ]


def test_navsim_e120_factory_rejects_p0_only_stop_before_initializing_models(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.cvoi.stage = "stop_calibrated"
    config.cvoi.protocol_version = "formal_v2_navsim_e120_h4_v3"
    config.cvoi.controller_lineage = "p0_controller"

    with (
        patch.object(
            runtime_module,
            "init_encoder_for_full_state_warmstart",
            side_effect=AssertionError("invalid direct Stop must fail before model initialization"),
        ),
        pytest.raises(ValueError, match="Stop requires.*value_guided"),
    ):
        create_navsim_cvoi_model_runtime(
            config=config,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


@pytest.mark.parametrize("symlink", [False, True])
def test_navsim_e120_direct_policy_restore_accepts_copy_or_in_stage_symlink_and_is_rng_neutral(
    tmp_path: Path,
    symlink: bool,
) -> None:
    results_root = tmp_path / "cvoi_manual_full"
    path = _write_direct_policy(
        results_root,
        stage="p0",
        payload=_direct_policy_payload(
            stage="p0",
            encoder_value=1.0,
            predictor_value=3.0,
            planner_value=7.0,
        ),
        symlink=symlink,
    )
    encoder = _FrozenModule(1.0)
    predictor = _FrozenModule(0.0)
    planner = _Planner()
    authoritative_reader = runtime_module.read_formal_v2_navsim_e120_direct_checkpoint

    def read_with_rng_probe(checkpoint_path):
        torch.rand(5)
        return authoritative_reader(checkpoint_path)

    before_rng = torch.random.get_rng_state().clone()
    with patch.object(
        runtime_module,
        "read_formal_v2_navsim_e120_direct_checkpoint",
        side_effect=read_with_rng_probe,
    ):
        restored = runtime_module._restore_navsim_e120_direct_policy_checkpoint(
            encoder=encoder,
            predictor=predictor,
            planner=planner,
            path=path,
            expected_stage="p0",
            results_root=results_root,
        )

    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert restored["stage"] == "p0"
    assert restored["branch_id"] == "p0_uniform"
    assert restored["epoch"] == 35
    assert encoder.weight.item() == pytest.approx(1.0)
    assert predictor.weight.item() == pytest.approx(3.0)
    assert planner.bias.item() == pytest.approx(7.0)
    assert planner.calls == []


def test_navsim_e120_direct_policy_restore_rejects_fixed_path_drift_before_reading(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "cvoi_manual_full"
    wrong_path = results_root / "p0" / "run" / "checkpoint.pt"
    wrong_path.parent.mkdir(parents=True)
    torch.save(_direct_policy_payload(stage="p0"), wrong_path)

    with (
        patch.object(
            runtime_module,
            "read_formal_v2_navsim_e120_direct_checkpoint",
            side_effect=AssertionError("fixed path drift must fail before direct reading"),
        ),
        pytest.raises(ValueError, match="path must be exactly.*p0_selected.pt"),
    ):
        runtime_module._restore_navsim_e120_direct_policy_checkpoint(
            encoder=_FrozenModule(1.0),
            predictor=_FrozenModule(),
            planner=_Planner(),
            path=wrong_path,
            expected_stage="p0",
            results_root=results_root,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_direct_policy_payload(stage="p1"), "stage mismatch"),
        (
            {
                **_direct_policy_payload(stage="p0"),
                "lineage": {
                    **_direct_policy_payload(stage="p0")["lineage"],
                    "branch_id": "p1_full",
                },
            },
            "branch",
        ),
        (_direct_policy_payload(stage="p0", epoch=11), "selected epoch"),
    ],
)
def test_navsim_e120_direct_policy_restore_rejects_stage_branch_or_candidate_drift(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    results_root = tmp_path / "cvoi_manual_full"
    path = _write_direct_policy(results_root, stage="p0", payload=payload)
    planner = _Planner()

    with pytest.raises(ValueError, match=message):
        runtime_module._restore_navsim_e120_direct_policy_checkpoint(
            encoder=_FrozenModule(1.0),
            predictor=_FrozenModule(),
            planner=planner,
            path=path,
            expected_stage="p0",
            results_root=results_root,
        )

    assert planner.calls == []


@pytest.mark.parametrize(
    ("checkpoint_predictor_factory", "target_predictor_factory", "message"),
    [
        (_FrozenModule, _TwoStateModule, "keys mismatch.*missing"),
        (_TwoStateModule, _FrozenModule, "keys mismatch.*unexpected"),
        (_VectorModule, _FrozenModule, "shapes mismatch"),
    ],
)
def test_navsim_e120_direct_policy_restore_rejects_role_key_or_shape_drift(
    tmp_path: Path,
    checkpoint_predictor_factory,
    target_predictor_factory,
    message: str,
) -> None:
    results_root = tmp_path / "cvoi_manual_full"
    path = _write_direct_policy(
        results_root,
        stage="p0",
        payload=_direct_policy_payload(
            stage="p0",
            predictor=checkpoint_predictor_factory(3.0),
        ),
    )
    planner = _Planner()

    with pytest.raises(ValueError, match=message):
        runtime_module._restore_navsim_e120_direct_policy_checkpoint(
            encoder=_FrozenModule(1.0),
            predictor=target_predictor_factory(0.0),
            planner=planner,
            path=path,
            expected_stage="p0",
            results_root=results_root,
        )

    assert planner.calls == []


def test_navsim_e120_direct_policy_restore_rejects_encoder_tensor_mismatch_before_policy_load(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "cvoi_manual_full"
    path = _write_direct_policy(
        results_root,
        stage="p1",
        payload=_direct_policy_payload(
            stage="p1",
            encoder_value=2.0,
            predictor_value=5.0,
            planner_value=9.0,
        ),
    )
    predictor = _FrozenModule(0.0)
    planner = _Planner()

    with pytest.raises(ValueError, match="encoder tensor.*warm-start"):
        runtime_module._restore_navsim_e120_direct_policy_checkpoint(
            encoder=_FrozenModule(1.0),
            predictor=predictor,
            planner=planner,
            path=path,
            expected_stage="p1",
            results_root=results_root,
        )

    assert predictor.weight.item() == pytest.approx(0.0)
    assert planner.bias.item() == pytest.approx(0.0)
    assert planner.calls == []


def test_retired_navsim_e120_v1_factory_is_rejected_before_model_initialization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.cvoi.protocol_version = "formal_v2_navsim_e120_v1"

    with (
        patch.object(runtime_module, "init_encoder", side_effect=AssertionError("must reject before init")),
        pytest.raises(ValueError, match="supports only direct NavSim-e120"),
    ):
        create_navsim_cvoi_model_runtime(
            config=config,
            device=torch.device("cpu"),
            _allow_cpu_for_tests=True,
        )


def test_p0_controller_factory_does_not_initialize_or_restore_p1(tmp_path: Path) -> None:
    def configure(config):
        config.cvoi.controller_lineage = "p0_controller"
        config.cvoi.guided_planner_checkpoint = None
        config.value_guidance.enabled = False

    _, runtime, _, _, _, planners, _, handoff_calls = _build_runtime(tmp_path, configure=configure)

    assert runtime.planner_p1 is None
    assert handoff_calls == ["p0"]
    assert planners[0].bias.item() == pytest.approx(1.0)


def test_encode_uses_observed_inputs_zero_fills_future_and_rolls_only_h4(tmp_path: Path) -> None:
    _, runtime, *_ = _build_runtime(tmp_path)
    captured: dict[str, object] = {}

    def fake_forward_main_context(_encoder, context_clips, **kwargs):
        captured["encoder_clip"] = context_clips.detach().clone()
        captured["encoder_kwargs"] = kwargs
        return torch.ones(1, 4 * 128, 8)

    def fake_build_timeline(*, actions, states, extrinsics, driving_command, ego_dynamics, **kwargs):
        del kwargs
        captured["timeline"] = (actions, states, extrinsics, driving_command, ego_dynamics)
        aligned_actions = actions.clone()
        aligned_states = states.clone()
        aligned_extrinsics = extrinsics.clone()
        aligned_command = driving_command.clone()
        aligned_dynamics = ego_dynamics.clone()
        aligned_actions[:, 3:] = 7.0
        for tensor in (aligned_states, aligned_extrinsics, aligned_command, aligned_dynamics):
            tensor[:, 4:] = 7.0
        return PredictorTimelineInputs(
            raw_num_frames=10,
            frame_stride=1,
            num_observed_steps=4,
            num_time_steps=10,
            num_future_steps=6,
            tokens_per_frame=128,
            actions=aligned_actions,
            states=aligned_states,
            extrinsics=aligned_extrinsics,
            driving_command=aligned_command,
            ego_dynamics=aligned_dynamics,
        )

    def fake_rollout(_step_predictor, **kwargs):
        captured["rollout"] = kwargs
        for _ in range(4):
            _step_predictor(
                kwargs["z_context"],
                kwargs["actions"],
                kwargs["states"],
                kwargs["extrinsics"],
            )
        future = torch.arange(4 * 128 * 8, dtype=torch.float32).reshape(1, 4 * 128, 8)
        return None, future

    with (
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.forward_main_context",
            side_effect=fake_forward_main_context,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.build_predictor_timeline_inputs",
            side_effect=fake_build_timeline,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.make_predictor_step_fn",
            return_value=lambda z, actions, states, extrinsics: z,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.rollout_latent_predictions",
            side_effect=fake_rollout,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.prepare_inference_consistent_status_vector",
            return_value=torch.ones(1, 8),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.build_observed_action_trajectory_history",
            return_value=torch.zeros(1, 4, 3),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.time.perf_counter",
            side_effect=(0.0, 0.001, 0.001, 0.003, 0.003, 0.006, 0.006, 0.010),
        ),
    ):
        encoded = runtime.encode_batch(_model_batch())

    assert encoded.z_observed.shape == (1, 4, 128, 8)
    assert encoded.z_future.shape == (1, 4, 128, 8)
    assert len(encoded.model_contexts) == 1
    assert encoded.model_contexts[0].rollout_latency_ms_by_horizon == pytest.approx((0.0, 1.0, 3.0, 6.0, 10.0))
    assert torch.equal(captured["encoder_clip"], torch.ones(1, 3, 4, 8, 8))
    actions, states, extrinsics, driving_command, ego_dynamics = captured["timeline"]
    assert actions.shape[1] == 9
    assert states.shape[1] == extrinsics.shape[1] == driving_command.shape[1] == ego_dynamics.shape[1] == 10
    assert torch.count_nonzero(actions[:, 3:]) == 0
    for tensor in (states, extrinsics, driving_command, ego_dynamics):
        assert torch.count_nonzero(tensor[:, 4:]) == 0
    assert captured["rollout"]["rollout_end_step"] == 8
    assert captured["rollout"]["num_total"] == 10
    assert captured["rollout"]["compute_tf"] is False
    assert torch.count_nonzero(captured["rollout"]["actions"][:, 3:]) == 0
    for name in ("states", "extrinsics"):
        assert torch.count_nonzero(captured["rollout"][name][:, 4:]) == 0


def test_runtime_rejects_non_online_metadata_at_model_boundary(tmp_path: Path) -> None:
    _, runtime, *_ = _build_runtime(tmp_path)

    with pytest.raises(ValueError, match="metadata key"):
        runtime.encode_batch(_model_batch(metadata={"cf_is_hazard": torch.tensor([True])}))
    with pytest.raises(ValueError, match="camera_intrinsics.*observed"):
        runtime.encode_batch(
            _model_batch(metadata={"camera_intrinsics": torch.eye(3).reshape(1, 1, 1, 3, 3).expand(1, 1, 5, 3, 3)})
        )


def test_h0_and_guided_evaluation_keep_six_pose_output_and_common_random_seed(tmp_path: Path) -> None:
    _, runtime, _, _, _, planners, dual_value, _ = _build_runtime(tmp_path)
    context = SimpleNamespace(
        status_feature=torch.ones(1, 8),
        z_first_frame=None,
        z_observed_for_planner=torch.ones(1, 4 * 128, 8),
        action_history=torch.zeros(1, 4, 3),
        anchor_state=torch.zeros(1, 6),
        rollout_latency_ms_by_horizon=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    observed = torch.ones(1, 4, 128, 8)
    empty = torch.empty(1, 0, 128, 8)

    h0 = runtime.evaluate_guided_horizon(
        context=context,
        z_observed=observed,
        raw_prefix=empty,
        horizon=0,
        apply_guidance=False,
        seed=91,
    )
    assert h0.pred_trajs.shape == (1, 2, 6, 3)
    assert torch.allclose(h0.confidences, torch.tensor([[0.2, 0.8]]))
    assert h0.guidance_steps == 0

    prefix = torch.ones(1, 2, 128, 8)
    first = runtime.evaluate_guided_horizon(
        context=context,
        z_observed=observed,
        raw_prefix=prefix,
        horizon=2,
        apply_guidance=True,
        seed=1234,
    )
    second = runtime.evaluate_guided_horizon(
        context=context,
        z_observed=observed,
        raw_prefix=prefix,
        horizon=2,
        apply_guidance=True,
        seed=1234,
    )
    assert first.guidance_steps == second.guidance_steps == 2
    assert first.pred_trajs.shape[-2:] == (6, 3)
    assert torch.equal(first.pred_trajs, second.pred_trajs)
    assert torch.equal(first.confidences, second.confidences)
    assert dual_value.future_frame_counts.count(2) >= 2
    assert planners[1].calls[-1]["z_future"].shape == (1, 2 * 128, 8)
    assert "gt_trajectory" not in planners[1].calls[-1]["kwargs"]

    p0 = runtime.evaluate_unguided_prefix(
        context=context,
        z_observed=observed,
        prefix=prefix,
        horizon=2,
        seed=3,
    )
    assert p0.guidance_steps == 0
    assert planners[0].calls[-1]["z_future"].shape == (1, 2 * 128, 8)

    p1_unguided = runtime.evaluate_p1_unguided_prefix(
        context=context,
        z_observed=observed,
        prefix=prefix,
        horizon=2,
        seed=3,
    )
    assert p1_unguided.guidance_steps == 0
    assert torch.equal(planners[1].calls[-1]["z_future"], prefix.flatten(1, 2))


def test_horizon_latency_uses_adaptive_compute_and_excludes_common_planner_forward(tmp_path: Path) -> None:
    _, runtime, *_ = _build_runtime(tmp_path)
    context = NavSimCvoiPlannerContext(
        status_feature=torch.ones(1, 8),
        z_first_frame=None,
        z_observed_for_planner=torch.ones(1, 4 * 128, 8),
        action_history=torch.zeros(1, 4, 3),
        anchor_state=torch.zeros(1, 6),
        rollout_latency_ms_by_horizon=(0.0, 2.0, 5.0, 9.0, 14.0),
    )
    observed = torch.ones(1, 4, 128, 8)
    prefix = torch.ones(1, 2, 128, 8)

    with patch.object(runtime, "_guided_prefix", return_value=(prefix.flatten(1, 2), 2, 4.0)):
        unguided = runtime.evaluate_unguided_prefix(
            context=context,
            z_observed=observed,
            prefix=prefix,
            horizon=2,
            seed=5,
        )
        guided = runtime.evaluate_guided_horizon(
            context=context,
            z_observed=observed,
            raw_prefix=prefix,
            horizon=2,
            apply_guidance=True,
            seed=5,
        )

    assert unguided.latency_ms == pytest.approx(5.0)
    assert guided.latency_ms == pytest.approx(9.0)


def test_retained_navsim_e120_runtime_executes_evaluation_guidance_k1_through_k8(tmp_path: Path) -> None:
    def configure_evaluation(config) -> None:
        config.cvoi.stage = "evaluation"

    _, runtime, *_ = _build_runtime(tmp_path, configure=configure_evaluation)
    context = NavSimCvoiPlannerContext(
        status_feature=torch.ones(1, 8),
        z_first_frame=None,
        z_observed_for_planner=torch.ones(1, 4 * 128, 8),
        action_history=torch.zeros(1, 4, 3),
        anchor_state=torch.zeros(1, 6),
        rollout_latency_ms_by_horizon=(0.0, 2.0, 5.0, 9.0, 14.0),
    )
    observed = torch.ones(1, 4, 128, 8)
    prefix = torch.ones(1, 2, 128, 8)

    results = [
        runtime.evaluate_guided_horizon(
            context=context,
            z_observed=observed,
            raw_prefix=prefix,
            horizon=2,
            apply_guidance=True,
            seed=239,
            guidance_steps=guidance_steps,
        )
        for guidance_steps in (1, 2, 4, 8)
    ]
    h0 = runtime.evaluate_guided_horizon(
        context=context,
        z_observed=observed,
        raw_prefix=prefix[:, :0],
        horizon=0,
        apply_guidance=False,
        seed=239,
        guidance_steps=4,
    )

    assert [result.guidance_steps for result in results] == [1, 2, 4, 8]
    assert h0.guidance_steps == 0
    assert all(result.pred_trajs.shape[-2:] == (6, 3) for result in results)


def test_horizon_evaluation_rejects_nonfinite_rollout_latency(tmp_path: Path) -> None:
    _, runtime, *_ = _build_runtime(tmp_path)
    context = NavSimCvoiPlannerContext(
        status_feature=torch.ones(1, 8),
        z_first_frame=None,
        z_observed_for_planner=torch.ones(1, 4 * 128, 8),
        action_history=torch.zeros(1, 4, 3),
        anchor_state=torch.zeros(1, 6),
        rollout_latency_ms_by_horizon=(0.0, float("nan"), 2.0, 3.0),
    )

    with pytest.raises(ValueError, match="rollout latency"):
        runtime.evaluate_unguided_prefix(
            context=context,
            z_observed=torch.ones(1, 4, 128, 8),
            prefix=torch.ones(1, 1, 128, 8),
            horizon=1,
            seed=5,
        )


def test_bfloat16_latents_use_float32_dual_value_and_return_bfloat16_guidance(tmp_path: Path) -> None:
    torch.manual_seed(7)
    dual_value = PrefixDualValueModel(embed_dim=8, hidden_dim=16, num_layers=1)
    _, runtime, _, _, _, planners, _, _ = _build_runtime(tmp_path, dual_value=dual_value)
    context = SimpleNamespace(
        status_feature=torch.ones(1, 8, dtype=torch.bfloat16),
        z_first_frame=None,
        z_observed_for_planner=torch.zeros(1, 4 * 128, 8, dtype=torch.bfloat16),
        action_history=torch.zeros(1, 4, 3, dtype=torch.bfloat16),
        anchor_state=torch.zeros(1, 6, dtype=torch.bfloat16),
        rollout_latency_ms_by_horizon=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    observed = torch.zeros(1, 4, 128, 8, dtype=torch.bfloat16)
    prefix = torch.zeros(1, 2, 128, 8, dtype=torch.bfloat16)

    result = runtime.evaluate_guided_horizon(
        context=context,
        z_observed=observed,
        raw_prefix=prefix,
        horizon=2,
        apply_guidance=True,
        seed=17,
    )

    guided = planners[1].calls[-1]["z_future"]
    assert next(dual_value.parameters()).dtype == torch.float32
    assert guided.dtype == torch.bfloat16
    assert torch.count_nonzero(guided) > 0
    assert result.guidance_steps == 2


def test_multiview_fusion_uses_dedicated_raw_encoder_token_resolver(tmp_path: Path) -> None:
    captured: list[int] = []

    def configure(config):
        config.multiview.enabled = True

    def fake_init_fusion(_config, *, embed_dim, raw_tokens_per_frame, device):
        del embed_dim, device
        captured.append(raw_tokens_per_frame)
        return None

    with (
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime."
            "resolve_main_encoder_raw_tokens_per_frame",
            return_value=2048,
            create=True,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime._init_multiview_fusion",
            side_effect=fake_init_fusion,
        ),
    ):
        _build_runtime(tmp_path, configure=configure)

    assert captured == [2048]


def test_runtime_rejects_no_aux_inference_consistent_predictor(tmp_path: Path) -> None:
    def configure(config):
        config.train.predictor_no_aux_input = True

    with pytest.raises(ValueError, match="predictor_no_aux_input=false"):
        _build_runtime(tmp_path, configure=configure)


def test_online_session_rolls_only_requested_steps_with_exact_zero_future_aux_slices(tmp_path: Path) -> None:
    _, runtime, *_ = _build_runtime(tmp_path)
    events: list[object] = []
    predictor_slices: list[tuple[int, int, int]] = []

    def fake_forward_main_context(_encoder, context_clips, **_kwargs):
        events.append("encoder")
        assert context_clips.shape == (1, 3, 4, 8, 8)
        return torch.ones(1, 4 * 128, 8)

    def fake_build_timeline(*, actions, states, extrinsics, driving_command, ego_dynamics, **_kwargs):
        assert torch.count_nonzero(actions[:, 3:]) == 0
        assert torch.count_nonzero(states[:, 4:]) == 0
        assert torch.count_nonzero(extrinsics[:, 4:]) == 0
        assert torch.count_nonzero(driving_command[:, 4:]) == 0
        assert torch.count_nonzero(ego_dynamics[:, 4:]) == 0
        return PredictorTimelineInputs(
            raw_num_frames=10,
            frame_stride=1,
            num_observed_steps=4,
            num_time_steps=10,
            num_future_steps=6,
            tokens_per_frame=128,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
        )

    def step_predictor(z_prefix, actions_step, states_step, extrinsics_step):
        predictor_slices.append((actions_step.shape[1], states_step.shape[1], extrinsics_step.shape[1]))
        events.append(("predictor", len(predictor_slices)))
        next_tokens = z_prefix.new_full((1, 128, 8), float(len(predictor_slices)))
        return torch.cat([z_prefix, next_tokens], dim=1)

    with (
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.forward_main_context",
            side_effect=fake_forward_main_context,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.build_predictor_timeline_inputs",
            side_effect=fake_build_timeline,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.make_predictor_step_fn",
            return_value=step_predictor,
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.prepare_inference_consistent_status_vector",
            return_value=torch.ones(1, 8),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.build_observed_action_trajectory_history",
            return_value=torch.zeros(1, 4, 3),
        ),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime.build_ego_relative_diffusion_anchor",
            return_value=torch.zeros(1, 6),
        ),
        patch.object(runtime, "_encode_batch_with_policy", side_effect=AssertionError("eager API is forbidden")),
        patch(
            "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime._time_tensor_operation",
            side_effect=AssertionError("nested timing is forbidden"),
        ),
    ):
        session = runtime.start_online_session(_model_batch(), policy="p1")
        raw_prefix = torch.empty(1, 0, 8)
        for next_horizon in (1, 2, 3):
            next_tokens = runtime.rollout_online_step(session, raw_prefix, next_horizon=next_horizon)
            raw_prefix = torch.cat([raw_prefix, next_tokens], dim=1)

    assert isinstance(session, runtime_module.NavSimCvoiOnlineSession)
    assert session.policy == "p1"
    assert not hasattr(session.model_context, "rollout_latency_ms_by_horizon")
    assert events == ["encoder", ("predictor", 1), ("predictor", 2), ("predictor", 3)]
    assert predictor_slices == [(4, 4, 4), (5, 5, 5), (6, 6, 6)]
    assert raw_prefix.shape == (1, 3 * 128, 8)


def test_online_p0_value_and_terminal_prefix_use_zero_field_without_guidance(tmp_path: Path) -> None:
    _, runtime, *_ = _build_runtime(tmp_path)
    session = runtime_module.NavSimCvoiOnlineSession(
        z_observed=torch.ones(1, 4 * 128, 8),
        model_context=runtime_module._NavSimCvoiOnlineModelContext(
            status_feature=torch.ones(1, 8),
            z_first_frame=None,
            z_observed_for_planner=torch.ones(1, 4 * 128, 8),
            action_history=torch.zeros(1, 4, 3),
            anchor_state=torch.zeros(1, 6),
        ),
        predictor_inputs=object(),
        step_predictor=lambda *_args: torch.empty(0),
        policy="p0",
    )
    raw_prefix = torch.ones(1, 2 * 128, 8)

    values = runtime.online_value_features(
        session,
        raw_prefix,
        horizon=2,
        controller_lineage="p0_controller",
    )
    terminal, diagnostics = runtime.prepare_online_terminal_prefix(
        session,
        raw_prefix,
        horizon=2,
        controller_lineage="p0_controller",
        guidance_steps=None,
    )

    assert torch.equal(values["field_value"], torch.zeros(1))
    assert torch.equal(values["stop_value"], torch.ones(1))
    assert torch.equal(terminal, raw_prefix)
    assert terminal.requires_grad is False
    assert diagnostics == {
        "guidance_steps": 0.0,
        "guidance_skipped_h0": 0.0,
        "delta_norm": 0.0,
        "field_value_before": 0.0,
        "field_value_after": 0.0,
    }


@pytest.mark.parametrize("policy", ["p0", "p1"])
def test_online_session_rejects_eager_or_unknown_policy_pair(tmp_path: Path, policy: str) -> None:
    def configure(config):
        if policy == "p0":
            config.cvoi.controller_lineage = "p0_controller"
            config.cvoi.guided_planner_checkpoint = None
            config.value_guidance.enabled = False

    _, runtime, *_ = _build_runtime(tmp_path, configure=configure)

    with pytest.raises(ValueError, match="policy"):
        runtime.start_online_session(_model_batch(), policy="eager")
    if policy == "p0":
        with pytest.raises(RuntimeError, match="P1"):
            runtime.start_online_session(_model_batch(), policy="p1")
