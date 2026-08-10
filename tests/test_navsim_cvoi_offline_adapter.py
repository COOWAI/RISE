"""Tests for the direct NavSim-e120 offline Value adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from app.vjepa_cowa_world_model.training import navsim_cvoi_offline_adapter as offline_adapter_module
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    CvoiPlannerEvaluation,
    NavSimCvoiEncodedBatch,
    NavSimCvoiModelBatch,
    build_navsim_cvoi_model_batch,
    navsim_cvoi_raw_prefix,
    validate_navsim_cvoi_encoded_batch,
)
from app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter import (
    CVOI_NAVSIM_OFFLINE_ADAPTER_FACTORY,
    NavSimCvoiOfflineAdapter,
    create_navsim_cvoi_offline_adapter,
)
from app.vjepa_cowa_world_model.training.navsim_data import CF_QUALITY_SCHEMA


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cvoi=SimpleNamespace(
            stage="field_warmup",
            protocol_version="formal_v2_navsim_e120_h4_v3",
            controller_lineage="value_guided",
            max_horizon=4,
            tokens_per_frame=None,
            offline_runtime_factory="tests.fake_runtime:create",
            field_warmup_domain="real_cf",
            field_calibration_num_perturbations=2,
            field_calibration_perturbation_scale=0.05,
            field_calibration_max_delta_norm=0.25,
            ablation_signature=SimpleNamespace(
                cf_field_supervision="hazard_quality",
                field_calibration_mode="local_geometry",
                evaluation_seed=239,
            ),
        ),
        data=SimpleNamespace(
            fps=2,
            navsim=SimpleNamespace(train_roots=[{"name": "real"}], val_roots=[]),
        ),
        train=SimpleNamespace(num_observed_frames=1, predictor_inference_consistent=True),
        meta=SimpleNamespace(seed=17),
    )


def _mixed_batch() -> tuple[object, ...]:
    batch_size = 2
    num_frames = 5
    metadata = {
        "dataset_domain": ["real", "counterfactual"],
        "stable_sample_id": ["navsim:real:a", "navsim:cf:b"],
        "base_scene_id": ["scene-a", "scene-b"],
        "cf_annotation_valid": torch.tensor([False, True]),
        "cf_is_hazard": torch.tensor([False, True]),
        "cf_hazard_type": ["", "自车行为引起"],
        "cf_quality_present": torch.tensor([False, True]),
        "cf_quality": torch.tensor([float("nan"), 0.2]),
        "cf_quality_schema": [None, CF_QUALITY_SCHEMA],
        "cf_quality_source": [None, "trajectory_quality_sidecar"],
        "camera_names": ["CAM_F0"],
        "camera_intrinsics": torch.eye(3).reshape(1, 1, 1, 3, 3).expand(batch_size, 1, num_frames, 3, 3),
        "camera2ego": torch.eye(4).reshape(1, 1, 1, 4, 4).expand(batch_size, 1, num_frames, 4, 4),
    }
    context_frames = torch.zeros(batch_size, 3, num_frames, 8, 8)
    context_frames[:, :, 1:] = 99.0
    actions = torch.full((batch_size, num_frames - 1, 3), 99.0)
    states = torch.zeros(batch_size, num_frames, 7)
    states[:, 1:] = 99.0
    extrinsics = torch.zeros(batch_size, num_frames, 7)
    extrinsics[:, 1:] = 99.0
    driving_command = torch.zeros(batch_size, num_frames, 4)
    driving_command[:, 1:] = 99.0
    ego_dynamics = torch.zeros(batch_size, num_frames, 4)
    ego_dynamics[:, 1:] = 99.0
    return (
        context_frames,
        actions,
        states,
        extrinsics,
        [None] * batch_size,
        driving_command,
        ego_dynamics,
        object(),
        object(),
        object(),
        None,
        metadata,
    )


def _single_domain_batch(domain: str) -> tuple[object, ...]:
    if domain not in {"real", "counterfactual"}:
        raise ValueError(domain)
    source = _mixed_batch()
    index = 0 if domain == "real" else 1
    batch = list(source)
    for position in (0, 1, 2, 3, 5, 6):
        batch[position] = source[position][index : index + 1]
    batch[4] = [source[4][index]]
    metadata = {}
    for key, value in source[11].items():
        if key == "camera_names":
            metadata[key] = value
        elif isinstance(value, torch.Tensor):
            metadata[key] = value[index : index + 1]
        elif isinstance(value, list):
            metadata[key] = [value[index]]
        else:
            metadata[key] = value
    batch[11] = metadata
    return tuple(batch)


def _matched_real_counterfactual_batch() -> tuple[object, ...]:
    batch = list(_mixed_batch())
    metadata = dict(batch[11])
    metadata["base_scene_id"] = ["scene-a", "scene-a"]
    metadata["window_start_pos"] = torch.tensor([10, 10], dtype=torch.long)
    batch[11] = metadata
    return tuple(batch)


class _Runtime:
    embed_dim = 4

    def __init__(self) -> None:
        self.model_batches: list[NavSimCvoiModelBatch] = []
        self.p1_model_batches: list[NavSimCvoiModelBatch] = []
        self.unguided_calls = 0
        self.guided_calls = 0
        self.guided_prefix_lengths: list[int] = []

    def encode_batch(self, batch: NavSimCvoiModelBatch) -> NavSimCvoiEncodedBatch:
        self.model_batches.append(batch)
        assert not hasattr(batch, "agent_boxes")
        assert "cf_is_hazard" not in batch.metadata
        assert "stable_sample_id" not in batch.metadata
        assert "dataset_domain" not in batch.metadata
        video_time_dim = 2 if batch.context_frames.ndim == 5 else 3
        assert batch.context_frames.shape[video_time_dim] == 1
        assert batch.actions.shape[1] == 0
        assert batch.states.shape[1] == 1
        assert batch.extrinsics.shape[1] == 1
        assert batch.driving_command.shape[1] == 1
        assert batch.ego_dynamics.shape[1] == 1
        assert batch.metadata["camera_intrinsics"].shape[2] == 1
        assert batch.metadata["camera2ego"].shape[2] == 1
        batch_size = batch.context_frames.shape[0]
        return NavSimCvoiEncodedBatch(
            z_observed=torch.zeros(batch_size, 2, 2, self.embed_dim),
            z_future=torch.zeros(batch_size, 4, 2, self.embed_dim),
            model_contexts=tuple({"row": index} for index in range(batch_size)),
        )

    def encode_p1_batch(self, batch: NavSimCvoiModelBatch) -> NavSimCvoiEncodedBatch:
        encoded = self.encode_batch(batch)
        self.model_batches.pop()
        self.p1_model_batches.append(batch)
        return NavSimCvoiEncodedBatch(
            z_observed=encoded.z_observed,
            z_future=torch.ones_like(encoded.z_future),
            model_contexts=encoded.model_contexts,
        )

    @staticmethod
    def _planner_result(*, guidance_steps: int) -> CvoiPlannerEvaluation:
        return CvoiPlannerEvaluation(
            pred_trajs=torch.zeros(1, 2, 3, 3),
            confidences=torch.tensor([[0.9, 0.1]]),
            latency_ms=1.5,
            guidance_steps=guidance_steps,
        )

    def evaluate_unguided_prefix(self, *, context, z_observed, prefix, horizon, seed):
        del context, z_observed, prefix, horizon, seed
        self.unguided_calls += 1
        return self._planner_result(guidance_steps=0)

    def evaluate_guided_horizon(self, *, context, z_observed, raw_prefix, horizon, apply_guidance, seed):
        del context, z_observed, horizon, seed
        self.guided_prefix_lengths.append(int(raw_prefix.shape[1]))
        self.guided_calls += 1
        return self._planner_result(guidance_steps=2 if apply_guidance else 0)


def _adapter(
    tmp_path: Path,
    *,
    stage: str = "field_warmup",
    batch: tuple[object, ...] | None = None,
    controller_lineage: str = "value_guided",
    calibration_mode: str = "local_geometry",
) -> tuple[NavSimCvoiOfflineAdapter, _Runtime]:
    runtime = _Runtime()
    config = _config(tmp_path)
    config.cvoi.stage = stage
    config.cvoi.controller_lineage = controller_lineage
    config.cvoi.ablation_signature.field_calibration_mode = calibration_mode
    adapter = NavSimCvoiOfflineAdapter(
        config=config,
        device=torch.device("cpu"),
        loader=[batch or _mixed_batch()],
        runtime=runtime,
    )
    return adapter, runtime


def test_adapter_reexports_world4drive_model_boundary_helpers() -> None:
    assert offline_adapter_module.NavSimCvoiModelBatch is NavSimCvoiModelBatch
    assert offline_adapter_module.NavSimCvoiEncodedBatch is NavSimCvoiEncodedBatch
    assert offline_adapter_module.build_navsim_cvoi_model_batch is build_navsim_cvoi_model_batch
    assert offline_adapter_module.navsim_cvoi_raw_prefix is navsim_cvoi_raw_prefix
    assert offline_adapter_module.validate_navsim_cvoi_encoded_batch is validate_navsim_cvoi_encoded_batch


@pytest.mark.parametrize(
    ("stage", "controller_lineage", "expected_p1_calls"),
    (
        ("field_warmup", "value_guided", 0),
        ("field_calibrated", "value_guided", 0),
        ("stop_calibrated", "value_guided", 1),
    ),
)
def test_navsim_e120_offline_stage_encodes_with_its_bound_policy_pair(
    tmp_path: Path,
    stage: str,
    controller_lineage: str,
    expected_p1_calls: int,
) -> None:
    adapter, runtime = _adapter(
        tmp_path,
        stage=stage,
        batch=_single_domain_batch("real"),
        controller_lineage=controller_lineage,
    )

    encoded = adapter._encode(_single_domain_batch("real"))

    assert len(runtime.p1_model_batches) == expected_p1_calls
    assert len(runtime.model_batches) == 1 - expected_p1_calls
    assert torch.all(encoded.z_future == float(expected_p1_calls))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("dataset_domain", None),
        ("stable_sample_id", 7),
        ("base_scene_id", object()),
    ),
)
def test_navsim_e120_value_batches_reject_non_string_identity_and_domain_metadata(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    batch = list(_mixed_batch())
    metadata = dict(batch[11])
    values = list(metadata[field])
    values[0] = invalid_value
    metadata[field] = values
    batch[11] = metadata
    adapter, _ = _adapter(tmp_path, batch=tuple(batch))

    with pytest.raises(ValueError, match=field):
        list(adapter.value_batches("field_warmup", 0, provenance=None))


def test_navsim_e120_field_uses_route_free_quality_and_never_requires_real_geometry(tmp_path: Path) -> None:
    adapter, runtime = _adapter(tmp_path, batch=_matched_real_counterfactual_batch())

    batches = list(adapter.value_batches("field_warmup", 0, provenance=None))

    assert len(batches) == 1
    assert batches[0]["adapter_schema"] == "navsim_e120_quality_field_batch_v1"
    assert batches[0]["real_quality_targets"].shape == (1, 4)
    assert "real_geometry_targets" not in batches[0]
    assert runtime.unguided_calls == 4


def test_navsim_e120_calibration_and_stop_use_explicit_quality_targets(tmp_path: Path) -> None:
    batch = _single_domain_batch("real")
    calibration, calibration_runtime = _adapter(tmp_path, stage="field_calibrated", batch=batch)

    calibrated_batches = list(calibration.value_batches("field_calibrated", 0, provenance=None))

    assert calibrated_batches[0]["adapter_schema"] == "navsim_e120_local_quality_calibration_v1"
    assert calibrated_batches[0]["real_quality_targets"].shape == (3, 4)
    assert "real_geometry_targets" not in calibrated_batches[0]
    assert calibration_runtime.unguided_calls == 12

    stop, stop_runtime = _adapter(tmp_path, stage="stop_calibrated", batch=batch)
    stop_batches = list(stop.value_batches("stop_calibrated", 0, provenance=None))

    assert stop_batches[0]["adapter_schema"] == "navsim_e120_stop_quality_v1"
    assert stop_batches[0]["stop_quality_targets"].shape == (1, 5)
    assert "stop_targets" not in stop_batches[0]
    assert stop_runtime.guided_calls == 5


def test_navsim_e120_stop_rejects_p0_controller_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Stop requires controller_lineage='value_guided'"):
        _adapter(
            tmp_path,
            stage="stop_calibrated",
            batch=_single_domain_batch("real"),
            controller_lineage="p0_controller",
        )


@pytest.mark.parametrize(
    ("configured_stage", "requested_stage"),
    (
        ("field_warmup", "stop_calibrated"),
        ("stop_calibrated", "field_warmup"),
    ),
)
def test_direct_navsim_e120_value_batches_reject_stage_mismatch_before_encoding(
    tmp_path: Path,
    configured_stage: str,
    requested_stage: str,
) -> None:
    adapter, runtime = _adapter(
        tmp_path,
        stage=configured_stage,
        batch=_single_domain_batch("real"),
    )

    with pytest.raises(ValueError, match="must match configured cvoi.stage"):
        list(adapter.value_batches(requested_stage, 0, provenance=None))

    assert runtime.model_batches == []
    assert runtime.p1_model_batches == []
    assert runtime.unguided_calls == 0
    assert runtime.guided_calls == 0


def test_model_boundary_slices_multiview_and_proposal_video_to_observed_frames(tmp_path: Path) -> None:
    batch = list(_matched_real_counterfactual_batch())
    batch_size = batch[2].shape[0]
    num_frames = batch[2].shape[1]
    multiview = torch.zeros(batch_size, 2, 3, num_frames, 8, 8)
    multiview[:, :, :, 1:] = 99.0
    proposal = torch.zeros_like(multiview)
    proposal[:, :, :, 1:] = 99.0
    batch[0] = multiview
    batch[10] = proposal
    adapter, runtime = _adapter(tmp_path, batch=tuple(batch))

    list(adapter.value_batches("field_warmup", 0, provenance=None))

    assert runtime.model_batches[0].context_frames.shape[3] == 1
    assert runtime.model_batches[0].proposal_context_frames.shape[3] == 1


def test_navsim_e120_field_validation_uses_independent_real_and_matched_pair_loaders(tmp_path: Path) -> None:
    runtime = _Runtime()
    config = _config(tmp_path)
    adapter = NavSimCvoiOfflineAdapter(
        config=config,
        device=torch.device("cpu"),
        loader=[_mixed_batch()],
        field_validation_loaders={
            "real": [_single_domain_batch("real")],
            "matched_real_counterfactual": [_matched_real_counterfactual_batch()],
        },
        runtime=runtime,
    )

    real = list(adapter.field_validation_batches(domain="real", provenance=None))
    matched = list(adapter.field_validation_batches(domain="matched_real_counterfactual", provenance=None))

    assert real[0]["dataset_domains"] == ["real"]
    assert matched[0]["dataset_domains"] == ["real", "counterfactual"]
    assert matched[0]["cf_hazard_pair_real_indices"].tolist() == [0]
    assert matched[0]["cf_hazard_pair_counterfactual_indices"].tolist() == [1]
    assert matched[0]["cf_hazard_pair_keys"] == [("scene-a", 10)]
    assert "cvoi_provenance" not in real[0]
    assert "cvoi_provenance" not in matched[0]


def test_strict_real_only_adapter_has_no_counterfactual_validation_loader(tmp_path: Path) -> None:
    runtime = _Runtime()
    config = _config(tmp_path)
    config.cvoi.ablation_signature.cf_field_supervision = "none"
    adapter = NavSimCvoiOfflineAdapter(
        config=config,
        device=torch.device("cpu"),
        loader=[_single_domain_batch("real")],
        field_validation_loaders={"real": [_single_domain_batch("real")]},
        runtime=runtime,
    )

    assert len(list(adapter.field_validation_batches(domain="real", provenance=None))) == 1
    with pytest.raises(ValueError, match="matched_real_counterfactual.*not configured"):
        list(adapter.field_validation_batches(domain="matched_real_counterfactual", provenance=None))


def _configure_existing_real_roots(config: SimpleNamespace, tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    sensor_blobs_path = tmp_path / "sensor-blobs"
    data_path.mkdir()
    sensor_blobs_path.mkdir()
    root = {
        "name": "real",
        "domain": "real",
        "data_path": str(data_path.resolve()),
        "sensor_blobs_path": str(sensor_blobs_path.resolve()),
    }
    config.data.navsim.train_roots = [root]
    config.data.navsim.val_roots = [dict(root)]


def test_direct_root_validation_resolves_formal_scene_filter_from_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _configure_existing_real_roots(config, tmp_path)
    config.data.navsim.train_roots[0]["scene_filter_yaml"] = "configs/navsim/scene_filters/navtrain.yaml"
    monkeypatch.chdir(tmp_path)

    offline_adapter_module._validate_direct_navsim_roots(config)


def test_direct_root_validation_rejects_arbitrary_absolute_scene_filter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _configure_existing_real_roots(config, tmp_path)
    arbitrary_filter = tmp_path / "arbitrary-scene-filter.yaml"
    arbitrary_filter.write_text("{}", encoding="utf-8")
    config.data.navsim.train_roots[0]["scene_filter_yaml"] = str(arbitrary_filter.resolve())

    with pytest.raises(ValueError, match="exact repository-relative path"):
        offline_adapter_module._validate_direct_navsim_roots(config)


def test_direct_navsim_e120_factory_validates_roots_and_builds_adapter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.cvoi.stage = "field_calibrated"
    _configure_existing_real_roots(config, tmp_path)
    runtime = _Runtime()

    with (
        patch.object(offline_adapter_module, "load_navsim_cvoi_runtime", return_value=runtime) as load_runtime,
        patch.object(offline_adapter_module, "create_validation_transforms", return_value=object()),
        patch.object(
            offline_adapter_module,
            "create_train_dataloader",
            return_value=([_single_domain_batch("real")], object()),
        ) as create_loader,
    ):
        adapter = create_navsim_cvoi_offline_adapter(config=config, device=torch.device("cpu"))

    assert isinstance(adapter, NavSimCvoiOfflineAdapter)
    assert CVOI_NAVSIM_OFFLINE_ADAPTER_FACTORY.endswith(":create_navsim_cvoi_offline_adapter")
    load_runtime.assert_called_once_with(config, device=torch.device("cpu"))
    create_loader.assert_called_once()
    assert not hasattr(adapter, "audit_signature")


def test_direct_navsim_e120_counterfactual_root_rejects_locked_annotation_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    data_path = tmp_path / "data"
    sensor_blobs_path = tmp_path / "sensor-blobs"
    pose_overlay_path = tmp_path / "poses"
    for path in (data_path, sensor_blobs_path, pose_overlay_path):
        path.mkdir()
    annotations_path = tmp_path / "navsim_train.json"
    annotations_path.write_text("[]", encoding="utf-8")
    trajectory_quality_path = tmp_path / "trajectory-quality.json"
    trajectory_quality_path.write_text("{}", encoding="utf-8")
    config.data.navsim.train_roots = [
        {
            "name": "navsim_cf_train",
            "domain": "counterfactual",
            "data_path": str(data_path.resolve()),
            "sensor_blobs_path": str(sensor_blobs_path.resolve()),
            "pose_overlay_path": str(pose_overlay_path.resolve()),
            "annotations_path": str(annotations_path.resolve()),
            "trajectory_quality_path": str(trajectory_quality_path.resolve()),
        }
    ]
    config.data.navsim.val_roots = []

    with (
        patch.object(
            offline_adapter_module,
            "validate_formal_v2_navsim_cf_annotation_source",
            side_effect=ValueError("source_sha256 mismatch"),
        ) as validate_annotations,
        pytest.raises(ValueError, match="source_sha256 mismatch"),
    ):
        offline_adapter_module._validate_direct_navsim_roots(config)

    validate_annotations.assert_called_once_with(annotations_path.resolve(), root_name="navsim_cf_train")


@pytest.mark.parametrize(
    ("field_warmup_domain", "cf_field_supervision", "expected_validation_domains"),
    (
        ("real_cf", "hazard_quality", ["real", "matched_real_counterfactual"]),
        ("real_cf", "quality_only", ["real", "counterfactual"]),
        ("real", "none", ["real"]),
    ),
)
def test_navsim_e120_factory_builds_only_required_independent_validation_loaders(
    tmp_path: Path,
    field_warmup_domain: str,
    cf_field_supervision: str,
    expected_validation_domains: list[str],
) -> None:
    config = _config(tmp_path)
    config.cvoi.field_warmup_domain = field_warmup_domain
    config.cvoi.ablation_signature.cf_field_supervision = cf_field_supervision
    _configure_existing_real_roots(config, tmp_path)
    runtime = _Runtime()
    validation_loaders = {
        "real": [_single_domain_batch("real")],
        "counterfactual": [_single_domain_batch("counterfactual")],
        "matched_real_counterfactual": [_matched_real_counterfactual_batch()],
    }

    def create_validation_loader(_config, *, rank, world_size, transform, validation_domain):
        assert rank == 0
        assert world_size == 1
        assert transform is not None
        return validation_loaders[validation_domain], object()

    with (
        patch.object(offline_adapter_module, "load_navsim_cvoi_runtime", return_value=runtime),
        patch.object(offline_adapter_module, "create_validation_transforms", return_value=object()),
        patch.object(
            offline_adapter_module,
            "create_train_dataloader",
            return_value=([_mixed_batch()], object()),
        ),
        patch.object(
            offline_adapter_module,
            "create_val_dataloader",
            side_effect=create_validation_loader,
        ) as create_val,
    ):
        adapter = create_navsim_cvoi_offline_adapter(config=config, device=torch.device("cpu"))

    assert [call.kwargs["validation_domain"] for call in create_val.call_args_list] == expected_validation_domains
    assert set(adapter.field_validation_loaders) == set(expected_validation_domains)
