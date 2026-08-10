"""Strict adaptation from mixed NavSim batches to CVoI Field supervision."""

from __future__ import annotations

import pytest
import torch

from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    NavSimCvoiEncodedBatch,
    NavSimCvoiModelBatch,
    build_navsim_cvoi_model_batch,
    navsim_cvoi_raw_prefix,
    validate_navsim_cvoi_encoded_batch,
)
from app.vjepa_cowa_world_model.training.navsim_cvoi_batch import (
    NAVSIM_CVOI_FIELD_ADAPTER_SCHEMA,
    RealGeometryTargetRequest,
    adapt_navsim_cvoi_field_batch,
    adapt_navsim_e120_quality_field_batch,
)
from app.vjepa_cowa_world_model.training.navsim_data import CF_QUALITY_SCHEMA


def _mixed_navsim_batch() -> tuple[object, ...]:
    batch_size = 3
    num_frames = 4
    agent_boxes = torch.zeros(batch_size, num_frames, 256, 7)
    agent_mask = torch.zeros(batch_size, num_frames, 256, dtype=torch.bool)
    agent_boxes[0, :, :2, 3:6] = torch.tensor([4.0, 2.0, 1.5])
    agent_boxes[2, :, :1, 3:6] = torch.tensor([4.5, 2.2, 1.6])
    agent_mask[0, :, :2] = True
    agent_mask[2, :, :1] = True
    states = torch.zeros(batch_size, num_frames, 7)
    metadata = {
        "dataset_domain": ["real", "counterfactual", "real"],
        "stable_sample_id": ["navsim:real:a", "navsim:cf:b", "navsim:real:c"],
        "base_scene_id": ["scene-a", "scene-b", "scene-c"],
        "cf_annotation_valid": torch.tensor([False, True, False]),
        "cf_is_hazard": torch.tensor([False, True, False]),
        "cf_hazard_type": ["", "自车行为引起", ""],
        "geometry_present": torch.tensor([True, False, True]),
        "future_agent_geometry_valid": torch.tensor([True, False, True]),
        "agent_geometry_truncated": [False, None, False],
        "geometry_source": ["logged_nuscenes_gt", None, "logged_nuscenes_gt"],
        "geometry_coordinate_frame": ["per_frame_ego", None, "per_frame_ego"],
        "raw_agent_count": [torch.full((num_frames,), 2), None, torch.ones(num_frames, dtype=torch.long)],
        "cf_quality_present": torch.tensor([False, True, False]),
        "cf_quality": torch.tensor([float("nan"), 0.25, float("nan")]),
        "cf_quality_schema": [None, CF_QUALITY_SCHEMA, None],
        "cf_quality_source": [None, "trajectory_quality_sidecar", None],
    }
    return (
        torch.zeros(batch_size, 3, num_frames, 8, 8),
        torch.zeros(batch_size, num_frames - 1, 3),
        states,
        torch.zeros(batch_size, num_frames, 7),
        [None] * batch_size,
        torch.zeros(batch_size, num_frames, 4),
        torch.zeros(batch_size, num_frames, 4),
        agent_boxes,
        agent_mask,
        torch.zeros(batch_size, num_frames, 200, 200),
        None,
        metadata,
    )


def _latents() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(3, 2, 5, 4), torch.randn(3, 3, 5, 4)


def test_world4drive_runtime_owns_the_public_model_batch_boundary() -> None:
    from app.vjepa_cowa_world_model.training import navsim_cvoi_offline_adapter

    assert navsim_cvoi_offline_adapter.NavSimCvoiModelBatch is NavSimCvoiModelBatch
    assert navsim_cvoi_offline_adapter.NavSimCvoiEncodedBatch is NavSimCvoiEncodedBatch
    assert navsim_cvoi_offline_adapter.build_navsim_cvoi_model_batch is build_navsim_cvoi_model_batch
    assert navsim_cvoi_offline_adapter.navsim_cvoi_raw_prefix is navsim_cvoi_raw_prefix
    assert navsim_cvoi_offline_adapter.validate_navsim_cvoi_encoded_batch is validate_navsim_cvoi_encoded_batch


def test_mixed_adapter_routes_cf_labels_and_real_callback_without_geometry_leak() -> None:
    z_observed, z_future = _latents()
    requests: list[RealGeometryTargetRequest] = []

    def target_provider(request: RealGeometryTargetRequest) -> torch.Tensor:
        requests.append(request)
        assert request.batch_indices.tolist() == [0, 2]
        assert request.sample_ids == ("navsim:real:a", "navsim:real:c")
        assert request.group_ids == ("scene-a", "scene-c")
        assert request.agent_boxes.shape == (2, 4, 256, 7)
        assert request.agent_mask.shape == (2, 4, 256)
        assert request.agent_mask.any()
        assert request.states.shape == (2, 4, 7)
        assert request.raw_agent_count.tolist() == [[2, 2, 2, 2], [1, 1, 1, 1]]
        assert request.geometry_source == ("logged_nuscenes_gt", "logged_nuscenes_gt")
        assert request.geometry_coordinate_frame == ("per_frame_ego", "per_frame_ego")
        return torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.8, 0.9]])

    adapted = adapt_navsim_cvoi_field_batch(
        _mixed_navsim_batch(),
        z_observed=z_observed,
        z_future=z_future,
        real_geometry_target_provider=target_provider,
    )

    assert len(requests) == 1
    assert adapted["adapter_schema"] == NAVSIM_CVOI_FIELD_ADAPTER_SCHEMA
    assert adapted["dataset_domains"] == ["real", "counterfactual", "real"]
    assert adapted["real_mask"].tolist() == [True, False, True]
    assert adapted["counterfactual_mask"].tolist() == [False, True, False]
    assert adapted["real_geometry_targets"].shape == (2, 3)
    assert adapted["real_group_ids"] == ["scene-a", "scene-c"]
    assert adapted["cf_hazard"].tolist() == [False, True, False]
    assert adapted["cf_hazard_types"] == ["", "自车行为引起", ""]
    assert torch.isnan(adapted["cf_quality"][[0, 2]]).all()
    assert adapted["cf_quality"][1].item() == pytest.approx(0.25)
    assert adapted["stable_sample_ids"] == ["navsim:real:a", "navsim:cf:b", "navsim:real:c"]


@pytest.mark.parametrize(
    ("mode", "removed", "present", "absent"),
    [
        (
            "none",
            {
                "cf_annotation_valid",
                "cf_is_hazard",
                "cf_hazard_type",
                "cf_quality_present",
                "cf_quality",
                "cf_quality_schema",
                "cf_quality_source",
            },
            set(),
            {"cf_hazard", "cf_hazard_types", "cf_quality"},
        ),
        (
            "hazard_only",
            {"cf_quality_present", "cf_quality", "cf_quality_schema", "cf_quality_source"},
            {"cf_hazard", "cf_hazard_types"},
            {"cf_quality"},
        ),
        (
            "quality_only",
            {"cf_annotation_valid", "cf_is_hazard", "cf_hazard_type"},
            {"cf_quality"},
            {"cf_hazard", "cf_hazard_types"},
        ),
    ],
)
def test_adapter_does_not_require_or_emit_disabled_cf_labels(
    mode: str,
    removed: set[str],
    present: set[str],
    absent: set[str],
) -> None:
    batch = list(_mixed_navsim_batch())
    metadata = dict(batch[-1])
    for key in removed:
        del metadata[key]
    batch[-1] = metadata
    z_observed, z_future = _latents()

    adapted = adapt_navsim_cvoi_field_batch(
        tuple(batch),
        z_observed=z_observed,
        z_future=z_future,
        real_geometry_target_provider=lambda request: torch.ones(2, 3),
        cf_field_supervision=mode,
    )

    assert present.issubset(adapted)
    assert absent.isdisjoint(adapted)


def test_mixed_adapter_requires_explicit_real_geometry_target_provider() -> None:
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="real_geometry_target_provider.*required"):
        adapt_navsim_cvoi_field_batch(
            _mixed_navsim_batch(),
            z_observed=z_observed,
            z_future=z_future,
        )


@pytest.mark.parametrize(
    "invalid_target",
    [
        torch.ones(3, 3),
        torch.tensor([[0.2, float("nan"), 0.4], [0.5, 0.6, 0.7]]),
        torch.tensor([[0.2, 1.2, 0.4], [0.5, 0.6, 0.7]]),
    ],
)
def test_mixed_adapter_rejects_invalid_real_geometry_callback_output(invalid_target: torch.Tensor) -> None:
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="real geometry target provider"):
        adapt_navsim_cvoi_field_batch(
            _mixed_navsim_batch(),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: invalid_target,
        )


def test_mixed_adapter_rejects_domain_geometry_disagreement() -> None:
    batch = list(_mixed_navsim_batch())
    metadata = dict(batch[-1])
    metadata["geometry_present"] = torch.tensor([True, True, True])
    batch[-1] = metadata
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="geometry_present.*domain"):
        adapt_navsim_cvoi_field_batch(
            tuple(batch),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def test_mixed_adapter_rejects_missing_cf_quality() -> None:
    batch = list(_mixed_navsim_batch())
    metadata = dict(batch[-1])
    metadata["cf_quality_present"] = torch.tensor([False, False, False])
    metadata["cf_quality"] = torch.full((3,), float("nan"))
    batch[-1] = metadata
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="counterfactual.*quality"):
        adapt_navsim_cvoi_field_batch(
            tuple(batch),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def test_mixed_adapter_rejects_nonzero_cf_geometry_transport() -> None:
    batch = list(_mixed_navsim_batch())
    boxes = batch[7].clone()
    boxes[1, 0, 0, 0] = 1.0
    batch[7] = boxes
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="counterfactual.*geometry transport"):
        adapt_navsim_cvoi_field_batch(
            tuple(batch),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def test_cf_only_adapter_does_not_require_real_target_provider() -> None:
    batch = list(_mixed_navsim_batch())
    keep = torch.tensor([False, True, False])
    for index in (0, 1, 2, 3, 5, 6, 7, 8, 9):
        batch[index] = batch[index][keep]
    metadata = batch[-1]
    batch[-1] = {
        key: value[keep] if isinstance(value, torch.Tensor) else [value[1]] for key, value in metadata.items()
    }
    z_observed, z_future = _latents()

    adapted = adapt_navsim_cvoi_field_batch(
        tuple(batch),
        z_observed=z_observed[keep],
        z_future=z_future[keep],
    )

    assert adapted["dataset_domains"] == ["counterfactual"]
    assert "real_geometry_targets" not in adapted
    assert adapted["cf_quality"].tolist() == pytest.approx([0.25])


def test_flat_latents_require_tokens_per_frame_to_divide_observed_and_future() -> None:
    with pytest.raises(ValueError, match="z_observed.*tokens_per_frame"):
        adapt_navsim_cvoi_field_batch(
            _mixed_navsim_batch(),
            z_observed=torch.randn(3, 5, 4),
            z_future=torch.randn(3, 6, 4),
            tokens_per_frame=2,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def test_real_geometry_request_requires_states_and_boxes_on_same_timeline() -> None:
    batch = list(_mixed_navsim_batch())
    batch[2] = batch[2][:, :-1]
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="states.*agent_boxes.*time"):
        adapt_navsim_cvoi_field_batch(
            tuple(batch),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def test_real_raw_agent_count_must_match_mask_and_cf_count_must_be_null() -> None:
    batch = list(_mixed_navsim_batch())
    metadata = dict(batch[-1])
    metadata["raw_agent_count"] = [torch.full((4,), 3), None, torch.ones(4, dtype=torch.long)]
    batch[-1] = metadata
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="raw_agent_count.*agent_mask"):
        adapt_navsim_cvoi_field_batch(
            tuple(batch),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def test_cf_quality_requires_canonical_schema_and_sidecar_provenance() -> None:
    batch = list(_mixed_navsim_batch())
    metadata = dict(batch[-1])
    metadata["cf_quality_source"] = [None, "manual_label", None]
    batch[-1] = metadata
    z_observed, z_future = _latents()

    with pytest.raises(ValueError, match="trajectory quality.*source"):
        adapt_navsim_cvoi_field_batch(
            tuple(batch),
            z_observed=z_observed,
            z_future=z_future,
            real_geometry_target_provider=lambda request: torch.ones(2, 3),
        )


def _matched_e120_batch(*, hazard_type: str = "非自车行为引起") -> tuple[object, ...]:
    batch = list(_mixed_navsim_batch())
    metadata = dict(batch[-1])
    metadata["base_scene_id"] = ["scene-b", "scene-b", "scene-c"]
    metadata["window_start_pos"] = torch.tensor([10, 10, 20], dtype=torch.long)
    metadata["cf_hazard_type"] = ["", hazard_type, ""]
    batch[-1] = metadata
    return tuple(batch)


def test_e120_adapter_builds_explicit_matched_real_hazard_pairs_without_relabeling() -> None:
    z_observed, z_future = _latents()

    adapted = adapt_navsim_e120_quality_field_batch(
        _matched_e120_batch(),
        z_observed=z_observed,
        z_future=z_future,
        real_quality_target_provider=lambda request: torch.ones(2, 3),
        cf_field_supervision="hazard_only",
    )

    assert adapted["cf_hazard"].tolist() == [False, True, False]
    assert adapted["cf_hazard_types"] == ["", "非自车行为引起", ""]
    assert adapted["cf_hazard_pair_real_indices"].tolist() == [0]
    assert adapted["cf_hazard_pair_counterfactual_indices"].tolist() == [1]
    assert adapted["cf_hazard_pair_keys"] == [("scene-b", 10)]


def test_e120_adapter_fails_before_provider_when_factual_pair_is_missing() -> None:
    batch = list(_matched_e120_batch())
    metadata = dict(batch[-1])
    metadata["window_start_pos"] = torch.tensor([11, 10, 20], dtype=torch.long)
    batch[-1] = metadata
    provider_called = False

    def provider(_request):
        nonlocal provider_called
        provider_called = True
        return torch.ones(2, 3)

    with pytest.raises(ValueError, match="missing matched real factual sample.*scene-b.*10"):
        adapt_navsim_e120_quality_field_batch(
            tuple(batch),
            z_observed=_latents()[0],
            z_future=_latents()[1],
            real_quality_target_provider=provider,
            cf_field_supervision="hazard_only",
        )
    assert provider_called is False


def test_e120_adapter_rejects_non_allowlisted_hazard_instead_of_treating_it_as_safe() -> None:
    with pytest.raises(ValueError, match="exact accident_type allowlist"):
        adapt_navsim_e120_quality_field_batch(
            _matched_e120_batch(hazard_type="有事故但与自车无关"),
            z_observed=_latents()[0],
            z_future=_latents()[1],
            real_quality_target_provider=lambda request: torch.ones(2, 3),
            cf_field_supervision="hazard_only",
        )
