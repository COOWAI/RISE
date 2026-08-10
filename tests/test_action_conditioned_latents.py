"""Contract tests for fixed-observation action scenario construction."""

import json
import math
from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from app.vjepa_cowa_world_model.evaluation.action_conditioned_latents import (
    SCENARIO_NAMES,
    ActionScenarioConfig,
    InsufficientActionAnchorError,
    build_action_scenarios,
    compute_all_scenario_metrics,
    compute_pairwise_latent_metrics,
    integrate_ego_actions,
)


def _actions(dtype=torch.float32):
    return torch.tensor(
        [
            [[1.0, 2.0, 0.3], [2.0, -1.0, -0.2], [4.0, 8.0, 0.7], [5.0, 9.0, 0.8]],
            [[3.0, 4.0, 0.4], [3.5, -2.0, -0.3], [6.0, 7.0, 0.6], [7.0, 8.0, 0.7]],
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_builds_full_cloned_scenarios_with_fixed_observation_prefix(dtype):
    actions = _actions(dtype).repeat_interleave(2, dim=1)[:, ::2]
    assert not actions.is_contiguous()
    result = build_action_scenarios(actions, 3, ActionScenarioConfig())

    assert type(result) is dict
    assert tuple(result) == SCENARIO_NAMES
    for scenario in result.values():
        assert scenario.shape == actions.shape
        assert torch.equal(scenario[:, :2], actions[:, :2])
        assert scenario.data_ptr() != actions.data_ptr()
        assert scenario.untyped_storage().data_ptr() != actions.untyped_storage().data_ptr()
    assert len({item.untyped_storage().data_ptr() for item in result.values()}) == len(result)
    assert torch.equal(result["factual"], actions)
    original = actions.clone()
    result["brake"][0, 2, 0] = 123
    assert torch.equal(actions, original)


def test_brake_and_accelerate_use_each_batch_anchor_and_monotone_scales():
    actions = _actions()
    result = build_action_scenarios(actions, 3, ActionScenarioConfig())
    brake, accelerate = result["brake"], result["accelerate"]

    assert torch.equal(brake[:, 2:, 1:], torch.zeros_like(brake[:, 2:, 1:]))
    assert torch.equal(accelerate[:, 2:, 1:], torch.zeros_like(accelerate[:, 2:, 1:]))
    assert torch.all(brake[:, 2:, 0].diff(dim=1) <= 0)
    assert torch.all(accelerate[:, 2:, 0].diff(dim=1) >= 0)
    anchor = actions[:, 1, 0]
    assert torch.allclose(brake[:, -1, 0], torch.zeros_like(anchor))
    assert torch.allclose(accelerate[:, -1, 0], anchor * 1.5)


def test_turns_are_exact_mirrors_of_constant_curvature_chord():
    actions = _actions()
    config = ActionScenarioConfig(turn_yaw_radians_per_step=0.2)
    result = build_action_scenarios(actions, 3, config)
    anchor = actions[:, 1, 0]
    expected_x = anchor * math.sin(0.2) / 0.2
    expected_y = anchor * (1.0 - math.cos(0.2)) / 0.2

    left, right = result["left_turn"], result["right_turn"]
    assert torch.allclose(left[:, 2:, 0], expected_x[:, None].expand(-1, 2))
    assert torch.allclose(right[:, 2:, 0], left[:, 2:, 0])
    assert torch.allclose(left[:, 2:, 1], expected_y[:, None].expand(-1, 2))
    assert torch.allclose(right[:, 2:, 1], -left[:, 2:, 1])
    assert torch.allclose(left[:, 2:, 2], torch.full_like(left[:, 2:, 2], 0.2))
    assert torch.allclose(right[:, 2:, 2], -left[:, 2:, 2])


def test_explicit_anchor_is_broadcast_without_minimum_threshold():
    actions = _actions()
    result = build_action_scenarios(
        actions, 3, ActionScenarioConfig(min_anchor_forward_m=100.0, anchor_forward_override_m=0.1)
    )
    assert torch.allclose(result["accelerate"][:, -1, 0], torch.full((2,), 0.15))


def test_inferred_low_anchor_raises_typed_candidate_rejection():
    actions = _actions()
    actions[:, 1, 0] = torch.tensor([0.25, 0.1])

    with pytest.raises(InsufficientActionAnchorError, match="exceed min_anchor_forward_m"):
        build_action_scenarios(actions, 3, ActionScenarioConfig(min_anchor_forward_m=0.25))


def test_single_future_transition_applies_terminal_scales():
    actions = _actions()[:, :2]
    result = build_action_scenarios(actions, 2, ActionScenarioConfig())
    anchor = actions[:, 0, 0]
    assert torch.allclose(result["brake"][:, 1, 0], torch.zeros_like(anchor))
    assert torch.allclose(result["accelerate"][:, 1, 0], anchor * 1.5)


@pytest.mark.parametrize(
    ("actions", "match"),
    [
        ("not-a-tensor", "Tensor"),
        (torch.ones(2, 3), "shape"),
        (torch.ones(1, 2, 2), "shape"),
        (torch.empty(0, 2, 3), "non-empty"),
        (torch.empty(1, 0, 3), "non-empty"),
        (torch.ones(1, 2, 3, dtype=torch.int64), "dtype"),
        (torch.ones(1, 2, 3, dtype=torch.float16), "float32 or torch.float64"),
        (torch.ones(1, 2, 3, dtype=torch.bfloat16), "float32 or torch.float64"),
        (torch.tensor([[[float("nan"), 0.0, 0.0]]]), "finite"),
        (torch.tensor([[[float("inf"), 0.0, 0.0]]]), "finite"),
    ],
)
def test_actions_validation_rejects_invalid_inputs(actions, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build_action_scenarios(actions, 2, ActionScenarioConfig())
    with pytest.raises((TypeError, ValueError), match=match):
        integrate_ego_actions(actions)


@pytest.mark.parametrize("steps", [True, 2.0, 1, 5])
def test_observation_boundary_validation(steps):
    with pytest.raises((TypeError, ValueError)):
        build_action_scenarios(_actions(), steps, ActionScenarioConfig())


@pytest.mark.parametrize(
    "config",
    [
        "wrong",
        replace(ActionScenarioConfig(), min_anchor_forward_m=float("nan")),
        replace(ActionScenarioConfig(), min_anchor_forward_m=True),
        replace(ActionScenarioConfig(), min_anchor_forward_m=-0.1),
        replace(ActionScenarioConfig(), brake_final_scale=float("nan")),
        replace(ActionScenarioConfig(), brake_final_scale=1.0),
        replace(ActionScenarioConfig(), brake_final_scale=-0.1),
        replace(ActionScenarioConfig(), accelerate_final_scale=float("inf")),
        replace(ActionScenarioConfig(), accelerate_final_scale=1.0),
        replace(ActionScenarioConfig(), turn_yaw_radians_per_step=float("nan")),
        replace(ActionScenarioConfig(), turn_yaw_radians_per_step=0.0),
        replace(ActionScenarioConfig(), turn_yaw_radians_per_step=math.pi),
        replace(ActionScenarioConfig(), anchor_forward_override_m=0.0),
        replace(ActionScenarioConfig(), anchor_forward_override_m=True),
    ],
)
def test_config_validation(config):
    with pytest.raises((TypeError, ValueError)):
        build_action_scenarios(_actions(), 3, config)


@pytest.mark.parametrize(
    "config",
    [
        replace(ActionScenarioConfig(), brake_final_scale=Fraction(1, 2)),
        replace(ActionScenarioConfig(), accelerate_final_scale=Fraction(3, 2)),
        replace(ActionScenarioConfig(), turn_yaw_radians_per_step=Fraction(1, 5)),
    ],
)
def test_config_rejects_non_builtin_numeric_scalars_before_tensor_construction(config):
    with pytest.raises(TypeError, match="built-in int or float"):
        build_action_scenarios(_actions(), 3, config)


@pytest.mark.parametrize(
    "config",
    [
        replace(ActionScenarioConfig(), brake_final_scale=0.999999999),
        replace(ActionScenarioConfig(), accelerate_final_scale=1.0 + 1e-9),
    ],
)
def test_float32_rejects_configuration_values_that_quantize_to_invalid_boundaries(config):
    with pytest.raises(ValueError, match="dtype=torch.float32.*cast value"):
        build_action_scenarios(_actions(), 3, config)


def test_float32_rejects_tiny_turn_with_no_representable_lateral_chord():
    config = replace(ActionScenarioConfig(), turn_yaw_radians_per_step=1e-23)
    with pytest.raises(ValueError, match="turn_lateral"):
        build_action_scenarios(_actions(), 3, config)


def test_float32_rejects_scenario_overflow():
    actions = torch.zeros((1, 2, 3), dtype=torch.float32)
    actions[0, 0, 0] = 3e38
    with pytest.raises(ValueError, match="finite"):
        build_action_scenarios(actions, 2, ActionScenarioConfig())


def test_config_rejects_huge_builtin_integer_without_raw_overflow_error():
    config = replace(ActionScenarioConfig(), brake_final_scale=10**400)
    with pytest.raises(ValueError, match="brake_final_scale"):
        build_action_scenarios(_actions(), 3, config)


def test_integration_rejects_non_finite_float32_trajectory():
    actions = torch.zeros((1, 2, 3), dtype=torch.float32)
    actions[..., 0] = 3e38
    with pytest.raises(ValueError, match="integrated trajectory.*non-finite.*torch.float32"):
        integrate_ego_actions(actions)


def test_inferred_anchor_must_exceed_configured_minimum_for_every_batch_item():
    actions = _actions()
    actions[1, 1, 0] = 0.25
    with pytest.raises(ValueError, match="anchor"):
        build_action_scenarios(actions, 3, ActionScenarioConfig(min_anchor_forward_m=0.25))


def test_integrates_ego_actions_wraps_yaw_and_preserves_autograd():
    actions = torch.tensor(
        [[[1.0, 0.0, math.pi / 2], [1.0, 0.0, math.pi / 2], [-1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    poses = integrate_ego_actions(actions)

    expected_final = torch.tensor([2.0, 1.0])
    assert poses.shape == (1, 4, 3)
    assert torch.allclose(poses[0, -1, :2], expected_final, atol=1e-6)
    assert -math.pi <= poses[0, 2, 2].item() <= math.pi
    assert torch.allclose(poses[0, 2, 2], torch.tensor(-math.pi), atol=1e-6)
    poses.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()


def test_scenario_builder_preserves_autograd_with_finite_gradients():
    actions = _actions().requires_grad_()
    scenarios = build_action_scenarios(actions, 3, ActionScenarioConfig())
    sum(scenario.sum() for scenario in scenarios.values()).backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()


def test_float32_long_yaw_integration_has_bounded_circular_error():
    actions = torch.zeros((1, 10_000, 3), dtype=torch.float32)
    actions[..., 2] = 0.001
    final_yaw = integrate_ego_actions(actions)[0, -1, 2]
    expected_yaw = torch.atan2(torch.sin(torch.tensor(10.0)), torch.cos(torch.tensor(10.0)))
    circular_error = torch.atan2(torch.sin(final_yaw - expected_yaw), torch.cos(final_yaw - expected_yaw))
    assert circular_error.abs() < 2e-4


def test_integration_handles_batches_without_mutating_input():
    actions = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]])
    before = actions.clone()
    poses = integrate_ego_actions(actions)
    assert torch.equal(actions, before)
    assert torch.allclose(poses[:, -1, :2], torch.tensor([[1.0, 0.0], [0.0, 2.0]]))


def test_pairwise_latent_metrics_known_values():
    left = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    right = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])

    metrics = compute_pairwise_latent_metrics(left, right)

    expected_l2_mean = (math.sqrt(2.0) + 1.0) / 2.0
    expected_relative_l2 = [math.sqrt(2.0) / (1.0 + 1e-8), 1.0 / (1.5 + 1e-8)]
    assert metrics["future_steps"] == 1
    assert metrics["cosine_mean_per_step"] == [0.5]
    assert metrics["l2_mean_per_step"][0] == pytest.approx(expected_l2_mean, abs=1e-12)
    assert metrics["l2_max_per_step"][0] == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert metrics["relative_l2_mean_per_step"][0] == pytest.approx(sum(expected_relative_l2) / 2.0, abs=1e-12)
    assert metrics["norm_drift_mean_per_step"] == pytest.approx([0.5], abs=1e-12)
    assert metrics["cosine_mean"] == pytest.approx(0.5, abs=1e-12)
    assert metrics["l2_mean"] == pytest.approx(expected_l2_mean, abs=1e-12)
    assert metrics["l2_max"] == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert metrics["relative_l2_mean"] == pytest.approx(sum(expected_relative_l2) / 2.0, abs=1e-12)
    assert metrics["norm_drift_mean"] == pytest.approx(0.5, abs=1e-12)


def test_pairwise_latent_metrics_reduce_batch_and_patch_dimensions_and_are_symmetric():
    left = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]])
    right = torch.zeros_like(left)

    metrics = compute_pairwise_latent_metrics(left, right)
    reverse = compute_pairwise_latent_metrics(right, left)

    assert metrics["cosine_mean_per_step"] == [0.0, 0.0]
    assert metrics["l2_mean_per_step"] == pytest.approx([3.5, 5.5], abs=1e-12)
    assert metrics["l2_max_per_step"] == [6.0, 8.0]
    expected_relative = [
        sum(value / (0.5 * value + 1e-8) for value in (1.0, 2.0, 5.0, 6.0)) / 4.0,
        sum(value / (0.5 * value + 1e-8) for value in (3.0, 4.0, 7.0, 8.0)) / 4.0,
    ]
    assert metrics["relative_l2_mean_per_step"] == pytest.approx(expected_relative, abs=1e-12)
    assert metrics["norm_drift_mean_per_step"] == pytest.approx([3.5, 5.5], abs=1e-12)
    assert metrics == reverse


def test_pairwise_latent_metrics_identical_zero_and_large_low_precision_values_are_finite_without_mutation():
    identical = torch.full((2, 3, 4, 5), 3.0, dtype=torch.float32)
    before = identical.clone()
    metrics = compute_pairwise_latent_metrics(identical, identical)
    assert metrics["cosine_mean"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["l2_mean"] == metrics["l2_max"] == metrics["relative_l2_mean"] == 0.0
    assert metrics["norm_drift_mean"] == 0.0
    assert torch.equal(identical, before)

    zeros = torch.zeros((1, 2, 1, 3), dtype=torch.float16)
    zero_metrics = compute_pairwise_latent_metrics(zeros, zeros)
    assert zero_metrics["cosine_mean_per_step"] == [0.0, 0.0]
    assert all(math.isfinite(value) for value in zero_metrics["relative_l2_mean_per_step"])

    large = torch.full((1, 1, 1, 2), 60_000.0, dtype=torch.float16)
    large_metrics = compute_pairwise_latent_metrics(large, -large)
    assert all(math.isfinite(value) for value in large_metrics.values() if isinstance(value, float))
    assert all(math.isfinite(value) for value in large_metrics["l2_mean_per_step"])


def test_pairwise_latent_metrics_handles_float64_extremes_without_intermediate_overflow():
    identical = torch.full((1, 1, 1, 2), 1e200, dtype=torch.float64)
    identical_metrics = compute_pairwise_latent_metrics(identical, identical)
    assert identical_metrics["cosine_mean"] == pytest.approx(1.0, abs=1e-12)
    assert identical_metrics["l2_mean"] == 0.0
    assert identical_metrics["norm_drift_mean"] == 0.0

    left = torch.full((1, 1, 2, 1), 1e308, dtype=torch.float64)
    right = torch.zeros_like(left)
    metrics = compute_pairwise_latent_metrics(left, right)
    assert metrics["l2_mean_per_step"] == pytest.approx([1e308], rel=1e-12)
    assert metrics["l2_mean"] == pytest.approx(1e308, rel=1e-12)
    assert math.isfinite(metrics["l2_mean"])


def test_pairwise_latent_metrics_preserves_finfo_max_identical_norms_in_scaled_form():
    largest = torch.finfo(torch.float64).max
    identical = torch.full((1, 1, 1, 2), largest, dtype=torch.float64)

    metrics = compute_pairwise_latent_metrics(identical, identical)

    assert metrics["cosine_mean"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["l2_mean"] == 0.0
    assert metrics["relative_l2_mean"] == 0.0
    assert metrics["norm_drift_mean"] == 0.0


def test_pairwise_latent_metrics_reports_representable_huge_norm_drift():
    largest = torch.finfo(torch.float64).max
    left = torch.full((1, 1, 1, 1), largest / 2.0, dtype=torch.float64)
    right = torch.full_like(left, largest / 4.0)

    metrics = compute_pairwise_latent_metrics(left, right)

    assert metrics["norm_drift_mean"] == pytest.approx(largest / 4.0, rel=1e-12)


def test_pairwise_latent_metrics_preserves_small_norm_clamp_formula():
    left = torch.tensor([[[[1e-5]]]], dtype=torch.float64)
    right = torch.tensor([[[[2e-5]]]], dtype=torch.float64)
    metrics = compute_pairwise_latent_metrics(left, right)

    expected_cosine = (1e-5 * 2e-5) / 1e-8
    expected_relative_l2 = 1e-5 / (0.5 * 1e-5 + 0.5 * 2e-5 + 1e-8)
    assert metrics["cosine_mean"] == pytest.approx(expected_cosine, abs=1e-12)
    assert metrics["relative_l2_mean"] == pytest.approx(expected_relative_l2, abs=1e-12)


def test_pairwise_latent_metrics_matches_direct_formula_for_ordinary_random_values():
    generator = torch.Generator().manual_seed(7)
    left = torch.randn((2, 3, 4, 5), generator=generator, dtype=torch.float64)
    right = torch.randn((2, 3, 4, 5), generator=generator, dtype=torch.float64)
    metrics = compute_pairwise_latent_metrics(left, right)

    dot = (left * right).sum(dim=-1)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    cosine = dot / (left_norm * right_norm).clamp_min(1e-8)
    l2 = torch.linalg.vector_norm(left - right, dim=-1)
    relative_l2 = l2 / (0.5 * (left_norm + right_norm) + 1e-8)
    norm_drift = (left_norm - right_norm).abs()

    assert metrics["cosine_mean_per_step"] == pytest.approx(cosine.mean(dim=(0, 2)).tolist(), abs=1e-12)
    assert metrics["l2_mean_per_step"] == pytest.approx(l2.mean(dim=(0, 2)).tolist(), abs=1e-12)
    assert metrics["l2_max_per_step"] == pytest.approx(l2.amax(dim=(0, 2)).tolist(), abs=1e-12)
    assert metrics["relative_l2_mean_per_step"] == pytest.approx(relative_l2.mean(dim=(0, 2)).tolist(), abs=1e-12)
    assert metrics["norm_drift_mean_per_step"] == pytest.approx(norm_drift.mean(dim=(0, 2)).tolist(), abs=1e-12)
    assert metrics["cosine_mean"] == pytest.approx(cosine.mean().item(), abs=1e-12)
    assert metrics["l2_mean"] == pytest.approx(l2.mean().item(), abs=1e-12)
    assert metrics["l2_max"] == pytest.approx(l2.max().item(), abs=1e-12)
    assert metrics["relative_l2_mean"] == pytest.approx(relative_l2.mean().item(), abs=1e-12)
    assert metrics["norm_drift_mean"] == pytest.approx(norm_drift.mean().item(), abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pairwise_latent_metrics_casts_low_precision_inputs_before_multiplication(dtype):
    left = torch.tensor([[[[300.0, 400.0]]]], dtype=dtype)
    right = torch.tensor([[[[0.0, 500.0]]]], dtype=dtype)
    metrics = compute_pairwise_latent_metrics(left, right)

    assert metrics["cosine_mean"] == pytest.approx(0.8, abs=1e-12)
    assert metrics["l2_mean"] == pytest.approx(math.sqrt(100_000.0), abs=1e-12)
    assert metrics["relative_l2_mean"] == pytest.approx(math.sqrt(100_000.0) / (500.0 + 1e-8), abs=1e-12)
    assert metrics["norm_drift_mean"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_pairwise_latent_metrics_returns_python_known_values_on_cuda(dtype):
    left = torch.tensor([[[[3.0, 4.0]]]], dtype=dtype, device="cuda")
    right = torch.tensor([[[[0.0, 5.0]]]], dtype=dtype, device="cuda")
    metrics = compute_pairwise_latent_metrics(left, right)

    assert metrics["cosine_mean"] == pytest.approx(0.8, abs=1e-12)
    assert metrics["l2_mean"] == pytest.approx(math.sqrt(10.0), abs=1e-12)
    assert metrics["relative_l2_mean"] == pytest.approx(math.sqrt(10.0) / (5.0 + 1e-8), abs=1e-12)
    assert metrics["norm_drift_mean"] == 0.0
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


@pytest.mark.parametrize(
    "left,right,match",
    [
        ("not-a-tensor", torch.ones(1, 1, 1, 1), "left must be a torch Tensor"),
        (torch.ones(1, 1, 1), torch.ones(1, 1, 1), r"shape \[B, F, P, D\]"),
        (torch.empty(0, 1, 1, 1), torch.empty(0, 1, 1, 1), "non-empty"),
        (torch.empty(1, 0, 1, 1), torch.empty(1, 0, 1, 1), "non-empty"),
        (torch.empty(1, 1, 0, 1), torch.empty(1, 1, 0, 1), "non-empty"),
        (torch.empty(1, 1, 1, 0), torch.empty(1, 1, 1, 0), "non-empty"),
        (torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 2), "same shape"),
        (torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1, dtype=torch.float64), "same dtype"),
        (torch.ones(1, 1, 1, 1, dtype=torch.int64), torch.ones(1, 1, 1, 1, dtype=torch.int64), "dtype"),
        (torch.ones(1, 1, 1, 1, dtype=torch.complex64), torch.ones(1, 1, 1, 1, dtype=torch.complex64), "dtype"),
        (torch.tensor([[[[float("nan")]]]]), torch.ones(1, 1, 1, 1), "finite"),
        (torch.ones(1, 1, 1, 1), torch.tensor([[[[float("inf")]]]]), "finite"),
    ],
)
def test_pairwise_latent_metrics_rejects_invalid_inputs(left, right, match):
    with pytest.raises((TypeError, ValueError), match=match):
        compute_pairwise_latent_metrics(left, right)


def test_pairwise_latent_metrics_rejects_non_strided_layout():
    sparse = torch.sparse_coo_tensor(indices=[[0], [0], [0], [0]], values=[1.0], size=(1, 1, 1, 1))
    with pytest.raises(TypeError, match="strided"):
        compute_pairwise_latent_metrics(sparse, sparse)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to construct tensors on distinct devices")
def test_pairwise_latent_metrics_rejects_device_mismatch():
    with pytest.raises(ValueError, match="same device"):
        compute_pairwise_latent_metrics(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1, device="cuda"))


def test_pairwise_latent_metrics_detaches_autograd_and_returns_json_safe_python_values():
    left = torch.ones((1, 2, 1, 2), requires_grad=True)
    right = torch.zeros_like(left, requires_grad=True)
    metrics = compute_pairwise_latent_metrics(left, right)

    assert left.grad is None
    assert right.grad is None
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())
    json.dumps(metrics)


def test_all_scenario_metrics_uses_exact_keys_order_and_flattened_pair_metrics():
    latents = {name: torch.full((1, 2, 1, 1), float(index + 1)) for index, name in enumerate(reversed(SCENARIO_NAMES))}
    records = compute_all_scenario_metrics(latents)

    assert len(records) == 10
    assert [(record["scenario_a"], record["scenario_b"]) for record in records] == [
        ("factual", "brake"),
        ("factual", "left_turn"),
        ("factual", "right_turn"),
        ("factual", "accelerate"),
        ("brake", "left_turn"),
        ("brake", "right_turn"),
        ("brake", "accelerate"),
        ("left_turn", "right_turn"),
        ("left_turn", "accelerate"),
        ("right_turn", "accelerate"),
    ]
    direct = compute_pairwise_latent_metrics(latents["factual"], latents["brake"])
    assert records[0] == {"scenario_a": "factual", "scenario_b": "brake", **direct}
    json.dumps(records)


@pytest.mark.parametrize(
    "latents,match",
    [
        ("not-a-mapping", "Mapping"),
        ({name: torch.ones(1, 1, 1, 1) for name in SCENARIO_NAMES[:-1]}, "missing"),
        ({**{name: torch.ones(1, 1, 1, 1) for name in SCENARIO_NAMES}, "extra": torch.ones(1, 1, 1, 1)}, "extra"),
        ({name: torch.ones(1, 1, 1, 1) for name in SCENARIO_NAMES} | {"brake": torch.ones(1, 1, 1, 2)}, "same shape"),
    ],
)
def test_all_scenario_metrics_rejects_invalid_mapping_contract(latents, match):
    with pytest.raises((TypeError, ValueError), match=match):
        compute_all_scenario_metrics(latents)
