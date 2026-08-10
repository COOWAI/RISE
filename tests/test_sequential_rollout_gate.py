import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixValueOutput
from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    CVOI_FORMAL_V2_GATE_FEATURE_MODES,
    CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
    CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES,
    SequentialRolloutGate,
    apply_cvoi_formal_v2_gate_feature_mask,
    apply_cvoi_formal_v2_navsim_e120_gate_feature_mask,
    apply_cvoi_gate_feature_mask,
    build_sequential_gate_features,
    compute_cvoi_targets,
    extract_prefix_gate_values,
    sequential_gate_loss,
)


def test_cvoi_targets_encode_roll_roll_stop_curve():
    targets = compute_cvoi_targets(
        task_scores=torch.tensor([[0.62, 0.74, 0.78, 0.74]]),
        compute_costs=torch.zeros(1, 4),
        lambda_compute=torch.tensor([0.0]),
    )

    assert torch.allclose(targets.delta_utility, torch.tensor([[0.16, 0.04, -0.04]]), atol=1e-6)
    assert torch.equal(targets.continue_target, torch.tensor([[True, True, False]]))


def test_compute_penalty_can_move_stop_decision_to_h0():
    targets = compute_cvoi_targets(
        task_scores=torch.tensor([[0.50, 0.60, 0.65]]),
        compute_costs=torch.tensor([[0.0, 0.5, 1.0]]),
        lambda_compute=torch.tensor([0.5]),
    )

    assert targets.delta_utility[0, 0] < 0
    assert not bool(targets.continue_target[0, 0])


def test_gate_features_have_only_online_values_and_stable_shape():
    features = build_sequential_gate_features(
        pooled_observed=torch.tensor([[1.0, 2.0]]),
        pooled_prefix=torch.tensor([[3.0, 4.0]]),
        field_value=torch.tensor([0.4]),
        stop_value=torch.tensor([0.5]),
        previous_stop_value=torch.tensor([0.3]),
        horizon=torch.tensor([2]),
        max_horizon=3,
        current_cost=torch.tensor([0.6]),
        next_cost=torch.tensor([1.0]),
        lambda_compute=torch.tensor([0.05]),
    )

    assert features.shape == (1, 11)
    assert torch.allclose(features[0, 4:], torch.tensor([0.4, 0.5, 0.2, 2.0 / 3.0, 0.6, 1.0, 0.05]))


def test_gate_feature_masks_preserve_capacity_and_remove_only_registered_value_columns():
    features = torch.arange(22, dtype=torch.float32).reshape(2, 11)

    full = apply_cvoi_gate_feature_mask(features, latent_dim=2, mode="full")
    no_stop = apply_cvoi_gate_feature_mask(features, latent_dim=2, mode="gate_no_stop_value")
    no_summary = apply_cvoi_gate_feature_mask(
        features,
        latent_dim=2,
        mode="gate_no_explicit_value_summary",
    )

    assert full.shape == no_stop.shape == no_summary.shape == features.shape
    assert torch.equal(full, features)
    assert torch.equal(no_stop[:, 5], torch.zeros(2))
    assert torch.equal(no_stop[:, :5], features[:, :5])
    assert torch.equal(no_stop[:, 6:], features[:, 6:])
    assert torch.equal(no_summary[:, 4:7], torch.zeros(2, 3))
    assert torch.equal(no_summary[:, :4], features[:, :4])
    assert torch.equal(no_summary[:, 7:], features[:, 7:])


def test_gate_feature_mask_rejects_unknown_mode_and_dimension_drift():
    with pytest.raises(ValueError, match="gate feature mode"):
        apply_cvoi_gate_feature_mask(torch.zeros(1, 11), latent_dim=2, mode="small")
    with pytest.raises(ValueError, match="feature schema"):
        apply_cvoi_gate_feature_mask(torch.zeros(1, 10), latent_dim=2, mode="full")


def test_formal_v2_gate_feature_modes_mask_only_the_declared_value_columns():
    features = torch.arange(22, dtype=torch.float32).reshape(2, 11)

    full = apply_cvoi_formal_v2_gate_feature_mask(features, latent_dim=2, mode="full")
    without_field = apply_cvoi_formal_v2_gate_feature_mask(features, latent_dim=2, mode="without_field")
    without_stop = apply_cvoi_formal_v2_gate_feature_mask(features, latent_dim=2, mode="without_stop")

    assert CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA == "cvoi_gate_online_features_v2"
    assert CVOI_FORMAL_V2_GATE_FEATURE_MODES == frozenset({"full", "without_field", "without_stop"})
    assert torch.equal(full, features)

    # without_field removes only Field; Stop level and Stop slope remain available.
    assert torch.equal(without_field[:, 4], torch.zeros(2))
    assert torch.equal(without_field[:, :4], features[:, :4])
    assert torch.equal(without_field[:, 5:], features[:, 5:])

    # without_stop removes both Stop level and its slope, while retaining Field.
    assert torch.equal(without_stop[:, 4], features[:, 4])
    assert torch.equal(without_stop[:, 5:7], torch.zeros(2, 2))
    assert torch.equal(without_stop[:, :5], features[:, :5])
    assert torch.equal(without_stop[:, 7:], features[:, 7:])

    # Both ablations retain horizon, compute costs, and the conditional lambda.
    assert torch.equal(without_field[:, 7:], features[:, 7:])
    assert torch.equal(without_stop[:, 7:], features[:, 7:])


@pytest.mark.parametrize("mode", ["gate_no_stop_value", "gate_no_explicit_value_summary", "unknown"])
def test_formal_v2_gate_feature_mask_rejects_legacy_and_unknown_modes(mode: str):
    with pytest.raises(ValueError, match="Formal v2 Gate feature mode"):
        apply_cvoi_formal_v2_gate_feature_mask(torch.zeros(1, 11), latent_dim=2, mode=mode)


def test_navsim_e120_value_summary_mask_removes_only_three_scalar_columns() -> None:
    features = torch.arange(22, dtype=torch.float32).reshape(2, 11)

    masked = apply_cvoi_formal_v2_navsim_e120_gate_feature_mask(
        features,
        latent_dim=2,
        mode="without_value_summary",
    )

    assert CVOI_FORMAL_V2_GATE_FEATURE_MODES == frozenset({"full", "without_field", "without_stop"})
    assert CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES == frozenset(
        {"full", "without_field", "without_stop", "without_value_summary"}
    )
    assert torch.equal(masked[:, 4:7], torch.zeros(2, 3))
    assert torch.equal(masked[:, :4], features[:, :4])
    assert torch.equal(masked[:, 7:], features[:, 7:])
    with pytest.raises(ValueError, match="Formal v2 Gate feature mode"):
        apply_cvoi_formal_v2_gate_feature_mask(
            features,
            latent_dim=2,
            mode="without_value_summary",
        )


def test_gate_uses_delta_sign_and_forces_stop_at_max_horizon():
    assert SequentialRolloutGate.should_roll(torch.tensor([0.01]), horizon=1, max_horizon=3).item()
    assert not SequentialRolloutGate.should_roll(torch.tensor([0.0]), horizon=1, max_horizon=3).item()
    assert not SequentialRolloutGate.should_roll(torch.tensor([1.0]), horizon=3, max_horizon=3).item()


def test_h0_gate_value_semantics_use_zero_field_and_observed_stop():
    values = extract_prefix_gate_values(
        PrefixValueOutput(
            field_values=torch.empty(2, 0),
            stop_values=torch.tensor([[0.3], [0.7]]),
        )
    )

    assert torch.equal(values["field_value"], torch.zeros(2))
    assert torch.allclose(values["stop_value"], torch.tensor([0.3, 0.7]))
    assert torch.equal(values["previous_stop_value"], values["stop_value"])


def test_nonzero_prefix_gate_values_use_last_field_and_stop_slope():
    values = extract_prefix_gate_values(
        PrefixValueOutput(
            field_values=torch.tensor([[0.1, 0.4]]),
            stop_values=torch.tensor([[0.2, 0.3, 0.6]]),
        )
    )

    assert values["field_value"].item() == pytest.approx(0.4)
    assert values["stop_value"].item() == pytest.approx(0.6)
    assert values["previous_stop_value"].item() == pytest.approx(0.3)


def test_sequential_gate_loss_backpropagates():
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    features = torch.randn(3, gate.feature_dim)
    predicted = gate(features)
    result = sequential_gate_loss(
        predicted,
        target_delta=torch.tensor([0.2, -0.1, 0.0]),
        continue_target=torch.tensor([True, False, False]),
    )

    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None for parameter in gate.parameters())


def test_gate_predicted_delta_is_nonincreasing_in_compute_penalty():
    torch.manual_seed(0)
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    shared_state = torch.randn(128, gate.feature_dim)
    low_penalty = shared_state.clone()
    high_penalty = shared_state.clone()
    low_penalty[:, -1] = 0.0
    high_penalty[:, -1] = 2.0

    low_delta = gate(low_penalty)
    high_delta = gate(high_penalty)

    assert torch.all(high_delta <= low_delta)


def test_cvoi_targets_reject_non_monotonic_cost():
    try:
        compute_cvoi_targets(
            task_scores=torch.tensor([[0.5, 0.6, 0.7]]),
            compute_costs=torch.tensor([[0.0, 1.0, 0.5]]),
            lambda_compute=torch.tensor([0.1]),
        )
    except ValueError as exc:
        assert "non-decreasing" in str(exc)
    else:
        raise AssertionError("non-monotonic compute cost must fail")
