import inspect

import pytest
import torch

from app.vjepa_cowa_world_model.training.runtimes import sequential_rollout_runtime as rollout_runtime_module
from app.vjepa_cowa_world_model.training.runtimes.sequential_rollout_runtime import run_sequential_rollout


class _Gate(torch.nn.Module):
    def __init__(self, deltas, *, max_horizon=3):
        super().__init__()
        self.deltas = list(deltas)
        self.feature_dim = 11
        self.max_horizon = max_horizon

    def forward(self, features):
        horizon = int(round(float(features[0, 7] * self.max_horizon)))
        return torch.tensor([self.deltas[horizon]], device=features.device)


def _value_features(observed, prefix, horizon):
    del observed, prefix
    return {"field_value": float(horizon), "stop_value": 0.1 * horizon}


def test_runtime_defers_predictor_token_finite_reduction_until_terminal_stop() -> None:
    source = inspect.getsource(rollout_runtime_module.run_sequential_rollout)

    assert "or not torch.isfinite(next_tokens).all()" not in source
    assert "require_finite_rollout_tokens" not in source


def test_runtime_exposes_strict_nonfinite_predictor_failure_at_caller_boundary() -> None:
    stop_calls = []

    result = run_sequential_rollout(
        observed_latent=torch.ones(1, 1, 2),
        gate=_Gate([1.0]),
        max_horizon=1,
        lambda_compute=0.05,
        compute_costs=[0.0, 1.0],
        rollout_step=lambda prefix, horizon: torch.full((1, 1, 2), float("nan")),
        value_features=_value_features,
        stop_and_plan=lambda prefix, horizon, apply_guidance: stop_calls.append(horizon),
    )

    with pytest.raises(ValueError, match="rollout_step.*finite"):
        result.require_finite_rollout_tokens()

    assert stop_calls == [1]


def test_runtime_stop_at_h0_skips_rollout_and_guidance():
    calls = {"rollout": 0, "stop": []}

    def rollout_step(prefix, next_horizon):
        calls["rollout"] += 1
        return torch.ones(1, 2, 2) * next_horizon

    def stop_and_plan(prefix, horizon, apply_guidance):
        calls["stop"].append((horizon, apply_guidance, prefix.shape[1]))
        return "planned"

    result = run_sequential_rollout(
        observed_latent=torch.ones(1, 2, 2),
        gate=_Gate([-0.1]),
        max_horizon=3,
        lambda_compute=0.05,
        compute_costs=[0.0, 0.4, 0.7, 1.0],
        rollout_step=rollout_step,
        value_features=_value_features,
        stop_and_plan=stop_and_plan,
    )

    assert result.stop_horizon == 0
    assert calls == {"rollout": 0, "stop": [(0, False, 0)]}


def test_runtime_rolls_raw_prefix_then_guides_only_after_stop():
    prefixes_seen = []
    stop_calls = []

    def rollout_step(prefix, next_horizon):
        prefixes_seen.append(prefix.clone())
        return torch.full((1, 2, 2), float(next_horizon))

    def stop_and_plan(prefix, horizon, apply_guidance):
        stop_calls.append((prefix.clone(), horizon, apply_guidance))
        return {"horizon": horizon}

    result = run_sequential_rollout(
        observed_latent=torch.ones(1, 2, 2),
        gate=_Gate([0.2, 0.1, -0.1]),
        max_horizon=3,
        lambda_compute=0.05,
        compute_costs=[0.0, 0.4, 0.7, 1.0],
        rollout_step=rollout_step,
        value_features=_value_features,
        stop_and_plan=stop_and_plan,
    )

    assert result.stop_horizon == 2
    assert len(prefixes_seen) == 2
    assert torch.equal(prefixes_seen[1], torch.ones(1, 2, 2))
    assert stop_calls[0][1:] == (2, True)
    assert stop_calls[0][0].shape == (1, 4, 2)


def test_runtime_forces_stop_at_h_without_querying_extra_rollout():
    calls = {"rollout": 0, "value_horizons": []}

    def rollout_step(prefix, next_horizon):
        calls["rollout"] += 1
        return torch.full((1, 1, 2), float(next_horizon))

    def value_features(observed, prefix, horizon):
        calls["value_horizons"].append(horizon)
        return _value_features(observed, prefix, horizon)

    result = run_sequential_rollout(
        observed_latent=torch.ones(1, 1, 2),
        gate=_Gate([1.0, 1.0, 1.0, 1.0]),
        max_horizon=3,
        lambda_compute=0.05,
        compute_costs=[0.0, 0.3, 0.6, 1.0],
        rollout_step=rollout_step,
        value_features=value_features,
        stop_and_plan=lambda prefix, horizon, apply_guidance: horizon,
    )

    assert result.stop_horizon == 3
    assert calls["rollout"] == 3
    assert calls["value_horizons"] == [0, 1, 2]


def test_runtime_requires_batch_size_one():
    try:
        run_sequential_rollout(
            observed_latent=torch.ones(2, 1, 2),
            gate=_Gate([-1.0]),
            max_horizon=3,
            lambda_compute=0.05,
            compute_costs=[0.0, 0.3, 0.6, 1.0],
            rollout_step=lambda prefix, horizon: prefix,
            value_features=_value_features,
            stop_and_plan=lambda prefix, horizon, apply_guidance: None,
        )
    except ValueError as exc:
        assert "batch size 1" in str(exc)
    else:
        raise AssertionError("batch size >1 must fail")


def test_runtime_rejects_nonzero_h0_field_sentinel():
    with pytest.raises(ValueError, match="h=0 field_value"):
        run_sequential_rollout(
            observed_latent=torch.ones(1, 1, 2),
            gate=_Gate([-1.0]),
            max_horizon=3,
            lambda_compute=0.05,
            compute_costs=[0.0, 0.3, 0.6, 1.0],
            rollout_step=lambda prefix, horizon: torch.ones(1, 1, 2),
            value_features=lambda observed, prefix, horizon: {"field_value": 1.0, "stop_value": 0.0},
            stop_and_plan=lambda prefix, horizon, apply_guidance: None,
        )


def test_runtime_uses_navsim_e120_gate_mask_and_masks_stop_level_and_slope():
    class RecordingGate(_Gate):
        def __init__(self):
            super().__init__([1.0, -1.0], max_horizon=4)
            self.features_seen = []

        def forward(self, features):
            self.features_seen.append(features.clone())
            return super().forward(features)

    gate = RecordingGate()
    result = run_sequential_rollout(
        observed_latent=torch.ones(1, 1, 2),
        gate=gate,
        max_horizon=4,
        lambda_compute=0.005,
        compute_costs=[0.0, 0.25, 0.5, 0.75, 1.0],
        rollout_step=lambda prefix, horizon: torch.ones(1, 1, 2),
        value_features=_value_features,
        stop_and_plan=lambda prefix, horizon, apply_guidance: horizon,
        gate_feature_mode="without_stop",
        gate_feature_protocol="formal_v2_navsim_e120_h4_v3",
    )

    assert result.stop_horizon == 1
    scalar_start = 4
    assert float(gate.features_seen[1][0, scalar_start]) == pytest.approx(1.0)
    assert torch.equal(gate.features_seen[1][0, scalar_start + 1 : scalar_start + 3], torch.zeros(2))


def test_runtime_navsim_e120_value_summary_protocol_masks_all_value_scalars() -> None:
    class RecordingGate(_Gate):
        def __init__(self):
            super().__init__([1.0, -1.0], max_horizon=4)
            self.features_seen = []

        def forward(self, features):
            self.features_seen.append(features.clone())
            return super().forward(features)

    gate = RecordingGate()
    result = run_sequential_rollout(
        observed_latent=torch.ones(1, 1, 2),
        gate=gate,
        max_horizon=4,
        lambda_compute=0.005,
        compute_costs=[0.0, 0.25, 0.5, 0.75, 1.0],
        rollout_step=lambda prefix, horizon: torch.ones(1, 1, 2),
        value_features=_value_features,
        stop_and_plan=lambda prefix, horizon, apply_guidance: horizon,
        gate_feature_mode="without_value_summary",
        gate_feature_protocol="formal_v2_navsim_e120_h4_v3",
    )

    assert result.stop_horizon == 1
    assert torch.equal(gate.features_seen[1][0, 4:7], torch.zeros(3))
    assert torch.equal(gate.features_seen[1][0, :4], torch.ones(4))
    assert torch.equal(
        gate.features_seen[1][0, 7:],
        torch.tensor([0.25, 0.25, 0.5, 0.005]),
    )


def test_runtime_rejects_retired_generic_formal_v2_protocol() -> None:
    with pytest.raises(ValueError, match="gate_feature_protocol"):
        run_sequential_rollout(
            observed_latent=torch.ones(1, 1, 2),
            gate=_Gate([-1.0], max_horizon=4),
            max_horizon=4,
            lambda_compute=0.005,
            compute_costs=[0.0, 0.25, 0.5, 0.75, 1.0],
            rollout_step=lambda prefix, horizon: torch.ones(1, 1, 2),
            value_features=_value_features,
            stop_and_plan=lambda prefix, horizon, apply_guidance: horizon,
            gate_feature_mode="full",
            gate_feature_protocol="formal_v2",
        )
