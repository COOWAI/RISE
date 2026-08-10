"""Focused contract tests for the NavSim-e120 H0--H4 migration."""

import pytest
import torch

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    FormalV2NavSimE120HorizonExposureState,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_MAX_HORIZON,
    FORMAL_V2_NAVSIM_P0_POLICIES,
)
from app.vjepa_cowa_world_model.training.runtimes.sequential_rollout_runtime import run_sequential_rollout


def test_navsim_e120_protocol_has_five_horizons_and_h4_selection() -> None:
    assert FORMAL_V2_NAVSIM_MAX_HORIZON == 4
    assert dict(FORMAL_V2_NAVSIM_P0_POLICIES) == {
        "uniform": (0.2, 0.2, 0.2, 0.2, 0.2),
        "extremes": (0.5, 0.0, 0.0, 0.0, 0.5),
        "short_heavy": (0.225, 0.225, 0.225, 0.225, 0.1),
        "no_full": (0.25, 0.25, 0.25, 0.25, 0.0),
    }


def test_navsim_e120_exposure_state_tracks_exactly_five_bins() -> None:
    state = FormalV2NavSimE120HorizonExposureState()
    state.record(horizon=4, batch_size=3)
    assert state.snapshot(device=torch.device("cpu")) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 3}
    with pytest.raises(ValueError, match=r"\[0, 1, 2, 3, 4\]"):
        state.record(horizon=5, batch_size=1)


def test_sequential_runtime_accepts_profile_specific_h4_without_changing_legacy_default() -> None:
    class AlwaysRoll(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features.new_ones((1,))

    result = run_sequential_rollout(
        observed_latent=torch.zeros(1, 1, 2),
        gate=AlwaysRoll(),
        max_horizon=4,
        lambda_compute=0.0,
        compute_costs=[0.0, 1.0, 2.0, 3.0, 4.0],
        rollout_step=lambda raw_prefix, horizon: torch.full((1, 1, 2), float(horizon)),
        value_features=lambda observed, raw_prefix, horizon: {
            "field_value": 0.0 if horizon == 0 else float(horizon),
            "stop_value": float(horizon),
        },
        stop_and_plan=lambda raw_prefix, horizon, guided: (horizon, guided),
    )
    assert result.stop_horizon == 4
    assert result.decisions == ["ROLL", "ROLL", "ROLL", "ROLL", "STOP"]
