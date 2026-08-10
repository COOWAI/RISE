"""Real-consumer integration tests for diffusion refinement and proposals."""

from __future__ import annotations

import torch

from app.vjepa_cowa_world_model.models.diffusion_refinement_decoder import DiffusionRefinementDecoder
from app.vjepa_cowa_world_model.models.proposal_providers import DiffusionProposalProvider

BATCH_SIZE = 2
PROVIDER_KWARGS = {
    "encoder_dim": 8,
    "num_poses": 3,
    "status_dim": 4,
    "hidden_dim": 16,
    "depth": 1,
    "heads": 4,
    "dropout": 0.0,
    "mlp_ratio": 2.0,
    "traj_dim": 4,
    "sde_beta_min": 0.1,
    "sde_beta_max": 20.0,
    "num_samples": 2,
    "inference_steps": 2,
    "tokens_per_frame": 3,
    "num_modes": 2,
    "use_last_frame_only": False,
    "independent_modes": False,
    "use_anchor_frame": False,
    "cls_loss_weight": 1.0,
    "reg_loss_weight": 1.0,
    "vel_loss_weight": 0.5,
    "yaw_loss_weight": 0.5,
    "conf_temperature": 1.5,
    "cls_th": 2.0,
    "cls_ignore": 0.2,
    "command_dim": 0,
    "trajectory_token_mode": "per_pose_token",
    "adaln_version": "v2",
    "mode_token_expansion": False,
    "proposal_hidden_dim": 8,
    "use_action_history": False,
    "action_history_dim": 3,
    "num_observed_frames": 1,
    "observed_token_mode": "none",
    "use_z_context": True,
}


def test_diffusion_refinement_executes_two_real_rounds_without_an_observed_stream() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3301)
        decoder = DiffusionRefinementDecoder(
            encoder_dim=8,
            hidden_dim=16,
            depth=1,
            heads=4,
            dropout=0.0,
            mlp_ratio=2.0,
            num_poses=3,
            status_dim=4,
            traj_dim=4,
            num_modes=2,
            tokens_per_frame=3,
            trajectory_token_mode="per_pose_token",
            adaln_version="v2",
            observed_token_mode="none",
        ).eval()
        z_context = torch.randn(BATCH_SIZE, 6, 8)
        status = torch.randn(BATCH_SIZE, 4)
        proposal = torch.randn(BATCH_SIZE, 2, 3, 3)
        proposal_logits = torch.randn(BATCH_SIZE, 2)

    with torch.no_grad():
        refinement_rounds, final_trajectory = decoder.forward_iterative(
            z_context=z_context,
            status_feature=status,
            proposal_traj=proposal,
            proposal_logits=proposal_logits,
            proposal_features=None,
            predictor_rollout_fn=None,
            num_rounds=2,
        )

    assert decoder.refine_core.observed_token_mode == "none"
    assert len(refinement_rounds) == 2
    for round_output in refinement_rounds:
        assert round_output["trajectories"].shape == (BATCH_SIZE, 2, 3, 3)
        assert round_output["confidences"].shape == (BATCH_SIZE, 2)
        assert torch.isfinite(round_output["trajectories"]).all()
        assert torch.isfinite(round_output["confidences"]).all()
    assert final_trajectory is not None
    assert final_trajectory.shape == (BATCH_SIZE, 3, 3)
    assert torch.isfinite(final_trajectory).all()


def test_diffusion_proposal_provider_forwards_explicit_none_and_returns_real_features() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3302)
        provider = DiffusionProposalProvider(**PROVIDER_KWARGS).eval()
        z_context = torch.randn(BATCH_SIZE, 6, 8)
        status = torch.randn(BATCH_SIZE, 4)
        inference_noise = torch.randn(BATCH_SIZE, 2, 3, 4)

    with torch.no_grad():
        output = provider(
            z_context=z_context,
            status_feature=status,
            history_traj=None,
            inference_noise=inference_noise,
        )

    assert provider.core.observed_token_mode == "none"
    assert output["trajectories"].shape == (BATCH_SIZE, 2, 3, 3)
    assert output["confidences"].shape == (BATCH_SIZE, 2)
    assert output["proposal_features"].shape == (BATCH_SIZE, 2, 3, 8)
    assert torch.isfinite(output["trajectories"]).all()
    assert torch.isfinite(output["confidences"]).all()
    assert torch.isfinite(output["proposal_features"]).all()


def test_diffusion_proposal_provider_preserves_legacy_positional_use_z_context_argument() -> None:
    legacy_positional_args = (
        8,
        3,
        4,
        16,
        1,
        4,
        0.0,
        2.0,
        4,
        0.1,
        20.0,
        2,
        2,
        3,
        2,
        False,
        False,
        False,
        1.0,
        1.0,
        0.5,
        0.5,
        1.5,
        2.0,
        0.2,
        0,
        "per_pose_token",
        "v2",
        False,
        8,
        False,
        3,
        1,
        False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3303)
        provider = DiffusionProposalProvider(*legacy_positional_args).eval()
        z_context = torch.randn(BATCH_SIZE, 6, 8)
        status = torch.randn(BATCH_SIZE, 4)
        inference_noise = torch.randn(BATCH_SIZE, 2, 3, 4)

    assert provider.core.use_z_context is False
    assert provider.core.observed_token_mode == "none"

    with torch.no_grad():
        output = provider(
            z_context=z_context,
            status_feature=status,
            history_traj=None,
            inference_noise=inference_noise,
        )

    assert output["trajectories"].shape == (BATCH_SIZE, 2, 3, 3)
    assert output["confidences"].shape == (BATCH_SIZE, 2)
    assert torch.isfinite(output["trajectories"]).all()
    assert torch.isfinite(output["confidences"]).all()
