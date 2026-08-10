"""Parity test for the shared predictor autoregressive rollout.

Guards that predictor_autoregressive_rollout reproduces exactly the inline rollout that
val_command and viz run_inference previously duplicated, for both inference-consistent and
teacher-forced init. Pure tensor logic with a deterministic mock step (no model/GPU).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402
import torch  # noqa: E402

from app.vjepa_cowa_world_model.training.predictor_stepping import (  # noqa: E402
    predictor_autoregressive_rollout,
    rollout_latent_predictions,
    validate_empty_future_planner_conditions,
)


def _mock_step(tpf):
    # Deterministic next-frame: depends on the current rollout length so each step differs.
    def step(_z, actions, states, extrinsics):
        nxt = _z[:, -tpf:] + float(_z.size(1))
        return nxt  # caller takes [:, -tpf:], which is the whole thing here

    return step


def _inline(step, z, actions, states, extrinsics, num_obs, tpf, num_total, ic, z_tf, rollout_end_step=None):
    rollout_end = num_total if rollout_end_step is None else rollout_end_step
    if ic:
        _z = z[:, : num_obs * tpf]
        start = num_obs
    else:
        _z = torch.cat([z[:, :tpf], z_tf[:, :tpf]], dim=1)
        start = 2
    for k in range(start, rollout_end):
        if k == num_total - 1:
            a, s, e = actions, states[:, :-1], extrinsics[:, :-1]
        else:
            a, s, e = actions[:, :k], states[:, :k], extrinsics[:, :k]
        nxt = step(_z, a, s, e)[:, -tpf:]
        _z = torch.cat([_z, nxt], dim=1)
    return _z[:, num_obs * tpf :] if ic else _z[:, tpf:]


def _inputs(num_total, tpf, D=4, B=2):
    z = torch.randn(B, num_total * tpf, D)
    actions = torch.randn(B, num_total - 1, 3)
    states = torch.randn(B, num_total, 7)
    extrinsics = torch.randn(B, num_total, 7)
    z_tf = torch.randn(B, (num_total - 1) * tpf, D)
    return z, actions, states, extrinsics, z_tf


def test_matches_inline_inference_consistent():
    num_total, tpf, num_obs = 6, 3, 2
    z, actions, states, extrinsics, z_tf = _inputs(num_total, tpf)
    step = _mock_step(tpf)
    out = predictor_autoregressive_rollout(
        step,
        z,
        actions,
        states,
        extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tpf,
        num_total=num_total,
        predictor_inference_consistent=True,
    )
    exp = _inline(step, z, actions, states, extrinsics, num_obs, tpf, num_total, True, None)
    torch.testing.assert_close(out, exp)
    assert out.shape[1] == (num_total - num_obs) * tpf


def test_matches_inline_teacher_forced():
    num_total, tpf, num_obs = 5, 2, 2
    z, actions, states, extrinsics, z_tf = _inputs(num_total, tpf)
    step = _mock_step(tpf)
    out = predictor_autoregressive_rollout(
        step,
        z,
        actions,
        states,
        extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tpf,
        num_total=num_total,
        predictor_inference_consistent=False,
        z_tf=z_tf,
    )
    exp = _inline(step, z, actions, states, extrinsics, num_obs, tpf, num_total, False, z_tf)
    torch.testing.assert_close(out, exp)


def test_rollout_end_step_stops_autoregressive_rollout_early():
    num_total, tpf, num_obs = 7, 2, 3
    rollout_end_step = 5
    z, actions, states, extrinsics, _ = _inputs(num_total, tpf)
    step = _mock_step(tpf)

    out = predictor_autoregressive_rollout(
        step,
        z,
        actions,
        states,
        extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tpf,
        num_total=num_total,
        predictor_inference_consistent=True,
        rollout_end_step=rollout_end_step,
    )
    exp = _inline(
        step,
        z,
        actions,
        states,
        extrinsics,
        num_obs,
        tpf,
        num_total,
        True,
        None,
        rollout_end_step=rollout_end_step,
    )

    torch.testing.assert_close(out, exp)
    assert out.shape[1] == (rollout_end_step - num_obs) * tpf


def test_rollout_end_step_at_observed_prefix_returns_empty_future_without_calling_predictor():
    num_total, tpf, num_obs = 6, 2, 3
    z, actions, states, extrinsics, _ = _inputs(num_total, tpf)
    calls = []

    def step(_z, _actions, _states, _extrinsics):
        calls.append(_z.shape[1])
        return _z[:, -tpf:]

    out = predictor_autoregressive_rollout(
        step,
        z,
        actions,
        states,
        extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tpf,
        num_total=num_total,
        predictor_inference_consistent=True,
        rollout_end_step=num_obs,
    )

    assert calls == []
    assert out.shape == (z.shape[0], 0, z.shape[-1])


def test_rollout_end_step_must_be_after_observed_prefix_and_within_timeline():
    num_total, tpf, num_obs = 6, 2, 3
    z, actions, states, extrinsics, _ = _inputs(num_total, tpf)
    for rollout_end_step in (num_obs - 1, num_total + 1):
        try:
            predictor_autoregressive_rollout(
                _mock_step(tpf),
                z,
                actions,
                states,
                extrinsics,
                num_obs=num_obs,
                tokens_per_frame=tpf,
                num_total=num_total,
                predictor_inference_consistent=True,
                rollout_end_step=rollout_end_step,
            )
            raise AssertionError("expected ValueError for invalid rollout_end_step")
        except ValueError:
            pass


def test_rollout_zero_is_rejected_for_teacher_forced_init():
    num_total, tpf, num_obs = 6, 2, 3
    z, actions, states, extrinsics, z_tf = _inputs(num_total, tpf)

    try:
        predictor_autoregressive_rollout(
            _mock_step(tpf),
            z,
            actions,
            states,
            extrinsics,
            num_obs=num_obs,
            tokens_per_frame=tpf,
            num_total=num_total,
            predictor_inference_consistent=False,
            z_tf=z_tf,
            rollout_end_step=num_obs,
        )
        raise AssertionError("expected ValueError for rollout 0 with teacher-forced init")
    except ValueError as exc:
        assert "rollout 0" in str(exc)


def test_teacher_forced_requires_z_tf():
    num_total, tpf, num_obs = 4, 2, 2
    z, actions, states, extrinsics, _ = _inputs(num_total, tpf)
    try:
        predictor_autoregressive_rollout(
            _mock_step(tpf),
            z,
            actions,
            states,
            extrinsics,
            num_obs=num_obs,
            tokens_per_frame=tpf,
            num_total=num_total,
            predictor_inference_consistent=False,
            z_tf=None,
        )
        raise AssertionError("expected ValueError when z_tf is None in non-IC mode")
    except ValueError:
        pass


class _Config:
    class _Train:
        predictor_inference_consistent = True

    train = _Train()


def test_rollout_latent_predictions_stops_at_dynamic_rollout_end_step():
    num_total, tpf, num_obs = 7, 2, 3
    z, actions, states, extrinsics, _ = _inputs(num_total, tpf)
    step_calls = []

    def step(_z, _actions, _states, _extrinsics):
        step_calls.append(_z.shape[1] // tpf)
        return _z[:, -tpf:] + float(_z.size(1))

    z_tf, z_ar = rollout_latent_predictions(
        step,
        config=_Config(),
        z_context=z,
        actions=actions,
        states=states,
        extrinsics=extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tpf,
        num_total=num_total,
        compute_tf=False,
        needs_ar_rollout=True,
        planner_only_error_context=False,
        validate_ic_prefix=True,
        rollout_end_step=6,
    )

    assert step_calls == [3, 4, 5]
    assert z_tf.shape[1] == (6 - 1) * tpf
    assert z_ar.shape[1] == (6 - num_obs) * tpf


def test_rollout_latent_predictions_allows_zero_future_for_planner_only_ic_path():
    num_total, tpf, num_obs = 7, 2, 3
    z, actions, states, extrinsics, _ = _inputs(num_total, tpf)
    step_calls = []

    def step(_z, _actions, _states, _extrinsics):
        step_calls.append(_z.shape[1] // tpf)
        return _z[:, -tpf:]

    _, z_ar = rollout_latent_predictions(
        step,
        config=_Config(),
        z_context=z,
        actions=actions,
        states=states,
        extrinsics=extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tpf,
        num_total=num_total,
        compute_tf=False,
        needs_ar_rollout=True,
        planner_only_error_context=True,
        validate_ic_prefix=True,
        rollout_end_step=num_obs,
    )

    assert step_calls == []
    assert z_ar.shape == (z.shape[0], 0, z.shape[-1])


def test_empty_future_planner_conditions_require_non_future_context_tokens():
    z_empty = torch.zeros(2, 0, 4)

    try:
        validate_empty_future_planner_conditions(z_empty)
        raise AssertionError("expected ValueError when empty z_ar has no non-future planner context")
    except ValueError as exc:
        assert "rollout_future_steps=0" in str(exc)

    validate_empty_future_planner_conditions(
        z_empty,
        action_history=torch.zeros(2, 3, 3),
    )


def test_rollout0_validation_skips_only_value_guidance_and_keeps_context_builders():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(repo_root, "app", "vjepa_cowa_world_model", "val_command.py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    assert "should_apply_value_guidance" in source
    assert "allow_empty_rollout_skip=empty_rollout_is_intentional" in source
    assert "planner_conditioning = _prepare_validation_planner_conditioning(" in source
    assert "action_history = build_observed_action_trajectory_history(" in source
    assert "validate_empty_future_planner_conditions(" in source


def test_rollout0_planner_conditioning_keeps_observed_tokens_action_history_and_context() -> None:
    from app.vjepa_cowa_world_model.val_command import _prepare_validation_planner_conditioning

    batch_size, tokens_per_frame, observed_steps = 2, 3, 2
    z = torch.arange(batch_size * 5 * tokens_per_frame * 4, dtype=torch.float32).view(
        batch_size, 5 * tokens_per_frame, 4
    )
    actions = torch.zeros(batch_size, 4, 3)
    actions[:, 0, 0] = 1.0
    empty_future = z[:, :0]

    conditioning = _prepare_validation_planner_conditioning(
        z_ar_planner=empty_future,
        z=z,
        predictor_actions=actions,
        tokens_per_frame=tokens_per_frame,
        observed_steps=observed_steps,
        use_z_context=True,
        use_observed_tokens=True,
        use_action_history=True,
        action_history_dim=3,
        timestep_sec=0.5,
        predictor_frame_stride=1,
    )

    assert conditioning["z_context"].shape == (batch_size, tokens_per_frame, 4)
    assert conditioning["z_observed"].shape == (batch_size, observed_steps * tokens_per_frame, 4)
    assert conditioning["action_history"].shape[0] == batch_size
    assert conditioning["action_history"].numel() > 0


def test_validation_only_marks_explicit_rollout0_as_intentional():
    from app.vjepa_cowa_world_model.val_command import _validation_empty_rollout_is_intentional

    assert _validation_empty_rollout_is_intentional(validation_rollout_horizon=0, budget_profile=None)
    assert _validation_empty_rollout_is_intentional(
        validation_rollout_horizon=None,
        budget_profile=type("Profile", (), {"rollout_future_steps": 0})(),
    )
    assert not _validation_empty_rollout_is_intentional(validation_rollout_horizon=None, budget_profile=None)
    assert not _validation_empty_rollout_is_intentional(validation_rollout_horizon=2, budget_profile=None)


def test_latent_dit_planner_validation_samples_true_short_prefix(monkeypatch) -> None:
    from app.vjepa_cowa_world_model import val_command

    calls = []

    def fake_sample(**kwargs):
        calls.append(kwargs)
        return kwargs["initial_noise"]

    monkeypatch.setattr(val_command, "sample_latent_dit_predictor", fake_sample)
    monkeypatch.setattr(
        val_command,
        "resolve_latent_dit_sampler_params",
        lambda config: type("Params", (), {"as_kwargs": lambda self: {}})(),
    )
    full_noise = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    output = val_command._sample_latent_dit_validation_horizon(
        predictor=object(),
        z_context=torch.zeros(2, 2, 3),
        predictor_inputs=object(),
        tokens_per_frame=2,
        num_observed_steps=1,
        runtime_normalize_reps=False,
        config=object(),
        full_initial_noise=full_noise,
        validation_rollout_horizon=1,
    )

    assert len(calls) == 1
    assert calls[0]["future_steps"] == 1
    assert calls[0]["initial_noise"].shape == (2, 2, 3)
    torch.testing.assert_close(calls[0]["initial_noise"], full_noise[:, :2])
    torch.testing.assert_close(output, full_noise[:, :2])


def test_latent_dit_rollout0_never_calls_sampler(monkeypatch) -> None:
    from app.vjepa_cowa_world_model import val_command

    monkeypatch.setattr(
        val_command,
        "sample_latent_dit_predictor",
        lambda **kwargs: pytest.fail("rollout0 must not call the Latent-DiT sampler"),
    )
    output = val_command._sample_latent_dit_validation_horizon(
        predictor=object(),
        z_context=torch.zeros(2, 2, 3),
        predictor_inputs=object(),
        tokens_per_frame=2,
        num_observed_steps=1,
        runtime_normalize_reps=False,
        config=object(),
        full_initial_noise=torch.zeros(2, 4, 3),
        validation_rollout_horizon=0,
    )

    assert output.shape == (2, 0, 3)


def test_validation_unwraps_distributed_multiview_module_before_forward() -> None:
    from app.vjepa_cowa_world_model.val_command import _unwrap_validation_module

    class Core(torch.nn.Module):
        def forward(self, value):
            return value + 1

    class Wrapper:
        def __init__(self, module):
            self.module = module

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("validation must not execute wrapper forward collectives")

    core = Core()
    wrapper = Wrapper(core)

    assert _unwrap_validation_module(wrapper) is core
    assert _unwrap_validation_module(wrapper)(torch.tensor(1)).item() == 2
