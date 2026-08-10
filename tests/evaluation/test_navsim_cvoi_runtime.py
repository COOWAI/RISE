from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixValueOutput
from app.vjepa_cowa_world_model.training.cvoi_execution import common_random_numbers
from tests.evaluation.test_navsim_stage3_alignment import _load_navsim_agent


class _FixedGate(torch.nn.Module):
    def __init__(self, deltas: list[float], *, latent_dim: int) -> None:
        super().__init__()
        self.feature_dim = 2 * latent_dim + 7
        self._deltas = iter(deltas)
        self.inputs: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.inputs.append(features.detach().clone())
        return features.new_tensor([next(self._deltas)])


class _DifferentiableDualValue(torch.nn.Module):
    def forward(
        self,
        observed: torch.Tensor,
        future: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> PrefixValueOutput:
        batch_size, _, embed_dim = future.shape
        future_frames = future.reshape(batch_size, -1, tokens_per_frame, embed_dim)
        field = future_frames.mean(dim=(2, 3))
        observed_stop = observed.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return PrefixValueOutput(
            field_values=field,
            stop_values=torch.cat([observed_stop, field], dim=1),
        )


class _BombDualValue(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("K=0 must not call Value Guidance")


class _CountingDualValue(_DifferentiableDualValue):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, *args, **kwargs) -> PrefixValueOutput:
        self.call_count += 1
        return super().forward(*args, **kwargs)


def test_navsim_agent_uses_retained_common_random_numbers_owner() -> None:
    agent_class = _load_navsim_agent()

    assert agent_class._forward_stage12.__globals__["common_random_numbers"] is common_random_numbers


def _agent(deltas: list[float]):
    agent_class = _load_navsim_agent()
    agent = object.__new__(agent_class)
    agent._predictor = torch.nn.Identity()
    agent._runtime_normalize_reps = False
    agent._cvoi_gate = _FixedGate(deltas, latent_dim=4)
    agent._cvoi_dual_value = _DifferentiableDualValue()
    agent._cvoi_dual_value_adapter = agent._cvoi_dual_value
    agent._last_cvoi_trace = None
    agent._cvoi_direct_runtime = None
    agent._cvoi_evaluation_guidance_steps = None
    agent._cvoi_evaluation_forced_horizon = None
    agent._config = SimpleNamespace(
        meta=SimpleNamespace(dtype="float32"),
        cvoi=SimpleNamespace(
            enabled=True,
            protocol_version="formal_v2_navsim_e120_h4_v3",
            stage="evaluation",
            evaluation_mode="controller",
            controller_lineage="value_guided",
            max_horizon=4,
            lambda_compute=0.005,
            compute_costs=[0.0, 1.0, 2.0, 3.0, 4.0],
            ablation_signature=SimpleNamespace(gate_feature_mode="full"),
        ),
        value_guidance=SimpleNamespace(
            enabled=True,
            steps=2,
            objective="last",
            step_size=0.0,
            max_delta_norm=0.25,
            detach_output=True,
        ),
    )
    return agent


def _predictor_inputs() -> SimpleNamespace:
    return SimpleNamespace(
        tokens_per_frame=2,
        num_observed_steps=1,
        num_future_steps=4,
        actions=torch.zeros(1, 4, 7),
        states=torch.zeros(1, 5, 7),
        extrinsics=torch.zeros(1, 5, 7),
        driving_command=None,
        ego_dynamics=None,
    )


def test_navsim_p0_agent_selects_unguided_planner_checkpoint() -> None:
    agent_class = _load_navsim_agent()
    agent = object.__new__(agent_class)
    agent._config = SimpleNamespace(
        cvoi=SimpleNamespace(
            controller_lineage="p0_controller",
            unguided_planner_checkpoint="/artifacts/p0.pt",
            guided_planner_checkpoint=None,
        )
    )

    assert agent._configured_cvoi_planner_checkpoint() == "/artifacts/p0.pt"


def _install_toy_step_factory(agent):
    globals_dict = agent._run_cvoi_sequential_prefix.__func__.__globals__
    original = globals_dict["make_predictor_step_fn"]

    def factory(*args, **kwargs):
        del args, kwargs

        def step(rolled, actions, states, extrinsics):
            del actions, states, extrinsics
            next_tokens = rolled[:, -2:] + 1.0
            return torch.cat([rolled, next_tokens], dim=1)

        return step

    globals_dict["make_predictor_step_fn"] = factory
    return globals_dict, original


def test_navsim_cvoi_full_rollout_matches_fixed_h4_raw_prefix_before_guidance() -> None:
    agent = _agent([1.0, 1.0, 1.0, 1.0])
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    expected = torch.cat(
        [torch.full((1, 2, 4), value) for value in (1.0, 2.0, 3.0, 4.0)],
        dim=1,
    )
    assert torch.equal(output, expected)
    assert agent.get_last_cvoi_trace()["stop_horizon"] == 4
    assert agent.get_last_cvoi_trace()["decisions"] == ["ROLL", "ROLL", "ROLL", "ROLL", "STOP"]
    assert agent.get_last_cvoi_trace()["guidance"]["guidance_steps"] == 2.0


def test_navsim_cvoi_all_stop_is_observed_only_and_skips_guidance() -> None:
    agent = _agent([-0.1])
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    trace = agent.get_last_cvoi_trace()
    assert output.shape == (1, 0, 4)
    assert trace["stop_horizon"] == 0
    assert trace["decisions"] == ["STOP"]
    assert trace["guidance"]["guidance_steps"] == 0.0
    assert trace["guidance"]["guidance_skipped_h0"] == 1.0


def test_navsim_p0_controller_forces_zero_field_gate_scalar_and_never_guides() -> None:
    agent = _agent([1.0, -0.1])
    agent._config.cvoi.controller_lineage = "p0_controller"
    agent._config.value_guidance.enabled = False
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert len(agent._cvoi_gate.inputs) == 2
    assert float(agent._cvoi_gate.inputs[1][0, 8]) == 0.0
    trace = agent.get_last_cvoi_trace()
    assert trace["stop_horizon"] == 1
    assert trace["guidance"]["guidance_steps"] == 0.0


def test_navsim_cvoi_sequential_runtime_supports_explicit_evaluation_guidance_k8() -> None:
    agent = _agent([1.0, 1.0, 1.0, 1.0])
    agent.set_cvoi_evaluation_guidance_steps(8)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert agent.get_last_cvoi_trace()["guidance"]["guidance_steps"] == 8.0


def test_navsim_cvoi_k0_uses_raw_future_without_calling_guidance() -> None:
    agent = _agent([1.0, 1.0, 1.0])
    agent._cvoi_dual_value_adapter = _BombDualValue()
    agent.set_cvoi_evaluation_guidance_steps(0)
    agent.set_cvoi_evaluation_forced_horizon(1)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        planner_input = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert torch.equal(planner_input, torch.ones(1, 2, 4))
    trace = agent.get_last_cvoi_trace()
    assert trace["stop_horizon"] == 1
    assert trace["guidance"]["guidance_steps"] == 0.0


def test_navsim_cvoi_evaluation_guidance_override_is_strict_and_resettable() -> None:
    agent = _agent([-0.1])

    for value in (5, 6, True, 1.5):
        try:
            agent.set_cvoi_evaluation_guidance_steps(value)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid guidance override accepted: {value!r}")

    agent.set_cvoi_evaluation_guidance_steps(8)
    assert agent._cvoi_evaluation_guidance_steps == 8
    agent.set_cvoi_evaluation_guidance_steps(None)
    assert agent._cvoi_evaluation_guidance_steps is None

    agent._config.cvoi.stage = "stop_calibrated"
    try:
        agent.set_cvoi_evaluation_guidance_steps(2)
    except ValueError as exc:
        assert "evaluation" in str(exc)
    else:
        raise AssertionError("non-evaluation guidance override was accepted")


@pytest.mark.parametrize("steps", (0, 1, 2, 4, 8))
def test_navsim_e120_evaluation_accepts_the_e120_guidance_grid(steps: int) -> None:
    agent = _agent([-0.1])

    agent.set_cvoi_evaluation_guidance_steps(steps)

    assert agent._cvoi_evaluation_guidance_steps == steps


def test_navsim_e120_value_summary_agent_preserves_protocol_for_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent([1.0, -0.1])
    agent._config.cvoi.ablation_signature = SimpleNamespace(gate_feature_mode="without_value_summary")
    captured_protocols: list[str] = []

    def toy_step_factory(*args, **kwargs):
        del args, kwargs

        def step(rolled, actions, states, extrinsics):
            del actions, states, extrinsics
            next_tokens = rolled[:, -2:] + 1.0
            return torch.cat([rolled, next_tokens], dim=1)

        return step

    module_globals = {
        id(
            agent._run_cvoi_sequential_prefix.__func__.__globals__
        ): agent._run_cvoi_sequential_prefix.__func__.__globals__
    }
    for globals_dict in module_globals.values():
        original_rollout = globals_dict["run_sequential_rollout"]

        def capture_rollout(*, _original=original_rollout, **kwargs):
            captured_protocols.append(kwargs["gate_feature_protocol"])
            return _original(**kwargs)

        monkeypatch.setitem(globals_dict, "make_predictor_step_fn", toy_step_factory)
        monkeypatch.setitem(globals_dict, "run_sequential_rollout", capture_rollout)

    agent._run_cvoi_sequential_prefix(
        torch.ones(1, 2, 4),
        _predictor_inputs(),
        tokens_per_frame=2,
    )
    assert captured_protocols == ["formal_v2_navsim_e120_h4_v3"]
    assert torch.equal(agent._cvoi_gate.inputs[1][0, 8:11], torch.zeros(3))


def test_navsim_agent_gate_override_rejects_retired_generic_formal_v2_protocol() -> None:
    agent = _agent([])
    agent._config.cvoi.protocol_version = "formal_v2"

    with pytest.raises(ValueError, match="NavSim-e120"):
        agent.set_cvoi_evaluation_gate(_FixedGate([], latent_dim=4).eval(), feature_mode="full")


def test_navsim_cvoi_forced_horizon_bypasses_gate_and_rolls_exact_prefix() -> None:
    agent = _agent([])
    agent.set_cvoi_evaluation_forced_horizon(2)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert torch.equal(
        output,
        torch.cat(
            [torch.full((1, 2, 4), value) for value in (1.0, 2.0)],
            dim=1,
        ),
    )
    assert agent._cvoi_gate.inputs == []
    trace = agent.get_last_cvoi_trace()
    assert trace["stop_horizon"] == 2
    assert trace["decisions"] == []
    assert trace["predicted_deltas"] == []
    assert trace["guidance"]["guidance_steps"] == 2.0


def test_navsim_cvoi_forced_h0_skips_gate_rollout_and_guidance() -> None:
    agent = _agent([])
    agent.set_cvoi_evaluation_forced_horizon(0)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert output.shape == (1, 0, 4)
    assert agent._cvoi_gate.inputs == []
    trace = agent.get_last_cvoi_trace()
    assert trace["stop_horizon"] == 0
    assert trace["guidance"]["guidance_steps"] == 0.0
    assert trace["guidance"]["guidance_skipped_h0"] == 1.0


def test_navsim_p0_forced_h4_runs_without_gate_or_value() -> None:
    agent = _agent([])
    agent._config.cvoi.evaluation_mode = "p0_forced"
    agent._config.cvoi.controller_lineage = "p0_controller"
    agent._config.value_guidance.enabled = False
    agent._cvoi_gate = None
    agent._cvoi_dual_value = None
    agent._cvoi_dual_value_adapter = None
    agent.set_cvoi_evaluation_guidance_steps(0)
    agent.set_cvoi_evaluation_forced_horizon(4)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert output.shape == (1, 8, 4)
    assert agent.get_last_cvoi_trace()["guidance"]["guidance_steps"] == 0.0


def test_navsim_p1_forced_k0_runs_without_gate_or_value() -> None:
    agent = _agent([])
    agent._config.cvoi.evaluation_mode = "p1_field_forced"
    agent._cvoi_gate = None
    agent._cvoi_dual_value = None
    agent._cvoi_dual_value_adapter = None
    agent.set_cvoi_evaluation_guidance_steps(0)
    agent.set_cvoi_evaluation_forced_horizon(2)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert output.shape == (1, 4, 4)
    assert agent.get_last_cvoi_trace()["guidance"]["guidance_steps"] == 0.0


def test_navsim_p1_forced_h0_short_circuits_before_predictor_and_value() -> None:
    agent = _agent([])
    agent._config.cvoi.evaluation_mode = "p1_field_forced"
    agent._cvoi_gate = None
    agent._cvoi_dual_value = None
    agent._cvoi_dual_value_adapter = None
    agent.set_cvoi_evaluation_guidance_steps(8)
    agent.set_cvoi_evaluation_forced_horizon(0)
    globals_dict = agent._run_cvoi_sequential_prefix.__func__.__globals__
    original = globals_dict["make_predictor_step_fn"]

    def bomb_factory(*args, **kwargs):
        raise AssertionError("H0 must not construct or call the predictor step")

    globals_dict["make_predictor_step_fn"] = bomb_factory
    try:
        output = agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert output.shape == (1, 0, 4)
    trace = agent.get_last_cvoi_trace()
    assert trace["stop_horizon"] == 0
    assert trace["guidance"]["guidance_steps"] == 0.0
    assert trace["guidance"]["guidance_skipped_h0"] == 1.0


def test_navsim_p1_forced_nonzero_horizon_and_k_call_only_field_guidance_value() -> None:
    agent = _agent([])
    agent._config.cvoi.evaluation_mode = "p1_field_forced"
    agent._cvoi_gate = None
    value = _CountingDualValue()
    agent._cvoi_dual_value = value
    agent._cvoi_dual_value_adapter = value
    agent.set_cvoi_evaluation_guidance_steps(2)
    agent.set_cvoi_evaluation_forced_horizon(2)
    globals_dict, original = _install_toy_step_factory(agent)
    try:
        agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["make_predictor_step_fn"] = original

    assert value.call_count == 4  # before + K gradient calls + after


def test_navsim_online_controller_requires_gate_and_stop_value() -> None:
    for missing in ("gate", "value"):
        agent = _agent([-0.1])
        if missing == "gate":
            agent._cvoi_gate = None
        else:
            agent._cvoi_dual_value = None
            agent._cvoi_dual_value_adapter = None
        try:
            agent._run_cvoi_sequential_prefix(
                torch.zeros(1, 2, 4),
                _predictor_inputs(),
                tokens_per_frame=2,
            )
        except RuntimeError as exc:
            assert "Gate" in str(exc) or "Value" in str(exc) or "value" in str(exc)
        else:
            raise AssertionError(f"online controller accepted missing {missing}")


def test_navsim_cvoi_path_selection_rejects_missing_gate_or_forced_horizon_before_full_ar() -> None:
    agent = _agent([])
    agent._cvoi_gate = None
    agent._cvoi_evaluation_forced_horizon = None

    try:
        agent._should_run_cvoi_sequential_prefix()
    except RuntimeError as exc:
        assert "forced horizon" in str(exc) or "Gate" in str(exc)
    else:
        raise AssertionError("CVoI path selector allowed the full autoregressive rollout")

    agent._cvoi_evaluation_forced_horizon = 0
    assert agent._should_run_cvoi_sequential_prefix() is True
    agent._config.cvoi.enabled = False
    agent._cvoi_evaluation_forced_horizon = None
    assert agent._should_run_cvoi_sequential_prefix() is False


def test_navsim_online_controller_uses_configured_lambda_without_mutating_config() -> None:
    agent = _agent([-0.1])
    globals_dict = agent._run_cvoi_sequential_prefix.__func__.__globals__
    original_rollout = globals_dict["run_sequential_rollout"]
    captured = {}

    def fake_rollout(**kwargs):
        captured["lambda_compute"] = kwargs["lambda_compute"]
        return SimpleNamespace(
            stop_horizon=0,
            decisions=["STOP"],
            predicted_deltas=[],
            planner_output=torch.empty(1, 0, 4),
            require_finite_rollout_tokens=lambda: None,
        )

    globals_dict["run_sequential_rollout"] = fake_rollout
    globals_dict, original_step = _install_toy_step_factory(agent)
    try:
        agent._run_cvoi_sequential_prefix(
            torch.zeros(1, 2, 4),
            _predictor_inputs(),
            tokens_per_frame=2,
        )
    finally:
        globals_dict["run_sequential_rollout"] = original_rollout
        globals_dict["make_predictor_step_fn"] = original_step

    assert captured["lambda_compute"] == 0.005
    assert agent._config.cvoi.lambda_compute == 0.005


def test_stage12_missing_gate_and_forced_horizon_fails_before_full_ar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent([])
    agent._encoder = torch.nn.Linear(1, 1)
    agent._token_ae = None
    agent._multiview_fusion = None
    agent._tokens_per_frame = 2
    agent._cvoi_latency_mode = False
    agent._last_cvoi_planner_output = None
    agent._last_cvoi_latency_components = None
    agent._budget_controller = None
    agent._cvoi_gate = None
    agent._cvoi_evaluation_forced_horizon = None
    agent._config.data = SimpleNamespace(num_target_frames=4, fps=2)
    globals_dict = agent._forward_stage12.__func__.__globals__
    monkeypatch.setitem(globals_dict, "cvoi_execution_autocast", lambda config, device: nullcontext())
    monkeypatch.setitem(
        globals_dict,
        "forward_main_context",
        lambda *args, **kwargs: torch.zeros(1, 2, 4),
    )
    monkeypatch.setitem(globals_dict, "use_latent_dit_predictor", lambda config: False)
    monkeypatch.setitem(globals_dict, "use_parallel_predictor", lambda config: False)
    monkeypatch.setitem(globals_dict, "build_predictor_timeline_inputs", lambda **kwargs: _predictor_inputs())
    monkeypatch.setitem(globals_dict, "enforce_cvoi_zero_future_aux", lambda inputs: inputs)
    ar_calls: list[object] = []

    def bomb_full_ar(*args: object, **kwargs: object) -> object:
        del args, kwargs
        ar_calls.append(object())
        raise AssertionError("full AR must never run without a Gate or forced horizon")

    monkeypatch.setattr(agent, "_run_predictor_ar", bomb_full_ar)
    features = {
        "video_clip": torch.zeros(1, 3, 2, 4, 4),
        "states": torch.zeros(1, 2, 7),
        "actions": torch.zeros(1, 1, 7),
        "extrinsics": torch.zeros(1, 2, 7),
        "cvoi_rng_seed": 1,
    }

    with pytest.raises(RuntimeError, match="forced horizon|Gate"):
        agent._forward_stage12(features)

    assert ar_calls == []


@pytest.mark.parametrize("num_poses", (7, 9))
def test_direct_epdms_stage12_rejects_raw_planner_pose_count_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
    num_poses: int,
) -> None:
    agent = _agent([])
    agent._encoder = torch.nn.Linear(1, 1)
    agent._token_ae = None
    agent._multiview_fusion = None
    agent._tokens_per_frame = 2
    agent._cvoi_latency_mode = False
    agent._last_cvoi_planner_output = None
    agent._last_cvoi_latency_components = None
    agent._budget_controller = None
    agent._cvoi_gate = None
    agent._cvoi_evaluation_forced_horizon = 0
    agent._cvoi_direct_runtime = SimpleNamespace()
    agent._cvoi_trajectory_sampling = SimpleNamespace(num_poses=8)
    agent._z_ar_mode = "full"
    agent._use_z_context = False
    agent._use_observed_tokens = False
    agent._stage12_planner_uses_full_context = False
    agent._config.data = SimpleNamespace(num_target_frames=4, fps=2)
    agent._config.planner = SimpleNamespace(
        planner_type="multimodal",
        use_action_history_for_planner=False,
    )

    class WrongPoseCountPlanner(torch.nn.Module):
        def forward(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            del args, kwargs
            return {
                "trajectories": torch.zeros(1, 2, num_poses, 3),
                "confidences": torch.zeros(1, 2),
            }

    agent._planner = WrongPoseCountPlanner()
    globals_dict = agent._forward_stage12.__func__.__globals__
    monkeypatch.setitem(globals_dict, "cvoi_execution_autocast", lambda config, device: nullcontext())
    monkeypatch.setitem(globals_dict, "forward_main_context", lambda *args, **kwargs: torch.zeros(1, 2, 4))
    monkeypatch.setitem(globals_dict, "use_latent_dit_predictor", lambda config: False)
    monkeypatch.setitem(globals_dict, "use_parallel_predictor", lambda config: False)
    monkeypatch.setitem(globals_dict, "build_predictor_timeline_inputs", lambda **kwargs: _predictor_inputs())
    monkeypatch.setitem(globals_dict, "enforce_cvoi_zero_future_aux", lambda inputs: inputs)
    monkeypatch.setitem(globals_dict, "common_random_numbers", lambda seed: nullcontext())
    monkeypatch.setitem(globals_dict, "validate_empty_future_planner_conditions", lambda *args, **kwargs: None)
    monkeypatch.setitem(globals_dict, "value_planning_method1_enabled", lambda config: False)

    def run_direct_prefix(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        agent._last_cvoi_trace = {"stop_horizon": 0}
        return torch.zeros(1, 2, 4)

    monkeypatch.setattr(agent, "_run_cvoi_sequential_prefix", run_direct_prefix)
    monkeypatch.setattr(agent, "_select_cvoi_direct_planner", lambda horizon: agent._planner)
    monkeypatch.setattr(agent, "_build_status_feature", lambda *args, **kwargs: torch.zeros(1, 7))
    features = {
        "video_clip": torch.zeros(1, 3, 2, 4, 4),
        "states": torch.zeros(1, 2, 7),
        "actions": torch.zeros(1, 1, 7),
        "extrinsics": torch.zeros(1, 2, 7),
        "cvoi_rng_seed": 1,
    }

    with pytest.raises(ValueError, match="num_poses|pose count|8"):
        agent._forward_stage12(features)


def test_navsim_cvoi_forced_horizon_override_is_strict_and_resettable() -> None:
    agent = _agent([])

    for value in (-1, 5, True, 1.5):
        try:
            agent.set_cvoi_evaluation_forced_horizon(value)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid forced horizon accepted: {value!r}")

    agent.set_cvoi_evaluation_forced_horizon(4)
    assert agent._cvoi_evaluation_forced_horizon == 4
    agent.set_cvoi_evaluation_forced_horizon(None)
    assert agent._cvoi_evaluation_forced_horizon is None

    agent._config.cvoi.stage = "stop_calibrated"
    try:
        agent.set_cvoi_evaluation_forced_horizon(2)
    except ValueError as exc:
        assert "evaluation" in str(exc)
    else:
        raise AssertionError("non-evaluation forced horizon was accepted")


def test_navsim_ordinary_initialize_retains_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_class = _load_navsim_agent()
    agent = object.__new__(agent_class)
    agent._device_str = "cuda"
    agent.training_config_path = ""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="training_config_path"):
        agent.initialize()
