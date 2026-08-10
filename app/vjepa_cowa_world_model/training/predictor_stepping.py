"""Shared predictor step-function factory.

统一原先散落在 val_command / utils/viz/inference / evaluation/navsim_agent /
training/rl_policy 中的嵌套 `_step_predictor` 闭包：
prepare_predictor_aux_inputs -> call_predictor_with_aux -> 可选 layer_norm。
"""

from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.predictor_aux import call_predictor_with_aux, prepare_predictor_aux_inputs

_UNSET = object()


def validate_empty_future_planner_conditions(
    z_ar: torch.Tensor,
    *,
    z_context: Optional[torch.Tensor] = None,
    z_observed: Optional[torch.Tensor] = None,
    action_history: Optional[torch.Tensor] = None,
) -> None:
    """Fail loud if rollout 0 leaves the planner without any context tokens."""

    if int(z_ar.shape[1]) > 0:
        return
    has_z_context = z_context is not None and int(z_context.shape[1]) > 0
    has_z_observed = z_observed is not None and int(z_observed.shape[1]) > 0
    has_action_history = action_history is not None and int(action_history.shape[1]) > 0
    if has_z_context or has_z_observed or has_action_history:
        return
    raise ValueError(
        "rollout_future_steps=0 produces empty z_ar; planner requires at least one non-future context token "
        "source such as z_context, observed tokens, or action_history"
    )


def make_predictor_step_fn(
    predictor,
    config,
    num_observed_steps,
    *,
    driving_command=None,
    ego_dynamics=None,
    predictor_no_aux_input=_UNSET,
    normalize_reps=False,
    no_grad=False,
    expected_state_dim=None,
    state_dim=None,
    use_drive_command=None,
):
    """返回 (_z, _a, _s, _e) -> z_out 的 predictor 单步函数。

    predictor_no_aux_input 仅在显式给出时透传（保持与原各闭包的调用面一致）。
    expected_state_dim 给出时校验 aux states 维度（原 rl_policy 行为）。
    """

    extra = {}
    if predictor_no_aux_input is not _UNSET:
        extra["predictor_no_aux_input"] = predictor_no_aux_input

    def _step_predictor(_z, _a, _s, _e):
        aux_inputs = prepare_predictor_aux_inputs(
            actions=_a,
            states=_s,
            extrinsics=_e,
            config=config,
            num_observed_steps=num_observed_steps,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            **extra,
        )
        if expected_state_dim is not None:
            if aux_inputs.states is None:
                raise ValueError(
                    f"Predictor expects state features (state_encoder dim {expected_state_dim}) but the aux "
                    "policy produced states=None; enable use_states_for_predictor or drop the state encoder."
                )
            if aux_inputs.states.shape[-1] != expected_state_dim:
                raise ValueError(
                    "Predictor state feature dimension mismatch: "
                    f"prepared {aux_inputs.states.shape[-1]} dims but predictor.state_encoder expects "
                    f"{expected_state_dim}; state_dim={state_dim}, use_drive_command={use_drive_command}"
                )
        ctx = torch.no_grad() if no_grad else nullcontext()
        with ctx:
            z_out = call_predictor_with_aux(predictor, _z, aux_inputs)
        if normalize_reps:
            z_out = F.layer_norm(z_out, (z_out.size(-1),))
        return z_out

    return _step_predictor


def predictor_autoregressive_rollout(
    step_predictor,
    z,
    actions,
    states,
    extrinsics,
    *,
    num_obs: int,
    tokens_per_frame: int,
    num_total: int,
    predictor_inference_consistent: bool,
    z_tf=None,
    rollout_end_step: Optional[int] = None,
):
    """Shared encoder->predictor autoregressive rollout (single source for train-val and viz).

    Reproduces verbatim the rollout that ``val_command.validate_one_epoch`` and the visualization
    ``run_inference`` previously each inlined: an inference-consistent or teacher-forced init,
    then a step-by-step autoregressive predictor unroll, returning the future-token sequence
    ``z_ar``. Keeping one copy means validation and visualization cannot silently diverge from the
    predictor forward.

    Args:
        step_predictor: the per-step predictor closure (see ``make_predictor_step_fn``).
        z: encoder tokens ``[B, num_total * tokens_per_frame, D]``.
        actions/states/extrinsics: predictor conditioning, sliced per autoregressive step.
        num_obs: number of observed frames (rollout start step in inference-consistent mode).
        tokens_per_frame: tokens per frame.
        num_total: total frame steps to roll out to.
        predictor_inference_consistent: if True, init from the observed tokens (no teacher forcing);
            otherwise init from frame 0 + the teacher-forced frame 1 (``z_tf`` required).
        z_tf: teacher-forced tokens; required only when ``predictor_inference_consistent`` is False.
        rollout_end_step: optional exclusive timeline end step. Defaults to ``num_total``.

    Returns:
        z_ar: the predicted future tokens (observed/initial prefix stripped).
    """
    rollout_end = int(num_total) if rollout_end_step is None else int(rollout_end_step)
    if rollout_end > int(num_total):
        raise ValueError(
            f"rollout_end_step must be <= num_total; got rollout_end_step={rollout_end}, "
            f"num_total={int(num_total)}"
        )
    if predictor_inference_consistent:
        if rollout_end < int(num_obs):
            raise ValueError(
                f"rollout_end_step must satisfy num_obs <= rollout_end_step <= num_total for "
                f"inference-consistent rollout; got num_obs={int(num_obs)}, rollout_end_step={rollout_end}, "
                f"num_total={int(num_total)}"
            )
        _z = z[:, : num_obs * tokens_per_frame]
        start_step = num_obs
    else:
        if rollout_end <= int(num_obs):
            raise ValueError(
                "rollout 0 is only defined for predictor_inference_consistent=True; "
                f"got num_obs={int(num_obs)}, rollout_end_step={rollout_end}"
            )
        if z_tf is None:
            raise ValueError(
                "predictor_autoregressive_rollout requires z_tf (teacher-forced tokens) when "
                "predictor_inference_consistent=False."
            )
        _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
        start_step = 2

    if rollout_end < start_step:
        raise ValueError(f"rollout_end_step={rollout_end} is before rollout start step {start_step}")

    for k in range(start_step, rollout_end):
        # action[t] is the transition t->t+1, so there are N-1 actions for N frames: the last step uses all
        # action-transitions but only N-1 prior states/extrinsics (NOT a future-state leak — verified
        # by tests/test_predictor_rollout.py). Mirror of training/lines/planner_world_model.py.
        if k == num_total - 1:
            _a_full, _s_k, _e_k = actions, states[:, :-1], extrinsics[:, :-1]
        else:
            _a_full, _s_k, _e_k = actions[:, :k], states[:, :k], extrinsics[:, :k]
        _z_nxt = step_predictor(_z, _a_full, _s_k, _e_k)[:, -tokens_per_frame:]
        _z = torch.cat([_z, _z_nxt], dim=1)

    if predictor_inference_consistent:
        return _z[:, num_obs * tokens_per_frame :]
    return _z[:, tokens_per_frame:]


def rollout_latent_predictions(
    step_predictor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    config,
    z_context: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    num_obs: int,
    tokens_per_frame: int,
    num_total: int,
    compute_tf: bool,
    needs_ar_rollout: bool,
    planner_only_error_context: bool,
    validate_ic_prefix: bool,
    rollout_end_step: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Shared teacher-forcing + AR rollout skeleton for training lines.

    The caller owns predictor aux-input assembly and normalization inside ``step_predictor``. This helper
    only moves the duplicated sequence logic from ``world_model`` and ``planner_world_model``:
    optional teacher forcing, optional AR rollout, and explicit planner-world-model fail-loud checks.
    """
    rollout_end = int(num_total) if rollout_end_step is None else int(rollout_end_step)
    if rollout_end < 0 or rollout_end > int(num_total):
        raise ValueError(f"rollout_end_step must be in [0, {int(num_total)}], got {rollout_end}")

    z_tf = None
    if compute_tf:
        tf_input_tokens = (rollout_end - 1) * int(tokens_per_frame)
        if tf_input_tokens <= 0:
            raise ValueError(f"rollout_end_step={rollout_end} leaves no teacher-forcing input tokens")
        tf_steps = rollout_end - 1
        z_tf = step_predictor(
            z_context[:, :tf_input_tokens],
            actions[:, :tf_steps],
            states[:, :tf_steps],
            extrinsics[:, :tf_steps],
        )

    if not needs_ar_rollout:
        return z_tf, None

    if config.train.predictor_inference_consistent:
        num_observed_tokens = num_obs * tokens_per_frame
        if z_context.size(1) < num_observed_tokens:
            raise ValueError(
                "Observed token sequence is shorter than predictor observed prefix: "
                f"tokens={z_context.size(1)}, requested={num_observed_tokens}"
            )
        rolled = z_context[:, : num_obs * tokens_per_frame]
        start_step = num_obs
    else:
        if rollout_end <= int(num_obs):
            raise ValueError(
                "rollout 0 is only defined for predictor_inference_consistent=True; "
                f"got num_obs={int(num_obs)}, rollout_end_step={rollout_end}"
            )
        if z_tf is None:
            if planner_only_error_context:
                raise ValueError(
                    "planner-only non-parallel rollout requires predictor_inference_consistent=True "
                    "because teacher forcing is skipped without future images."
                )
            raise ValueError(
                "rollout_latent_predictions requires teacher-forced tokens when "
                "predictor_inference_consistent=False."
            )
        rolled = torch.cat([z_context[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
        start_step = 2
    if rollout_end < start_step:
        raise ValueError(f"rollout_end_step={rollout_end} is before rollout start step {start_step}")

    for step in range(start_step, rollout_end):
        if step == num_total - 1:
            actions_full = actions
            states_step = states[:, :-1]
            extrinsics_step = extrinsics[:, :-1]
        else:
            actions_full = actions[:, :step]
            states_step = states[:, :step]
            extrinsics_step = extrinsics[:, :step]
        next_tokens = step_predictor(
            rolled,
            actions_full,
            states_step,
            extrinsics_step,
        )[:, -tokens_per_frame:]
        rolled = torch.cat([rolled, next_tokens], dim=1)

    if config.train.predictor_inference_consistent:
        z_ar = rolled[:, num_obs * tokens_per_frame :]
        if validate_ic_prefix:
            if not (
                torch.allclose(
                    rolled[:, : num_obs * tokens_per_frame],
                    z_context[:, : num_obs * tokens_per_frame],
                    rtol=1e-3,
                    atol=1e-3,
                )
            ):
                raise AssertionError("Initial part of _z does not match z")
    else:
        z_ar = rolled[:, tokens_per_frame:]

    if z_tf is None:
        z_tf = rolled[:, tokens_per_frame:]
    return z_tf, z_ar
