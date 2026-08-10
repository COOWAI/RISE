"""Predictor auxiliary-input policies for ablation experiments."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch

from app.vjepa_cowa_world_model.utils import mask_future_actions as mask_unknown_actions
from app.vjepa_cowa_world_model.utils import prepare_inference_consistent_states

PREDICTOR_AUX_POLICIES = {"auto", "full", "inference_consistent", "mask_future", "no_state", "no_aux"}


@dataclass(frozen=True)
class PredictorAuxInputs:
    """Prepared side inputs and masks for AC predictor calls."""

    actions: Optional[torch.Tensor]
    states: Optional[torch.Tensor]
    extrinsics: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
    state_mask: Optional[torch.Tensor]


def _get_train_attr(config: Any, name: str, default: Any = None) -> Any:
    train = getattr(config, "train", None)
    if train is None:
        return default
    if hasattr(train, name):
        value = getattr(train, name)
        return default if value is None else value
    if isinstance(train, dict):
        return train.get(name, default)
    return default


def _require_train_attr(config: Any, name: str) -> Any:
    """fail-loud: 影响监督/条件语义的 train.* 字段必须显式存在，禁止兜默认值。"""
    train = getattr(config, "train", None)
    if train is not None:
        if isinstance(train, dict):
            if name in train and train[name] is not None:
                return train[name]
        elif hasattr(train, name):
            value = getattr(train, name)
            if value is not None:
                return value
    raise ValueError(f"config.train.{name} 缺失（或为 None）；该字段影响 predictor 输入语义，禁止用默认值兜底。")


def _get_predictor_use_drive_command(config: Any) -> bool:
    train = getattr(config, "train", None)
    if train is not None:
        if hasattr(train, "use_drive_command"):
            return bool(getattr(train, "use_drive_command"))
        if isinstance(train, dict) and "use_drive_command" in train:
            return bool(train["use_drive_command"])
        if hasattr(train, "use_drive_command_for_predictor"):
            return bool(getattr(train, "use_drive_command_for_predictor"))
        if isinstance(train, dict) and "use_drive_command_for_predictor" in train:
            return bool(train["use_drive_command_for_predictor"])
    # fail-loud: use_drive_command 决定 state 向量是否含 4 维 one-hot，缺失即报错，禁止默认 True。
    raise ValueError(
        "config.train.use_drive_command（或 use_drive_command_for_predictor）缺失；"
        "该字段决定 predictor state 向量维度语义，禁止用默认值兜底。"
    )


def normalize_predictor_aux_policy(policy: Any) -> str:
    """Normalize predictor auxiliary input policy aliases."""
    value = str(policy or "auto").strip().lower().replace("-", "_")
    aliases = {
        "none": "no_aux",
        "zero": "no_aux",
        "zero_aux": "no_aux",
        "nostate": "no_state",
        "zero_state": "no_state",
        "ic": "inference_consistent",
        "infer_consistent": "inference_consistent",
        "future_mask": "mask_future",
    }
    value = aliases.get(value, value)
    if value not in PREDICTOR_AUX_POLICIES:
        raise ValueError(
            "train.predictor_aux_policy must be one of " f"{sorted(PREDICTOR_AUX_POLICIES)}, got {policy!r}"
        )
    return value


def resolve_predictor_aux_policy(config: Any, override: Optional[str] = None) -> str:
    """Resolve policy from new ablation field and legacy booleans."""
    policy = normalize_predictor_aux_policy(
        override if override is not None else _get_train_attr(config, "predictor_aux_policy", "auto")
    )
    if policy != "auto":
        return policy
    if bool(_get_train_attr(config, "predictor_no_aux_input", False)):
        return "no_aux"
    if bool(_get_train_attr(config, "predictor_inference_consistent", False)):
        return "inference_consistent"
    if bool(_get_train_attr(config, "use_states_for_predictor", True)):
        return "full"
    return "no_state"


def _mask_unknown_temporal(
    tensor: Optional[torch.Tensor], num_known: int
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if tensor is None:
        return None, None
    batch_size, num_steps = tensor.shape[:2]
    masked = torch.zeros_like(tensor)
    temporal_mask = torch.ones(batch_size, num_steps, dtype=torch.bool, device=tensor.device)
    if num_known > 0:
        known_steps = min(int(num_known), num_steps)
        masked[:, :known_steps] = tensor[:, :known_steps]
        temporal_mask[:, :known_steps] = False
    return masked, temporal_mask


def _zero_like(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return torch.zeros_like(tensor) if tensor is not None else None


def _slice_temporal_prefix(tensor: Optional[torch.Tensor], num_steps: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim >= 3:
        return tensor[:, :num_steps]
    return tensor


def _validate_raw_state_inputs(states: torch.Tensor, state_dim: int, policy: str) -> None:
    if state_dim < 7 or states.shape[-1] < 7:
        raise ValueError(
            f"train.predictor_aux_policy={policy!r} requires raw states with "
            f"last dim >= 7 and train.state_dim >= 7, got states.shape[-1]={states.shape[-1]} "
            f"and train.state_dim={state_dim}"
        )


def _prepare_mask_future_states(
    states: torch.Tensor,
    *,
    num_known: int,
    state_dim: int,
    use_drive_command: bool,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_steps = states.shape[:2]
    known_steps = min(max(int(num_known), 0), num_steps)

    if known_steps == 0:
        probe = prepare_inference_consistent_states(
            states[:, :1],
            num_observed=1,
            driving_command=_slice_temporal_prefix(driving_command, 1),
            ego_dynamics=_slice_temporal_prefix(ego_dynamics, 1),
            state_dim=state_dim,
            use_drive_command=use_drive_command,
        )
        dim = probe.shape[-1]
        masked = states.new_zeros(batch_size, num_steps, dim)
        state_mask = torch.ones(batch_size, num_steps, dtype=torch.bool, device=states.device)
        return masked, state_mask

    known_vectors = []
    for step_idx in range(known_steps):
        prefix_steps = step_idx + 1
        state_seq = prepare_inference_consistent_states(
            states[:, :prefix_steps],
            num_observed=prefix_steps,
            driving_command=_slice_temporal_prefix(driving_command, prefix_steps),
            ego_dynamics=_slice_temporal_prefix(ego_dynamics, prefix_steps),
            state_dim=state_dim,
            use_drive_command=use_drive_command,
        )
        known_vectors.append(state_seq[:, -1])

    known_states = torch.stack(known_vectors, dim=1)
    masked = states.new_zeros(batch_size, num_steps, known_states.shape[-1])
    masked[:, :known_steps] = known_states
    state_mask = torch.ones(batch_size, num_steps, dtype=torch.bool, device=states.device)
    state_mask[:, :known_steps] = False
    return masked, state_mask


def prepare_predictor_aux_inputs(
    *,
    actions: Optional[torch.Tensor],
    states: Optional[torch.Tensor],
    extrinsics: Optional[torch.Tensor],
    config: Any,
    num_observed_steps: int,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    policy: Optional[str] = None,
    predictor_no_aux_input: Optional[bool] = None,
) -> PredictorAuxInputs:
    """Prepare action/state/extrinsic inputs under the configured ablation policy."""
    if predictor_no_aux_input is not None and bool(predictor_no_aux_input):
        return PredictorAuxInputs(
            actions=_zero_like(actions),
            states=_zero_like(states),
            extrinsics=_zero_like(extrinsics),
            action_mask=None,
            state_mask=None,
        )

    resolved_policy = resolve_predictor_aux_policy(config, override=policy)
    num_known = max(0, int(num_observed_steps) - 1)

    if resolved_policy == "no_aux":
        return PredictorAuxInputs(
            actions=_zero_like(actions),
            states=_zero_like(states),
            extrinsics=_zero_like(extrinsics),
            action_mask=None,
            state_mask=None,
        )

    if resolved_policy == "no_state":
        return PredictorAuxInputs(
            actions=actions,
            states=_zero_like(states),
            extrinsics=extrinsics,
            action_mask=None,
            state_mask=None,
        )

    if resolved_policy == "full":
        return PredictorAuxInputs(
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            action_mask=None,
            state_mask=None,
        )

    if resolved_policy in {"inference_consistent", "mask_future"}:
        actions_input, action_mask = mask_unknown_actions(actions, num_known) if actions is not None else (None, None)
        if states is None:
            states_input, state_mask = None, None
        else:
            # fail-loud: state_dim 影响 state 向量构造语义，必须来自显式配置，禁止用 states.shape[-1] 兜底。
            state_dim = int(_require_train_attr(config, "state_dim"))
            _validate_raw_state_inputs(states, state_dim, resolved_policy)
            use_drive_command = _get_predictor_use_drive_command(config)
            if resolved_policy == "inference_consistent":
                states_input = prepare_inference_consistent_states(
                    states,
                    num_observed=int(num_observed_steps),
                    driving_command=driving_command,
                    ego_dynamics=ego_dynamics,
                    state_dim=state_dim,
                    use_drive_command=use_drive_command,
                )
                state_mask = None
            else:
                # actions 与 states 的已知步数不同：action[t] 跨帧 t→t+1，最后一个观测帧的
                # action 属于未来（known = N-1）；state[t] 是帧 t 时刻量，当前帧（N-1）的
                # state 在推理时已观测（known = N），不得连同未来一起 mask。
                states_input, state_mask = _prepare_mask_future_states(
                    states,
                    num_known=int(num_observed_steps),
                    state_dim=state_dim,
                    use_drive_command=use_drive_command,
                    driving_command=driving_command,
                    ego_dynamics=ego_dynamics,
                )
        extrinsics_input, _ = _mask_unknown_temporal(extrinsics, int(num_observed_steps))
        return PredictorAuxInputs(
            actions=actions_input,
            states=states_input,
            extrinsics=extrinsics_input,
            action_mask=action_mask,
            state_mask=state_mask,
        )

    raise AssertionError(f"Unhandled predictor aux policy: {resolved_policy}")


def _predictor_accepts_state_mask(predictor: torch.nn.Module) -> bool:
    core = predictor.module if hasattr(predictor, "module") else predictor
    # fail-loud (point 25): 签名探测失败不再静默返回 False（会导致 state_mask 被丢弃），直接抛出。
    signature = inspect.signature(core.forward)
    return "state_mask" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def call_predictor_with_aux(
    predictor: torch.nn.Module,
    tokens: torch.Tensor,
    aux_inputs: PredictorAuxInputs,
) -> torch.Tensor:
    """Call a predictor; if a state_mask was computed it MUST be consumed by the predictor."""
    kwargs = {"action_mask": aux_inputs.action_mask}
    if aux_inputs.state_mask is not None:
        # fail-loud (point 25): 已经算出 state_mask 就必须传给 predictor，否则 inference-consistent
        # 掩码策略会被静默丢弃，造成训练/推理不一致。
        if not _predictor_accepts_state_mask(predictor):
            raise TypeError(
                "Predictor.forward does not accept 'state_mask' but a state_mask was computed; "
                "禁止静默丢弃 state_mask（会破坏 inference-consistent 掩码策略）。"
            )
        kwargs["state_mask"] = aux_inputs.state_mask
    return predictor(tokens, aux_inputs.actions, aux_inputs.states, aux_inputs.extrinsics, **kwargs)
