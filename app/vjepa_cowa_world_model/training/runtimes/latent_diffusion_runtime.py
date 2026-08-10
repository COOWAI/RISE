"""Runtime helpers for the latent-token DiT predictor."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.counterfactual_supervision import distributed_masked_mean
from app.vjepa_cowa_world_model.training.predictor_aux import prepare_predictor_aux_inputs
from app.vjepa_cowa_world_model.training.prefix_schedule import (
    PrefixDistribution,
    PrefixSample,
    resolve_prefix_distribution,
    sample_prefix,
)
from app.vjepa_cowa_world_model.training.runtimes.latent_action_runtime import (
    build_future_action_targets,
    build_last_observed_action_state_tokens,
    denormalize_joint_actions,
    normalize_joint_actions,
)
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module


@dataclass(frozen=True)
class LatentDiTRuntimeOutput:
    """Output bundle for train/eval latent DiT forwarding."""

    z_ar: torch.Tensor
    loss: torch.Tensor
    flow_loss: torch.Tensor
    x0_loss: torch.Tensor
    x0_pred: torch.Tensor
    velocity_pred: torch.Tensor
    velocity_target: torch.Tensor
    future_token_indices: Optional[torch.Tensor] = None
    objective: str = "flow_matching"
    objective_loss: Optional[torch.Tensor] = None
    action_loss: Optional[torch.Tensor] = None
    action_objective_loss: Optional[torch.Tensor] = None
    action_loss_sum: Optional[torch.Tensor] = None
    action_num_samples: Optional[torch.Tensor] = None
    action_x0_pred: Optional[torch.Tensor] = None
    action_velocity_pred: Optional[torch.Tensor] = None
    action_velocity_target: Optional[torch.Tensor] = None
    target_future_actions: Optional[torch.Tensor] = None
    active_prefix_steps: Optional[int] = None
    prefix_distribution: Optional[PrefixDistribution] = None


def use_latent_dit_predictor(config) -> bool:
    """Return whether train.predictor_type selects the latent DiT predictor."""
    return str(config.train.predictor_type).lower() == "latent_dit"


def use_sampled_latent_dit_planner_input(config) -> bool:
    """Return whether planner training should consume sampled latent DiT predictions."""
    planner_input = str(config.train.latent_dit_planner_input).lower()
    return use_latent_dit_predictor(config) and planner_input == "sample" and not bool(config.train.predictor_train)


# Canonical latent-DiT multi-step sampler defaults. Kept in ONE place so train, validation
# and deployment cannot silently disagree on how the future tokens are sampled.
_DIT_SAMPLER_DEFAULTS = {
    "num_inference_steps": 8,
    "sampler_type": "heun",
    "schedule_type": "cosine",
    "temperature": 1.0,
}


@dataclass(frozen=True)
class LatentDiTSamplerParams:
    """Resolved latent-DiT sampler hyperparameters shared by train/val/deploy."""

    num_inference_steps: int
    sampler_type: str
    schedule_type: str
    temperature: float

    def as_kwargs(self) -> dict:
        return {
            "num_inference_steps": self.num_inference_steps,
            "sampler_type": self.sampler_type,
            "schedule_type": self.schedule_type,
            "temperature": self.temperature,
        }


def _read_predictor_dit_field(config: Any, field: str, default: Any) -> Any:
    """Read config.predictor_dit.<field> tolerant of dataclass/namespace/dict configs."""
    dit = getattr(config, "predictor_dit", None)
    if dit is None and isinstance(config, dict):
        dit = config.get("predictor_dit")
    if dit is None:
        return default
    value = dit.get(field, default) if isinstance(dit, dict) else getattr(dit, field, default)
    return default if value is None else value


def _latent_dit_objective(config: Any, predictor: Optional[torch.nn.Module] = None) -> str:
    default = "flow_matching"
    if predictor is not None:
        default = str(getattr(unwrap_module(predictor), "objective", default))
    objective = str(_read_predictor_dit_field(config, "objective", default)).lower()
    if objective not in {"flow_matching", "x0_prediction"}:
        raise ValueError(
            "predictor_dit.objective must be one of " f"{['flow_matching', 'x0_prediction']}, got {objective!r}"
        )
    return objective


def _joint_action_enabled(config: Any) -> bool:
    return bool(_read_predictor_dit_field(config, "joint_action_enabled", False))


def _joint_action_loss_weight(config: Any) -> float:
    weight = float(_read_predictor_dit_field(config, "joint_action_loss_weight", 0.0))
    if weight < 0.0:
        raise ValueError(f"predictor_dit.joint_action_loss_weight must be >= 0, got {weight}")
    return weight


def _joint_action_dim(config: Any, predictor: Optional[torch.nn.Module] = None) -> int:
    default = 3
    if predictor is not None:
        default = int(getattr(unwrap_module(predictor), "joint_action_dim", default))
    dim = int(_read_predictor_dit_field(config, "joint_action_dim", default))
    if dim <= 0:
        raise ValueError(f"predictor_dit.joint_action_dim must be positive, got {dim}")
    return dim


def _joint_action_state_dim(config: Any, predictor: Optional[torch.nn.Module] = None) -> int:
    default = 7
    if predictor is not None:
        default = int(getattr(unwrap_module(predictor), "joint_action_state_dim", default))
    dim = int(_read_predictor_dit_field(config, "joint_action_state_dim", default))
    if dim <= 0:
        raise ValueError(f"predictor_dit.joint_action_state_dim must be positive, got {dim}")
    return dim


def _joint_action_scale(config: Any) -> tuple[float, ...]:
    raw_scale = _read_predictor_dit_field(config, "joint_action_scale", (8.0, 4.0, 1.0))
    return tuple(float(value) for value in raw_scale)


def _joint_action_noise_mode(config: Any) -> str:
    mode = str(_read_predictor_dit_field(config, "joint_action_noise_mode", "shared")).lower()
    if mode not in {"shared", "decoupled"}:
        raise ValueError("predictor_dit.joint_action_noise_mode must be one of ['shared', 'decoupled']")
    return mode


def _joint_action_inference_noise_mode(config: Any, predictor: Optional[torch.nn.Module] = None) -> str:
    default = "shared"
    if predictor is not None:
        default = str(getattr(unwrap_module(predictor), "joint_action_inference_noise_mode", default))
    mode = str(_read_predictor_dit_field(config, "joint_action_inference_noise_mode", default)).lower()
    if mode not in {"shared", "decoupled"}:
        raise ValueError("predictor_dit.joint_action_inference_noise_mode must be one of ['shared', 'decoupled']")
    return mode


def _joint_video_final_noise(config: Any, predictor: Optional[torch.nn.Module] = None) -> float:
    default = 0.0
    if predictor is not None:
        default = float(getattr(unwrap_module(predictor), "joint_video_final_noise", default))
    value = float(_read_predictor_dit_field(config, "joint_video_final_noise", default))
    if not 0.0 <= value < 1.0:
        raise ValueError(f"predictor_dit.joint_video_final_noise must be in [0.0, 1.0), got {value}")
    if value > 0.0 and _joint_action_inference_noise_mode(config, predictor) != "decoupled":
        raise ValueError(
            "predictor_dit.joint_video_final_noise > 0 requires "
            "predictor_dit.joint_action_inference_noise_mode='decoupled'"
        )
    return value


def _joint_action_horizon(predictor: torch.nn.Module, target_future_tokens: torch.Tensor) -> int:
    core = unwrap_module(predictor)
    horizon = int(getattr(core, "num_future_steps", 0))
    if horizon <= 0:
        horizon = int(target_future_tokens.shape[1])
    if horizon <= 0:
        raise ValueError(f"joint action horizon must be positive, got {horizon}")
    return horizon


def _call_latent_dit_predictor(
    predictor: torch.nn.Module,
    predictor_kwargs: dict,
    *,
    joint_action_enabled: bool,
) -> torch.Tensor | dict:
    core = unwrap_module(predictor)
    if joint_action_enabled and hasattr(core, "forward_joint"):
        signature = inspect.signature(core.forward)
        supports_joint_forward = "noisy_future_actions" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if not supports_joint_forward:
            return core.forward_joint(**predictor_kwargs)
    if joint_action_enabled and type(core).forward is torch.nn.Module.forward:
        return core.forward_joint(**predictor_kwargs)
    return predictor(**predictor_kwargs)


def _merge_model_outputs(
    metadata_condition_mask: torch.Tensor,
    pred_cond: torch.Tensor | dict,
    pred_uncond: torch.Tensor | dict,
) -> torch.Tensor | dict:
    if isinstance(pred_cond, dict) or isinstance(pred_uncond, dict):
        if not isinstance(pred_cond, dict) or not isinstance(pred_uncond, dict):
            raise TypeError("conditional and unconditional latent-DiT outputs must have matching types")
        merged = {}
        for key, cond_value in pred_cond.items():
            uncond_value = pred_uncond[key]
            if not torch.is_tensor(cond_value):
                merged[key] = cond_value
                continue
            merged[key] = torch.where(metadata_condition_mask[:, None, None], cond_value, uncond_value)
        return merged
    return torch.where(metadata_condition_mask[:, None, None], pred_cond, pred_uncond)


def _world_prediction(model_output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["world_pred"]
    return model_output


def _action_prediction(model_output: torch.Tensor | dict) -> torch.Tensor:
    if not isinstance(model_output, dict) or "action_pred" not in model_output:
        raise ValueError("joint action latent-DiT forward must return {'world_pred', 'action_pred'}")
    return model_output["action_pred"]


def _metadata_conditioning_policy(config: Any) -> str:
    policy = str(_read_predictor_dit_field(config, "metadata_conditioning_policy", "auto")).lower()
    if policy not in {"auto", "always", "never"}:
        raise ValueError(
            "predictor_dit.metadata_conditioning_policy must be one of " f"['auto', 'always', 'never'], got {policy!r}"
        )
    return policy


def _metadata_condition_dropout(config: Any) -> float:
    dropout = float(_read_predictor_dit_field(config, "metadata_condition_dropout", 0.0))
    if dropout < 0.0 or dropout > 1.0:
        raise ValueError(f"predictor_dit.metadata_condition_dropout must be in [0, 1], got {dropout}")
    return dropout


def _metadata_guidance_scale(config: Any) -> float:
    return float(_read_predictor_dit_field(config, "metadata_guidance_scale", 1.0))


def latent_dit_masked_inpainting_enabled(config: Any) -> bool:
    """Return whether latent-DiT active-token masked/inpainting mode is enabled."""
    return bool(_read_predictor_dit_field(config, "masked_inpainting_enabled", False))


def _metadata_can_request_unconditional(config: Any) -> bool:
    return _metadata_conditioning_policy(config) != "never" and _metadata_condition_dropout(config) > 0.0


def _supports_metadata_condition_mask(predictor: torch.nn.Module) -> bool:
    return bool(getattr(unwrap_module(predictor), "supports_metadata_condition_mask", False))


def _masked_sample_return_full(config: Any) -> bool:
    return bool(_read_predictor_dit_field(config, "masked_sample_return_full", False))


def _build_future_token_indices(
    *,
    future_start_step: int,
    future_steps: int,
    tokens_per_frame: int,
    num_future_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    future_start_step = int(future_start_step)
    future_steps = int(future_steps)
    tokens_per_frame = int(tokens_per_frame)
    num_future_tokens = int(num_future_tokens)
    if future_start_step < 0:
        raise ValueError(f"future_start_step must be >= 0, got {future_start_step}")
    if future_steps <= 0:
        raise ValueError(f"future_steps must be positive, got {future_steps}")
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    start_token = future_start_step * tokens_per_frame
    end_token = start_token + future_steps * tokens_per_frame
    if end_token > num_future_tokens:
        raise ValueError(
            f"requested future window [{future_start_step}, {future_start_step + future_steps}) "
            f"exceeds num_future_tokens={num_future_tokens} with tokens_per_frame={tokens_per_frame}"
        )
    return torch.arange(start_token, end_token, device=device, dtype=torch.long)


def _select_masked_train_prefix(config: Any, *, total_future_steps: int, device: torch.device) -> PrefixSample:
    """Draw one cumulative masked Latent-DiT training prefix."""
    full_prefix_prob = float(_read_predictor_dit_field(config, "masked_train_full_prefix_prob", 0.25))
    min_prefix_steps = int(_read_predictor_dit_field(config, "masked_train_min_prefix_steps", 1))
    if full_prefix_prob < 1.0 and min_prefix_steps == 0:
        raise ValueError(
            "masked Latent-DiT predictor supervision cannot include h=0; "
            "set predictor_dit.masked_train_min_prefix_steps >= 1"
        )
    distribution = resolve_prefix_distribution(
        enabled=True,
        horizon_steps=total_future_steps,
        full_prefix_prob=full_prefix_prob,
        min_prefix_steps=min_prefix_steps,
        max_non_full_prefix_steps=_read_predictor_dit_field(
            config,
            "masked_train_max_non_full_prefix_steps",
            None,
        ),
    )
    sample = sample_prefix(distribution, device=device)
    if sample.prefix_steps == 0:
        raise ValueError(
            "masked Latent-DiT predictor supervision cannot train on h=0; "
            "set predictor_dit.masked_train_min_prefix_steps >= 1"
        )
    return sample


def _resolve_metadata_condition_mask(
    predictor_inputs: Any,
    *,
    config: Any,
    num_observed_steps: int,
    batch_size: int,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    policy = _metadata_conditioning_policy(config)
    if policy == "never":
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    elif policy == "always":
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    else:
        observed_metadata_valid_mask = getattr(predictor_inputs, "observed_metadata_valid_mask", None)
        if observed_metadata_valid_mask is not None:
            if not torch.is_tensor(observed_metadata_valid_mask):
                raise TypeError("predictor_inputs.observed_metadata_valid_mask must be a torch.Tensor when provided")
            values = observed_metadata_valid_mask.to(device=device, dtype=torch.bool)
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(
                    "observed_metadata_valid_mask must have shape [B], got "
                    f"{tuple(values.shape)} for batch_size={batch_size}"
                )
            mask = values
        else:
            metadata_valid_mask = getattr(predictor_inputs, "metadata_valid_mask", None)
            if metadata_valid_mask is None:
                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                if not torch.is_tensor(metadata_valid_mask):
                    raise TypeError("predictor_inputs.metadata_valid_mask must be a torch.Tensor when provided")
                values = metadata_valid_mask.to(device=device, dtype=torch.bool)
                if values.ndim != 2 or values.shape[0] != batch_size:
                    raise ValueError(
                        "metadata_valid_mask must have shape [B, T], got "
                        f"{tuple(values.shape)} for batch_size={batch_size}"
                    )
                observed_steps = min(int(num_observed_steps), values.shape[1])
                if observed_steps <= 0:
                    raise ValueError(f"num_observed_steps must be positive, got {num_observed_steps}")
                mask = values[:, :observed_steps].all(dim=1)

    dropout = _metadata_condition_dropout(config) if training else 0.0
    if dropout > 0.0:
        keep = torch.rand(batch_size, device=device) >= dropout
        mask = mask & keep
    return mask


def _metadata_uncond_kwargs(predictor_kwargs: dict) -> dict:
    kwargs = dict(predictor_kwargs)
    kwargs["actions"] = None
    kwargs["states"] = None
    kwargs["extrinsics"] = None
    return kwargs


def _ensure_metadata_unconditional_supported(predictor: torch.nn.Module, *, needed: bool) -> None:
    if not needed:
        return
    core = unwrap_module(predictor)
    conditioning_mode = str(getattr(core, "conditioning_mode", "temporal_aux_tokens")).lower()
    if conditioning_mode == "mean":
        raise ValueError(
            "LatentDiT metadata-unconditional training/sampling requires "
            "predictor_dit.conditioning_mode='temporal_aux_tokens'. The 'mean' conditioner requires actions."
        )


def resolve_latent_dit_sampler_params(config: Any) -> LatentDiTSamplerParams:
    """Single source of truth for the latent-DiT multi-step sampler hyperparameters.

    Train (``train_navsim_v2`` planner-input sampling), validation (``predictor_validation`` /
    ``val_command``) and deployment (``navsim_agent`` PDMS) must all sample the future tokens
    with the SAME num_inference_steps / sampler_type / schedule_type / temperature. Resolving
    them here removes the train(direct attribute, fail-loud) vs eval(getattr-with-default,
    fail-soft) skew that could otherwise let the sampler silently desync between train and infer.
    """
    return LatentDiTSamplerParams(
        num_inference_steps=int(
            _read_predictor_dit_field(config, "num_inference_steps", _DIT_SAMPLER_DEFAULTS["num_inference_steps"])
        ),
        sampler_type=str(_read_predictor_dit_field(config, "sampler_type", _DIT_SAMPLER_DEFAULTS["sampler_type"])),
        schedule_type=str(_read_predictor_dit_field(config, "schedule_type", _DIT_SAMPLER_DEFAULTS["schedule_type"])),
        temperature=float(_read_predictor_dit_field(config, "temperature", _DIT_SAMPLER_DEFAULTS["temperature"])),
    )


def _uses_anchor_frame(predictor: torch.nn.Module) -> bool:
    return bool(getattr(unwrap_module(predictor), "use_anchor_frame", False))


def prepare_latent_dit_aux_inputs(
    predictor_inputs: Any,
    *,
    config: Any | None,
    num_observed_steps: int,
) -> SimpleNamespace:
    """Align latent-DiT side inputs with AC predictor inference-time masking."""
    actions = getattr(predictor_inputs, "actions", None)
    states = getattr(predictor_inputs, "states", None)
    extrinsics = getattr(predictor_inputs, "extrinsics", None)

    # fail-loud (point 26): config 为 None 会完全绕过 inference-consistent 掩码策略（actions/states/
    # extrinsics 原样返回）。调用方必须显式传入 config，禁止漏传后静默跳过掩码。
    if config is None:
        raise ValueError(
            "prepare_latent_dit_aux_inputs: config is None — 这会绕过 inference-consistent 掩码策略；"
            "调用方必须传入 config。"
        )

    aux_inputs = prepare_predictor_aux_inputs(
        actions=actions,
        states=states,
        extrinsics=extrinsics,
        config=config,
        num_observed_steps=int(num_observed_steps),
        driving_command=getattr(predictor_inputs, "driving_command", None),
        ego_dynamics=getattr(predictor_inputs, "ego_dynamics", None),
    )
    return SimpleNamespace(
        actions=aux_inputs.actions,
        states=aux_inputs.states,
        extrinsics=aux_inputs.extrinsics,
        metadata_valid_mask=getattr(predictor_inputs, "metadata_valid_mask", None),
        observed_metadata_valid_mask=getattr(predictor_inputs, "observed_metadata_valid_mask", None),
    )


def forward_latent_dit_predictor_train(
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    h_target: torch.Tensor,
    predictor_inputs,
    *,
    tokens_per_frame: int,
    num_observed_steps: int,
    runtime_normalize_reps: bool,
    config: Any,
    imitation_mask: Optional[torch.Tensor] = None,
    distributed_action_normalization: bool = True,
    timesteps: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    action_timesteps: Optional[torch.Tensor] = None,
    action_noise: Optional[torch.Tensor] = None,
    masked_prefix_steps: Optional[int] = None,
    metadata_condition_training: bool = True,
) -> LatentDiTRuntimeOutput:
    """Train latent DiT against target future tokens and return planner tokens."""
    observed_tokens = z_context[:, : int(num_observed_steps) * int(tokens_per_frame)]
    full_target_future_tokens = h_target[:, int(num_observed_steps) * int(tokens_per_frame) :]
    joint_action_enabled = _joint_action_enabled(config)
    aux_inputs = prepare_latent_dit_aux_inputs(
        predictor_inputs,
        config=config,
        num_observed_steps=num_observed_steps,
    )

    future_token_indices = None
    known_future_tokens = None
    known_future_token_indices = None
    target_future_tokens = full_target_future_tokens
    prefix_sample = None
    if latent_dit_masked_inpainting_enabled(config):
        if full_target_future_tokens.shape[1] % int(tokens_per_frame) != 0:
            raise ValueError(
                "masked latent-DiT training requires full target future tokens to be frame-aligned: "
                f"tokens={full_target_future_tokens.shape[1]}, tokens_per_frame={tokens_per_frame}"
            )
        total_future_steps = int(full_target_future_tokens.shape[1]) // int(tokens_per_frame)
        if masked_prefix_steps is None:
            prefix_sample = _select_masked_train_prefix(
                config,
                total_future_steps=total_future_steps,
                device=full_target_future_tokens.device,
            )
        else:
            explicit_prefix = int(masked_prefix_steps)
            if explicit_prefix <= 0 or explicit_prefix > total_future_steps:
                raise ValueError(
                    "masked_prefix_steps must be within the positive target horizon: "
                    f"got {explicit_prefix}, total_future_steps={total_future_steps}"
                )
            prefix_sample = PrefixSample(
                prefix_steps=explicit_prefix,
                distribution=PrefixDistribution(
                    horizon_steps=total_future_steps,
                    prefix_steps=(explicit_prefix,),
                    probabilities=(1.0,),
                ),
            )
        active_token_count = prefix_sample.prefix_steps * int(tokens_per_frame)
        if active_token_count != full_target_future_tokens.shape[1]:
            future_token_indices = _build_future_token_indices(
                future_start_step=0,
                future_steps=prefix_sample.prefix_steps,
                tokens_per_frame=tokens_per_frame,
                num_future_tokens=full_target_future_tokens.shape[1],
                device=full_target_future_tokens.device,
            )
            target_future_tokens = full_target_future_tokens.index_select(1, future_token_indices)

    if target_future_tokens.shape[1] % int(tokens_per_frame) != 0:
        raise ValueError(
            "latent-DiT target future tokens must be frame-aligned for joint action training: "
            f"tokens={target_future_tokens.shape[1]}, tokens_per_frame={tokens_per_frame}"
        )
    active_future_steps = int(target_future_tokens.shape[1]) // int(tokens_per_frame)

    batch_size = target_future_tokens.shape[0]
    objective = _latent_dit_objective(config, predictor)
    if timesteps is None:
        t = torch.sigmoid(torch.randn(batch_size, device=target_future_tokens.device)).clamp(1e-5, 1.0 - 1e-5)
    else:
        if not torch.is_tensor(timesteps):
            raise TypeError("timesteps must be a torch.Tensor when provided")
        if tuple(timesteps.shape) != (batch_size,):
            raise ValueError(f"timesteps must have shape {(batch_size,)}, got {tuple(timesteps.shape)}")
        if timesteps.device != target_future_tokens.device:
            raise ValueError(
                "timesteps device must match target_future_tokens: "
                f"{timesteps.device} != {target_future_tokens.device}"
            )
        if not timesteps.dtype.is_floating_point:
            raise TypeError(f"timesteps must use a floating dtype, got {timesteps.dtype}")
        if bool(((timesteps <= 0.0) | (timesteps >= 1.0)).any().item()):
            raise ValueError("timesteps values must be strictly between 0 and 1")
        t = timesteps
    if noise is None:
        noise = torch.randn_like(target_future_tokens)
    else:
        if not torch.is_tensor(noise):
            raise TypeError("noise must be a torch.Tensor when provided")
        if tuple(noise.shape) != tuple(target_future_tokens.shape):
            raise ValueError(f"noise must have shape {tuple(target_future_tokens.shape)}, got {tuple(noise.shape)}")
        if noise.device != target_future_tokens.device or noise.dtype != target_future_tokens.dtype:
            raise ValueError(
                "noise device/dtype must match target_future_tokens: "
                f"{noise.device}/{noise.dtype} != {target_future_tokens.device}/{target_future_tokens.dtype}"
            )
    t_expand = t[:, None, None]
    x_t = (1.0 - t_expand) * noise + t_expand * target_future_tokens
    velocity_target = target_future_tokens - noise

    target_future_actions = None
    action_x_t = None
    action_t = None
    action_t_expand = None
    action_velocity_target = None
    action_state_tokens = None
    if joint_action_enabled:
        if not bool(getattr(unwrap_module(predictor), "joint_action_enabled", False)):
            raise ValueError("config enables joint action latent-DiT but predictor.joint_action_enabled is False")
        action_dim = _joint_action_dim(config, predictor)
        action_horizon = active_future_steps
        raw_future_actions = build_future_action_targets(
            actions=getattr(predictor_inputs, "actions", None),
            num_observed_steps=num_observed_steps,
            num_future_steps=action_horizon,
            action_dim=action_dim,
            future_start_step=0,
        )
        target_future_actions = normalize_joint_actions(raw_future_actions, _joint_action_scale(config))
        action_state_tokens = build_last_observed_action_state_tokens(
            states=getattr(predictor_inputs, "states", None),
            num_observed_steps=num_observed_steps,
            num_future_steps=action_horizon,
            state_dim=_joint_action_state_dim(config, predictor),
        ).to(device=target_future_actions.device, dtype=target_future_actions.dtype)
        shared_action_t = t[:, None].expand(-1, action_horizon)
        if action_timesteps is None:
            if _joint_action_noise_mode(config) == "shared":
                action_t = shared_action_t
            else:
                action_t = torch.sigmoid(
                    torch.randn(
                        batch_size,
                        action_horizon,
                        device=target_future_actions.device,
                        dtype=target_future_actions.dtype,
                    )
                ).clamp(1e-5, 1.0 - 1e-5)
        else:
            if not torch.is_tensor(action_timesteps):
                raise TypeError("action_timesteps must be a torch.Tensor when provided")
            expected_action_t_shape = (batch_size, action_horizon)
            if tuple(action_timesteps.shape) != expected_action_t_shape:
                raise ValueError(
                    f"action_timesteps must have shape {expected_action_t_shape}, "
                    f"got {tuple(action_timesteps.shape)}"
                )
            if (
                action_timesteps.device != target_future_actions.device
                or action_timesteps.dtype != target_future_actions.dtype
            ):
                raise ValueError(
                    "action_timesteps device/dtype must match target_future_actions: "
                    f"{action_timesteps.device}/{action_timesteps.dtype} != "
                    f"{target_future_actions.device}/{target_future_actions.dtype}"
                )
            if bool(((action_timesteps <= 0.0) | (action_timesteps >= 1.0)).any().item()):
                raise ValueError("action_timesteps values must be strictly between 0 and 1")
            if _joint_action_noise_mode(config) == "shared" and not torch.equal(action_timesteps, shared_action_t):
                raise ValueError("shared joint action noise mode requires action_timesteps to equal world timesteps")
            action_t = action_timesteps
        if action_noise is None:
            action_noise = torch.randn_like(target_future_actions)
        else:
            if not torch.is_tensor(action_noise):
                raise TypeError("action_noise must be a torch.Tensor when provided")
            if tuple(action_noise.shape) != tuple(target_future_actions.shape):
                raise ValueError(
                    f"action_noise must have shape {tuple(target_future_actions.shape)}, "
                    f"got {tuple(action_noise.shape)}"
                )
            if (
                action_noise.device != target_future_actions.device
                or action_noise.dtype != target_future_actions.dtype
            ):
                raise ValueError(
                    "action_noise device/dtype must match target_future_actions: "
                    f"{action_noise.device}/{action_noise.dtype} != "
                    f"{target_future_actions.device}/{target_future_actions.dtype}"
                )
        action_t_expand = action_t[:, :, None]
        action_x_t = (1.0 - action_t_expand) * action_noise + action_t_expand * target_future_actions
        action_velocity_target = target_future_actions - action_noise

    predictor_kwargs = {
        "noisy_future_tokens": x_t,
        "timesteps": t,
        "observed_tokens": observed_tokens,
        "actions": aux_inputs.actions,
        "states": aux_inputs.states,
        "extrinsics": aux_inputs.extrinsics,
    }
    if joint_action_enabled:
        predictor_kwargs.update(
            {
                "noisy_future_actions": action_x_t,
                "action_timesteps": action_t,
                "action_state_tokens": action_state_tokens,
                "target_future_actions": target_future_actions,
            }
        )
    if future_token_indices is not None:
        predictor_kwargs["future_token_indices"] = future_token_indices
    if known_future_tokens is not None:
        predictor_kwargs["known_future_tokens"] = known_future_tokens
        predictor_kwargs["known_future_token_indices"] = known_future_token_indices
    if _uses_anchor_frame(predictor):
        # The anchor must come from the SAME encoder the model sees at inference. The sampling
        # path (sample_latent_dit_predictor) uses observed_tokens[:, -tokens_per_frame:], i.e. the
        # student/context encoder's last observed frame. Using h_target (the EMA/target encoder)
        # here makes the anchor off-distribution at deployment whenever target != context (encoder
        # trainable, multiview enabled, or a distinct target checkpoint). Keep train == infer.
        predictor_kwargs["anchor_tokens"] = observed_tokens[:, -int(tokens_per_frame) :]

    metadata_condition_mask = _resolve_metadata_condition_mask(
        predictor_inputs,
        config=config,
        num_observed_steps=num_observed_steps,
        batch_size=batch_size,
        device=target_future_tokens.device,
        training=bool(metadata_condition_training),
    )
    can_request_uncond = bool(metadata_condition_training) and _metadata_can_request_unconditional(config)
    needs_uncond = bool((~metadata_condition_mask).any().item()) or can_request_uncond
    _ensure_metadata_unconditional_supported(predictor, needed=needs_uncond)

    use_masked_forward = _metadata_conditioning_policy(config) != "never" and (
        can_request_uncond or not bool(metadata_condition_mask.all().item())
    )
    if use_masked_forward:
        if not _supports_metadata_condition_mask(predictor):
            raise ValueError(
                "LatentDiT metadata condition dropout/validity masking requires predictor support for "
                "metadata_condition_mask so training can stay in one DDP forward."
            )
        predictor_kwargs["metadata_condition_mask"] = metadata_condition_mask
        model_output = _call_latent_dit_predictor(
            predictor,
            predictor_kwargs,
            joint_action_enabled=joint_action_enabled,
        )
    elif bool(metadata_condition_mask.all().item()):
        model_output = _call_latent_dit_predictor(
            predictor,
            predictor_kwargs,
            joint_action_enabled=joint_action_enabled,
        )
    elif bool((~metadata_condition_mask).all().item()):
        model_output = _call_latent_dit_predictor(
            predictor,
            _metadata_uncond_kwargs(predictor_kwargs),
            joint_action_enabled=joint_action_enabled,
        )
    else:
        pred_cond = _call_latent_dit_predictor(
            predictor,
            predictor_kwargs,
            joint_action_enabled=joint_action_enabled,
        )
        pred_uncond = _call_latent_dit_predictor(
            predictor,
            _metadata_uncond_kwargs(predictor_kwargs),
            joint_action_enabled=joint_action_enabled,
        )
        model_output = _merge_model_outputs(metadata_condition_mask, pred_cond, pred_uncond)

    model_pred = _world_prediction(model_output)
    if objective == "flow_matching":
        velocity_pred = model_pred
        x0_pred = x_t + (1.0 - t_expand) * velocity_pred
    else:
        x0_pred = model_pred
        velocity_pred = (x0_pred - x_t) / (1.0 - t_expand).clamp_min(1e-5)
    x0_loss = F.mse_loss(x0_pred, target_future_tokens)
    z_ar = x0_pred
    if runtime_normalize_reps:
        z_ar = F.layer_norm(z_ar, (z_ar.size(-1),))

    flow_loss = F.mse_loss(velocity_pred, velocity_target)
    x0_loss_weight = float(unwrap_module(predictor).x0_loss_weight)
    if objective == "flow_matching":
        objective_loss = flow_loss + x0_loss_weight * x0_loss
    else:
        objective_loss = x0_loss

    action_loss = None
    action_objective_loss = None
    action_loss_sum = None
    action_num_samples = None
    action_x0_pred = None
    action_velocity_pred = None
    if joint_action_enabled:
        action_pred = _action_prediction(model_output)
        if objective == "flow_matching":
            action_velocity_pred = action_pred
            action_x0_pred = action_x_t + (1.0 - action_t_expand) * action_velocity_pred
            action_per_sample_loss = (
                F.mse_loss(action_velocity_pred, action_velocity_target, reduction="none").flatten(1).mean(dim=1)
            )
        else:
            action_x0_pred = action_pred
            action_velocity_pred = (action_x0_pred - action_x_t) / (1.0 - action_t_expand).clamp_min(1e-5)
            action_per_sample_loss = (
                F.mse_loss(action_x0_pred, target_future_actions, reduction="none").flatten(1).mean(dim=1)
            )
        if imitation_mask is None:
            action_mask = torch.ones(batch_size, dtype=torch.bool, device=action_per_sample_loss.device)
        else:
            if imitation_mask.dtype != torch.bool or imitation_mask.ndim != 1:
                raise ValueError(
                    "imitation_mask must be a bool tensor with shape [B], "
                    f"got dtype={imitation_mask.dtype}, shape={tuple(imitation_mask.shape)}"
                )
            if int(imitation_mask.shape[0]) != batch_size:
                raise ValueError(
                    f"imitation_mask length {int(imitation_mask.shape[0])} != latent-DiT batch {batch_size}"
                )
            action_mask = imitation_mask.to(device=action_per_sample_loss.device)

        action_loss_sum = (action_per_sample_loss * action_mask.to(dtype=action_per_sample_loss.dtype)).sum()
        action_num_samples = action_mask.sum().to(dtype=action_per_sample_loss.dtype)
        if distributed_action_normalization and imitation_mask is not None:
            action_objective_loss = distributed_masked_mean(
                action_per_sample_loss,
                action_mask,
                name="latent-DiT joint action imitation",
            )
        else:
            action_objective_loss = action_loss_sum / action_num_samples.clamp_min(1.0)
        action_loss = action_objective_loss
    loss = objective_loss
    if action_loss is not None:
        loss = loss + _joint_action_loss_weight(config) * action_loss
    return LatentDiTRuntimeOutput(
        z_ar=z_ar,
        loss=loss,
        flow_loss=flow_loss,
        x0_loss=x0_loss,
        x0_pred=x0_pred,
        velocity_pred=velocity_pred,
        velocity_target=velocity_target,
        future_token_indices=future_token_indices,
        objective=objective,
        objective_loss=objective_loss,
        action_loss=action_loss,
        action_objective_loss=action_objective_loss,
        action_loss_sum=action_loss_sum,
        action_num_samples=action_num_samples,
        action_x0_pred=action_x0_pred,
        action_velocity_pred=action_velocity_pred,
        action_velocity_target=action_velocity_target,
        target_future_actions=target_future_actions,
        active_prefix_steps=None if prefix_sample is None else prefix_sample.prefix_steps,
        prefix_distribution=None if prefix_sample is None else prefix_sample.distribution,
    )


@torch.no_grad()
def sample_latent_dit_predictor(
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    predictor_inputs,
    *,
    tokens_per_frame: int,
    num_observed_steps: int,
    runtime_normalize_reps: bool,
    num_inference_steps: int = 8,
    sampler_type: str = "heun",
    schedule_type: str = "cosine",
    temperature: float = 1.0,
    config: Any,
    future_start_step: int = 0,
    future_steps: Optional[int] = None,
    known_future_tokens: Optional[torch.Tensor] = None,
    known_future_start_step: int = 0,
    initial_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample clean future tokens for NavSim evaluation."""
    observed_tokens = z_context[:, : int(num_observed_steps) * int(tokens_per_frame)]
    predictor_core = unwrap_module(predictor)
    num_future_tokens = int(getattr(predictor_core, "num_future_tokens", 0))
    if num_future_tokens <= 0:
        raise ValueError("latent_dit predictor must expose positive num_future_tokens")
    if num_future_tokens % int(tokens_per_frame) != 0:
        raise ValueError(
            f"latent_dit predictor num_future_tokens={num_future_tokens} is not divisible by "
            f"tokens_per_frame={tokens_per_frame}"
        )
    total_future_steps = num_future_tokens // int(tokens_per_frame)
    masked_enabled = latent_dit_masked_inpainting_enabled(config)
    future_token_indices = None
    if future_steps is not None:
        requested_steps = int(future_steps)
        requested_full_horizon = int(future_start_step) == 0 and requested_steps == total_future_steps
        if not masked_enabled and not requested_full_horizon:
            raise ValueError(
                "LatentDiT short/dynamic rollout requires predictor_dit.masked_inpainting_enabled=true; "
                "the current checkpoint/config only supports the full fixed future horizon."
            )
        if masked_enabled:
            future_token_indices = _build_future_token_indices(
                future_start_step=int(future_start_step),
                future_steps=requested_steps,
                tokens_per_frame=tokens_per_frame,
                num_future_tokens=num_future_tokens,
                device=observed_tokens.device,
            )
    elif int(future_start_step) != 0:
        raise ValueError("future_start_step must be 0 when future_steps is None")

    known_future_token_indices = None
    if known_future_tokens is not None:
        if not masked_enabled:
            raise ValueError("known_future_tokens conditioning requires predictor_dit.masked_inpainting_enabled=true")
        if known_future_tokens.ndim != 3:
            raise ValueError(f"known_future_tokens must have shape [B, N, D], got {tuple(known_future_tokens.shape)}")
        if known_future_tokens.shape[1] % int(tokens_per_frame) != 0:
            raise ValueError(
                f"known_future_tokens length {known_future_tokens.shape[1]} is not divisible by "
                f"tokens_per_frame={tokens_per_frame}"
            )
        known_steps = int(known_future_tokens.shape[1]) // int(tokens_per_frame)
        known_future_token_indices = _build_future_token_indices(
            future_start_step=int(known_future_start_step),
            future_steps=known_steps,
            tokens_per_frame=tokens_per_frame,
            num_future_tokens=num_future_tokens,
            device=observed_tokens.device,
        )
    aux_inputs = prepare_latent_dit_aux_inputs(
        predictor_inputs,
        config=config,
        num_observed_steps=num_observed_steps,
    )
    sample_kwargs = {
        "observed_tokens": observed_tokens,
        "actions": aux_inputs.actions,
        "states": aux_inputs.states,
        "extrinsics": aux_inputs.extrinsics,
        "num_inference_steps": num_inference_steps,
        "sampler_type": sampler_type,
        "schedule_type": schedule_type,
        "temperature": temperature,
    }
    if initial_noise is not None:
        sample_kwargs["initial_noise"] = initial_noise
    if future_token_indices is not None:
        sample_kwargs["future_token_indices"] = future_token_indices
    if known_future_tokens is not None:
        sample_kwargs["known_future_tokens"] = known_future_tokens
        sample_kwargs["known_future_token_indices"] = known_future_token_indices
    metadata_condition_mask = _resolve_metadata_condition_mask(
        predictor_inputs,
        config=config,
        num_observed_steps=num_observed_steps,
        batch_size=observed_tokens.shape[0],
        device=observed_tokens.device,
        training=False,
    )
    needs_uncond = bool((~metadata_condition_mask).any().item()) or abs(_metadata_guidance_scale(config) - 1.0) > 1e-8
    _ensure_metadata_unconditional_supported(predictor_core, needed=needs_uncond)
    sample_kwargs["metadata_condition_mask"] = metadata_condition_mask
    sample_kwargs["metadata_guidance_scale"] = _metadata_guidance_scale(config)
    if _uses_anchor_frame(predictor):
        sample_kwargs["anchor_tokens"] = observed_tokens[:, -int(tokens_per_frame) :]
    z_ar = predictor_core.sample(**sample_kwargs)
    if runtime_normalize_reps:
        z_ar = F.layer_norm(z_ar, (z_ar.size(-1),))
    if future_token_indices is not None and _masked_sample_return_full(config):
        z_full = z_ar.new_zeros(z_ar.shape[0], num_future_tokens, z_ar.shape[-1])
        z_full.index_copy_(1, future_token_indices.to(device=z_ar.device), z_ar)
        z_ar = z_full
    return z_ar


@torch.no_grad()
def sample_latent_dit_joint_action_predictor(
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    predictor_inputs,
    *,
    tokens_per_frame: int,
    num_observed_steps: int,
    runtime_normalize_reps: bool,
    num_inference_steps: int = 8,
    sampler_type: str = "euler",
    schedule_type: str = "cosine",
    temperature: float = 1.0,
    config: Any,
    initial_world_noise: Optional[torch.Tensor] = None,
    initial_action_noise: Optional[torch.Tensor] = None,
) -> SimpleNamespace:
    """Sample clean future tokens plus denormalized future action predictions."""
    observed_tokens = z_context[:, : int(num_observed_steps) * int(tokens_per_frame)]
    predictor_core = unwrap_module(predictor)
    if not bool(getattr(predictor_core, "joint_action_enabled", False)):
        raise ValueError("sample_latent_dit_joint_action_predictor requires predictor.joint_action_enabled=True")
    if not _joint_action_enabled(config):
        raise ValueError("sample_latent_dit_joint_action_predictor requires predictor_dit.joint_action_enabled=true")

    num_future_steps = int(getattr(predictor_core, "num_future_steps", 0))
    if num_future_steps <= 0:
        raise ValueError("joint latent-DiT predictor must expose positive num_future_steps")
    action_state_tokens = build_last_observed_action_state_tokens(
        states=getattr(predictor_inputs, "states", None),
        num_observed_steps=num_observed_steps,
        num_future_steps=num_future_steps,
        state_dim=_joint_action_state_dim(config, predictor_core),
    ).to(device=observed_tokens.device, dtype=observed_tokens.dtype)
    aux_inputs = prepare_latent_dit_aux_inputs(
        predictor_inputs,
        config=config,
        num_observed_steps=num_observed_steps,
    )

    sample = predictor_core.sample_joint(
        observed_tokens=observed_tokens,
        action_state_tokens=action_state_tokens,
        actions=aux_inputs.actions,
        states=aux_inputs.states,
        extrinsics=aux_inputs.extrinsics,
        num_inference_steps=num_inference_steps,
        sampler_type=sampler_type,
        schedule_type=schedule_type,
        temperature=temperature,
        return_diagnostics=True,
        joint_action_inference_noise_mode=_joint_action_inference_noise_mode(config, predictor_core),
        joint_video_final_noise=_joint_video_final_noise(config, predictor_core),
        initial_world_noise=initial_world_noise,
        initial_action_noise=initial_action_noise,
    )
    z_ar = sample["samples"]
    if runtime_normalize_reps:
        z_ar = F.layer_norm(z_ar, (z_ar.size(-1),))
    normalized_actions = sample["actions"]
    actions = denormalize_joint_actions(normalized_actions, _joint_action_scale(config))
    return SimpleNamespace(
        z_ar=z_ar,
        actions=actions,
        normalized_actions=normalized_actions,
        diagnostics=sample,
    )
