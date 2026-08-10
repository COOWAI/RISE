"""Manual mode expansion for single-trajectory staged proposals."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _as_mode_tensor(
    values: Optional[list[float]],
    *,
    num_modes: int,
    device: torch.device,
    dtype: torch.dtype,
    default: list[float],
    name: str,
) -> torch.Tensor:
    source = default if values is None else values
    if len(source) != num_modes:
        raise ValueError(f"{name} must contain {num_modes} values, got {len(source)}")
    return torch.tensor(source, device=device, dtype=dtype)


def _default_lateral_offsets(num_modes: int) -> list[float]:
    offsets = [0.0]
    magnitude = 0.8
    while len(offsets) < num_modes:
        offsets.append(magnitude)
        if len(offsets) < num_modes:
            offsets.append(-magnitude)
        magnitude += 0.8
    return offsets[:num_modes]


def _default_yaw_offsets_deg(num_modes: int) -> list[float]:
    offsets = [0.0]
    magnitude = 4.0
    while len(offsets) < num_modes:
        offsets.append(magnitude)
        if len(offsets) < num_modes:
            offsets.append(-magnitude)
        magnitude += 4.0
    return offsets[:num_modes]


def _default_speed_scales(num_modes: int) -> list[float]:
    scales = [1.0 for _ in range(num_modes)]
    if num_modes > 1:
        scales[-1] = 0.75
    return scales


def _wrap_yaw(yaw: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(yaw), torch.cos(yaw))


def _select_top_conf_mode(
    trajectories: torch.Tensor,
    confidences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if trajectories.shape[0] != confidences.shape[0] or trajectories.shape[1] != confidences.shape[1]:
        raise ValueError(
            "manual expansion requires trajectories and confidences to have the same number of modes, "
            f"got trajectories={tuple(trajectories.shape)} confidences={tuple(confidences.shape)}"
        )
    if trajectories.shape[1] < 1:
        raise ValueError("manual expansion requires at least one proposal mode")

    top_idx = confidences.argmax(dim=1)
    traj_idx = top_idx.view(-1, 1, 1, 1).expand(-1, 1, trajectories.shape[2], trajectories.shape[3])
    conf_idx = top_idx.view(-1, 1)
    return trajectories.gather(1, traj_idx), confidences.gather(1, conf_idx)


def expand_single_mode_proposal(proposal_out: dict[str, torch.Tensor], config: Any) -> dict[str, torch.Tensor]:
    """Expand a frozen proposal into deterministic K anchor modes.

    Parameters
    ----------
    proposal_out : dict
        Contains ``trajectories`` [B, M, num_poses, 3] and ``confidences`` [B, M].
        If M > 1, the top-confidence mode is selected as the expansion anchor.
    config : object
        Proposal config with ``num_modes`` and optional manual expansion lists.

    Returns
    -------
    dict
        Proposal-like output with expanded ``trajectories`` [B, K, num_poses, 3]
        and ``confidences`` [B, K]. ``proposal_features`` is intentionally dropped
        because mode features must be regenerated from expanded trajectories.
    """
    trajectories = proposal_out["trajectories"]
    confidences = proposal_out["confidences"]
    if trajectories.ndim != 4:
        raise ValueError(f"manual expansion expects trajectories [B, M, T, 3], got {tuple(trajectories.shape)}")
    if confidences.ndim != 2:
        raise ValueError(f"manual expansion expects confidences [B, M], got {tuple(confidences.shape)}")
    trajectories, confidences = _select_top_conf_mode(trajectories, confidences)

    num_modes = int(_config_value(config, "num_modes"))
    if num_modes < 1:
        raise ValueError(f"num_modes must be >= 1, got {num_modes}")

    device = trajectories.device
    dtype = trajectories.dtype
    num_poses = trajectories.shape[2]
    ramp_power = float(_config_value(config, "manual_ramp_power", 1.5))
    if ramp_power <= 0:
        raise ValueError(f"manual_ramp_power must be > 0, got {ramp_power}")

    lateral_offsets = _as_mode_tensor(
        _config_value(config, "manual_lateral_offsets"),
        num_modes=num_modes,
        device=device,
        dtype=dtype,
        default=_default_lateral_offsets(num_modes),
        name="manual_lateral_offsets",
    )
    yaw_offsets_deg = _as_mode_tensor(
        _config_value(config, "manual_yaw_offsets_deg"),
        num_modes=num_modes,
        device=device,
        dtype=dtype,
        default=_default_yaw_offsets_deg(num_modes),
        name="manual_yaw_offsets_deg",
    )
    speed_scales = _as_mode_tensor(
        _config_value(config, "manual_speed_scales"),
        num_modes=num_modes,
        device=device,
        dtype=dtype,
        default=_default_speed_scales(num_modes),
        name="manual_speed_scales",
    )

    ramp = torch.linspace(1.0 / num_poses, 1.0, steps=num_poses, device=device, dtype=dtype).pow(ramp_power)
    base = trajectories[:, :1]
    expanded = base.repeat(1, num_modes, 1, 1).clone()
    expanded[..., 0] = expanded[..., 0] * speed_scales.view(1, num_modes, 1)
    expanded[..., 1] = expanded[..., 1] + ramp.view(1, 1, num_poses) * lateral_offsets.view(1, num_modes, 1)
    yaw_offsets = yaw_offsets_deg * (math.pi / 180.0)
    expanded[..., 2] = _wrap_yaw(expanded[..., 2] + ramp.view(1, 1, num_poses) * yaw_offsets.view(1, num_modes, 1))

    confidence_temperature = float(_config_value(config, "manual_confidence_temperature", 1.0))
    if confidence_temperature <= 0:
        raise ValueError(f"manual_confidence_temperature must be > 0, got {confidence_temperature}")
    prior_penalty = torch.ones(num_modes, device=confidences.device, dtype=confidences.dtype)
    prior_penalty[0] = 0.0
    expanded_confidences = confidences[:, :1] - prior_penalty.view(1, num_modes) / confidence_temperature

    result = {key: value for key, value in proposal_out.items() if key != "proposal_features"}
    result["trajectories"] = expanded
    result["confidences"] = expanded_confidences
    return result
