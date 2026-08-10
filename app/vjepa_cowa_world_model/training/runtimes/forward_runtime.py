"""Phase 1 — single-source runtime spec + forward helpers for the encoder-direct path.

The framework consistency contract: training and eval/viz build the SAME
``EncoderDirectRuntimeSpec`` from one config and drive the SAME pure helpers, so their
encode / status / action-history / planner-call inputs cannot drift. This first slice
formalizes the *encoder-direct* path only (preprocessing/clip-prep parity is a later slice).

The underlying resolvers and ops still live in ``training/runtimes/encoder_token_runtime.py`` and
``utils/status_features.py``; this module bundles the resolved values into one frozen,
hashable spec and exposes spec-driven helpers that mirror the existing train/eval call sites
(``training/lines/planner_encoder_only.py`` and ``evaluation/navsim_agent.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from app.vjepa_cowa_world_model.training.runtimes.encoder_token_runtime import (
    is_vjepa_img_encoder,
    resolve_action_history_dt,
    resolve_encoder_direct_num_time_steps,
    resolve_encoder_direct_tokens_per_frame,
)
from app.vjepa_cowa_world_model.utils import (
    build_observed_action_trajectory_history,
    prepare_inference_consistent_status_vector,
    resolve_effective_planner_status_dim,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
)

RUNTIME_SPEC_VERSION = 1


@dataclass(frozen=True)
class EncoderDirectRuntimeSpec:
    """Frozen, hashable bundle of every resolved value the encoder-direct forward needs.

    Built once from config; train and eval build it identically, then read these fields
    instead of re-deriving from config, so the two paths cannot diverge. ``status_state_dim``
    (what ``prepare_inference_consistent_status_vector(state_dim=...)`` consumes) is kept
    distinct from ``planner_status_dim`` (what the planner module is constructed with) on
    purpose. ``action_history_frames`` is intentionally NOT a field — train and eval resolve
    it differently (train reads ``planner.num_observed_frames``; eval adapts from checkpoint).
    """

    version: int
    runtime_mode: str
    backbone_is_vjepa: bool
    vjepa_use_causal_attention: bool
    num_observed_frames: int
    num_target_frames: int
    num_poses: int
    tokens_per_frame: int
    tubelet_size: int
    num_time_steps: int
    normalize_reps: bool
    status_feature_mode: str
    status_state_dim: int
    planner_status_dim: int
    use_drive_command: bool
    command_dim: int
    use_action_history: bool
    action_history_dim: int
    action_history_dt: float

    @classmethod
    def from_config(cls, config: Any, encoder: Optional[Any] = None) -> "EncoderDirectRuntimeSpec":
        num_observed = int(config.train.num_observed_frames)
        num_target = int(config.data.num_target_frames)
        use_command = bool(resolve_planner_use_drive_command(config))
        # Mirror init_encoder_direct_planner's command_dim resolution exactly.
        if (
            use_command
            and bool(config.planner.split_status_embedding)
            and bool(config.train.predictor_inference_consistent)
        ):
            command_dim = 4
        else:
            command_dim = 0
        return cls(
            version=RUNTIME_SPEC_VERSION,
            runtime_mode="encoder_direct",
            backbone_is_vjepa=bool(is_vjepa_img_encoder(config)),
            vjepa_use_causal_attention=bool(config.model.vjepa_use_causal_attention),
            num_observed_frames=num_observed,
            num_target_frames=num_target,
            num_poses=num_target - num_observed,
            tokens_per_frame=int(resolve_encoder_direct_tokens_per_frame(config, encoder)),
            tubelet_size=int(config.data.tubelet_size),
            num_time_steps=int(resolve_encoder_direct_num_time_steps(config, encoder)),
            normalize_reps=bool(config.loss.normalize_reps),
            status_feature_mode="inference_consistent",
            status_state_dim=int(resolve_planner_status_dim(config)),
            planner_status_dim=int(resolve_effective_planner_status_dim(config)),
            use_drive_command=use_command,
            command_dim=command_dim,
            use_action_history=bool(config.planner.use_action_history_for_planner),
            action_history_dim=int(config.planner.action_history_dim),
            action_history_dt=float(resolve_action_history_dt(config)),
        )

    def canonical_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in sorted(asdict(self).items())}

    def content_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- pure, spec-driven helpers (shared verbatim by train + eval/viz) -----------------------


def encode_encoder_direct_context(
    spec: EncoderDirectRuntimeSpec, encoder: Any, context_clips: torch.Tensor
) -> torch.Tensor:
    """Encode observed context clips into planner token features, then optionally layer-norm.

    Mirrors ``encoder_token_runtime.forward_encoder_direct_tokens`` + the
    ``if config.loss.normalize_reps: F.layer_norm`` step both call sites apply afterwards,
    folded into one consistent step keyed only off the spec.
    """
    if spec.backbone_is_vjepa:
        z = encoder(
            context_clips,
            num_observed_frames=spec.num_observed_frames,
            use_causal_attention=spec.vjepa_use_causal_attention,
        )
    else:
        batch_size, _, max_num_frames, _, _ = context_clips.shape
        encoder_input = build_tubelet_encoder_input(context_clips, spec.tubelet_size)
        z = encoder([encoder_input])[0]
        z = z.view(batch_size, max_num_frames, -1, z.size(-1)).flatten(1, 2)
    if spec.normalize_reps:
        z = F.layer_norm(z, (z.size(-1),))
    return z


def slice_encoder_direct_observed_tokens(spec: EncoderDirectRuntimeSpec, z: torch.Tensor) -> torch.Tensor:
    """Select the observed-frame tokens fed to the planner."""
    if spec.backbone_is_vjepa:
        return z
    return z[:, : spec.num_observed_frames * spec.tokens_per_frame]


def build_encoder_direct_status_feature(
    spec: EncoderDirectRuntimeSpec,
    states: torch.Tensor,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the inference-consistent planner status vector ``[B, status_state_dim]``."""
    return prepare_inference_consistent_status_vector(
        states,
        num_observed=spec.num_observed_frames,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        state_dim=spec.status_state_dim,
        use_drive_command=spec.use_drive_command,
    )


def build_encoder_direct_action_history(
    spec: EncoderDirectRuntimeSpec, actions: torch.Tensor, num_observed_frames: int
) -> Optional[torch.Tensor]:
    """Build the planner action-history tensor, or ``None`` when disabled.

    ``num_observed_frames`` is passed explicitly (not taken from the spec) because train and
    eval resolve it differently — train reads ``planner.num_observed_frames`` while eval
    adapts it from the loaded checkpoint.
    """
    if not spec.use_action_history:
        return None
    return build_observed_action_trajectory_history(
        actions,
        num_observed_frames=int(num_observed_frames),
        action_history_dim=spec.action_history_dim,
        dt=spec.action_history_dt,
    )


def forward_encoder_direct_planner(
    planner: Any,
    tokens: torch.Tensor,
    status_feature: torch.Tensor,
    action_history: Optional[torch.Tensor] = None,
    inference_noise: Optional[torch.Tensor] = None,
) -> Any:
    """Invoke the planner with the encoder-direct contract."""
    kwargs = {"action_history": action_history}
    if inference_noise is not None:
        kwargs["inference_noise"] = inference_noise
    return planner(tokens, status_feature, **kwargs)


class ForwardRuntime:
    """Thin object binding an ``EncoderDirectRuntimeSpec`` to its encoder + planner.

    Train and eval/viz each construct this from the same (checkpoint) config and call the
    same methods, so their forward inputs are identical by construction.
    """

    def __init__(self, spec: EncoderDirectRuntimeSpec, encoder: Any = None, planner: Any = None):
        self.spec = spec
        self.encoder = encoder
        self.planner = planner

    @classmethod
    def encoder_direct_from_config(cls, config: Any, encoder: Any = None, planner: Any = None) -> "ForwardRuntime":
        return cls(EncoderDirectRuntimeSpec.from_config(config, encoder), encoder=encoder, planner=planner)

    def encode_context(self, context_clips: torch.Tensor) -> torch.Tensor:
        return encode_encoder_direct_context(self.spec, self.encoder, context_clips)

    def observed_tokens(self, z: torch.Tensor) -> torch.Tensor:
        return slice_encoder_direct_observed_tokens(self.spec, z)

    def status_feature(
        self,
        states: torch.Tensor,
        driving_command: Optional[torch.Tensor] = None,
        ego_dynamics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return build_encoder_direct_status_feature(self.spec, states, driving_command, ego_dynamics)

    def action_history(self, actions: torch.Tensor, num_observed_frames: int) -> Optional[torch.Tensor]:
        return build_encoder_direct_action_history(self.spec, actions, num_observed_frames)

    def forward_planner(
        self,
        tokens: torch.Tensor,
        status_feature: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Any:
        return forward_encoder_direct_planner(
            self.planner,
            tokens,
            status_feature,
            action_history,
            inference_noise=inference_noise,
        )
