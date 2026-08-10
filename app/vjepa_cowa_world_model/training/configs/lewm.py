"""Latent world-model + refinement config dataclasses (renamed from the old lewm/stage2/stage3 schema).

Section keys in YAML/config are: `world_model` (was `lewm`/`le-wm`), `refinement` (was `stage2`),
`refinement_gated` (was `stage3`). The pre-restructure class names are kept as aliases at the bottom for
any pickled configs still in circulation; `parse.py` remaps the old YAML keys at parse time.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorldModelConfig:
    """Latent world-model JEPA pipeline config (encoder + predictor; SIGReg). Was: lewm."""

    enabled: bool = False
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    projector_hidden_dim: int = 2048
    embed_dim: int = 192
    num_subspaces: int = 1
    subspace_dim: Optional[int] = None
    init_mode: str = "orthogonal_frozen"
    theta: float = 0.0


@dataclass
class StageLossWeightsConfig:
    """Stage-2 planner 损失权重。"""

    prop: float = 1.0
    refine: float = 1.0
    anchor: float = 0.0
    div: float = 0.0


@dataclass
class RefinementConfig:
    """Proposal + refinement planner config (frozen proposal provider; iterative rounds). Was: stage2."""

    num_modes: int = 6
    inference_num_rounds: int = 1
    predictor_rollout_seconds: Optional[float] = None
    detach_zfut: bool = True
    refine_stop_grad_to_proposer: bool = True
    refine_use_random_predictor_latent: bool = False
    refine_keep_initial_actions: bool = False
    predictor_finetune: bool = False
    checkpoint: Optional[str] = None
    lambdas: StageLossWeightsConfig = field(default_factory=StageLossWeightsConfig)


@dataclass
class RefinementGatedConfig:
    """Refinement planner with selective input-gating (the refine_use_* flags). Was: stage3."""

    num_rounds: int = 2
    round_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    predictor_rollout_seconds: Optional[float] = None
    grad_checkpoint: bool = True
    predictor_finetune: bool = False
    checkpoint: Optional[str] = None
    refine_use_z_context: bool = True
    refine_use_status_feature: bool = True
    refine_use_proposal_traj: bool = True
    refine_use_proposal_logits: bool = True
    refine_use_proposal_features: bool = True
    refine_use_predictor_rollout: bool = True
    refine_use_random_predictor_latent: bool = False
    refine_keep_initial_actions: bool = False
    use_multimodal_final: bool = False


# Backward-compat aliases (pre-restructure class names) for any pickled configs still in circulation.
LeWMConfig = WorldModelConfig
Stage2Config = RefinementConfig
Stage3Config = RefinementGatedConfig
