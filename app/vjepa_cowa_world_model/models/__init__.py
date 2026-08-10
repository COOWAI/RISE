# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
模型模块

包含:
- MultiModalTemporalPlanner: 统一的多模态时序 Planner
- DiffusionPlanner: 基于扩散模型的 Planner
- PrefixConditionedDiffusionPlanner: 带 rollout 前缀条件训练的扩散 Planner 变体
"""

# NOTE: the active flow planner is FlowMatchingDiffusionPlanner; diffusion_head's DiTFlowMatching /
# FlowMatchingScheduler / flow_matching_loss are an alternative, currently-unused implementation and are
# intentionally NOT re-exported here (import directly from .diffusion_head if ever needed).
from .diffusion_planner import DiffusionPlanner
from .diffusion_refinement_decoder import DiffusionRefinementDecoder
from .flow_matching_diffusion_planner import FlowMatchingDiffusionPlanner
from .latent_dit_predictor import LatentDiTPredictor
from .lewm_modules import PredProjectorMLP, ProjectionMLP, ProjectorMLP
from .multiview_fusion import PETRMultiViewFusion

# from .multimodal_planner import MultiModalTemporalPlanner
from .prefix_conditioned_diffusion_planner import PrefixConditionedDiffusionPlanner
from .proposal_providers import (
    DiffusionProposalProvider,
    HistoryKinematicProposalProvider,
    TransformerProposalProvider,
    build_proposal_provider,
)

# from .refinement_decoder import RefinementDecoder
from .seeded_diffusion_planner import PrefixConditionedSeededDiffusionPlanner, SeededDiffusionPlanner

__all__ = [
    # "MultiModalTemporalPlanner",
    "TransformerProposalProvider",
    "DiffusionProposalProvider",
    "HistoryKinematicProposalProvider",
    "build_proposal_provider",
    "RefinementDecoder",
    "DiffusionPlanner",
    "FlowMatchingDiffusionPlanner",
    "DiffusionRefinementDecoder",
    "build_refinement_decoder",
    "resolve_refinement_core_type",
    "ProjectorMLP",
    "PredProjectorMLP",
    "ProjectionMLP",
    "SeededDiffusionPlanner",
    "PrefixConditionedDiffusionPlanner",
    "PrefixConditionedSeededDiffusionPlanner",
    "PETRMultiViewFusion",
    "LatentDiTPredictor",
    "TemporalTrajectoryValueHead",
]


def __getattr__(name):
    if name == "MultiModalTemporalPlanner":
        from .multimodal_planner import MultiModalTemporalPlanner

        return MultiModalTemporalPlanner
    if name == "DiffusionPlanner":
        from .diffusion_planner import DiffusionPlanner

        return DiffusionPlanner
    if name == "FlowMatchingDiffusionPlanner":
        from .flow_matching_diffusion_planner import FlowMatchingDiffusionPlanner

        return FlowMatchingDiffusionPlanner
    if name == "DiffusionRefinementDecoder":
        from .diffusion_refinement_decoder import DiffusionRefinementDecoder

        return DiffusionRefinementDecoder
    if name == "build_refinement_decoder":
        from .refinement_decoder_factory import build_refinement_decoder

        return build_refinement_decoder
    if name == "resolve_refinement_core_type":
        from .refinement_decoder_factory import resolve_refinement_core_type

        return resolve_refinement_core_type
    if name == "HistoryKinematicProposalProvider":
        from .proposal_providers import HistoryKinematicProposalProvider

        return HistoryKinematicProposalProvider
    if name == "TransformerProposalProvider":
        from .proposal_providers import TransformerProposalProvider

        return TransformerProposalProvider
    if name == "DiffusionProposalProvider":
        from .proposal_providers import DiffusionProposalProvider

        return DiffusionProposalProvider
    if name == "build_proposal_provider":
        from .proposal_providers import build_proposal_provider

        return build_proposal_provider
    if name == "RefinementDecoder":
        from .refinement_decoder import RefinementDecoder

        return RefinementDecoder
    if name == "SeededDiffusionPlanner":
        from .seeded_diffusion_planner import SeededDiffusionPlanner

        return SeededDiffusionPlanner
    if name == "PrefixConditionedDiffusionPlanner":
        from .prefix_conditioned_diffusion_planner import PrefixConditionedDiffusionPlanner

        return PrefixConditionedDiffusionPlanner
    if name == "PrefixConditionedSeededDiffusionPlanner":
        from .seeded_diffusion_planner import PrefixConditionedSeededDiffusionPlanner

        return PrefixConditionedSeededDiffusionPlanner
    if name == "TemporalTrajectoryValueHead":
        from .trajectory_value import TemporalTrajectoryValueHead

        return TemporalTrajectoryValueHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
