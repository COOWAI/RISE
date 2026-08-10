"""Evaluation integrations with lazy NavSim-dependent public exports."""

from typing import Any

__all__ = ("VJEPAWorldModelAgent", "VJEPAFeatureBuilder")


def __getattr__(name: str) -> Any:
    """Load NavSim integrations only when callers request their public class."""

    if name == "VJEPAWorldModelAgent":
        from app.vjepa_cowa_world_model.evaluation.navsim_agent import VJEPAWorldModelAgent

        globals()[name] = VJEPAWorldModelAgent
        return VJEPAWorldModelAgent
    if name == "VJEPAFeatureBuilder":
        from app.vjepa_cowa_world_model.evaluation.navsim_feature_builder import VJEPAFeatureBuilder

        globals()[name] = VJEPAFeatureBuilder
        return VJEPAFeatureBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
