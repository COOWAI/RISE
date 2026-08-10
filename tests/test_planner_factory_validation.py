"""Fail-loud validation in the planner factory.

init_planner must reject an unknown planner.planner_type instead of silently falling through
to the transformer default. The check fires before any model construction, so this needs only a
tiny stub config (no encoder/GPU).
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vjepa_cowa_world_model.training.model_factories.planner import init_planner  # noqa: E402


def _cfg(use_planner, planner_type, policy_output_source="planner"):
    return types.SimpleNamespace(
        predictor_dynamic_rollout=types.SimpleNamespace(enabled=False),
        planner=types.SimpleNamespace(
            use_planner=use_planner,
            planner_type=planner_type,
            policy_output_source=policy_output_source,
            diff_train_prefix_conditioning=False,
        ),
    )


def test_unknown_planner_type_raises():
    with pytest.raises(ValueError) as exc:
        init_planner(_cfg(True, "diffsion"), encoder_dim=384, device="cpu")  # typo
    assert "planner_type" in str(exc.value)


def test_disabled_planner_returns_none_without_validating_type():
    # use_planner=False short-circuits before the type check (returns None).
    assert init_planner(_cfg(False, "whatever"), encoder_dim=384, device="cpu") is None


def test_unknown_policy_output_source_raises():
    with pytest.raises(ValueError) as exc:
        init_planner(_cfg(True, "transformer", policy_output_source="unknown"), encoder_dim=384, device="cpu")
    assert "policy_output_source" in str(exc.value)


def test_direct_joint_action_policy_returns_none_without_instantiating_planner():
    planner = init_planner(_cfg(True, "whatever", policy_output_source="joint_action"), encoder_dim=384, device="cpu")
    assert planner is None
