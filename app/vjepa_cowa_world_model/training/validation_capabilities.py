"""Fail-fast runtime capability checks for validation-suite execution."""

from __future__ import annotations

from typing import Any, Set


def validate_validation_suite_execution_contract(
    config: Any,
    *,
    line_name: str,
    declared_executors: Set[str],
    active_consumers: Set[str],
) -> None:
    """Reject suites whose configured training line cannot actually execute them."""

    if not bool(config.validation_suite.enabled):
        return
    if int(config.meta.val_freq) <= 0:
        raise ValueError("validation_suite requires meta.val_freq > 0")
    if not active_consumers:
        raise ValueError(f"validation_suite has no active consumer in {line_name}")

    supported_consumers = {"predictor", "planner"}
    unsupported = set(active_consumers) - supported_consumers
    if unsupported:
        raise ValueError(f"validation_suite has unsupported consumer(s) in {line_name}: {sorted(unsupported)}")
    if bool(config.budget_controller.enabled):
        raise ValueError("validation_suite cannot execute with budget_controller.enabled=true")
    if str(config.planner.policy_output_source).lower() == "joint_action":
        raise ValueError("validation_suite cannot execute with planner.policy_output_source='joint_action'")

    missing = set(active_consumers) - set(declared_executors)
    if missing:
        raise ValueError(f"{line_name} does not execute validation_suite consumers: {sorted(missing)}")
