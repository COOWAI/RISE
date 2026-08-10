"""Shared training services (composition, used alongside pipeline.setup_run).

Small, single-responsibility helpers that several training lines need identically.
Import directly (e.g. ``from app.vjepa_cowa_world_model.training.services import
BestOpenLoopTracker``); these are deliberately kept out of the ``training`` package
facade so they stay free of import cycles with the line modules.
"""

from app.vjepa_cowa_world_model.training.services.best_checkpoint_tracker import (
    OPEN_LOOP_METRIC_SCHEMA,
    BestOpenLoopTracker,
    BestPredictorTracker,
)
from app.vjepa_cowa_world_model.training.services.best_value_tracker import (
    BEST_VALUE_EPOCH_KEY,
    BEST_VALUE_METRIC_SCHEMA,
    BEST_VALUE_METRIC_SCHEMA_KEY,
    BEST_VALUE_METRIC_SCHEMA_VERSION,
    BEST_VALUE_METRIC_SCHEMA_VERSION_KEY,
    BEST_VALUE_METRIC_SIGNATURE_KEY,
    BEST_VALUE_VALIDATION_LOSS_KEY,
    BestValueTracker,
    ValueMetricSignature,
    build_value_metric_signature,
)

__all__ = [
    "BEST_VALUE_EPOCH_KEY",
    "BEST_VALUE_METRIC_SCHEMA",
    "BEST_VALUE_METRIC_SCHEMA_KEY",
    "BEST_VALUE_METRIC_SCHEMA_VERSION",
    "BEST_VALUE_METRIC_SCHEMA_VERSION_KEY",
    "BEST_VALUE_METRIC_SIGNATURE_KEY",
    "BEST_VALUE_VALIDATION_LOSS_KEY",
    "BestOpenLoopTracker",
    "BestPredictorTracker",
    "BestValueTracker",
    "OPEN_LOOP_METRIC_SCHEMA",
    "ValueMetricSignature",
    "build_value_metric_signature",
]
