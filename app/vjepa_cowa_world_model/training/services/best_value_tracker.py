"""Best checkpoint selection state for value-head validation."""

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Dict, Optional, TypedDict

BEST_VALUE_VALIDATION_LOSS_KEY = "best_value_validation_loss"
BEST_VALUE_EPOCH_KEY = "best_value_epoch"
BEST_VALUE_METRIC_SCHEMA_KEY = "best_value_metric_schema"
BEST_VALUE_METRIC_SCHEMA_VERSION_KEY = "best_value_metric_schema_version"
BEST_VALUE_METRIC_SIGNATURE_KEY = "best_value_metric_signature"
BEST_VALUE_METRIC_SCHEMA = "fixed_gt_truncated_return"
BEST_VALUE_METRIC_SCHEMA_VERSION = 1

VALUE_RETURN_DEFINITION = "fixed_gt_truncated_return"
VALUE_RETURN_DEFINITION_VERSION = 1
VALUE_CALIBRATION_DEFINITION = "smooth_l1_per_timestep"
VALUE_CALIBRATION_DEFINITION_VERSION = 1
VALUE_EPISODE_SCORE_DEFINITION = "mean_predicted_timestep_values"
VALUE_EPISODE_SCORE_DEFINITION_VERSION = 1


class ValueMetricSignature(TypedDict):
    """Canonical inputs and definitions that determine value-validation loss."""

    gamma: float
    frame_stride: int
    bootstrap_horizon: int
    progress_weight: float
    comfort_weight: float
    episode_ranking_margin: float
    validation_calibration_weight: float
    validation_ranking_weight: float
    return_definition: str
    return_definition_version: int
    calibration_definition: str
    calibration_definition_version: int
    episode_score_definition: str
    episode_score_definition_version: int


_SIGNATURE_FLOAT_KEYS = (
    "gamma",
    "progress_weight",
    "comfort_weight",
    "episode_ranking_margin",
    "validation_calibration_weight",
    "validation_ranking_weight",
)
_SIGNATURE_INT_KEYS = (
    "frame_stride",
    "bootstrap_horizon",
    "return_definition_version",
    "calibration_definition_version",
    "episode_score_definition_version",
)
_SIGNATURE_DEFINITIONS = {
    "return_definition": VALUE_RETURN_DEFINITION,
    "calibration_definition": VALUE_CALIBRATION_DEFINITION,
    "episode_score_definition": VALUE_EPISODE_SCORE_DEFINITION,
}
_SIGNATURE_VERSIONS = {
    "return_definition_version": VALUE_RETURN_DEFINITION_VERSION,
    "calibration_definition_version": VALUE_CALIBRATION_DEFINITION_VERSION,
    "episode_score_definition_version": VALUE_EPISODE_SCORE_DEFINITION_VERSION,
}
_SIGNATURE_KEYS = frozenset(_SIGNATURE_FLOAT_KEYS + _SIGNATURE_INT_KEYS + tuple(_SIGNATURE_DEFINITIONS))

_STATE_KEYS = frozenset(
    (
        BEST_VALUE_METRIC_SCHEMA_KEY,
        BEST_VALUE_METRIC_SCHEMA_VERSION_KEY,
        BEST_VALUE_METRIC_SIGNATURE_KEY,
        BEST_VALUE_VALIDATION_LOSS_KEY,
        BEST_VALUE_EPOCH_KEY,
    )
)


def _config_field(config: object, field: str) -> object:
    if isinstance(config, Mapping):
        if field not in config:
            raise KeyError(f"value metric signature config is missing {field!r}")
        return config[field]
    if not hasattr(config, field):
        raise AttributeError(f"value metric signature config is missing {field!r}")
    return getattr(config, field)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite, got {normalized}")
    return normalized


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{field} must be >= 1, got {normalized}")
    return normalized


def build_value_metric_signature(config: object, frame_stride: int) -> ValueMetricSignature:
    """Build the canonical signature for fixed value validation.

    ``frame_stride`` must be the resolved runtime stride, rather than an
    unresolved config alias.  Numeric values remain typed numbers in checkpoint
    state; no string formatting or float ``repr`` participates in identity.
    """
    value_planning = _config_field(config, "value_planning")
    gamma = _finite_float(_config_field(value_planning, "gamma"), field="gamma")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    return {
        "gamma": gamma,
        "frame_stride": _positive_int(frame_stride, field="frame_stride"),
        "bootstrap_horizon": _positive_int(
            _config_field(value_planning, "bootstrap_horizon"),
            field="bootstrap_horizon",
        ),
        "progress_weight": _finite_float(
            _config_field(value_planning, "progress_weight"),
            field="progress_weight",
        ),
        "comfort_weight": _finite_float(
            _config_field(value_planning, "comfort_weight"),
            field="comfort_weight",
        ),
        "episode_ranking_margin": _finite_float(
            _config_field(value_planning, "episode_ranking_margin"),
            field="episode_ranking_margin",
        ),
        "validation_calibration_weight": _finite_float(
            _config_field(value_planning, "validation_calibration_weight"),
            field="validation_calibration_weight",
        ),
        "validation_ranking_weight": _finite_float(
            _config_field(value_planning, "validation_ranking_weight"),
            field="validation_ranking_weight",
        ),
        "return_definition": VALUE_RETURN_DEFINITION,
        "return_definition_version": VALUE_RETURN_DEFINITION_VERSION,
        "calibration_definition": VALUE_CALIBRATION_DEFINITION,
        "calibration_definition_version": VALUE_CALIBRATION_DEFINITION_VERSION,
        "episode_score_definition": VALUE_EPISODE_SCORE_DEFINITION,
        "episode_score_definition_version": VALUE_EPISODE_SCORE_DEFINITION_VERSION,
    }


def _validate_normalized_signature(signature: object) -> ValueMetricSignature:
    if not isinstance(signature, Mapping):
        raise TypeError(f"value metric signature must be a mapping, got {type(signature).__name__}")

    signature_keys = frozenset(signature.keys())
    missing = _SIGNATURE_KEYS - signature_keys
    if missing:
        raise KeyError(f"value metric signature is missing keys: {sorted(missing)}")
    unexpected = signature_keys - _SIGNATURE_KEYS
    if unexpected:
        raise KeyError(f"value metric signature has unexpected keys: {sorted(unexpected)}")

    for key in _SIGNATURE_FLOAT_KEYS:
        value = signature[key]
        if type(value) is not float:
            raise TypeError(f"value metric signature {key} must be a normalized float, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"value metric signature {key} must be finite, got {value}")

    for key in _SIGNATURE_INT_KEYS:
        value = signature[key]
        if type(value) is not int:
            raise TypeError(f"value metric signature {key} must be a normalized integer, got {value!r}")
        if value < 1:
            raise ValueError(f"value metric signature {key} must be >= 1, got {value}")

    for key, expected in _SIGNATURE_DEFINITIONS.items():
        value = signature[key]
        if value != expected:
            raise ValueError(f"value metric signature mismatch: {key} must be {expected!r}, got {value!r}")
    for key, expected in _SIGNATURE_VERSIONS.items():
        value = signature[key]
        if value != expected:
            raise ValueError(f"value metric signature mismatch: {key} must be {expected}, got {value}")

    if not 0.0 < signature["gamma"] <= 1.0:
        raise ValueError(f"value metric signature gamma must be in (0, 1], got {signature['gamma']}")

    return dict(signature)  # type: ignore[return-value]


class BestValueTracker:
    """Track the lowest finite value-validation loss seen so far.

    Equal losses do not replace the existing best, so the first checkpoint at a
    tied loss remains selected.  The serialized field names are deliberately
    value-specific and can be nested under ``best_value_tracker_state`` in a
    training checkpoint without sharing predictor-selection state.
    """

    def __init__(self, expected_metric_signature: ValueMetricSignature) -> None:
        self._metric_signature = _validate_normalized_signature(expected_metric_signature)
        self._loss: Optional[float] = None
        self.epoch = 0

    @property
    def loss(self) -> float:
        """Return the historical best, or ``inf`` before the first observation."""
        if self._loss is None:
            return float("inf")
        return self._loss

    def consider(self, validation_loss: float, epoch: int) -> bool:
        """Adopt a strictly lower finite validation loss.

        Parameters
        ----------
        validation_loss:
            Scalar value-head validation loss. NaN and infinities are invalid.
        epoch:
            Non-negative training epoch associated with the candidate.

        Returns
        -------
        bool:
            ``True`` only when the candidate becomes the new historical best.
        """
        candidate = self._validate_loss(validation_loss)
        candidate_epoch = self._validate_epoch(epoch)

        if self._loss is not None and candidate >= self._loss:
            return False

        self._loss = candidate
        self.epoch = candidate_epoch
        return True

    def state_dict(self) -> Dict[str, object]:
        """Return the checkpoint-safe historical selection state."""
        return {
            BEST_VALUE_METRIC_SCHEMA_KEY: BEST_VALUE_METRIC_SCHEMA,
            BEST_VALUE_METRIC_SCHEMA_VERSION_KEY: BEST_VALUE_METRIC_SCHEMA_VERSION,
            BEST_VALUE_METRIC_SIGNATURE_KEY: dict(self._metric_signature),
            BEST_VALUE_VALIDATION_LOSS_KEY: self._loss,
            BEST_VALUE_EPOCH_KEY: self.epoch,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict` with strict validation."""
        if not isinstance(state, Mapping):
            raise TypeError(f"BestValueTracker state must be a mapping, got {type(state).__name__}")

        state_keys = frozenset(state.keys())
        missing = _STATE_KEYS - state_keys
        if missing:
            raise KeyError(f"BestValueTracker state is missing keys: {sorted(missing)}")
        unexpected = state_keys - _STATE_KEYS
        if unexpected:
            raise KeyError(f"BestValueTracker state has unexpected keys: {sorted(unexpected)}")

        schema = state[BEST_VALUE_METRIC_SCHEMA_KEY]
        if schema != BEST_VALUE_METRIC_SCHEMA:
            raise ValueError(
                f"BestValueTracker metric schema mismatch: expected {BEST_VALUE_METRIC_SCHEMA!r}, got {schema!r}"
            )
        version = state[BEST_VALUE_METRIC_SCHEMA_VERSION_KEY]
        if isinstance(version, bool) or not isinstance(version, Integral):
            raise TypeError(f"BestValueTracker metric schema version must be an integer, got {version!r}")
        if int(version) != BEST_VALUE_METRIC_SCHEMA_VERSION:
            raise ValueError(
                "BestValueTracker metric schema version mismatch: "
                f"expected {BEST_VALUE_METRIC_SCHEMA_VERSION}, got {version}"
            )

        stored_signature = _validate_normalized_signature(state[BEST_VALUE_METRIC_SIGNATURE_KEY])
        if stored_signature != self._metric_signature:
            differing_keys = sorted(
                key for key in _SIGNATURE_KEYS if stored_signature[key] != self._metric_signature[key]
            )
            raise ValueError(
                "BestValueTracker metric signature mismatch for keys "
                f"{differing_keys}: expected {self._metric_signature!r}, got {stored_signature!r}"
            )

        stored_loss = state[BEST_VALUE_VALIDATION_LOSS_KEY]
        stored_epoch = self._validate_epoch(state[BEST_VALUE_EPOCH_KEY])
        if stored_loss is None:
            if stored_epoch != 0:
                raise ValueError("BestValueTracker empty history must use best_value_epoch=0")
            restored_loss = None
        else:
            restored_loss = self._validate_loss(stored_loss)

        self._loss = restored_loss
        self.epoch = stored_epoch

    @staticmethod
    def _validate_loss(value: object) -> float:
        if isinstance(value, bool):
            raise TypeError("best value validation loss must be a real finite scalar, not bool")
        try:
            loss = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("best value validation loss must be a real finite scalar") from exc
        if not math.isfinite(loss):
            raise ValueError(f"best value validation loss must be finite, got {loss}")
        return loss

    @staticmethod
    def _validate_epoch(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"best value epoch must be a non-negative integer, got {value!r}")
        epoch = int(value)
        if epoch < 0:
            raise ValueError(f"best value epoch must be non-negative, got {epoch}")
        return epoch
