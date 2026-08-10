"""Strict CSV field parsing shared by the retained direct NavTest EPDMS reader.

The direct scorer owns CSV inventory, schema, scenario coverage, summary-row,
and aggregate-mean validation.  This module only parses its scenario and
summary score fields without depending on the removed official-suite policy,
registry, trace, or report contracts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType
from typing import Optional, Protocol


class _CsvScoreProtocol(Protocol):
    """Structural component contract required by the direct CSV reader."""

    required_components: frozenset[str]
    nullable_components: frozenset[str]


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _unit_score(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _csv_number(value: str, *, field: str, unit_score: bool = False) -> float:
    """Parse one finite CSV number, optionally constrained to ``[0, 1]``."""

    if type(value) is not str:
        raise ValueError(f"{field} must be numeric text")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must not be blank")
    try:
        number = float(stripped)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    return _unit_score(number, field=field) if unit_score else _finite_number(number, field=field)


def _csv_components(
    row: Mapping[str, str], *, protocol: _CsvScoreProtocol, token: str
) -> Mapping[str, Optional[float]]:
    """Parse the protocol-declared component columns for one CSV row."""

    components: dict[str, Optional[float]] = {}
    for component in sorted(protocol.required_components):
        raw = row[component]
        if type(raw) is not str:
            raise ValueError(f"CSV token {token!r} component {component!r} must be numeric text")
        stripped = raw.strip()
        if stripped.lower() == "nan" or not stripped:
            if component not in protocol.nullable_components:
                raise ValueError(f"official score CSV token {token!r} has non-finite component {component!r}")
            components[component] = None
            continue
        components[component] = _csv_number(
            stripped,
            field=f"CSV token {token!r} component {component!r}",
        )
    return MappingProxyType(components)
