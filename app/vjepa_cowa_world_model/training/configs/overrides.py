"""Command-line config override helpers."""

from typing import Any, Dict, List, Sequence

import yaml


def apply_dotted_overrides(params: Dict[str, Any], assignments: Sequence[str]) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` assignments to a resolved config dict.

    Parameters
    ----------
    params:
        Resolved config dictionary to mutate in place.
    assignments:
        Repeatable command-line assignments. Values are parsed with ``yaml.safe_load``.

    Returns
    -------
    dict
        The same ``params`` object, returned for call-site convenience.
    """
    for assignment in assignments:
        key, value = _parse_assignment(assignment)
        parts = _split_dotted_key(key, assignment)
        cursor = params
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                existing = {}
                cursor[part] = existing
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Cannot apply override {assignment!r}: intermediate path component {part!r} "
                    f"already exists and is not a dict"
                )
            cursor = existing
        cursor[parts[-1]] = value
    return params


def get_dotted_value(params: Dict[str, Any], dotted_key: str) -> Any:
    """Return the value at ``dotted_key`` from ``params``.

    Raises ``ValueError`` when the key path is malformed or cannot be traversed as dictionaries.
    """
    parts = _split_dotted_key(dotted_key, dotted_key)
    cursor: Any = params
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            raise ValueError(f"Cannot read override value for {dotted_key!r}: missing path component {part!r}")
        cursor = cursor[part]
    return cursor


def _parse_assignment(assignment: str) -> tuple[str, Any]:
    if "=" not in assignment:
        raise ValueError(f"Config override assignment is missing '=': {assignment!r}")
    key, raw_value = assignment.split("=", 1)
    if not key:
        raise ValueError(f"Config override assignment has an empty key: {assignment!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse config override value for {assignment!r}: {exc}") from exc
    return key, value


def _split_dotted_key(dotted_key: str, assignment: str) -> List[str]:
    parts = dotted_key.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"Config override assignment has an empty dotted path component: {assignment!r}")
    return parts
