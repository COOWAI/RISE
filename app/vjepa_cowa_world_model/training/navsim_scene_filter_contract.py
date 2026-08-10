"""Strict, content-addressed NavSim scene-filter parsing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import resolve_formal_v2_navsim_scene_filter_path


def _canonical_set_sha256(values: tuple[str, ...]) -> str:
    encoded = json.dumps(sorted(values), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class NavSimSceneFilterContract:
    """Parsed scene-filter values plus the exact content identity consumed at runtime."""

    path: Path
    file_sha256: str
    log_names: tuple[str, ...]
    tokens: tuple[str, ...]
    log_name_set_sha256: str
    token_set_sha256: str

    def to_receipt(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "log_name_count": len(self.log_names),
            "log_name_set_sha256": self.log_name_set_sha256,
            "token_count": len(self.tokens),
            "token_set_sha256": self.token_set_sha256,
        }


def load_navsim_scene_filter_contract(path: str | Path) -> NavSimSceneFilterContract:
    """Load one official filter without silently normalizing duplicates or missing fields."""

    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("NavSim scene filter path must be a non-empty string or Path")
    source = Path(path)
    if not source.is_absolute():
        source = resolve_formal_v2_navsim_scene_filter_path(path)
    if source.is_symlink():
        raise ValueError(f"NavSim scene filter must not be a symlink: {source}")
    try:
        resolved = source.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"NavSim scene filter yaml does not exist: {source}") from exc
    if not resolved.is_file():
        raise ValueError(f"NavSim scene filter yaml must be a regular file: {resolved}")
    try:
        raw_bytes = resolved.read_bytes()
        payload = yaml.safe_load(raw_bytes)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to read NavSim scene filter yaml {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"NavSim scene filter yaml must parse to a mapping: {resolved}")

    normalized: dict[str, tuple[str, ...]] = {}
    for key in ("log_names", "tokens"):
        section = payload.get(key)
        if not isinstance(section, list) or not section:
            raise ValueError(f"NavSim scene filter yaml is missing a non-empty '{key}' list: {resolved}")
        if any(type(item) is not str or not item for item in section):
            raise ValueError(f"NavSim scene filter yaml '{key}' must contain non-empty strings: {resolved}")
        values = tuple(section)
        seen: set[str] = set()
        for value in values:
            if value in seen:
                raise ValueError(f"NavSim scene filter yaml contains duplicate {key} value {value!r}: {resolved}")
            seen.add(value)
        normalized[key] = values

    log_names = normalized["log_names"]
    tokens = normalized["tokens"]
    return NavSimSceneFilterContract(
        path=resolved,
        file_sha256=_sha256_bytes(raw_bytes),
        log_names=log_names,
        tokens=tokens,
        log_name_set_sha256=_canonical_set_sha256(log_names),
        token_set_sha256=_canonical_set_sha256(tokens),
    )
