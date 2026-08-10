"""Direct e120 full-state warm-start for the manually operated CVoI chain."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


FORMAL_V2_E120_CHECKPOINT_PATH = "/path/to/checkpoints/rise/e120.pt"
FORMAL_V2_E120_PARAMS_PRETRAIN_PATH = str(Path(FORMAL_V2_E120_CHECKPOINT_PATH).with_name("params-pretrain.yaml"))
FORMAL_V2_FULL_STATE_WARMSTART_ROLES = ("encoder", "predictor", "planner")
OBSERVED_SOURCE_EMBEDDING_KEY = "observed_source_embedding.weight"


def _load_torch() -> Any:
    """Import PyTorch only for state/checkpoint operations."""

    import torch

    return torch


def _normalize_parameter_key(raw_key: str, *, name: str) -> str:
    key = raw_key[7:] if raw_key.startswith("module.") else raw_key
    if not key:
        raise ValueError(f"{name} contains an empty normalized parameter key")
    return key


def _normalize_state(
    state: object,
    *,
    name: str,
    clone: bool,
) -> dict[str, torch.Tensor]:
    torch = _load_torch()
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a non-empty state mapping")
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            raise ValueError(f"{name} must map string parameter keys to tensors")
        key = _normalize_parameter_key(raw_key, name=name)
        if key in normalized:
            raise ValueError(f"{name} contains duplicate normalized parameter key {key!r}")
        normalized[key] = value.detach().clone(memory_format=torch.preserve_format) if clone else value
    return normalized


def _validate_observed_embedding(
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
    require_nonzero: bool,
) -> torch.Tensor:
    torch = _load_torch()
    embedding = state.get(OBSERVED_SOURCE_EMBEDDING_KEY)
    if not torch.is_tensor(embedding):
        raise ValueError(f"{name} must contain {OBSERVED_SOURCE_EMBEDDING_KEY!r}")
    if embedding.ndim != 2 or embedding.shape[0] != 2 or embedding.shape[1] <= 0:
        raise ValueError(f"{name} observed_source_embedding must have shape [2, D] with D > 0")
    if not torch.is_floating_point(embedding):
        raise ValueError(f"{name} observed_source_embedding must use a floating-point dtype")
    if not bool(torch.isfinite(embedding).all().item()):
        raise ValueError(f"{name} observed_source_embedding must contain only finite values")
    if require_nonzero and torch.count_nonzero(embedding).item() == 0:
        raise ValueError(f"{name} observed_source_embedding must not be all zeros")
    return embedding


def _require_exact_roles(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(role, str) for role in value):
        raise ValueError(f"{name} roles must be strings")
    expected = set(FORMAL_V2_FULL_STATE_WARMSTART_ROLES)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} roles mismatch: missing={missing}, unexpected={unexpected}")
    return value


def _validate_role_states(
    checkpoint_payload: Mapping[str, object],
    *,
    target_states: object,
) -> dict[str, dict[str, torch.Tensor]]:
    targets = _require_exact_roles(target_states, name="target_states")
    missing_source_roles = [role for role in FORMAL_V2_FULL_STATE_WARMSTART_ROLES if role not in checkpoint_payload]
    if missing_source_roles:
        raise ValueError(f"source checkpoint is missing required roles: {missing_source_roles}")

    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for role in FORMAL_V2_FULL_STATE_WARMSTART_ROLES:
        source = _normalize_state(checkpoint_payload[role], name=f"source {role} state", clone=True)
        target = _normalize_state(targets[role], name=f"target {role} state", clone=False)
        if role == "planner":
            _validate_observed_embedding(target, name="target Planner", require_nonzero=False)
            _validate_observed_embedding(source, name="source Planner", require_nonzero=True)
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        if missing or unexpected:
            raise ValueError(f"{role} parameter keys mismatch: missing={missing}, unexpected={unexpected}")
        for key, source_value in source.items():
            target_value = target[key]
            if tuple(source_value.shape) != tuple(target_value.shape):
                raise ValueError(
                    f"{role} parameter {key!r} shape mismatch: source={tuple(source_value.shape)}, "
                    f"target={tuple(target_value.shape)}"
                )
        prepared[role] = source
    return prepared


def prepare_formal_v2_full_state_warmstart(
    checkpoint_payload: object,
    *,
    target_states: object,
) -> dict[str, dict[str, torch.Tensor]]:
    """Validate and clone the three model-role states without provenance."""

    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("e120 checkpoint payload must be a mapping")
    return _validate_role_states(checkpoint_payload, target_states=target_states)


def _canonical_lexical_absolute_path(value: str | Path, *, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} path must be an absolute path")
    raw = str(value)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    if ".." in path.parts:
        raise ValueError(f"{name} path must not contain traversal ('..'): {raw}")
    if raw != str(path):
        raise ValueError(f"{name} path must be canonical in lexical form: {raw}")
    return path


def validate_formal_v2_full_state_warmstart_paths(
    checkpoint_path: str | Path,
    params_pretrain_path: str | Path,
) -> tuple[Path, Path]:
    """Validate one portable e120 checkpoint and adjacent params pair without filesystem access."""

    checkpoint = _canonical_lexical_absolute_path(checkpoint_path, name="e120 checkpoint")
    params = _canonical_lexical_absolute_path(params_pretrain_path, name="e120 params-pretrain")
    if checkpoint.suffix != ".pt":
        raise ValueError(f"e120 checkpoint path must have the .pt extension: {checkpoint}")
    if params.name != "params-pretrain.yaml":
        raise ValueError(f"e120 params-pretrain basename must be exactly 'params-pretrain.yaml': {params}")
    if checkpoint.parent != params.parent:
        raise ValueError("e120 checkpoint and params-pretrain paths must share the same parent directory")
    return checkpoint, params


def _module_target_states(modules: object) -> tuple[Mapping[str, object], dict[str, object]]:
    torch = _load_torch()
    normalized_modules = _require_exact_roles(modules, name="modules")
    targets: dict[str, object] = {}
    for role in FORMAL_V2_FULL_STATE_WARMSTART_ROLES:
        module = normalized_modules[role]
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"modules[{role!r}] must be a torch.nn.Module")
        targets[role] = module.state_dict()
    return normalized_modules, targets


def _normalized_to_raw_keys(state: object, *, name: str) -> dict[str, str]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a non-empty state mapping")
    keys: dict[str, str] = {}
    for raw_key in state:
        if not isinstance(raw_key, str):
            raise ValueError(f"{name} parameter keys must be strings")
        normalized = _normalize_parameter_key(raw_key, name=name)
        if normalized in keys:
            raise ValueError(f"{name} contains duplicate normalized parameter key {normalized!r}")
        keys[normalized] = raw_key
    return keys


def _require_direct_regular_file(path: Path, *, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    return path


def apply_formal_v2_full_state_warmstart_direct(
    checkpoint_path: str | Path,
    params_pretrain_path: str | Path,
    modules: object,
) -> None:
    """Apply one configured e120 initialization pair without hashes or receipts."""

    checkpoint, params = validate_formal_v2_full_state_warmstart_paths(
        checkpoint_path,
        params_pretrain_path,
    )
    _require_direct_regular_file(checkpoint, name="e120 checkpoint")
    _require_direct_regular_file(params, name="e120 params-pretrain")

    normalized_modules, target_states = _module_target_states(modules)
    payload = _load_torch().load(checkpoint, map_location="cpu", weights_only=False)
    prepared = prepare_formal_v2_full_state_warmstart(payload, target_states=target_states)
    for role in FORMAL_V2_FULL_STATE_WARMSTART_ROLES:
        module = normalized_modules[role]
        raw_keys = _normalized_to_raw_keys(module.state_dict(), name=f"target {role} state")
        load_state = {raw_keys[key]: value for key, value in prepared[role].items()}
        module.load_state_dict(load_state, strict=True)
