"""Split from training/config.py (verbatim node moves). Part: common."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


_PROPOSAL_PRETRAIN_YAML_NAMES = ("params-pretrain.yaml", "params-pretrain.yml")

# The image-availability policy is dataset-agnostic; the NAVSIM_ prefix predates the dataset decoupling.
IMAGE_REQUIRE_AUTO = "auto"

IMAGE_REQUIRE_ALL_FRAMES = "all_frames"

IMAGE_REQUIRE_OBSERVED_ONLY = "observed_only"

IMAGE_REQUIRE_POLICIES = {
    IMAGE_REQUIRE_AUTO,
    IMAGE_REQUIRE_ALL_FRAMES,
    IMAGE_REQUIRE_OBSERVED_ONLY,
}

# Back-compat aliases (existing imports + configs reference the NAVSIM_ names).
NAVSIM_IMAGE_REQUIRE_AUTO = IMAGE_REQUIRE_AUTO
NAVSIM_IMAGE_REQUIRE_ALL_FRAMES = IMAGE_REQUIRE_ALL_FRAMES
NAVSIM_IMAGE_REQUIRE_OBSERVED_ONLY = IMAGE_REQUIRE_OBSERVED_ONLY
NAVSIM_IMAGE_REQUIRE_POLICIES = IMAGE_REQUIRE_POLICIES

# Observed-token vocabulary is canonical in the planner_contracts module; re-export the constants
# here so the config layer (planner.py / parse.py) keeps importing them from configs.common.
from app.vjepa_cowa_world_model.models.planner_contracts import (  # noqa: E402, F401
    PLANNER_OBSERVED_TOKEN_CONCAT,
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    PLANNER_OBSERVED_TOKEN_MODES,
    PLANNER_OBSERVED_TOKEN_NONE,
)


def _parse_hw_resolution(value: Any, label: str) -> Tuple[int, int]:
    try:
        resolution = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must contain [height, width]") from exc
    if len(resolution) != 2:
        raise ValueError(f"{label} must contain [height, width]")
    return int(resolution[0]), int(resolution[1])


def _get_nested_value(config: Any, *keys: str, default=None):
    """Resolve a nested config value.

    point 24: 区分"键不存在"与"键存在但显式为 null"。
      - 路径上任一键不存在 → 返回 default；
      - 末端键显式写成 null（YAML 里用 null 关掉某项）→ 返回 None，不再静默套回 default。
    这样可避免"显式 null 被默认值覆盖"这种系统级放大器。
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            # current 为 None 或不含该属性 → 无法继续下钻，视为键不存在。
            return default
    return current


def _resolve_observed_frames_strict(config: Any) -> int:
    """Resolve train.num_observed_frames（兼容 YAML 别名 train.num_encoder_frames）。

    fail-loud: 该字段决定 observed/future 帧切分（监督语义），缺失时禁止兜默认值——
    历史上此处 default=1 与解析器/dataclass 的 default=2 不一致，校验与运行时
    各自假设不同帧数。与 parse_training_config 的别名解析保持同一语义。
    """
    value = _get_nested_value(config, "train", "num_encoder_frames", default=None)
    if value is None:
        value = _get_nested_value(config, "train", "num_observed_frames", default=None)
    if value is None:
        raise ValueError(
            "train.num_observed_frames (or alias train.num_encoder_frames) is required; " "禁止静默兜默认帧数。"
        )
    return int(value)


def is_multiview_per_view_output(config: Any) -> bool:
    """Return whether multi-view fusion should keep one token stream per camera."""
    return bool(_get_nested_value(config, "multiview", "enabled", default=False)) and (
        str(_get_nested_value(config, "multiview", "output_mode", default="fused")).lower() == "per_view"
    )


def resolve_multiview_num_views(config: Any) -> int:
    """Resolve the number of camera views used by multi-view token streams."""
    camera_names = _get_nested_value(config, "data", "navsim", "camera_names", default=None)
    if camera_names:
        return len(list(camera_names))
    return 1


def apply_multiview_token_multiplier(config: Any, tokens_per_frame: int) -> int:
    """Expand tokens/frame when per-view multi-camera outputs are concatenated."""
    tokens = int(tokens_per_frame)
    if not is_multiview_per_view_output(config):
        return tokens
    num_views = resolve_multiview_num_views(config)
    if num_views <= 0:
        raise ValueError("multiview per_view output requires at least one camera")
    return tokens * num_views


def apply_multiview_image_size_multiplier(config: Any, image_size: Any):
    """Return a virtual predictor grid size for per-view camera-token concatenation."""
    if not is_multiview_per_view_output(config):
        return image_size
    height, width = normalize_image_size(image_size)
    return height, width * resolve_multiview_num_views(config)


def normalize_image_size(image_size: Any) -> Tuple[int, int]:
    if isinstance(image_size, int):
        size = int(image_size)
        return size, size
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"image size must be an int or a 2-element sequence, got {image_size!r}")


def compute_tokens_per_frame(image_size: Any, patch_size: int) -> int:
    height, width = normalize_image_size(image_size)
    return int((height // patch_size) * (width // patch_size))


@lru_cache(maxsize=128)
def _load_proposal_pretrain_config(checkpoint_path: str) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path).expanduser()
    for name in _PROPOSAL_PRETRAIN_YAML_NAMES:
        yaml_path = ckpt_path.parent / name
        if not yaml_path.is_file():
            continue
        with open(yaml_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _get_proposal_pretrain_value(config: Any, *keys: str, default=None):
    checkpoint = _get_nested_value(config, "proposal", "checkpoint", default=None)
    if not checkpoint:
        return default
    return _get_adjacent_pretrain_value(checkpoint, *keys, default=default)


def _get_adjacent_pretrain_config(checkpoint: str) -> Dict[str, Any]:
    try:
        return _load_proposal_pretrain_config(str(checkpoint))
    except (OSError, yaml.YAMLError):
        return {}


def _get_adjacent_pretrain_value(checkpoint: str, *keys: str, default=None):
    pretrain_config = _get_adjacent_pretrain_config(checkpoint)
    if not pretrain_config:
        return default
    return _get_nested_value(pretrain_config, *keys, default=default)


def resolve_predictor_runtime_normalize_reps(config: Any) -> bool:
    """Resolve predictor/refiner representation normalization.

    Precedence (explicit > checkpoint metadata > experiment default), mirroring
    resolve_proposal_runtime_normalize_reps:
      1. meta.predictor_runtime_normalize_reps —— 显式开关（最高优先）
      2. predictor_checkpoint 旁 params-pretrain.yaml 的 loss.normalize_reps
      3. 实验 config 的 loss.normalize_reps（默认 True）
    选中的来源会打日志，使 checkpoint 对实验 YAML 的（原本静默的）覆盖可见。
    """
    explicit = _get_nested_value(config, "meta", "predictor_runtime_normalize_reps", default=None)
    if explicit is not None:
        logger.info(
            "predictor normalize_reps=%s (from explicit meta.predictor_runtime_normalize_reps)", bool(explicit)
        )
        return bool(explicit)
    predictor_checkpoint = _get_nested_value(config, "meta", "predictor_checkpoint", default=None)
    if predictor_checkpoint:
        pretrain_value = _get_adjacent_pretrain_value(
            str(predictor_checkpoint),
            "loss",
            "normalize_reps",
            default=None,
        )
        if pretrain_value is not None:
            logger.info(
                "predictor normalize_reps=%s (from predictor checkpoint metadata %s; overrides experiment "
                "loss.normalize_reps — set meta.predictor_runtime_normalize_reps to override explicitly)",
                bool(pretrain_value),
                predictor_checkpoint,
            )
            return bool(pretrain_value)
    value = bool(_get_nested_value(config, "loss", "normalize_reps", default=True))
    logger.info("predictor normalize_reps=%s (from experiment loss.normalize_reps)", value)
    return value


def resolve_effective_tokens_per_frame(config: Any) -> int:
    token_ae_enabled = bool(_get_nested_value(config, "token_ae", "enabled", default=False))
    if token_ae_enabled:
        num_latent_tokens = int(_get_nested_value(config, "token_ae", "num_latent_tokens", default=0))
        if num_latent_tokens <= 0:
            raise ValueError("token_ae.num_latent_tokens must be > 0 when token_ae.enabled=True")
        return num_latent_tokens

    data_tokens = _get_nested_value(config, "data", "tokens_per_frame", default=None)
    if data_tokens is not None:
        return int(data_tokens)

    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    return compute_tokens_per_frame(crop_size, patch_size)


def resolve_proposal_runtime_normalize_reps(config: Any) -> bool:
    """Resolve the representation normalization used by the frozen proposal branch."""
    explicit = _get_nested_value(config, "proposal", "runtime_normalize_reps", default=None)
    if explicit is not None:
        return bool(explicit)
    pretrain_value = _get_proposal_pretrain_value(config, "loss", "normalize_reps", default=None)
    if pretrain_value is not None:
        return bool(pretrain_value)
    return bool(_get_nested_value(config, "loss", "normalize_reps", default=True))


def resolve_proposal_use_token_ae(config: Any) -> bool:
    """Resolve whether the frozen proposal branch consumes TokenAE-compressed tokens."""
    explicit = _get_nested_value(config, "proposal", "use_token_ae", default=None)
    if explicit is not None:
        return bool(explicit)
    return False


def resolve_proposal_encoder_backbone(config: Any) -> str:
    """Resolve the independent proposal encoder backbone."""
    explicit = _get_nested_value(config, "proposal", "encoder_backbone", default=None)
    if explicit:
        return str(explicit)
    return str(_get_nested_value(config, "model", "backbone", default="vjepa2"))


def _is_vjepa_proposal_encoder_config(config: Any) -> bool:
    return resolve_proposal_encoder_backbone(config) == "vjepa_img_encoder"


def _get_encoder_static_attr(encoder: Optional[Any], name: str) -> Optional[int]:
    if encoder is None:
        return None
    core = encoder.module if hasattr(encoder, "module") else encoder
    value = getattr(core, name, None)
    return None if value is None else int(value)


def resolve_proposal_tokens_per_frame(config: Any, proposal_encoder: Optional[Any] = None) -> int:
    """Resolve tokens/frame for the frozen proposal branch, independent of predictor runtime."""
    if resolve_proposal_use_token_ae(config):
        return resolve_effective_tokens_per_frame(config)

    encoder_tokens_per_frame = _get_encoder_static_attr(proposal_encoder, "tokens_per_frame")
    if encoder_tokens_per_frame is not None:
        return encoder_tokens_per_frame

    if _is_vjepa_proposal_encoder_config(config):
        height, width = _get_nested_value(config, "proposal", "vjepa_resolution", default=None)
        patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
        return int((int(height) // patch_size) * (int(width) // patch_size))

    data_tokens = _get_nested_value(config, "data", "tokens_per_frame", default=None)
    if data_tokens is not None:
        return int(data_tokens)
    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    return compute_tokens_per_frame(crop_size, patch_size)


def resolve_proposal_num_time_steps(config: Any, proposal_encoder: Optional[Any] = None) -> int:
    """Resolve proposal encoder temporal token steps."""
    encoder_steps = _get_encoder_static_attr(proposal_encoder, "num_time_steps")
    if encoder_steps is not None:
        return encoder_steps

    num_observed = _resolve_observed_frames_strict(config)
    if _is_vjepa_proposal_encoder_config(config):
        num_frames = int(_get_nested_value(config, "proposal", "vjepa_num_frames", default=2))
        if num_frames <= 0:
            raise ValueError("proposal.vjepa_num_frames must be positive")
        if num_observed % num_frames != 0:
            raise ValueError(
                f"train.num_observed_frames ({num_observed}) must be divisible by "
                f"proposal.vjepa_num_frames ({num_frames})"
            )
        return num_observed // num_frames
    return num_observed


def is_vjepa_main_encoder_config(config: Any) -> bool:
    """Return whether the main encoder backbone is the V-JEPA image encoder."""
    return str(_get_nested_value(config, "model", "backbone", default="vjepa2")) == "vjepa_img_encoder"


def is_dinov2_main_encoder_config(config: Any) -> bool:
    """Return whether the main encoder backbone is the DINOv2 image encoder."""
    return str(_get_nested_value(config, "model", "backbone", default="vjepa2")) == "dinov2_img_encoder"


def is_factory_pretrained_main_encoder_config(config: Any) -> bool:
    """Return whether the main encoder is built by a pretrained-encoder factory."""
    return is_vjepa_main_encoder_config(config) or is_dinov2_main_encoder_config(config)


def _parse_strict_positive_hw_resolution(value: Any, label: str) -> Tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{label} must contain [height, width] as positive integers, got {value!r}")
    return value[0], value[1]


def _require_positive_config_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer for main encoder stride, got {value!r}")
    return value


def _get_encoder_core(encoder: Optional[Any]) -> Optional[Any]:
    return encoder.module if encoder is not None and hasattr(encoder, "module") else encoder


def _resolve_dinov2_main_encoder_geometry(config: Any) -> Tuple[int, int, int]:
    """Validate and resolve DINOv2's fixed predictor geometry and temporal stride."""
    model_name = _get_nested_value(config, "model", "dinov2_model_name", default=None)
    if model_name != "vit_large_patch14_reg4_dinov2":
        raise ValueError(
            "model.dinov2_model_name must be exactly 'vit_large_patch14_reg4_dinov2', " f"got {model_name!r}"
        )
    height, width = _parse_strict_positive_hw_resolution(
        _get_nested_value(config, "model", "dinov2_resolution", default=None),
        "model.dinov2_resolution",
    )
    crop_size = _parse_strict_positive_hw_resolution(
        _get_nested_value(config, "data", "crop_size", default=None), "data.crop_size"
    )
    if crop_size != (height, width):
        raise ValueError(
            "data.crop_size must exactly match model.dinov2_resolution, " f"got {crop_size} != {(height, width)}"
        )
    patch_size = _get_nested_value(config, "data", "patch_size", default=None)
    if type(patch_size) is not int or patch_size != 14:
        raise ValueError(f"data.patch_size must be exactly 14 for DINOv2, got {patch_size!r}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("model.dinov2_resolution must be divisible by data.patch_size")
    stride = _get_nested_value(config, "model", "dinov2_frame_stride", default=None)
    if type(stride) is not int or stride <= 0:
        raise ValueError(f"model.dinov2_frame_stride must be a positive integer, got {stride!r}")
    if bool(_get_nested_value(config, "token_ae", "enabled", default=False)):
        raise ValueError("DINOv2 main encoder does not support token_ae.enabled=True")
    return height, width, stride


def _validate_dinov2_encoder_metadata(encoder: Optional[Any], height: int, width: int, stride: int) -> None:
    """Ensure optional DINOv2 runtime metadata agrees with its fixed config contract."""
    core = _get_encoder_core(encoder)
    if core is None:
        return

    expected_tokens = (height // 14) * (width // 14)
    tokens_per_frame = getattr(core, "tokens_per_frame", None)
    if tokens_per_frame is not None:
        if type(tokens_per_frame) is not int or tokens_per_frame != expected_tokens:
            raise ValueError(
                "encoder.tokens_per_frame must equal the DINOv2 raw core token count "
                f"{expected_tokens}, got {tokens_per_frame!r}"
            )

    resolution = getattr(core, "resolution", None)
    if resolution is not None:
        encoder_resolution = _parse_strict_positive_hw_resolution(resolution, "encoder.resolution")
        if encoder_resolution != (height, width):
            raise ValueError(
                "encoder.resolution must match model.dinov2_resolution, "
                f"got {encoder_resolution} != {(height, width)}"
            )

    patch_size = getattr(core, "patch_size", None)
    if patch_size is not None and (type(patch_size) is not int or patch_size != 14):
        raise ValueError(f"encoder.patch_size must be exactly 14 for DINOv2, got {patch_size!r}")

    num_frames = getattr(core, "num_frames", None)
    if num_frames is not None and (type(num_frames) is not int or num_frames != stride):
        raise ValueError("encoder.num_frames must match model.dinov2_frame_stride " f"({stride}), got {num_frames!r}")


def _validate_main_encoder_frame_stride(config: Any, stride: int) -> None:
    """Require configured raw-frame windows to align to the main encoder stride."""
    total = _require_positive_config_int(
        _get_nested_value(config, "data", "num_target_frames", default=None), "data.num_target_frames"
    )
    observed = _require_positive_config_int(
        _get_nested_value(config, "train", "num_observed_frames", default=None), "train.num_observed_frames"
    )
    if total % stride != 0:
        raise ValueError(f"data.num_target_frames ({total}) must be divisible by main encoder stride ({stride})")
    if observed % stride != 0:
        raise ValueError(f"train.num_observed_frames ({observed}) must be divisible by main encoder stride ({stride})")
    future = total - observed
    if future <= 0:
        raise ValueError(
            f"future frame count ({future}) from data.num_target_frames and train.num_observed_frames "
            f"must be positive for main encoder stride ({stride})"
        )
    if future % stride != 0:
        raise ValueError(f"future frame count ({future}) must be divisible by main encoder stride ({stride})")


def _validate_vjepa_main_token_ae(config: Any) -> None:
    if not is_vjepa_main_encoder_config(config) or not bool(
        _get_nested_value(config, "token_ae", "enabled", default=False)
    ):
        return

    height, width = _parse_hw_resolution(
        _get_nested_value(config, "model", "vjepa_resolution", default=(256, 512)),
        "model.vjepa_resolution",
    )
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("model.vjepa_resolution must be divisible by data.patch_size")

    num_latent_tokens = int(_get_nested_value(config, "token_ae", "num_latent_tokens", default=0))
    if num_latent_tokens <= 0:
        raise ValueError("token_ae.num_latent_tokens must be > 0 when V-JEPA main encoder uses TokenAE")

    input_grid_size = _get_nested_value(config, "token_ae", "input_grid_size", default=None)
    if input_grid_size is not None:
        input_grid = _parse_hw_resolution(input_grid_size, "token_ae.input_grid_size")
        expected_grid = (height // patch_size, width // patch_size)
        if input_grid != expected_grid:
            raise ValueError(
                "token_ae.input_grid_size must match V-JEPA raw token grid " f"{expected_grid}, got {input_grid}"
            )


def resolve_main_encoder_raw_tokens_per_frame(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve raw tokens per main-encoder predictor step before TokenAE compression."""
    if is_vjepa_main_encoder_config(config):
        _validate_vjepa_main_token_ae(config)
        encoder_tokens = _get_encoder_static_attr(encoder, "tokens_per_frame")
        if encoder_tokens is not None:
            return encoder_tokens
        height, width = _parse_hw_resolution(
            _get_nested_value(config, "model", "vjepa_resolution", default=(256, 512)),
            "model.vjepa_resolution",
        )
        patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("model.vjepa_resolution must be divisible by data.patch_size")
        return int((height // patch_size) * (width // patch_size))

    if is_dinov2_main_encoder_config(config):
        height, width, stride = _resolve_dinov2_main_encoder_geometry(config)
        _validate_dinov2_encoder_metadata(encoder, height, width, stride)
        return int((height // 14) * (width // 14))

    data_tokens = _get_nested_value(config, "data", "tokens_per_frame", default=None)
    if data_tokens is not None:
        return int(data_tokens)
    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    return compute_tokens_per_frame(crop_size, patch_size)


def resolve_main_encoder_frame_stride(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve raw-frame stride represented by one main-encoder predictor step."""
    if not is_factory_pretrained_main_encoder_config(config):
        return 1

    if is_vjepa_main_encoder_config(config):
        _validate_vjepa_main_token_ae(config)
        encoder_stride = _get_encoder_static_attr(encoder, "num_frames")
        stride = (
            encoder_stride
            if encoder_stride is not None
            else int(_get_nested_value(config, "model", "vjepa_num_frames", default=2))
        )
        if stride <= 0:
            raise ValueError(f"model.vjepa_num_frames must be positive, got {stride}")
    else:
        height, width, stride = _resolve_dinov2_main_encoder_geometry(config)
        _validate_dinov2_encoder_metadata(encoder, height, width, stride)

    _validate_main_encoder_frame_stride(config, stride)
    return stride


def resolve_main_encoder_tokens_per_frame(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve tokens per main-encoder predictor step."""
    if is_vjepa_main_encoder_config(config):
        _validate_vjepa_main_token_ae(config)
        if bool(_get_nested_value(config, "token_ae", "enabled", default=False)):
            return apply_multiview_token_multiplier(config, resolve_effective_tokens_per_frame(config))
        return apply_multiview_token_multiplier(config, resolve_main_encoder_raw_tokens_per_frame(config, encoder))
    if is_dinov2_main_encoder_config(config):
        return apply_multiview_token_multiplier(config, resolve_main_encoder_raw_tokens_per_frame(config, encoder))
    return apply_multiview_token_multiplier(config, resolve_effective_tokens_per_frame(config))


def resolve_main_encoder_num_time_steps(config: Any, num_raw_frames: int, encoder: Optional[Any] = None) -> int:
    """Resolve predictor time steps for a raw temporal window."""
    if type(num_raw_frames) is not int or num_raw_frames <= 0:
        raise ValueError(f"num_raw_frames must be a positive integer for main encoder stride, got {num_raw_frames!r}")
    raw_frames = num_raw_frames
    stride = resolve_main_encoder_frame_stride(config, encoder)
    if raw_frames % stride != 0:
        raise ValueError(f"num_raw_frames ({raw_frames}) must be divisible by main encoder stride ({stride})")
    return raw_frames // stride


def resolve_main_encoder_num_observed_steps(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve observed predictor steps for the main encoder."""
    if is_factory_pretrained_main_encoder_config(config):
        resolve_main_encoder_frame_stride(config, encoder)
    num_observed = _resolve_observed_frames_strict(config)
    return resolve_main_encoder_num_time_steps(config, num_raw_frames=num_observed, encoder=encoder)


def resolve_main_encoder_predictor_img_size(config: Any, encoder: Optional[Any] = None):
    """Resolve predictor image/grid size for the main encoder."""
    if is_vjepa_main_encoder_config(config):
        _validate_vjepa_main_token_ae(config)
        core = encoder.module if hasattr(encoder, "module") else encoder
        resolution = getattr(core, "resolution", None) if core is not None else None
        if resolution is None:
            resolution = _get_nested_value(config, "model", "vjepa_resolution", default=(256, 512))
        predictor_size = _parse_hw_resolution(resolution, "model.vjepa_resolution")
        return apply_multiview_image_size_multiplier(config, predictor_size)
    if is_dinov2_main_encoder_config(config):
        height, width, stride = _resolve_dinov2_main_encoder_geometry(config)
        _validate_dinov2_encoder_metadata(encoder, height, width, stride)
        return apply_multiview_image_size_multiplier(config, (height, width))
    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    return apply_multiview_image_size_multiplier(config, crop_size)
