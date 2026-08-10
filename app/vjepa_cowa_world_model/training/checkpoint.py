# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Checkpoint 保存/加载模块

提供模型权重的保存和加载功能。
"""

import copy
import inspect
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler

from app.vjepa_cowa_world_model.training.validation_suite import require_checkpoint_validation_signature
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module as _unwrap_module
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _require_checkpoint_validation_signatures(
    checkpoint: Mapping[str, Any],
    *,
    expected_signatures: Optional[Mapping[str, Mapping[str, Any]]],
    checkpoint_name: str,
    match_mode: str = "exact",
) -> None:
    """Validate every requested protocol signature before any state is restored."""

    if expected_signatures is None:
        return
    if not isinstance(expected_signatures, Mapping) or not expected_signatures:
        raise ValueError("expected checkpoint validation signatures must be a non-empty mapping when provided")
    for key, expected in expected_signatures.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"checkpoint validation signature keys must be non-empty strings, got {key!r}")
        if not isinstance(expected, Mapping) or not expected:
            raise ValueError(f"expected checkpoint validation signature for {key!r} must be a non-empty mapping")
        require_checkpoint_validation_signature(
            checkpoint,
            key=key,
            expected=expected,
            checkpoint_name=checkpoint_name,
            match_mode=match_mode,
        )


# ---------------------------------------------------------------------------
# 异步 Checkpoint 保存
# ---------------------------------------------------------------------------


def _deep_copy_to_cpu(obj: Any) -> Any:
    """
    递归地将嵌套 dict/list 中的 Tensor 拷贝到 CPU，非 Tensor 值做 deepcopy。

    这样后台线程在做 torch.save 时不会与 GPU 训练产生数据竞争。
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _deep_copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        copied = [_deep_copy_to_cpu(v) for v in obj]
        return type(obj)(copied)
    else:
        return copy.deepcopy(obj)


class _AsyncSaver:
    """
    管理后台 checkpoint 写入线程。

    使用方式::

        saver.save(save_dict, path)   # 提交异步保存（自动等待上次完成）
        saver.wait()                  # 显式等待当前保存完成

    线程安全：同一时刻最多只有一个后台写入线程。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None
        self._path: Optional[str] = None

    def wait(self) -> None:
        """等待上一次异步保存完成。

        fail-loud: 后台线程无法直接向训练主线程抛异常，所以失败暂存在 self._error，
        在这里（下一次 save() 入口或训练脚本显式调用 wait_for_checkpoint_save() 时）
        直接 raise——禁止把保存失败降级为 warning 后继续训练。
        """
        with self._lock:
            if self._thread is not None:
                self._thread.join()
                self._thread = None
                if self._error is not None:
                    error, path = self._error, self._path
                    self._error = None
                    self._path = None
                    raise RuntimeError(f"异步 checkpoint 保存失败: {path}") from error

    def save(self, save_dict: Dict[str, Any], path: str) -> None:
        """
        在后台线程中执行 torch.save。

        会先等待上一次保存完成（上一次失败会在此处抛出），然后启动新的后台线程。
        save_dict 必须已经在 CPU 上（调用方负责 _deep_copy_to_cpu）。
        """

        def _worker() -> None:
            try:
                t0 = time.monotonic()
                # 原子落盘：先写临时文件并 fsync，再 os.replace 到目标路径。
                # 避免训练被抢占/中断时在目标路径留下半写的损坏 checkpoint（os.replace 在同一文件系统上原子）。
                tmp_path = f"{path}.tmp"
                with open(tmp_path, "wb") as f:
                    torch.save(save_dict, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
                elapsed = time.monotonic() - t0
                logger.info(f"异步 checkpoint 写入完成: {path} (耗时 {elapsed:.1f}s)")
            except Exception as e:
                self._error = e

        with self._lock:
            self.wait()  # 确保上一次写完；上一次失败在此 raise
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._path = path
            self._thread = threading.Thread(target=_worker, daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._thread = None
                self._path = None
                raise


# 模块级单例，所有 save_training_checkpoint 调用共享
_async_saver = _AsyncSaver()


def wait_for_checkpoint_save() -> None:
    """
    等待异步 checkpoint 保存完成；若该次保存失败则直接抛出 RuntimeError（fail-loud）。

    训练脚本应在以下时机调用：
    - 训练循环结束后、进程退出前
    - 需要确保 checkpoint 已落盘时（例如 evaluation 前加载 checkpoint）
    """
    _async_saver.wait()


def _load_checkpoint_broadcast(path: str, rank: int = 0, world_size: int = 1) -> Dict[str, Any]:
    """
    加载 checkpoint，多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank。

    避免所有 rank 同时从 NFS 读取同一个大文件（ViT-G ~4GB），减少 NFS 带宽压力。

    Parameters
    ----------
    path       : checkpoint 文件路径
    rank       : 当前进程 rank
    world_size : 总进程数

    Returns
    -------
    dict: checkpoint 状态字典
    """
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return torch.load(path, map_location="cpu")

    # NOTE:
    # Broadcasting a very large Python checkpoint object can transiently duplicate
    # memory (pickle + object reconstruction) and cause rank crashes/OOM.
    # Default to local load on each rank for robustness; allow explicit broadcast
    # fallback via env var when NFS pressure is the bottleneck.
    load_mode = os.environ.get("VJEPA_CHECKPOINT_LOAD_MODE", "local").strip().lower()
    if load_mode in {"local", "per_rank", "rank_local"}:
        if rank == 0:
            logger.info("Loading checkpoint locally on all ranks (mode=%s): %s", load_mode, path)
        else:
            logger.debug("Rank %d local checkpoint load: %s", rank, path)
        return torch.load(path, map_location="cpu")
    if load_mode not in {"broadcast", "rank0_broadcast"}:
        if rank == 0:
            logger.warning(
                "Unknown VJEPA_CHECKPOINT_LOAD_MODE=%s, fallback to local load on all ranks",
                load_mode,
            )
        return torch.load(path, map_location="cpu")

    if rank == 0:
        logger.info(f"Rank 0 loading checkpoint from disk: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        broadcast_list = [checkpoint]
    else:
        broadcast_list = [None]

    if rank == 0:
        logger.info("Broadcasting checkpoint object to all ranks...")
    dist.broadcast_object_list(broadcast_list, src=0)
    if rank == 0:
        logger.info("Checkpoint broadcast completed")

    checkpoint = broadcast_list[0]
    if checkpoint is None:
        raise RuntimeError("Received empty checkpoint object from broadcast")
    return checkpoint


_TORCH_LOAD_PARAMS = inspect.signature(torch.load).parameters


def _torch_load_checkpoint(
    path: str,
    *,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if "mmap" in _TORCH_LOAD_PARAMS:
        kwargs["mmap"] = True
    if "weights_only" in _TORCH_LOAD_PARAMS:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def is_peft_wrapped(model: nn.Module) -> bool:
    """判断模块（可能被 DDP 包装）是否为 PEFT/LoRA 包装的 PeftModel。"""
    unwrapped = _unwrap_module(model)
    return hasattr(unwrapped, "base_model") and hasattr(unwrapped.base_model, "model")


def extract_portable_predictor_state(predictor: nn.Module) -> Dict[str, Any]:
    """
    导出可移植（plain key）的 predictor state dict。

    - 普通 predictor：直接返回去除 DDP 前缀的 state dict。
    - PEFT/LoRA 包装的 predictor：在 deepcopy 上 merge_and_unload()，把 adapter 权重
      合并进 base 权重后导出 plain key state dict（原模型不受影响，可继续训练）。

    下游 eval / meta.predictor_checkpoint / 下一阶段训练都按 plain key 加载 predictor，
    禁止把 `base_model.model.*` / `*.base_layer.*` / `lora_*` 这类 PEFT key 写进
    checkpoint 的 "predictor" 条目。
    """
    unwrapped = _unwrap_module(predictor)
    if is_peft_wrapped(unwrapped):
        merged = copy.deepcopy(unwrapped).merge_and_unload()
        state = merged.state_dict()
        # fail-loud：合并后绝不允许残留 PEFT 标记 key，否则下游全部加载失败。
        leftover = [k for k in state if "lora_" in k or ".base_layer." in k or k.startswith("base_model.")]
        if leftover:
            raise RuntimeError(
                f"merge_and_unload 后 predictor state dict 仍含 PEFT key（前10个）: {leftover[:10]}；"
                "禁止保存不可移植的 predictor 权重。"
            )
        return state
    return {k[7:] if k.startswith("module.") else k: v for k, v in predictor.state_dict().items()}


def extract_peft_predictor_state(predictor: nn.Module) -> Dict[str, Any]:
    """
    导出 PEFT 包装 predictor 的精确 state（base + adapter 分离，去除 DDP 前缀）。

    与 extract_portable_predictor_state 配对使用：portable 版给下游消费（merged plain key），
    本函数的结果存为 checkpoint["predictor_peft"]，供 LoRA resume 精确恢复。
    """
    if not is_peft_wrapped(predictor):
        raise RuntimeError("extract_peft_predictor_state 仅适用于 PEFT 包装的 predictor。")
    return {k[7:] if k.startswith("module.") else k: v for k, v in predictor.state_dict().items()}


def load_predictor_peft_state(checkpoint: Dict[str, Any], predictor: nn.Module, source: str) -> None:
    """
    从 checkpoint["predictor_peft"] 精确恢复 PEFT 包装 predictor 的 adapter + base 权重。

    用于 LoRA 训练中断后的 resume：portable 的 "predictor"（merged）只能保证函数等价，
    不能恢复 adapter 的精确数值（optimizer 动量将与 adapter 错位），因此 resume 必须用
    精确的 PEFT state。缺失即报错，禁止静默用 merged 权重 + 全新 adapter 继续。
    """
    if not is_peft_wrapped(predictor):
        raise RuntimeError("load_predictor_peft_state 仅适用于 PEFT 包装的 predictor。")
    if "predictor_peft" not in checkpoint:
        raise RuntimeError(
            f"Resume LoRA 训练需要 checkpoint 含 'predictor_peft'（来源: {source}），"
            f"实际 keys: {sorted(checkpoint.keys())[:12]}。"
            "该 checkpoint 可能由修复前的代码保存（LoRA 权重未保存），无法精确 resume。"
        )
    peft_state = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint["predictor_peft"].items()}
    _unwrap_module(predictor).load_state_dict(peft_state, strict=True)
    logger.info("已从 'predictor_peft' 精确恢复 PEFT predictor（base + adapter，strict=True）")


def load_state_dict_helper(model: nn.Module, state_dict: Dict[str, Any], name: str) -> None:
    """
    通用状态字典加载辅助函数，自动处理 DDP 的 module. 前缀

    Args:
        model: 模型
        state_dict: 状态字典
        name: 模型名称 (用于日志)
    """
    model_unwrapped = model.module if hasattr(model, "module") else model

    # 移除 'module.' 前缀
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    if hasattr(model_unwrapped, "base_model") and hasattr(model_unwrapped.base_model, "model"):
        base_model = model_unwrapped.base_model.model
        # PEFT/LoRA 将 target module 的 weight/bias 移到 base_layer 下,
        # 例如 qkv.weight → qkv.base_layer.weight, 需要自动 remap
        model_keys = set(base_model.state_dict().keys())
        remapped = {}
        for k, v in new_state_dict.items():
            if k in model_keys:
                remapped[k] = v
            else:
                # 尝试插入 .base_layer
                for suffix in (".weight", ".bias"):
                    if k.endswith(suffix):
                        new_key = k[: -len(suffix)] + ".base_layer" + suffix
                        if new_key in model_keys:
                            remapped[new_key] = v
                            break
                else:
                    remapped[k] = v
        missing, unexpected = base_model.load_state_dict(remapped, strict=False)
    else:
        missing, unexpected = model_unwrapped.load_state_dict(new_state_dict, strict=False)

    # LoRA adapter 权重处理（point 18，智能判定）：
    #   - 若 checkpoint 完全不含 lora_* key（纯 base checkpoint，首次给模型套上全新 LoRA），
    #     则 adapter 缺失属于正常的新初始化，从 strict 检查中过滤掉；
    #   - 若 checkpoint 含有 lora_* key（本应是 LoRA checkpoint），则缺/多任何 adapter key
    #     都是真实不一致，保留在 missing/unexpected 中触发后面的 raise（禁止静默容忍缺 adapter）。
    _LORA_SUFFIXES = ("lora_A.", "lora_B.", "lora_embedding_A.", "lora_embedding_B.")
    checkpoint_has_lora = any(any(s in key for s in _LORA_SUFFIXES) for key in new_state_dict)
    if not checkpoint_has_lora:
        missing = [k for k in missing if not any(k.endswith(s) or s in k for s in _LORA_SUFFIXES)]
        unexpected = [k for k in unexpected if not any(k.endswith(s) or s in k for s in _LORA_SUFFIXES)]

    if name == "predictor":
        predictor_query = getattr(model_unwrapped, "future_query_tokens", None)
        loaded_query = False
        if predictor_query is not None:
            for query_key in ("future_query_tokens", "base_model.model.future_query_tokens"):
                query_value = new_state_dict.get(query_key)
                if query_value is None:
                    continue
                if tuple(query_value.shape) != tuple(predictor_query.shape):
                    raise RuntimeError(
                        "Checkpoint mismatch for predictor future_query_tokens: "
                        f"checkpoint_shape={tuple(query_value.shape)}, model_shape={tuple(predictor_query.shape)}"
                    )
                predictor_query.data.copy_(query_value.to(device=predictor_query.device, dtype=predictor_query.dtype))
                logger.info("Loaded predictor future_query_tokens from checkpoint key '%s'", query_key)
                loaded_query = True
                break

        # fail-loud (point 15): future_query_tokens 缺失会让 AR rollout 使用随机初始化的 query token，
        # 禁止只 warning 后从 missing 列表过滤掉。仅对"缺失"报错；"unexpected"维持原有兼容容忍
        # （例如把含 future_query_tokens 的 ckpt 加载进不使用 AR query token 的 predictor）。
        predictor_query_missing = [k for k in missing if k.endswith("future_query_tokens")]
        if predictor_query is not None and not loaded_query and predictor_query_missing:
            raise RuntimeError(
                f"Predictor future_query_tokens 未能从 checkpoint 加载 (missing={predictor_query_missing}). "
                "AR rollout 会使用随机初始化的 query token，禁止静默继续。"
            )
        # future_query_tokens 已加载（或 ckpt 含额外副本），从 strict 检查中移除以避免重复 raise。
        missing = [k for k in missing if not k.endswith("future_query_tokens")]
        unexpected = [k for k in unexpected if not k.endswith("future_query_tokens")]
        # 旧 predictor checkpoint 没有 state_mask_token；当前 inference-consistent 状态输入沿用
        # 观测状态广播逻辑，不向 predictor 传 state_mask，因此该 token 对旧路径未使用。
        missing = [k for k in missing if not k.endswith("state_mask_token")]
        # 旧 latent-DiT checkpoint 没有 masked/inpainting future-condition type embedding。默认 masked
        # inpainting 关闭时该参数不参与旧路径；开启新功能的训练会从零初始化并学习它。
        missing = [k for k in missing if not k.endswith("future_condition_type_embed")]
        # 旧 predictor checkpoint 没有多层输出头是可兼容的；它们只作为新监督/诊断头存在。
        _OPTIONAL_OUTPUT_HEAD_PREFIXES = ("layer_output_norms.", "layer_output_projs.")
        missing = [k for k in missing if not k.startswith(_OPTIONAL_OUTPUT_HEAD_PREFIXES)]

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for '{name}': "
            f"missing={len(missing)} keys, unexpected={len(unexpected)} keys.\n"
            f"  missing: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"  unexpected: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    logger.info(f"Loaded {name}: all keys matched (missing=0, unexpected=0)")


def load_pretrained_checkpoint(
    path: str,
    encoder: Optional[nn.Module],
    target_encoder: Optional[nn.Module],
    predictor: Optional[nn.Module],
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    load_encoder: bool = True,
    load_predictor: bool = False,
    load_seg: bool = True,
    load_planner: bool = True,
    context_encoder_key: str = "encoder",
    target_encoder_key: str = "target_encoder",
    rank: int = 0,
    world_size: int = 1,
    predictor_checkpoint: Optional[str] = None,
    expected_full_checkpoint_signatures: Optional[Mapping[str, Mapping[str, Any]]] = None,
    expected_predictor_checkpoint_signatures: Optional[Mapping[str, Mapping[str, Any]]] = None,
    full_checkpoint_signature_match: str = "exact",
    predictor_checkpoint_signature_match: str = "exact",
) -> Dict[str, bool]:
    """
    加载预训练 checkpoint

    多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank，
    避免所有 rank 同时从 NFS 读取同一个大文件。

    Args:
        path: checkpoint 路径
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        load_encoder: 是否加载 encoder 权重
        load_predictor: 是否加载 predictor 权重
        load_seg: 是否加载 seg 权重
        load_planner: 是否加载 planner 权重
        context_encoder_key: checkpoint 中 encoder 的 key 名称
            (默认 "encoder"，V-JEPA 2.1 可能是 "ema_encoder" 或 "target_encoder")
        target_encoder_key: checkpoint 中 target_encoder 的 key 名称
            (默认 "target_encoder")
        rank: 当前进程 rank（默认 0，向后兼容单卡场景）
        world_size: 总进程数（默认 1，向后兼容单卡场景）
        predictor_checkpoint: predictor 独立 checkpoint 路径，设置后优先从该文件加载 predictor
        expected_full_checkpoint_signatures: full checkpoint 必须精确匹配的验证协议签名
        expected_predictor_checkpoint_signatures: predictor 权重来源必须精确匹配的验证协议签名
        full_checkpoint_signature_match: full checkpoint 签名比较模式（exact/compatible）
        predictor_checkpoint_signature_match: predictor checkpoint 签名比较模式（exact/compatible）
    """
    loaded_modules = {
        "encoder": False,
        "target_encoder": False,
        "predictor": False,
        "seg_neck": False,
        "seg_head": False,
        "planner": False,
    }

    if path is None:
        if expected_full_checkpoint_signatures is not None or expected_predictor_checkpoint_signatures is not None:
            raise ValueError(
                "checkpoint validation signatures were requested but no full checkpoint path was provided"
            )
        return loaded_modules
    if expected_predictor_checkpoint_signatures is not None and not load_predictor:
        raise ValueError("expected_predictor_checkpoint_signatures requires load_predictor=True")
    if not os.path.exists(path):
        # fail-loud (point 16): 配置了 pretrain checkpoint 路径却不存在（多为拼错路径），直接报错，
        # 禁止 warning 后静默保留 init 权重继续训练。
        raise FileNotFoundError(f"Full pretrained checkpoint not found: {path}")

    logger.info(f"Loading full pretrained checkpoint from {path}")
    checkpoint = _load_checkpoint_broadcast(path, rank=rank, world_size=world_size)
    _require_checkpoint_validation_signatures(
        checkpoint,
        expected_signatures=expected_full_checkpoint_signatures,
        checkpoint_name=path,
        match_mode=full_checkpoint_signature_match,
    )

    predictor_source_checkpoint = checkpoint
    predictor_source_name = path
    if load_predictor:
        if predictor is None:
            raise ValueError(
                "load_predictor=True 但 predictor 模块为 None；"
                "请先构建 predictor，或显式设置 meta.load_predictor: false。"
            )
        if predictor_checkpoint:
            if not os.path.exists(predictor_checkpoint):
                raise FileNotFoundError(f"Predictor checkpoint not found: {predictor_checkpoint}")
            logger.info(f"Loading predictor from separate checkpoint: {predictor_checkpoint}")
            predictor_source_checkpoint = _load_checkpoint_broadcast(
                predictor_checkpoint,
                rank=rank,
                world_size=world_size,
            )
            predictor_source_name = predictor_checkpoint
        _require_checkpoint_validation_signatures(
            predictor_source_checkpoint,
            expected_signatures=expected_predictor_checkpoint_signatures,
            checkpoint_name=predictor_source_name,
            match_mode=predictor_checkpoint_signature_match,
        )
        if "predictor" not in predictor_source_checkpoint:
            raise KeyError(
                f"load_predictor=True but no 'predictor' key in {predictor_source_name} "
                f"(available keys: {list(predictor_source_checkpoint.keys())[:10]}). "
                "若 predictor 需从零初始化，请显式设置 meta.load_predictor: false。"
            )

    if load_encoder:
        # 支持多种 encoder key: encoder, ema_encoder, target_encoder
        encoder_state = None
        for key in [context_encoder_key, "encoder", "ema_encoder", "target_encoder"]:
            if key in checkpoint:
                encoder_state = checkpoint[key]
                logger.info(f"Found encoder weights under key '{key}'")
                break
        if encoder_state is None:
            # fail-loud: load_encoder=True 时 ckpt 必须含 encoder 权重，
            # 禁止 warning 后静默保留 init 权重继续训练。
            raise KeyError(
                f"load_encoder=True but no encoder weights found in {path} "
                f"(tried keys: {[context_encoder_key, 'encoder', 'ema_encoder', 'target_encoder']}, "
                f"available: {list(checkpoint.keys())[:10]}). "
                "若本次训练不需要从该 ckpt 加载 encoder，请显式设置 meta.load_encoder: false。"
            )
        load_state_dict_helper(encoder, encoder_state, "encoder")
        target_encoder.load_state_dict(encoder.state_dict())
        loaded_modules["encoder"] = True
        loaded_modules["target_encoder"] = True
        logger.info("Synchronized target_encoder with encoder")

    if load_predictor:
        load_state_dict_helper(predictor, predictor_source_checkpoint["predictor"], "predictor")
        loaded_modules["predictor"] = True

    if load_seg:
        if seg_neck is not None:
            if "seg_neck" not in checkpoint:
                raise KeyError(
                    f"load_seg=True but no 'seg_neck' key in {path} "
                    f"(available keys: {list(checkpoint.keys())[:10]}). "
                    "若 seg 模块需从零初始化，请显式设置 meta.load_seg: false。"
                )
            load_state_dict_helper(seg_neck, checkpoint["seg_neck"], "seg_neck")
            loaded_modules["seg_neck"] = True
        if seg_head is not None:
            if "seg_head" not in checkpoint:
                raise KeyError(
                    f"load_seg=True but no 'seg_head' key in {path} "
                    f"(available keys: {list(checkpoint.keys())[:10]}). "
                    "若 seg 模块需从零初始化，请显式设置 meta.load_seg: false。"
                )
            load_state_dict_helper(seg_head, checkpoint["seg_head"], "seg_head")
            loaded_modules["seg_head"] = True

    if load_planner and planner is not None:
        if "planner" not in checkpoint:
            # fail-loud: planner 模块已构建且 load_planner=True，ckpt 缺 planner 权重必须报错。
            raise KeyError(
                f"load_planner=True but no 'planner' key in {path} "
                f"(available keys: {list(checkpoint.keys())[:10]}). "
                "若 planner 从零训练，请显式设置 meta.load_planner: false。"
            )
        load_state_dict_helper(planner, checkpoint["planner"], "planner")
        loaded_modules["planner"] = True

    logger.info("Full pretrained checkpoint loaded successfully!")
    return loaded_modules


def load_multiview_fusion_from_checkpoint(
    path: str,
    multiview_fusion: nn.Module,
    target_multiview_fusion: Optional[nn.Module] = None,
    rank: int = 0,
    world_size: int = 1,
    expected_validation_signatures: Optional[Mapping[str, Mapping[str, Any]]] = None,
    validation_signature_match: str = "exact",
) -> bool:
    """Load PETR multi-view fusion weights from a predictor/planner checkpoint.

    Parameters
    ----------
    path:
        Checkpoint containing ``multiview_fusion`` and optionally
        ``target_multiview_fusion``.
    multiview_fusion:
        Student fusion module to initialize.
    target_multiview_fusion:
        Optional EMA target fusion module. If the checkpoint does not contain
        a target key, it is synchronized from the loaded student fusion.

    Returns
    -------
    bool
        True when weights were loaded.
    """
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Multi-view fusion checkpoint not found: {path}")

    logger.info(f"Loading multiview_fusion from checkpoint: {path}")
    checkpoint = _load_checkpoint_broadcast(path, rank=rank, world_size=world_size)
    _require_checkpoint_validation_signatures(
        checkpoint,
        expected_signatures=expected_validation_signatures,
        checkpoint_name=path,
        match_mode=validation_signature_match,
    )
    if "multiview_fusion" not in checkpoint:
        raise KeyError(f"No 'multiview_fusion' key in {path}; available keys: {list(checkpoint.keys())[:10]}")

    load_state_dict_helper(multiview_fusion, checkpoint["multiview_fusion"], "multiview_fusion")
    if target_multiview_fusion is not None:
        if "target_multiview_fusion" in checkpoint:
            load_state_dict_helper(
                target_multiview_fusion,
                checkpoint["target_multiview_fusion"],
                "target_multiview_fusion",
            )
        else:
            target_unwrapped = (
                target_multiview_fusion.module
                if hasattr(target_multiview_fusion, "module")
                else target_multiview_fusion
            )
            source_unwrapped = multiview_fusion.module if hasattr(multiview_fusion, "module") else multiview_fusion
            target_unwrapped.load_state_dict(source_unwrapped.state_dict())
            logger.warning(
                "No 'target_multiview_fusion' key in checkpoint; synchronized target fusion from multiview_fusion."
            )
    return True


def save_training_checkpoint(
    path: str,
    encoder: Optional[nn.Module],
    target_encoder: Optional[nn.Module],
    predictor: Optional[nn.Module],
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler: Any,
    wd_scheduler: Any,
    epoch: int,
    loss: float,
    batch_size: int,
    world_size: int,
    lr: float,
    rank: int,
    use_planner: bool = True,
    encoder_train: bool = True,
    encoder_ema: bool = True,
    predictor_train: bool = True,
    seg_head_train: bool = True,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存训练 checkpoint（只保存训练中实际更新的模块）

    冻结且未通过 EMA 更新的模块不会被保存，以减小 checkpoint 体积。
    resume 时，未保存的模块会从 pretrained checkpoint 加载。

    Args:
        path: checkpoint 保存路径
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        epoch: 当前 epoch
        loss: 当前损失
        batch_size: 批大小
        world_size: 进程数
        lr: 学习率
        rank: 当前进程 rank
        use_planner: 是否使用 planner
        encoder_train: encoder 是否参与训练（梯度更新）
        encoder_ema: target_encoder 是否通过 EMA 更新
        predictor_train: predictor 是否参与训练（保存判定还会自动 OR 上
            PEFT/LoRA 包装与任一 requires_grad=True 参数，调用方传 False 也不会丢微调权重）
        seg_head_train: seg_head/seg_neck 是否参与训练
    """
    if rank != 0:
        return

    save_dict = {
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "batch_size": batch_size,
        "world_size": world_size,
        "lr": lr,
    }

    # 记录保存了哪些模块（用于 debug / 日志）
    saved_modules = []
    skipped_modules = []

    # encoder: 仅在参与梯度训练时保存
    if encoder_train:
        save_dict["encoder"] = encoder.state_dict()
        saved_modules.append("encoder")
    else:
        skipped_modules.append("encoder")

    # target_encoder: 仅在 EMA 更新时保存（冻结且无 EMA 时与 pretrained 一致）
    if encoder_ema:
        save_dict["target_encoder"] = target_encoder.state_dict()
        saved_modules.append("target_encoder")
    else:
        skipped_modules.append("target_encoder")

    # predictor: 在"权重可能被更新"的任一情形下保存——判定在这里集中推导，
    # 不再依赖每个调用方把 predictor_train 传对（历史上两个 LoRA 脚本因此静默丢权重）：
    #   - predictor_train=True：调用方显式声明在训练；
    #   - PEFT/LoRA 包装：adapter 必然在训练（_set_predictor_lora_trainable 保证）；
    #   - 任一参数 requires_grad=True：覆盖 planner-finetune 等"解冻但 predictor_train=False"模式。
    # 方向性原则：多存是安全的（浪费字节），漏存会永久丢失微调结果。
    if predictor_train and predictor is None:
        raise ValueError("save_training_checkpoint: predictor_train=True 但 predictor=None；调用方错误。")
    predictor_should_save = predictor is not None and (
        predictor_train or is_peft_wrapped(predictor) or any(p.requires_grad for p in predictor.parameters())
    )
    if predictor_should_save:
        # "predictor" 永远是 plain key（PEFT 时为 merged 权重），保证下游 eval /
        # meta.predictor_checkpoint / 下一阶段训练可直接加载。
        save_dict["predictor"] = extract_portable_predictor_state(predictor)
        saved_modules.append("predictor")
        if is_peft_wrapped(predictor):
            # 精确的 PEFT state（base + adapter 分离），用于 LoRA resume。
            save_dict["predictor_peft"] = extract_peft_predictor_state(predictor)
            saved_modules.append("predictor_peft")
    else:
        skipped_modules.append("predictor")

    # seg_head / seg_neck: 仅在参与训练时保存
    if seg_head_train:
        if seg_head is not None:
            save_dict["seg_head"] = seg_head.state_dict()
            saved_modules.append("seg_head")
        if seg_neck is not None:
            save_dict["seg_neck"] = seg_neck.state_dict()
            saved_modules.append("seg_neck")
    else:
        if seg_head is not None:
            skipped_modules.append("seg_head")
        if seg_neck is not None:
            skipped_modules.append("seg_neck")

    # planner: 始终可训练，始终保存
    if use_planner and planner is not None:
        save_dict["planner"] = planner.state_dict()
        saved_modules.append("planner")
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
        saved_modules.append("scheduler")
    if wd_scheduler is not None:
        save_dict["wd_scheduler"] = wd_scheduler.state_dict()
        saved_modules.append("wd_scheduler")

    if extra_state is not None:
        save_dict.update(extra_state)

    # 保存元数据，方便 resume 时判断 checkpoint 内容
    save_dict["saved_modules"] = saved_modules

    # 异步保存：先将 state_dict 拷贝到 CPU，然后在后台线程中 torch.save
    save_dict_cpu = _deep_copy_to_cpu(save_dict)
    logger.info(f"checkpoint 异步保存已提交: {path}")
    logger.info(f"  保存的模块: {saved_modules}")
    if skipped_modules:
        logger.info(f"  跳过的冻结模块: {skipped_modules}（resume 时从 pretrained 加载）")
    _async_saver.save(save_dict_cpu, path)


def resume_from_checkpoint(
    resume_path: Optional[str],
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    seg_head: Optional[nn.Module],
    seg_neck: Optional[nn.Module],
    planner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler: Any,
    wd_scheduler: Any,
    use_planner: bool = True,
    load_planner: Optional[bool] = None,
    rank: int = 0,
    world_size: int = 1,
    use_broadcast: bool = True,
    model_only: bool = False,
    expected_validation_signatures: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> int:
    """
    从 checkpoint 恢复训练

    多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank。

    Args:
        resume_path: checkpoint 路径
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_head: seg_head 模型
        seg_neck: seg_neck 模型
        planner: planner 模型
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        use_planner: 是否使用 planner
        load_planner: 是否从 resume checkpoint 加载 planner；None 表示跟随 use_planner
        rank: 当前进程 rank（默认 0，向后兼容单卡场景）
        world_size: 总进程数（默认 1，向后兼容单卡场景）
        use_broadcast: 是否使用 broadcast 分发 checkpoint（False 则每个 rank 独立从磁盘读取）
        model_only: 仅加载模型权重，跳过 optimizer/scheduler/epoch（用于分阶段训练）
        expected_validation_signatures: resume checkpoint 必须精确匹配的验证协议签名

    Returns:
        int: 开始的 epoch（model_only=True 时始终返回 0）
    """
    if resume_path is None:
        return 0
    if not os.path.exists(resume_path):
        # fail-loud (point 17): resume_path 已解析为具体路径却不存在，禁止静默从 epoch 0 重新开始。
        raise FileNotFoundError(f"resume_path does not exist: {resume_path}")

    if model_only:
        logger.info("resume_model_only=True: 仅加载模型权重，跳过 optimizer/scheduler/epoch")

    should_load_planner = use_planner if load_planner is None else use_planner and load_planner
    if use_planner and planner is not None and not should_load_planner:
        logger.info("Skipping planner resume load because load_planner=False")

    start_epoch = resume_training_checkpoint(
        resume_path=resume_path,
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        seg_head=seg_head,
        seg_neck=seg_neck,
        planner=planner if should_load_planner else None,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        logger=logger,
        rank=rank,
        world_size=world_size,
        use_broadcast=use_broadcast,
        model_only=model_only,
        expected_validation_signatures=expected_validation_signatures,
    )

    # LoRA resume：通用 resume 只把 portable（merged）"predictor" 装进 base，adapter 仍是
    # 新初始化（merged base + 零 delta 函数等价，但 adapter 数值与 optimizer 动量错位）。
    # 分三种情形：
    #   1. ckpt 含 "predictor_peft" → 无论 model_only 与否都精确覆盖恢复（最优）。
    #   2. ckpt 不含且 model_only=True → 合法暖启动：plain predictor 装进 PEFT base、
    #      adapter 全新初始化、不加载 optimizer 状态（resume_training_checkpoint 本就支持），放行。
    #   3. ckpt 不含且 model_only=False → 真正的训练续跑缺精确 adapter 状态，
    #      optimizer 动量会与全新 adapter 错位 → fail-loud。
    if predictor is not None and is_peft_wrapped(predictor):
        peft_ckpt = _torch_load_checkpoint(resume_path, map_location="cpu")
        if "predictor_peft" in peft_ckpt:
            load_predictor_peft_state(peft_ckpt, predictor, source=resume_path)
        elif model_only:
            logger.info(
                "model_only resume：checkpoint 无 'predictor_peft'，按 plain predictor 暖启动 "
                "PEFT base（adapter 全新初始化，不恢复 optimizer 状态）。"
            )
        else:
            raise RuntimeError(
                f"Resume LoRA 训练（model_only=False）需要 checkpoint 含 'predictor_peft'（来源: {resume_path}），"
                f"实际 keys: {sorted(peft_ckpt.keys())[:12]}。"
                "该 checkpoint 可能由修复前的代码保存（LoRA 权重未保存），无法精确续跑；"
                "若只想用它的权重开新 LoRA 阶段，请设 meta.resume_model_only: true。"
            )
        del peft_ckpt

    return start_epoch


def load_checkpoint(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a checkpoint dictionary without restoring it."""
    if path is None or not os.path.exists(path):
        return None
    return _torch_load_checkpoint(path, map_location="cpu", weights_only=False)


def setup_checkpoint_paths(folder: str, resume_checkpoint: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    设置 checkpoint 路径

    Args:
        folder: 保存文件夹
        resume_checkpoint: 指定的恢复 checkpoint 文件名

    Returns:
        Dict[str, Optional[str]]: 包含各种 checkpoint 路径的字典
    """
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best_ade.pt")

    if resume_checkpoint is not None:
        # fail-loud (point 17): 显式指定了 resume_checkpoint，文件必须存在；缺失/拼错直接报错，
        # 禁止静默回落到 None 后从 epoch 0 重新训练。
        resume_path = os.path.join(folder, resume_checkpoint)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"Specified resume_checkpoint does not exist: {resume_path} "
                f"(folder={folder}, resume_checkpoint={resume_checkpoint})"
            )
    else:
        # 未显式指定：自动尝试 latest.pt；首次训练时它不存在属正常 → resume=None。
        resume_path = latest_path if os.path.exists(latest_path) else None

    return {
        "latest": latest_path,
        "best": best_path,
        "resume": resume_path,
    }


# vendored verbatim from app/vjepa_droid/utils.py
def resume_training_checkpoint(
    resume_path,
    encoder=None,
    target_encoder=None,
    predictor=None,
    seg_head=None,
    seg_neck=None,
    planner=None,
    optimizer=None,
    scaler=None,
    scheduler=None,
    wd_scheduler=None,
    logger=None,
    rank=0,
    world_size=1,
    use_broadcast=True,
    model_only=False,
    expected_validation_signatures=None,
):
    """
    从检查点恢复训练状态

    多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank，
    避免所有 rank 同时从 NFS 读取同一个大文件。

    Args:
        resume_path: 检查点文件路径
        encoder, target_encoder, predictor, seg_head, seg_neck, planner: 模型
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        logger: 日志记录器
        rank: 当前进程 rank（默认 0，向后兼容单卡场景）
        world_size: 总进程数（默认 1，向后兼容单卡场景）
        use_broadcast: 是否使用 broadcast 分发 checkpoint（False 则每个 rank 独立从磁盘读取）
        model_only: 仅加载模型权重，跳过 optimizer/scheduler/epoch（用于分阶段训练）
        expected_validation_signatures: checkpoint 必须精确匹配的验证协议签名

    Returns:
        start_epoch: 恢复的起始epoch（model_only=True 时始终返回 0）
    """
    import torch
    import torch.distributed as dist

    if logger is None:
        logger = logging.getLogger()

    logger.info(f"正在从 {resume_path} 恢复训练...")

    # 多卡场景：仅 rank 0 从磁盘读取，broadcast 给其他 rank
    if use_broadcast and world_size > 1 and dist.is_available() and dist.is_initialized():
        if rank == 0:
            logger.info(f"Rank 0 loading resume checkpoint from disk: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")
            broadcast_list = [checkpoint]
        else:
            broadcast_list = [None]
        logger.info(f"Rank {rank} waiting for resume checkpoint broadcast...")
        dist.broadcast_object_list(broadcast_list, src=0)
        checkpoint = broadcast_list[0]
        logger.info(f"Rank {rank} received resume checkpoint via broadcast")
    else:
        logger.info(f"Rank {rank} loading resume checkpoint from disk: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

    _require_checkpoint_validation_signatures(
        checkpoint,
        expected_signatures=expected_validation_signatures,
        checkpoint_name=resume_path,
    )

    def summarize_keys(keys, max_items=10):
        keys = list(keys)
        preview = keys[:max_items]
        suffix = "" if len(keys) <= max_items else f", ... (+{len(keys) - max_items} more)"
        return f"{preview}{suffix}"

    def load_state_dict(model, state_dict, name):
        """安全加载状态字典，处理 shape 不匹配的参数和 PEFT 模型"""
        if state_dict is None:
            return

        # 检测 PEFT (LoRA) 包装: DDP(PeftModel(base)) 或 PeftModel(base)
        model_unwrapped = model.module if hasattr(model, "module") else model
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        is_peft = hasattr(model_unwrapped, "base_model") and hasattr(model_unwrapped.base_model, "model")

        if is_peft:
            # PEFT 模型：剥离 module. 前缀后加载到底层 base_model.model
            base_model = model_unwrapped.base_model.model
            # PEFT/LoRA 将 target module 的 weight/bias 移到 base_layer 下,
            # 例如 qkv.weight → qkv.base_layer.weight, 需要自动 remap
            model_keys = set(base_model.state_dict().keys())
            remapped = {}
            for k, v in new_state_dict.items():
                if k in model_keys:
                    remapped[k] = v
                else:
                    # 尝试插入 .base_layer
                    for suffix in (".weight", ".bias"):
                        if k.endswith(suffix):
                            new_key = k[: -len(suffix)] + ".base_layer" + suffix
                            if new_key in model_keys:
                                remapped[new_key] = v
                                break
                    else:
                        remapped[k] = v
            missing, unexpected = base_model.load_state_dict(remapped, strict=False)
            # missing 中应只剩 LoRA adapter 参数 (lora_A/lora_B), 这是正常的
            lora_missing = [k for k in missing if "lora_" in k]
            non_lora_missing = [k for k in missing if "lora_" not in k]
            logger.info(
                f"已恢复 {name} 权重 (PEFT base model): "
                f"missing={len(missing)} (lora={len(lora_missing)}, other={len(non_lora_missing)}), "
                f"unexpected={len(unexpected)}"
            )
            if non_lora_missing:
                logger.warning(f"{name} PEFT base model non-LoRA missing keys: {non_lora_missing[:10]}")
            if unexpected:
                raise RuntimeError(
                    f"{name} PEFT base model has unexpected keys ({len(unexpected)}): {unexpected[:10]}"
                )
            return

        try:
            model_unwrapped.load_state_dict(new_state_dict, strict=True)
            logger.info(f"已恢复 {name} 权重")
        except Exception as strict_error:
            should_raise_mismatch = (not model_only) or name == "planner"
            try:
                missing, unexpected = model_unwrapped.load_state_dict(new_state_dict, strict=False)
            except Exception as relaxed_error:
                message = (
                    f"{name} checkpoint is incompatible with current "
                    f"{model_unwrapped.__class__.__name__}: {relaxed_error}"
                )
                if should_raise_mismatch:
                    raise RuntimeError(message) from relaxed_error
                logger.warning(message)
                return

            if name == "predictor":
                # Optional predictor parameters added after older checkpoints. Keep all other
                # missing/unexpected keys strict so real architecture mismatches still fail loud.
                missing = [k for k in missing if not k.endswith("state_mask_token")]
                missing = [k for k in missing if not k.endswith("future_condition_type_embed")]
                _OPTIONAL_OUTPUT_HEAD_PREFIXES = ("layer_output_norms.", "layer_output_projs.")
                missing = [k for k in missing if not k.startswith(_OPTIONAL_OUTPUT_HEAD_PREFIXES)]

            if missing or unexpected:
                message = (
                    f"{name} checkpoint is incompatible with current "
                    f"{model_unwrapped.__class__.__name__}: "
                    f"missing={len(missing)} {summarize_keys(missing)}, "
                    f"unexpected={len(unexpected)} {summarize_keys(unexpected)}. "
                    "If this checkpoint is only meant to initialize shared encoder/predictor weights, "
                    "disable loading this module (for example, meta.load_planner=False with resume_model_only=True)."
                )
                if should_raise_mismatch:
                    raise RuntimeError(message) from strict_error
                logger.warning(message)

    # 恢复模型权重
    if encoder is not None and "encoder" in checkpoint:
        load_state_dict(encoder, checkpoint["encoder"], "encoder")
    if target_encoder is not None and "target_encoder" in checkpoint:
        load_state_dict(target_encoder, checkpoint["target_encoder"], "target_encoder")
    if predictor is not None and "predictor" in checkpoint:
        load_state_dict(predictor, checkpoint["predictor"], "predictor")
    if seg_head is not None and "seg_head" in checkpoint:
        load_state_dict(seg_head, checkpoint["seg_head"], "seg_head")
    if seg_neck is not None and "seg_neck" in checkpoint:
        load_state_dict(seg_neck, checkpoint["seg_neck"], "seg_neck")
    if planner is not None and "planner" in checkpoint:
        load_state_dict(planner, checkpoint["planner"], "planner")

    # 恢复 optimizer 状态（full-resume 完整性：opt 在保存侧无条件写入，缺失即 checkpoint 损坏/不完整 → fail-loud）
    if not model_only and optimizer is not None:
        if "opt" not in checkpoint:
            raise KeyError(
                f"Full resume from {resume_path} requires 'opt' (optimizer state) but it is missing "
                "(checkpoint incomplete). Use model_only/resume_model_only to warm-start instead."
            )
        optimizer.load_state_dict(checkpoint["opt"])
        logger.info("已恢复 optimizer 状态")

    # 恢复 scaler 状态（"scaler" key 在保存侧总会写入；值为 None 表示保存时未启用 AMP）
    if not model_only and scaler is not None:
        if "scaler" not in checkpoint:
            raise KeyError(
                f"Full resume from {resume_path} requires the 'scaler' key but it is missing (checkpoint incomplete)."
            )
        if checkpoint["scaler"] is None:
            logger.warning("Checkpoint 'scaler' is None (saved without AMP); skipping GradScaler state restore.")
        else:
            scaler.load_state_dict(checkpoint["scaler"])
            logger.info("已恢复 scaler 状态")

    # 恢复 scheduler 状态（保存侧在 scheduler 存在时写入；full-resume 下当前有 scheduler 即要求其存在）
    if not model_only and scheduler is not None:
        if "scheduler" not in checkpoint:
            raise KeyError(
                f"Full resume from {resume_path} requires 'scheduler' state but it is missing (checkpoint incomplete)."
            )
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("已恢复 scheduler 状态")

    # 恢复 wd_scheduler 状态
    if not model_only and wd_scheduler is not None:
        if "wd_scheduler" not in checkpoint:
            raise KeyError(
                f"Full resume from {resume_path} requires 'wd_scheduler' state but it is missing "
                "(checkpoint incomplete)."
            )
        wd_scheduler.load_state_dict(checkpoint["wd_scheduler"])
        logger.info("已恢复 wd_scheduler 状态")

    # 获取 epoch
    if model_only:
        start_epoch = 0
        logger.info("model_only=True: 跳过 optimizer/scheduler/epoch，将从 epoch 0 开始新阶段训练")
    else:
        # fail-loud: full-resume 必须能恢复 epoch；缺失说明 checkpoint 损坏/不完整，
        # 静默从 epoch 0 重训会悄无声息地丢掉已训练进度。若要从该权重开新阶段训练，
        # 请显式走 model_only / resume_model_only 路径。
        if "epoch" not in checkpoint:
            raise KeyError(
                f"Full resume from {resume_path} requires 'epoch' in the checkpoint, but it is missing "
                "(checkpoint corrupt/incomplete). Use model_only/resume_model_only to warm-start instead."
            )
        start_epoch = checkpoint["epoch"]
        logger.info(f"已恢复 epoch，将从 epoch {start_epoch} 继续训练")

    logger.info(f"训练已从 {resume_path} 成功恢复！")

    del checkpoint
    return start_epoch
