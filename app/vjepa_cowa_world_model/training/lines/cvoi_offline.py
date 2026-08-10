"""Training line for offline real-geometry anchored CVoI stages."""

import torch

from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_execution import seed_cvoi_process
from app.vjepa_cowa_world_model.training.cvoi_offline import load_cvoi_offline_adapter, run_cvoi_offline_stage
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(args, resume_preempt=False):
    """Run the configured offline CVoI lifecycle stage."""

    if resume_preempt:
        raise ValueError("train_cvoi_offline does not support resume_preempt")
    config = parse_training_config(args)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("train_cvoi_offline requires world_size=1 to prevent concurrent checkpoint writers")
    if not torch.cuda.is_available():
        raise RuntimeError("train_cvoi_offline requires CUDA; CPU fallback would violate the signed CVoI contract")
    seed_cvoi_process(config)
    device = torch.device("cuda")
    logger.info("[cvoi_offline] stage=%s device=%s", config.cvoi.stage, device)
    if config.cvoi.stage in {"field_warmup", "field_calibrated", "stop_calibrated"}:
        adapter = load_cvoi_offline_adapter(config, device=device)
        return run_cvoi_offline_stage(config, device=device, adapter=adapter)
    return run_cvoi_offline_stage(config, device=device)
