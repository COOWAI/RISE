"""Training line for the continuous compute-budget controller."""

import torch

from app.vjepa_cowa_world_model.training.budget_controller_training import train_budget_controller_from_config
from app.vjepa_cowa_world_model.training.config import parse_training_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(args, resume_preempt=False):
    """Entry point used by ``--train-script train_budget_controller``."""
    config = parse_training_config(args)
    planner_runtime_modes = {"online_grpo", "oracle_collection"}
    if bool(config.budget_controller.enabled) and config.budget_controller.mode in planner_runtime_modes:
        logger.info(
            "[budget_controller] dispatching %s to planner_world_model runtime",
            config.budget_controller.mode,
        )
        from app.vjepa_cowa_world_model.training.lines.planner_world_model import main as planner_world_model_main

        return planner_world_model_main(args, resume_preempt=resume_preempt)

    del resume_preempt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[budget_controller] using device=%s", device)
    return train_budget_controller_from_config(config, device=device)
