"""Entry for --train-script train_latent_predictor: trains the world-model PREDICTOR.

Unifies what used to be split across two entries:
  - lewm predictor (latent-world-model target + SIGReg/projector)  -> old train_world_model
  - ema/standard predictor                                          -> old train_planner_world_model
                                                                       with planner.use_planner=false
Dispatches on the predictor method so a newcomer reaches for ONE predictor entry regardless of variant.
The routed line still validates its own config fail-loud; this layer only chooses the route.
"""

from app.vjepa_cowa_world_model.training.lines import planner_world_model, world_model

_LEWM_METHODS = ("lewm", "le-wm", "le_wm")


def _is_lewm_predictor(args):
    """True if the config asks for the LeWM predictor (method or a world_model/lewm section)."""
    predictor_cfg = args.get("predictor") or {}
    method = (
        str(
            args.get("method")
            or args.get("training_method")
            or args.get("predictor_method")
            or predictor_cfg.get("method")
            or ""
        )
        .strip()
        .lower()
    )
    if method in _LEWM_METHODS:
        return True
    if method == "ema":
        return False
    # No explicit method: fall back to the structural signal — lewm configs enable a world_model section.
    world_model_cfg = (
        args.get("world_model")
        or args.get("lewm")
        or args.get("le-wm")
        or args.get("le_wm")
        or predictor_cfg.get("lewm")
        or {}
    )
    return bool(world_model_cfg.get("enabled", False))


def main(args, resume_preempt=False):
    if _is_lewm_predictor(args):
        return world_model.main(args=args, resume_preempt=resume_preempt)
    return planner_world_model.main(args=args, resume_preempt=resume_preempt)
