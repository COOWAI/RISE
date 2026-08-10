"""
模型初始化模块

提供各种模型的初始化函数。
"""

# Facade: definitions live in training/model_factories/*; import surface unchanged.
# flake8: noqa: F401
from app.vjepa_cowa_world_model.training.model_factories.common import _is_main_process, compile_models, logger
from app.vjepa_cowa_world_model.training.model_factories.encoder import (
    _init_encoder_vjepa2_1,
    configure_pretrained_image_encoder_trainability,
    configure_vjepa_encoder_trainability,
    get_encoder_embed_dim,
    init_context_encoder,
    init_context_encoder_for_full_state_warmstart,
    init_encoder,
    init_encoder_for_full_state_warmstart,
    init_proposal_encoder,
    is_dinov2_encoder,
    is_pretrained_image_encoder,
    is_vjepa_encoder,
    should_save_main_encoder,
    validate_factory_pretrained_main_encoder_load_plan,
)
from app.vjepa_cowa_world_model.training.model_factories.planner import init_planner
from app.vjepa_cowa_world_model.training.model_factories.predictor import (
    _init_latent_dit_predictor,
    build_predictor_input_with_future_queries,
    init_predictor,
    init_predictor_for_ae,
    init_predictor_runtime_with_token_ae,
    load_frozen_token_ae,
    prepare_runtime_tokens,
    register_predictor_future_query_tokens,
    resolve_main_predictor_runtime_overrides,
    resolve_runtime_normalize_reps,
)
from app.vjepa_cowa_world_model.training.model_factories.segmentation import init_segmentation_modules
