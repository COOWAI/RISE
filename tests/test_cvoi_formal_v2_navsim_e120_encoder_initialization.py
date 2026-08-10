"""Formal-v2 e120 encoder construction must not consume a legacy checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.training.model_factories.encoder import (
    init_encoder,
    init_encoder_for_full_state_warmstart,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            protocol_version="formal_v2_navsim_e120_h4_v3",
            full_state_warmstart=SimpleNamespace(import_mode="full_state_warmstart"),
        ),
        data=SimpleNamespace(num_target_frames=12, patch_size=16, tubelet_size=2),
        meta=SimpleNamespace(load_encoder=False, pretrain_checkpoint_full=None, use_sdpa=True),
        model=SimpleNamespace(
            backbone="vjepa_img_encoder",
            vjepa_resolution=(256, 512),
            vjepa_num_frames=2,
            vjepa_checkpoint_key="target_encoder",
            vjepa_use_grid_mask=False,
            vjepa_use_causal_attention=False,
            model_name="vit_large",
            uniform_power=False,
            use_rope=True,
            use_activation_checkpointing=False,
        ),
    )


def test_full_state_warmstart_encoder_factory_constructs_without_legacy_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.vjepa_cowa_world_model.models import vjepa_img_encoder

    calls: list[dict[str, object]] = []

    class RecordingAdapter(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            calls.append(dict(kwargs))
            self.weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr(vjepa_img_encoder, "VJEPAImgEncoderAdapter", RecordingAdapter)

    encoder, target_encoder = init_encoder_for_full_state_warmstart(_config(), torch.device("cpu"))

    assert isinstance(encoder, RecordingAdapter)
    assert isinstance(target_encoder, RecordingAdapter)
    assert encoder is not target_encoder
    assert calls == [
        {
            "checkpoint_path": None,
            "resolution": (256, 512),
            "num_frames": 2,
            "max_num_observed_frames": 12,
            "checkpoint_key": "target_encoder",
            "model_name": "vit_large",
            "patch_size": 16,
            "tubelet_size": 2,
            "uniform_power": False,
            "use_rope": True,
            "use_sdpa": True,
            "use_activation_checkpointing": False,
            "use_grid_mask": False,
            "use_causal_attention": False,
        }
    ]


def test_generic_vjepa_factory_still_rejects_load_encoder_false() -> None:
    with pytest.raises(ValueError, match="requires meta.load_encoder=true"):
        init_encoder(_config(), torch.device("cpu"))


def test_training_facade_exports_full_state_warmstart_encoder_factory() -> None:
    from app.vjepa_cowa_world_model import training

    assert training.init_encoder_for_full_state_warmstart is init_encoder_for_full_state_warmstart


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda config: setattr(config.meta, "load_encoder", True), "meta.load_encoder=false"),
        (
            lambda config: setattr(config.meta, "pretrain_checkpoint_full", "/legacy.pt"),
            "meta.pretrain_checkpoint_full=null",
        ),
        (
            lambda config: setattr(config.cvoi, "protocol_version", "legacy_v1"),
            "formal_v2_navsim_e120_h4_v3",
        ),
        (
            lambda config: setattr(config.cvoi.full_state_warmstart, "import_mode", "legacy"),
            "import_mode='full_state_warmstart'",
        ),
    ],
)
def test_full_state_warmstart_encoder_factory_rejects_mixed_initialization(
    mutator,
    message: str,
) -> None:
    config = _config()
    mutator(config)

    with pytest.raises(ValueError, match=message):
        init_encoder_for_full_state_warmstart(config, torch.device("cpu"))
