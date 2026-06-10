from __future__ import annotations

import pytest

from brainrot_diffusion.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_sampling_checkpoint,
)
from brainrot_diffusion.conditioning import ConditionMapper
from brainrot_diffusion.config import config_from_dict
from brainrot_diffusion.ema import EMA
from brainrot_diffusion.model import build_model


def tiny_config():
    return config_from_dict(
        {
            "image_size": 16,
            "diffusion": {"timesteps": 8},
            "model": {
                "base_channels": 8,
                "channel_mults": [1, 2],
                "attention_resolutions": [],
                "emb_dim": 32,
            },
        }
    )


def test_ema_and_checkpoint_roundtrip(tmp_path) -> None:
    mapper = ConditionMapper()
    config = tiny_config()
    model = build_model(config, mapper)
    ema = EMA(model, decay=0.5)
    for param in model.parameters():
        param.data.add_(1.0)
    ema.update(model)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        ema_state=ema.state_dict(),
        optimizer=None,
        config=config.to_dict(),
        mapper=mapper,
        step=1,
        epoch=0,
        seed_metadata={"seed": 1},
    )
    payload = load_checkpoint(path)
    validate_sampling_checkpoint(payload)
    assert payload["progress"]["step"] == 1
    with pytest.raises(ValueError, match="missing"):
        validate_sampling_checkpoint({"model": {}})
