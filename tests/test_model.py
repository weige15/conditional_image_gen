from __future__ import annotations

import pytest
import torch

from brainrot_diffusion.conditions import build_condition_mappings
from brainrot_diffusion.model import AttentionUNet, ConditionalUNet, build_model_from_config


def _mappings():
    return build_condition_mappings(
        [
            {"animal": "cat", "object": "chair"},
            {"animal": "dog", "object": "pizza"},
        ]
    )


def _tiny_model() -> ConditionalUNet:
    mappings = _mappings()
    return ConditionalUNet(
        num_animals=mappings.num_animals,
        num_objects=mappings.num_objects,
        num_pairs=mappings.num_pairs,
        base_channels=8,
        embedding_dim=16,
        dropout=0.0,
    )


def test_forward_shape_for_batch_one_and_three() -> None:
    model = _tiny_model()
    for batch_size in (1, 3):
        x = torch.randn(batch_size, 3, 64, 64)
        timesteps = torch.arange(batch_size)
        conditions = {
            "animal_id": torch.zeros(batch_size, dtype=torch.long),
            "object_id": torch.zeros(batch_size, dtype=torch.long),
            "pair_id": torch.zeros(batch_size, dtype=torch.long),
        }

        out = model(x, timesteps, conditions)

        assert out.shape == x.shape


def test_null_condition_ids_are_supported() -> None:
    mappings = _mappings()
    assert mappings.null_ids is not None
    model = _tiny_model()

    out = model(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]), mappings.null_ids)

    assert out.shape == (2, 3, 64, 64)


def test_invalid_condition_ids_fail_clearly() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="out-of-range"):
        model(
            torch.randn(1, 3, 64, 64),
            torch.tensor([0]),
            {"animal_id": [99], "object_id": [0], "pair_id": [0]},
        )


def test_build_model_from_config_records_architecture_metadata() -> None:
    model = build_model_from_config(
        {"model": {"base_channels": 8, "embedding_dim": 16, "dropout": 0.0, "image_size": 64}},
        _mappings(),
    )

    metadata = model.metadata.as_dict()

    assert metadata["name"] == "compact_unet"
    assert metadata["image_size"] == 64
    assert metadata["num_animals"] == 3
    assert metadata["channel_multipliers"] == [1, 2, 4]


def test_attention_unet_uses_configurable_depth_and_attention() -> None:
    model = build_model_from_config(
        {
            "model": {
                "name": "attention_unet",
                "base_channels": 4,
                "channel_multipliers": [1, 2, 4],
                "num_res_blocks": 2,
                "attention_resolutions": [16],
                "embedding_dim": 16,
                "dropout": 0.0,
                "image_size": 64,
            }
        },
        _mappings(),
    )

    out = model(
        torch.randn(2, 3, 64, 64),
        torch.tensor([0, 1]),
        {"animal_id": [0, 1], "object_id": [0, 1], "pair_id": [0, 3]},
    )

    assert isinstance(model, AttentionUNet)
    assert out.shape == (2, 3, 64, 64)
    assert model.metadata.as_dict()["num_res_blocks"] == 2
    assert model.metadata.as_dict()["attention_resolutions"] == [16]
