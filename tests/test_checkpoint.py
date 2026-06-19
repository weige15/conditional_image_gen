from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainrot_diffusion.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
    validate_generation_compatibility,
)
from brainrot_diffusion.conditions import build_condition_mappings
from brainrot_diffusion.config import load_config
from brainrot_diffusion.diffusion import GaussianDiffusion
from brainrot_diffusion.model import ConditionalUNet


def _checkpoint_inputs():
    mappings = build_condition_mappings([{"animal": "cat", "object": "chair"}])
    model = ConditionalUNet(
        num_animals=mappings.num_animals,
        num_objects=mappings.num_objects,
        num_pairs=mappings.num_pairs,
        base_channels=8,
        embedding_dim=16,
        dropout=0.0,
    )
    diffusion = GaussianDiffusion(timesteps=10, schedule="linear")
    return mappings, model, diffusion


def test_save_load_checkpoint_round_trip(tmp_path: Path) -> None:
    mappings, model, diffusion = _checkpoint_inputs()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model_state=model.state_dict(),
        config=load_config("configs/default.yaml"),
        condition_mappings=mappings,
        diffusion=diffusion.to_metadata(),
        architecture=model.metadata.as_dict(),
        seed={"python": 123, "torch": 123},
        step=7,
    )
    loaded = load_checkpoint(path)

    assert loaded["step"] == 7
    assert loaded["condition_mappings"] == mappings.to_metadata()
    assert loaded["diffusion"]["timesteps"] == 10


def test_missing_required_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_checkpoint({"model": {}})


def test_generation_compatibility_rejects_unknown_labels(tmp_path: Path) -> None:
    mappings, model, diffusion = _checkpoint_inputs()
    payload = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model_state=model.state_dict(),
        config=load_config("configs/default.yaml"),
        condition_mappings=mappings,
        diffusion=diffusion.to_metadata(),
        architecture=model.metadata.as_dict(),
        seed={"python": 123},
        step=0,
    )

    validate_generation_compatibility(payload, [{"animal": "cat", "object": "chair"}])
    with pytest.raises(ValueError, match="unknown animal"):
        validate_generation_compatibility(payload, [{"animal": "dog", "object": "chair"}])


def test_cpu_map_location_loads_tensor_state(tmp_path: Path) -> None:
    mappings, model, diffusion = _checkpoint_inputs()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model_state=model.state_dict(),
        config=load_config("configs/default.yaml"),
        condition_mappings=mappings,
        diffusion=diffusion.to_metadata(),
        architecture=model.metadata.as_dict(),
        seed={"torch": 1},
        step=1,
        ema={"decay": 0.5, "shadow": {"input.weight": torch.zeros_like(model.state_dict()["input.weight"])}},
    )

    loaded = load_checkpoint(path, map_location="cpu")

    assert loaded["model"]["input.weight"].device.type == "cpu"

