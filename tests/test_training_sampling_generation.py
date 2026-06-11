from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from brainrot_diffusion.checkpoint import export_model_pth, load_checkpoint
from brainrot_diffusion.diffusion import GaussianDiffusion
from brainrot_diffusion.sample import denormalize_to_uint8, generate_from_checkpoint, sample_ddim
from brainrot_diffusion.train_loop import train
from conftest import tiny_config


def test_tiny_training_checkpoint_and_generation(tiny_dataset, tmp_path: Path):
    config = tiny_config(tiny_dataset, tmp_path)
    result = train(config)
    state = load_checkpoint(result.checkpoint_path)
    assert state["step"] == 2
    assert "shadow" in state["ema"]
    exported = tmp_path / "model.pth"
    export_model_pth(result.checkpoint_path, exported)
    assert load_checkpoint(exported)["step"] == 2
    written = generate_from_checkpoint(result.checkpoint_path, config, overwrite=True)
    assert len(written) == 2
    for path in written:
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == (64, 64)
    with pytest.raises(FileExistsError):
        generate_from_checkpoint(result.checkpoint_path, config, overwrite=False)


def test_sampling_tensor_conversion():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x_t, t, conditions):
            return torch.zeros_like(x_t)

    diffusion = GaussianDiffusion(timesteps=4)
    conditions = {
        "animal_id": torch.tensor([0]),
        "object_id": torch.tensor([0]),
        "pair_id": torch.tensor([0]),
    }
    image = sample_ddim(Fake(), diffusion, conditions, (1, 3, 64, 64), steps=2)
    assert image.shape == (1, 3, 64, 64)
    array = denormalize_to_uint8(image)
    assert array.dtype.name == "uint8"
    assert array.min() >= 0 and array.max() <= 255
    with pytest.raises(ValueError):
        sample_ddim(Fake(), diffusion, conditions, (1, 3, 64, 64), steps=0)
