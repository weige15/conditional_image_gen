from __future__ import annotations

import pytest
import torch

from brainrot_diffusion.diffusion import GaussianDiffusion


class ZeroModel(torch.nn.Module):
    def forward(self, x_t, timesteps, conditions):
        return torch.zeros_like(x_t)


def test_schedule_tensors_are_finite_and_expected_length() -> None:
    diffusion = GaussianDiffusion(timesteps=10, schedule="linear")

    assert diffusion.betas.shape == (10,)
    assert torch.isfinite(diffusion.alphas_cumprod).all()
    assert diffusion.to_metadata()["prediction_type"] == "epsilon"


def test_q_sample_is_deterministic_for_fixed_noise() -> None:
    diffusion = GaussianDiffusion(timesteps=10, schedule="linear")
    x = torch.zeros(2, 3, 64, 64)
    t = torch.tensor([0, 9])
    noise = torch.ones_like(x)

    first = diffusion.q_sample(x, t, noise)
    second = diffusion.q_sample(x, t, noise)

    assert torch.equal(first, second)
    assert first.shape == x.shape


def test_training_loss_is_scalar_and_finite() -> None:
    diffusion = GaussianDiffusion(timesteps=10, schedule="linear")
    loss = diffusion.training_loss(
        ZeroModel(),
        torch.zeros(2, 3, 64, 64),
        torch.tensor([1, 2]),
        {"animal_id": [0, 0], "object_id": [0, 0], "pair_id": [0, 0]},
        noise=torch.ones(2, 3, 64, 64),
    )

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_reverse_steps_preserve_shape() -> None:
    diffusion = GaussianDiffusion(timesteps=10, schedule="linear")
    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([5, 5])
    prev = torch.tensor([4, 4])
    predicted = torch.randn_like(x)

    assert diffusion.ddpm_step(x, t, predicted).shape == x.shape
    assert diffusion.ddim_step(x, t, prev, predicted).shape == x.shape


def test_invalid_diffusion_config_fails() -> None:
    with pytest.raises(ValueError, match="timesteps"):
        GaussianDiffusion(timesteps=0)
    with pytest.raises(ValueError, match="schedule"):
        GaussianDiffusion(timesteps=10, schedule="bad")

