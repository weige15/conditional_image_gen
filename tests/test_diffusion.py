from __future__ import annotations

import pytest
import torch

from brainrot_diffusion.diffusion import DiffusionSchedule


def test_schedule_q_sample_and_ddim_steps() -> None:
    diffusion = DiffusionSchedule(10, "cosine")
    assert diffusion.betas.shape == (10,)
    assert torch.all(diffusion.alphas_cumprod[1:] < diffusion.alphas_cumprod[:-1])
    x = torch.zeros(2, 3, 8, 8)
    noise = torch.randn_like(x)
    t = torch.tensor([0, 9])
    noisy = diffusion.q_sample(x, t, noise)
    assert noisy.shape == x.shape
    pred_x0 = diffusion.predict_x0_from_epsilon(noisy, t, noise)
    assert pred_x0.shape == x.shape
    assert diffusion.ddim_timesteps(4).tolist() == [9, 6, 3, 0]
    with pytest.raises(ValueError):
        diffusion.ddim_timesteps(11)
