from __future__ import annotations

import pytest
import torch

from brainrot_diffusion.diffusion import GaussianDiffusion
from brainrot_diffusion.guidance import combine_cfg, drop_conditions, make_null_condition_batch
from brainrot_diffusion.unet import ConditionalUNet


def _conditions(batch=2):
    return {
        "animal_id": torch.tensor([0, 1])[:batch],
        "object_id": torch.tensor([0, 1])[:batch],
        "pair_id": torch.tensor([0, 3])[:batch],
    }


def test_unet_forward_shape_and_null_path():
    model = ConditionalUNet(2, 2, 4, base_channels=8, channel_mults=[1, 2], blocks_per_level=1, dropout=0.0)
    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([0, 3])
    out = model(x, t, _conditions())
    assert out.shape == x.shape
    null = {"animal_id": torch.tensor([2, 2]), "object_id": torch.tensor([2, 2]), "pair_id": torch.tensor([4, 4])}
    assert model(x, t, null).shape == x.shape


def test_guidance_dropout_and_combination():
    conditions = _conditions()
    dropped = drop_conditions(conditions, 1.0, null_ids={"animal_id": 2, "object_id": 2, "pair_id": 4})
    assert dropped["animal_id"].tolist() == [2, 2]
    assert drop_conditions(conditions, 0.0)["animal_id"].tolist() == [0, 1]
    null = make_null_condition_batch(2, {"null_animal_id": 2, "null_object_id": 2, "null_pair_id": 4})
    assert null["pair_id"].tolist() == [4, 4]
    eps_uncond = torch.zeros(1, 3, 4, 4)
    eps_cond = torch.ones(1, 3, 4, 4)
    assert torch.allclose(combine_cfg(eps_uncond, eps_cond, 1.5), torch.full_like(eps_cond, 1.5))
    with pytest.raises(ValueError):
        combine_cfg(torch.zeros(1), torch.zeros(2), 1.0)
    with pytest.raises(ValueError):
        drop_conditions(conditions, 1.5)


def test_diffusion_shapes_loss_and_determinism():
    diffusion = GaussianDiffusion(timesteps=8)
    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([0, 7])
    noise = torch.randn_like(x)
    xt = diffusion.q_sample(x, t, noise)
    assert xt.shape == x.shape

    class Fake(torch.nn.Module):
        def forward(self, x_t, t, conditions):
            return torch.zeros_like(x_t)

    loss = diffusion.training_loss(Fake(), x, t, _conditions(), noise=noise)
    assert loss.ndim == 0 and torch.isfinite(loss)
    eps = torch.zeros_like(x)
    assert diffusion.p_sample_ddpm(x, t, eps).shape == x.shape
    assert diffusion.ddim_step(x, t, torch.tensor([0, -1]), eps).shape == x.shape
    torch.manual_seed(1)
    a = diffusion.q_sample(x, t, noise)
    torch.manual_seed(1)
    b = diffusion.q_sample(x, t, noise)
    assert torch.allclose(a, b)
