from __future__ import annotations

import torch

from brainrot_diffusion.conditioning import ConditionMapper
from brainrot_diffusion.model import ConditionalUNet


def test_model_forward_with_attention_and_null_ids() -> None:
    mapper = ConditionMapper()
    model = ConditionalUNet(
        image_size=16,
        base_channels=8,
        channel_mults=[1, 2],
        attention_resolutions=[8],
        emb_dim=32,
        num_animals=mapper.num_animals_with_null,
        num_objects=mapper.num_objects_with_null,
        num_pairs=mapper.num_pairs_with_null,
    )
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([0, 3])
    conditions = mapper.null_batch(2)
    out = model(x, t, conditions)
    assert out.shape == x.shape
