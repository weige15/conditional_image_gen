from __future__ import annotations

from brainrot_diffusion.checkpoint import save_checkpoint
from brainrot_diffusion.conditioning import ConditionMapper
from brainrot_diffusion.config import config_from_dict
from brainrot_diffusion.ema import EMA
from brainrot_diffusion.model import build_model
from brainrot_diffusion.sampling import generate_from_checkpoint

from .conftest import write_csv


def test_sampling_writes_pngs(tmp_path) -> None:
    generate_csv = tmp_path / "generate.csv"
    write_csv(
        generate_csv,
        [
            {"id": "a.png", "animal": "cat", "object": "car", "prompt": "a cat and a car"},
            {"id": "b.png", "animal": "dog", "object": "chair", "prompt": "a dog and a chair"},
        ],
    )
    config = config_from_dict(
        {
            "seed": 2,
            "image_size": 16,
            "paths": {"generate_csv": str(generate_csv), "output_dir": str(tmp_path / "out")},
            "diffusion": {"timesteps": 4},
            "model": {
                "base_channels": 4,
                "channel_mults": [1],
                "attention_resolutions": [],
                "emb_dim": 16,
            },
            "sampling": {
                "batch_size": 2,
                "ddim_steps": 2,
                "guidance_scale": 1.0,
                "overwrite": True,
            },
        }
    )
    mapper = ConditionMapper()
    model = build_model(config, mapper)
    ema = EMA(model)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model=model,
        ema_state=ema.state_dict(),
        optimizer=None,
        config=config.to_dict(),
        mapper=mapper,
        step=0,
        epoch=0,
        seed_metadata={"seed": 2},
    )
    manifest = generate_from_checkpoint(ckpt, config, device="cpu")
    assert manifest["count"] == 2
    assert (tmp_path / "out" / "a.png").exists()
