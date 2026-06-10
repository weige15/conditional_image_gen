from __future__ import annotations

from brainrot_diffusion.config import config_from_dict
from brainrot_diffusion.training import train

from .conftest import write_csv, write_png


def test_one_step_training_writes_checkpoint(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_png(image_dir / "a.png", size=(16, 16))
    write_png(image_dir / "b.png", size=(16, 16))
    csv_path = tmp_path / "train.csv"
    write_csv(
        csv_path,
        [
            {"id": "a.png", "animal": "cat", "object": "car"},
            {"id": "b.png", "animal": "dog", "object": "chair"},
        ],
    )
    config = config_from_dict(
        {
            "seed": 1,
            "image_size": 16,
            "paths": {
                "train_csv": str(csv_path),
                "train_image_dir": str(image_dir),
                "checkpoint_dir": str(tmp_path / "ckpts"),
            },
            "training": {"batch_size": 2, "num_workers": 0, "checkpoint_every": 1},
            "diffusion": {"timesteps": 8},
            "model": {
                "base_channels": 8,
                "channel_mults": [1, 2],
                "attention_resolutions": [],
                "emb_dim": 32,
            },
        }
    )
    result = train(config, max_steps=1, device="cpu")
    assert result["step"] == 1
    assert (tmp_path / "ckpts" / "checkpoint_step_1.pt").exists()
