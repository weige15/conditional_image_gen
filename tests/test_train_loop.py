from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from brainrot_diffusion.checkpoint import load_checkpoint
from brainrot_diffusion.config import apply_overrides, load_config
from brainrot_diffusion.train_loop import train


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), color).save(path)


def _tiny_train_config(tmp_path: Path) -> dict:
    image_dir = tmp_path / "trainset"
    image_dir.mkdir()
    _write_image(image_dir / "000001.png", (255, 0, 0))
    _write_image(image_dir / "000002.png", (0, 255, 0))
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("id,animal,object\n000001.png,cat,chair\n000002.png,dog,pizza\n", encoding="utf-8")
    generate_csv = tmp_path / "generate.csv"
    generate_csv.write_text(
        "id,animal,object,prompt\n000101.png,cat,chair,a cat and a chair\n",
        encoding="utf-8",
    )
    return apply_overrides(
        load_config("configs/default.yaml"),
        {
            "data.train_csv": str(train_csv),
            "data.generate_csv": str(generate_csv),
            "data.train_image_dir": str(image_dir),
            "model.base_channels": 4,
            "model.embedding_dim": 8,
            "model.dropout": 0.0,
            "diffusion.timesteps": 4,
            "diffusion.schedule": "linear",
            "training.batch_size": 1,
            "training.max_steps": 2,
            "training.save_every": 2,
            "training.learning_rate": 0.001,
            "training.num_workers": 0,
            "training.condition_dropout": 0.0,
            "sampling.steps": 2,
            "sampling.batch_size": 1,
            "checkpointing.checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    )


def test_tiny_cpu_training_writes_metadata_complete_checkpoint(tmp_path: Path, capsys) -> None:
    config = _tiny_train_config(tmp_path)

    result = train(config)
    output = capsys.readouterr().out
    checkpoint = load_checkpoint(result.final_checkpoint)

    assert result.final_checkpoint.exists()
    assert result.step == 2
    assert math.isfinite(result.last_loss)
    assert "training device=" in output
    assert "step=1/2" in output
    assert "saved checkpoint:" in output
    assert checkpoint["step"] == 2
    assert checkpoint["config"]["data"]["train_csv"] == config["data"]["train_csv"]
    assert checkpoint["condition_mappings"]["animals"] == ["cat", "dog"]
    assert checkpoint["diffusion"]["timesteps"] == 4
    assert checkpoint["architecture"]["base_channels"] == 4
    assert checkpoint["seed"]["torch"] == config["training"]["seed"]
