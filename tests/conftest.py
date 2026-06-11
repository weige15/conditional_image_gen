from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_png(path: Path, color: tuple[int, int, int] = (128, 64, 32), mode: str = "RGB", size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (size, size), color if mode == "RGB" else 128)
    image.save(path)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> dict[str, Path]:
    image_dir = tmp_path / "trainset"
    rows = [
        {"id": "000001.png", "animal": "cat", "object": "banana"},
        {"id": "000002.png", "animal": "dog", "object": "chair"},
    ]
    for index, row in enumerate(rows):
        write_png(image_dir / row["id"], (64 + index * 40, 100, 140))
    train_csv = tmp_path / "train.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "animal", "object"])
        writer.writeheader()
        writer.writerows(rows)
    generate_csv = tmp_path / "generate.csv"
    with generate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "animal", "object", "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {"prompt": f"a {row['animal']} and a {row['object']}"})
    return {"train_csv": train_csv, "generate_csv": generate_csv, "image_dir": image_dir}


def tiny_config(paths: dict[str, Path], tmp_path: Path, max_steps: int = 2) -> dict:
    return {
        "paths": {
            "train_csv": str(paths["train_csv"]),
            "train_image_dir": str(paths["image_dir"]),
            "generate_csv": str(paths["generate_csv"]),
            "output_dir": str(tmp_path / "generated_images"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "reference_dir": str(tmp_path / "ref"),
        },
        "data": {"image_size": 64, "random_horizontal_flip": False, "num_workers": 0},
        "model": {
            "image_channels": 3,
            "base_channels": 8,
            "channel_mults": [1, 2],
            "blocks_per_level": 1,
            "dropout": 0.0,
            "attention_resolutions": [],
        },
        "diffusion": {"timesteps": 4, "beta_schedule": "cosine"},
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "mixed_precision": False,
        },
        "ema": {"decay": 0.9},
        "training": {"seed": 7, "condition_dropout": 0.1, "log_every": 0, "save_every": 100, "resume": None},
        "sampling": {
            "sampler": "ddim",
            "batch_size": 2,
            "steps": 2,
            "eta": 0.0,
            "guidance_scale": 1.0,
            "seed": 7,
            "use_ema": True,
        },
    }
