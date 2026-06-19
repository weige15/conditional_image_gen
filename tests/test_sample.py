from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from brainrot_diffusion.checkpoint import save_checkpoint
from brainrot_diffusion.conditions import build_condition_mappings
from brainrot_diffusion.config import apply_overrides, load_config
from brainrot_diffusion.diffusion import GaussianDiffusion
from brainrot_diffusion.model import ConditionalUNet
from brainrot_diffusion.sample import generate_from_checkpoint


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, dict, Path]:
    torch.manual_seed(123)
    generate_csv = tmp_path / "generate.csv"
    generate_csv.write_text(
        "id,animal,object,prompt\n"
        "000101.png,cat,chair,a cat and a chair\n"
        "000102.png,dog,pizza,a dog and a pizza\n",
        encoding="utf-8",
    )
    mappings = build_condition_mappings(
        [
            {"animal": "cat", "object": "chair"},
            {"animal": "dog", "object": "pizza"},
        ]
    )
    model = ConditionalUNet(
        num_animals=mappings.num_animals,
        num_objects=mappings.num_objects,
        num_pairs=mappings.num_pairs,
        base_channels=4,
        embedding_dim=8,
        dropout=0.0,
    )
    diffusion = GaussianDiffusion(timesteps=4, schedule="linear")
    config = apply_overrides(
        load_config("configs/default.yaml"),
        {
            "data.generate_csv": str(generate_csv),
            "model.base_channels": 4,
            "model.embedding_dim": 8,
            "model.dropout": 0.0,
            "diffusion.timesteps": 4,
            "diffusion.schedule": "linear",
            "sampling.sampler": "ddim",
            "sampling.steps": 2,
            "sampling.batch_size": 2,
            "sampling.seed": 777,
            "sampling.guidance_scale": 1.0,
            "sampling.output_dir": str(tmp_path / "generated"),
        },
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model_state=model.state_dict(),
        config=config,
        condition_mappings=mappings,
        diffusion=diffusion.to_metadata(),
        architecture=model.metadata.as_dict(),
        seed={"torch": 123},
        step=0,
    )
    return checkpoint_path, config, generate_csv


def test_generate_writes_exact_rgb_pngs_and_is_deterministic(tmp_path: Path) -> None:
    checkpoint_path, config, _ = _tiny_checkpoint(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = generate_from_checkpoint(checkpoint_path, config=config, output_dir=first_dir)
    second = generate_from_checkpoint(checkpoint_path, config=config, output_dir=second_dir)

    assert [path.name for path in first.files] == ["000101.png", "000102.png"]
    assert [path.name for path in second.files] == ["000101.png", "000102.png"]
    assert (first_dir / "000101.png").read_bytes() == (second_dir / "000101.png").read_bytes()
    with Image.open(first_dir / "000101.png") as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (64, 64)


def test_generate_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    checkpoint_path, config, _ = _tiny_checkpoint(tmp_path)
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    (out_dir / "000101.png").write_bytes(b"already here")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_from_checkpoint(checkpoint_path, config=config, output_dir=out_dir, overwrite=False)


def test_generate_refuses_nonempty_output_directory_without_overwrite(tmp_path: Path) -> None:
    checkpoint_path, config, _ = _tiny_checkpoint(tmp_path)
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    (out_dir / "notes.txt").write_text("leftover", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        generate_from_checkpoint(checkpoint_path, config=config, output_dir=out_dir, overwrite=False)


def test_unknown_labels_fail_before_output_directory_is_created(tmp_path: Path) -> None:
    checkpoint_path, config, _ = _tiny_checkpoint(tmp_path)
    bad_csv = tmp_path / "bad_generate.csv"
    bad_csv.write_text(
        "id,animal,object,prompt\n000101.png,frog,chair,a frog and a chair\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "bad_out"

    with pytest.raises(ValueError, match="unknown animal"):
        generate_from_checkpoint(checkpoint_path, config=config, generate_csv=bad_csv, output_dir=out_dir)

    assert not out_dir.exists()
