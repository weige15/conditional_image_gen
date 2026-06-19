from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from brainrot_diffusion.conditions import build_condition_mappings
from brainrot_diffusion.data import (
    BrainrotTrainDataset,
    GenerationRequestDataset,
    load_image_tensor,
    read_generation_csv,
    read_train_csv,
)


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 64)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_read_train_csv_and_generation_csv_preserve_order(tmp_path: Path) -> None:
    train_csv = _write_csv(
        tmp_path / "train.csv",
        "id,animal,object\n000002.png,dog,pizza\n000001.png,cat,chair\n",
    )
    generate_csv = _write_csv(
        tmp_path / "generate.csv",
        "id,animal,object,prompt\n000101.png,cat,chair,a cat and a chair\n000102.png,dog,pizza,a dog and a pizza\n",
    )

    assert [row.id for row in read_train_csv(train_csv)] == ["000002.png", "000001.png"]
    assert [row.id for row in read_generation_csv(generate_csv)] == ["000101.png", "000102.png"]
    assert GenerationRequestDataset(generate_csv)[0].prompt == "a cat and a chair"


def test_csv_missing_columns_duplicates_empty_and_bad_ids_fail(tmp_path: Path) -> None:
    missing = _write_csv(tmp_path / "missing.csv", "id,animal\n000001.png,cat\n")
    duplicate = _write_csv(
        tmp_path / "duplicate.csv",
        "id,animal,object\n000001.png,cat,chair\n000001.png,dog,pizza\n",
    )
    empty = _write_csv(tmp_path / "empty.csv", "id,animal,object\n")
    bad_id = _write_csv(tmp_path / "bad_id.csv", "id,animal,object\nnested/000001.png,cat,chair\n")

    with pytest.raises(ValueError, match="missing required"):
        read_train_csv(missing)
    with pytest.raises(ValueError, match="duplicate id"):
        read_train_csv(duplicate)
    with pytest.raises(ValueError, match="no rows"):
        read_train_csv(empty)
    with pytest.raises(ValueError, match="without directories"):
        read_train_csv(bad_id)


def test_image_loading_rgb_normalized_and_dataset_conditions(tmp_path: Path) -> None:
    image_dir = tmp_path / "trainset"
    image_dir.mkdir()
    _write_image(image_dir / "000001.png", (255, 0, 0))
    train_csv = _write_csv(tmp_path / "train.csv", "id,animal,object\n000001.png,cat,chair\n")
    rows = read_train_csv(train_csv)
    mappings = build_condition_mappings(rows)

    tensor = load_image_tensor(image_dir / "000001.png")
    dataset = BrainrotTrainDataset(train_csv, image_dir, mappings=mappings)
    sample = dataset[0]

    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert tensor.max().item() <= 1.0
    assert tensor.min().item() >= -1.0
    assert sample["conditions"] == {"animal_id": 0, "object_id": 0, "pair_id": 0}


def test_image_loading_resizes_or_rejects(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "small.png", size=(32, 32))

    assert load_image_tensor(path).shape == (3, 64, 64)
    with pytest.raises(ValueError, match="expected image size"):
        load_image_tensor(path, resize=False)


def test_missing_or_unreadable_image_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_tensor(tmp_path / "missing.png")

    unreadable = tmp_path / "bad.png"
    unreadable.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="could not read image"):
        load_image_tensor(unreadable)


def test_strict_prompt_validation(tmp_path: Path) -> None:
    generate_csv = _write_csv(
        tmp_path / "generate.csv",
        "id,animal,object,prompt\n000101.png,cat,chair,a wrong prompt\n",
    )

    with pytest.raises(ValueError, match="unexpected prompt"):
        read_generation_csv(generate_csv, strict_prompt=True)

