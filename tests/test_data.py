from __future__ import annotations

import pytest

from brainrot_diffusion.conditioning import ConditionMapper
from brainrot_diffusion.data import BrainrotDataset, read_generate_records, read_train_records

from .conftest import write_csv, write_png


def test_dataset_loads_rgb_and_normalizes(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "a.png", size=(32, 32))
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path, [{"id": "a.png", "animal": "cat", "object": "car"}])
    dataset = BrainrotDataset(
        csv_path, images, ConditionMapper(), image_size=64, resize_policy="resize"
    )
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert -1 <= item["image"].min() <= item["image"].max() <= 1
    assert item["animal"].item() == ConditionMapper().animal_id("cat")


def test_missing_columns_and_files(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    bad_csv = tmp_path / "bad.csv"
    write_csv(bad_csv, [{"id": "x.png", "animal": "cat"}], fields=["id", "animal"])
    with pytest.raises(ValueError, match="object"):
        read_train_records(bad_csv)
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path, [{"id": "missing.png", "animal": "cat", "object": "car"}])
    dataset = BrainrotDataset(csv_path, images, ConditionMapper())
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_generate_duplicate_ids(tmp_path) -> None:
    csv_path = tmp_path / "generate.csv"
    write_csv(
        csv_path,
        [
            {"id": "a.png", "animal": "cat", "object": "car", "prompt": "a cat and a car"},
            {"id": "a.png", "animal": "dog", "object": "chair", "prompt": "a dog and a chair"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_generate_records(csv_path)
