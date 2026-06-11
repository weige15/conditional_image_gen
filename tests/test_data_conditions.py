from __future__ import annotations

import csv
from pathlib import Path

import pytest

from brainrot_diffusion.conditions import build_condition_mappings, condition_ids
from brainrot_diffusion.data import BrainrotTrainDataset, load_generation_requests


def test_mapping_and_image_loading(tiny_dataset):
    mappings = build_condition_mappings(tiny_dataset["train_csv"])
    assert mappings["null_animal_id"] == mappings["num_animals"]
    assert condition_ids("cat", "banana", mappings)["pair_id"] != condition_ids("dog", "chair", mappings)["pair_id"]
    dataset = BrainrotTrainDataset(tiny_dataset["train_csv"], tiny_dataset["image_dir"], mappings)
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert float(item["image"].min()) >= -1.0
    assert float(item["image"].max()) <= 1.0
    assert item["filename"] == "000001.png"


def test_generation_rejects_duplicates_and_unknown_labels(tiny_dataset, tmp_path: Path):
    mappings = build_condition_mappings(tiny_dataset["train_csv"])
    dup = tmp_path / "dup.csv"
    with dup.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "animal", "object", "prompt"])
        writer.writeheader()
        writer.writerow({"id": "x.png", "animal": "cat", "object": "banana", "prompt": "x"})
        writer.writerow({"id": "x.png", "animal": "cat", "object": "banana", "prompt": "x"})
    with pytest.raises(ValueError, match="Duplicate"):
        load_generation_requests(dup, mappings)
    unknown = tmp_path / "unknown.csv"
    with unknown.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "animal", "object", "prompt"])
        writer.writeheader()
        writer.writerow({"id": "x.png", "animal": "horse", "object": "banana", "prompt": "x"})
    with pytest.raises(ValueError, match="Unknown animal"):
        load_generation_requests(unknown, mappings)


def test_missing_columns_and_missing_file(tiny_dataset, tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,animal\nx.png,cat\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        BrainrotTrainDataset(bad, tiny_dataset["image_dir"])
    missing = BrainrotTrainDataset(tiny_dataset["train_csv"], tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        missing[0]
