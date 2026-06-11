from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .conditions import build_condition_mappings, condition_ids


@dataclass(frozen=True)
class GenerationRequest:
    image_id: str
    animal: str
    object: str
    prompt: str
    animal_id: int
    object_id: int
    pair_id: int

    @property
    def conditions(self) -> dict[str, int]:
        return {
            "animal_id": self.animal_id,
            "object_id": self.object_id,
            "pair_id": self.pair_id,
        }


def read_csv_rows(path: str | Path, required_columns: set[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def load_image_tensor(path: str | Path, image_size: int = 64, random_flip: bool = False) -> torch.Tensor:
    path = Path(path)
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            if random_flip and torch.rand(()) < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            array = np.asarray(image, dtype=np.float32)
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image {path}") from exc
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.div(127.5).sub(1.0)


class BrainrotTrainDataset(Dataset):
    def __init__(
        self,
        train_csv: str | Path,
        image_dir: str | Path,
        mappings: dict[str, Any] | None = None,
        image_size: int = 64,
        random_horizontal_flip: bool = False,
    ) -> None:
        self.train_csv = Path(train_csv)
        self.image_dir = Path(image_dir)
        self.rows = read_csv_rows(self.train_csv, {"id", "animal", "object"})
        self.mappings = mappings or build_condition_mappings(rows=self.rows)
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = self.image_dir / row["id"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing training image for id {row['id']}: {image_path}")
        conditions = condition_ids(row["animal"], row["object"], self.mappings)
        return {
            "image": load_image_tensor(image_path, self.image_size, self.random_horizontal_flip),
            **conditions,
            "filename": row["id"],
        }


def load_generation_requests(generate_csv: str | Path, mappings: dict[str, Any]) -> list[GenerationRequest]:
    path = Path(generate_csv)
    rows = read_csv_rows(path, {"id", "animal", "object", "prompt"})
    seen: set[str] = set()
    requests: list[GenerationRequest] = []
    for row in rows:
        image_id = row["id"]
        if image_id in seen:
            raise ValueError(f"Duplicate generation id in {path}: {image_id}")
        seen.add(image_id)
        ids = condition_ids(row["animal"], row["object"], mappings)
        requests.append(
            GenerationRequest(
                image_id=image_id,
                animal=row["animal"],
                object=row["object"],
                prompt=row["prompt"],
                animal_id=ids["animal_id"],
                object_id=ids["object_id"],
                pair_id=ids["pair_id"],
            )
        )
    return requests


class GenerationRequestDataset(Dataset):
    def __init__(self, generate_csv: str | Path, mappings: dict[str, Any]) -> None:
        self.requests = load_generation_requests(generate_csv, mappings)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, idx: int) -> GenerationRequest:
        return self.requests[idx]
