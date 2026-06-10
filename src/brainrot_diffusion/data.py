from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .conditioning import ConditionBatch, ConditionMapper


@dataclass(frozen=True)
class CsvRecord:
    id: str
    animal: str
    object: str
    prompt: str | None = None


def read_records(path: str | Path, required_columns: set[str]) -> list[CsvRecord]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required_columns - columns
        if missing:
            raise ValueError(f"{path} missing required column(s): {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            rows.append(
                CsvRecord(
                    id=row["id"],
                    animal=row["animal"],
                    object=row["object"],
                    prompt=row.get("prompt"),
                )
            )
    return rows


def read_train_records(path: str | Path) -> list[CsvRecord]:
    return read_records(path, {"id", "animal", "object"})


def read_generate_records(path: str | Path) -> list[CsvRecord]:
    records = read_records(path, {"id", "animal", "object", "prompt"})
    ids = [record.id for record in records]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError("generate.csv contains duplicate id(s): " + ", ".join(duplicates))
    return records


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    array = ((tensor + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


class BrainrotDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        mapper: ConditionMapper,
        *,
        image_size: int = 64,
        resize_policy: str = "resize",
        horizontal_flip: bool = False,
        flip_prob: float = 0.5,
        color_jitter: bool = False,
        color_jitter_strength: float = 0.08,
    ) -> None:
        self.records = read_train_records(csv_path)
        self.image_dir = Path(image_dir)
        self.mapper = mapper
        self.image_size = image_size
        self.resize_policy = resize_policy
        self.horizontal_flip = horizontal_flip
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter
        self.color_jitter_strength = color_jitter_strength
        if resize_policy not in {"resize", "validate"}:
            raise ValueError("resize_policy must be 'resize' or 'validate'")
        for record in self.records:
            self.mapper.encode(record.animal, record.object)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_id: str) -> Image.Image:
        path = self.image_dir / image_id
        if not path.exists():
            raise FileNotFoundError(f"missing training image: {path}")
        image = Image.open(path).convert("RGB")
        size = (self.image_size, self.image_size)
        if image.size != size:
            if self.resize_policy == "validate":
                raise ValueError(f"{path} has size {image.size}, expected {size}")
            image = image.resize(size, Image.Resampling.BICUBIC)
        return image

    def _augment(self, image: Image.Image) -> Image.Image:
        if self.horizontal_flip and random.random() < self.flip_prob:
            image = ImageOps.mirror(image)
        if self.color_jitter:
            strength = self.color_jitter_strength
            for enhancer_cls in (
                ImageEnhance.Color,
                ImageEnhance.Brightness,
                ImageEnhance.Contrast,
            ):
                factor = 1.0 + random.uniform(-strength, strength)
                image = enhancer_cls(image).enhance(factor)
        return image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        animal_id, object_id, pair_id = self.mapper.encode(record.animal, record.object)
        image = self._augment(self._load_image(record.id))
        return {
            "image": image_to_tensor(image),
            "animal": torch.tensor(animal_id, dtype=torch.long),
            "object": torch.tensor(object_id, dtype=torch.long),
            "pair": torch.tensor(pair_id, dtype=torch.long),
            "id": record.id,
        }


def batch_conditions(batch: dict[str, torch.Tensor]) -> ConditionBatch:
    return ConditionBatch(batch["animal"].long(), batch["object"].long(), batch["pair"].long())
