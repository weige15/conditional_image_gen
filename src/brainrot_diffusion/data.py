"""CSV and image loading for the Brainrot assignment dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .conditions import ConditionMappings


@dataclass(frozen=True)
class TrainRow:
    id: str
    animal: str
    object: str


@dataclass(frozen=True)
class GenerationRequest:
    id: str
    animal: str
    object: str
    prompt: str


def read_train_csv(path: str | Path) -> list[TrainRow]:
    rows = _read_required_csv(path, ("id", "animal", "object"))
    return [TrainRow(id=row["id"], animal=row["animal"], object=row["object"]) for row in rows]


def read_generation_csv(path: str | Path, *, strict_prompt: bool = False) -> list[GenerationRequest]:
    rows = _read_required_csv(path, ("id", "animal", "object", "prompt"))
    requests = [
        GenerationRequest(id=row["id"], animal=row["animal"], object=row["object"], prompt=row["prompt"])
        for row in rows
    ]
    if strict_prompt:
        for request in requests:
            expected = f"a {request.animal} and a {request.object}"
            if request.prompt != expected:
                raise ValueError(f"unexpected prompt for {request.id}: {request.prompt!r}")
    return requests


def load_generation_requests(path: str | Path, *, strict_prompt: bool = False) -> list[GenerationRequest]:
    return read_generation_csv(path, strict_prompt=strict_prompt)


def load_image_tensor(path: str | Path, *, image_size: int = 64, resize: bool = True) -> torch.Tensor:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                if not resize:
                    raise ValueError(f"expected image size {image_size}x{image_size}: {image_path}")
                image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"could not read image {image_path}: {exc}") from exc
    tensor = data.permute(2, 0, 1).float()
    return tensor.div(127.5).sub(1.0)


class BrainrotTrainDataset(Dataset):
    def __init__(
        self,
        train_csv: str | Path,
        image_dir: str | Path,
        *,
        mappings: ConditionMappings | None = None,
        image_size: int = 64,
        resize: bool = True,
    ) -> None:
        self.rows = read_train_csv(train_csv)
        self.image_dir = Path(image_dir)
        self.mappings = mappings
        self.image_size = image_size
        self.resize = resize

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = load_image_tensor(self.image_dir / row.id, image_size=self.image_size, resize=self.resize)
        sample: dict[str, object] = {
            "image": image,
            "id": row.id,
            "animal": row.animal,
            "object": row.object,
        }
        if self.mappings is not None:
            sample["conditions"] = self.mappings.encode(row.animal, row.object).as_dict()
        return sample


class GenerationRequestDataset(Dataset):
    def __init__(self, generate_csv: str | Path, *, strict_prompt: bool = False) -> None:
        self.requests = read_generation_csv(generate_csv, strict_prompt=strict_prompt)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> GenerationRequest:
        return self.requests[index]


def _read_required_csv(path: str | Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")
        rows: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for row_number, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row, required_columns, csv_path, row_number)
            if row["id"] in seen_ids:
                raise ValueError(f"duplicate id in {csv_path}: {row['id']}")
            seen_ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV contains no rows: {csv_path}")
    return rows


def _clean_row(
    row: dict[str, str | None],
    required_columns: Iterable[str],
    csv_path: Path,
    row_number: int,
) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for column in required_columns:
        value = row.get(column)
        if value is None or value.strip() == "":
            raise ValueError(f"missing value for {column!r} in {csv_path} row {row_number}")
        cleaned[column] = value.strip()
    if Path(cleaned["id"]).name != cleaned["id"]:
        raise ValueError(f"id must be a filename without directories in {csv_path} row {row_number}")
    return cleaned
