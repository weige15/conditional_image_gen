from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

CONDITION_KEYS = ("animal_id", "object_id", "pair_id")


def _require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> None:
    missing = required.difference(fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _pair_key(animal: str, obj: str) -> str:
    return f"{animal}|||{obj}"


def build_condition_mappings(train_csv: str | Path | None = None, rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    if rows is None:
        if train_csv is None:
            raise ValueError("Either train_csv or rows must be provided")
        path = Path(train_csv)
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _require_columns(reader.fieldnames, {"animal", "object"}, path)
            rows = list(reader)

    animals = sorted({row["animal"] for row in rows})
    objects = sorted({row["object"] for row in rows})
    if not animals or not objects:
        raise ValueError("Condition mappings require at least one animal and one object")

    animal_to_id = {name: idx for idx, name in enumerate(animals)}
    object_to_id = {name: idx for idx, name in enumerate(objects)}
    pair_to_id = {
        _pair_key(animal, obj): animal_to_id[animal] * len(objects) + object_to_id[obj]
        for animal in animals
        for obj in objects
    }
    return {
        "animal_to_id": animal_to_id,
        "object_to_id": object_to_id,
        "pair_to_id": pair_to_id,
        "id_to_animal": animals,
        "id_to_object": objects,
        "id_to_pair": [None] * (len(animals) * len(objects)),
        "num_animals": len(animals),
        "num_objects": len(objects),
        "num_pairs": len(animals) * len(objects),
        "null_animal_id": len(animals),
        "null_object_id": len(objects),
        "null_pair_id": len(animals) * len(objects),
    } | {
        "id_to_pair": [
            _pair_key(animal, obj)
            for animal in animals
            for obj in objects
        ]
    }


def condition_ids(animal: str, obj: str, mappings: dict[str, Any]) -> dict[str, int]:
    try:
        animal_id = int(mappings["animal_to_id"][animal])
    except KeyError as exc:
        raise ValueError(f"Unknown animal label {animal!r}") from exc
    try:
        object_id = int(mappings["object_to_id"][obj])
    except KeyError as exc:
        raise ValueError(f"Unknown object label {obj!r}") from exc
    pair_key = _pair_key(animal, obj)
    try:
        pair_id = int(mappings["pair_to_id"][pair_key])
    except KeyError as exc:
        raise ValueError(f"Unknown animal/object pair {animal!r}, {obj!r}") from exc
    return {"animal_id": animal_id, "object_id": object_id, "pair_id": pair_id}


def null_condition_ids(mappings: dict[str, Any]) -> dict[str, int]:
    return {
        "animal_id": int(mappings["null_animal_id"]),
        "object_id": int(mappings["null_object_id"]),
        "pair_id": int(mappings["null_pair_id"]),
    }


def mapping_sizes(mappings: dict[str, Any]) -> dict[str, int]:
    return {
        "num_animals": int(mappings["num_animals"]),
        "num_objects": int(mappings["num_objects"]),
        "num_pairs": int(mappings["num_pairs"]),
    }


def batch_conditions(items: list[dict[str, int]], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot batch an empty condition list")
    return {
        key: torch.tensor([int(item[key]) for item in items], dtype=torch.long, device=device)
        for key in CONDITION_KEYS
    }


def validate_mappings(mappings: dict[str, Any]) -> None:
    required = {
        "animal_to_id",
        "object_to_id",
        "pair_to_id",
        "num_animals",
        "num_objects",
        "num_pairs",
        "null_animal_id",
        "null_object_id",
        "null_pair_id",
    }
    missing = required.difference(mappings)
    if missing:
        raise ValueError(f"Condition mappings missing keys: {sorted(missing)}")
    if int(mappings["null_animal_id"]) != int(mappings["num_animals"]):
        raise ValueError("null_animal_id must equal num_animals")
    if int(mappings["null_object_id"]) != int(mappings["num_objects"]):
        raise ValueError("null_object_id must equal num_objects")
    if int(mappings["null_pair_id"]) != int(mappings["num_pairs"]):
        raise ValueError("null_pair_id must equal num_pairs")
