from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

ANIMALS = [
    "shark",
    "crocodile",
    "frog",
    "cat",
    "dog",
    "capybara",
    "elephant",
    "bird",
    "fish",
    "monkey",
]
OBJECTS = [
    "sneaker",
    "airplane",
    "coffee cup",
    "banana",
    "cactus",
    "toilet",
    "pizza",
    "drum",
    "car",
    "chair",
]


@dataclass(frozen=True)
class ConditionBatch:
    animal: torch.Tensor
    object: torch.Tensor
    pair: torch.Tensor

    def to(self, device: torch.device | str) -> ConditionBatch:
        return ConditionBatch(self.animal.to(device), self.object.to(device), self.pair.to(device))


class ConditionMapper:
    def __init__(self, animals: list[str] | None = None, objects: list[str] | None = None) -> None:
        self.animals = animals or list(ANIMALS)
        self.objects = objects or list(OBJECTS)
        self.animal_to_id = {name: idx for idx, name in enumerate(self.animals)}
        self.object_to_id = {name: idx for idx, name in enumerate(self.objects)}
        self.null_animal_id = len(self.animals)
        self.null_object_id = len(self.objects)
        self.null_pair_id = len(self.animals) * len(self.objects)

    @property
    def num_animals_with_null(self) -> int:
        return len(self.animals) + 1

    @property
    def num_objects_with_null(self) -> int:
        return len(self.objects) + 1

    @property
    def num_pairs_with_null(self) -> int:
        return len(self.animals) * len(self.objects) + 1

    def animal_id(self, label: str) -> int:
        try:
            return self.animal_to_id[label]
        except KeyError as exc:
            raise ValueError(f"unknown animal label: {label}") from exc

    def object_id(self, label: str) -> int:
        try:
            return self.object_to_id[label]
        except KeyError as exc:
            raise ValueError(f"unknown object label: {label}") from exc

    def pair_id_from_ids(self, animal_id: int, object_id: int) -> int:
        if animal_id == self.null_animal_id or object_id == self.null_object_id:
            return self.null_pair_id
        if not 0 <= animal_id < len(self.animals):
            raise ValueError(f"invalid animal id: {animal_id}")
        if not 0 <= object_id < len(self.objects):
            raise ValueError(f"invalid object id: {object_id}")
        return animal_id * len(self.objects) + object_id

    def encode(self, animal: str, object_label: str) -> tuple[int, int, int]:
        animal_id = self.animal_id(animal)
        object_id = self.object_id(object_label)
        return animal_id, object_id, self.pair_id_from_ids(animal_id, object_id)

    def encode_many(self, animals: list[str], objects: list[str]) -> ConditionBatch:
        encoded = [self.encode(a, o) for a, o in zip(animals, objects, strict=True)]
        animal, object_id, pair = zip(*encoded, strict=True)
        return ConditionBatch(
            torch.tensor(animal, dtype=torch.long),
            torch.tensor(object_id, dtype=torch.long),
            torch.tensor(pair, dtype=torch.long),
        )

    def null_batch(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> ConditionBatch:
        return ConditionBatch(
            torch.full((batch_size,), self.null_animal_id, dtype=torch.long, device=device),
            torch.full((batch_size,), self.null_object_id, dtype=torch.long, device=device),
            torch.full((batch_size,), self.null_pair_id, dtype=torch.long, device=device),
        )

    def apply_dropout(
        self,
        conditions: ConditionBatch,
        dropout_prob: float,
        *,
        generator: torch.Generator | None = None,
    ) -> ConditionBatch:
        if dropout_prob <= 0:
            return conditions
        mask = (
            torch.rand(
                conditions.animal.shape, device=conditions.animal.device, generator=generator
            )
            < dropout_prob
        )
        animal = conditions.animal.clone()
        object_id = conditions.object.clone()
        pair = conditions.pair.clone()
        animal[mask] = self.null_animal_id
        object_id[mask] = self.null_object_id
        pair[mask] = self.null_pair_id
        return ConditionBatch(animal, object_id, pair)

    def to_dict(self) -> dict[str, Any]:
        return {"animals": self.animals, "objects": self.objects}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionMapper:
        mapper = cls(list(data["animals"]), list(data["objects"]))
        mapper.validate()
        return mapper

    def validate(self) -> None:
        if self.animals != ANIMALS:
            raise ValueError("checkpoint animal vocabulary does not match assignment vocabulary")
        if self.objects != OBJECTS:
            raise ValueError("checkpoint object vocabulary does not match assignment vocabulary")
