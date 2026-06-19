"""Stable condition mappings for animal-object generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


NULL_LABEL = "__null__"


@dataclass(frozen=True)
class ConditionIDs:
    animal_id: int
    object_id: int
    pair_id: int

    def as_dict(self) -> dict[str, int]:
        return {
            "animal_id": self.animal_id,
            "object_id": self.object_id,
            "pair_id": self.pair_id,
        }


@dataclass(frozen=True)
class ConditionMappings:
    animals: tuple[str, ...]
    objects: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    include_null: bool = True

    def __post_init__(self) -> None:
        _reject_duplicates("animals", self.animals)
        _reject_duplicates("objects", self.objects)
        _reject_duplicates("pairs", self.pairs)
        if NULL_LABEL in self.animals or NULL_LABEL in self.objects:
            raise ValueError(f"{NULL_LABEL!r} is reserved")
        animal_set = set(self.animals)
        object_set = set(self.objects)
        for animal, object_name in self.pairs:
            if animal not in animal_set:
                raise ValueError(f"pair uses unknown animal: {animal}")
            if object_name not in object_set:
                raise ValueError(f"pair uses unknown object: {object_name}")

    @property
    def animal_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.animals)}

    @property
    def object_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.objects)}

    @property
    def pair_to_id(self) -> dict[tuple[str, str], int]:
        return {pair: index for index, pair in enumerate(self.pairs)}

    @property
    def num_animals(self) -> int:
        return len(self.animals) + int(self.include_null)

    @property
    def num_objects(self) -> int:
        return len(self.objects) + int(self.include_null)

    @property
    def num_pairs(self) -> int:
        return len(self.pairs) + int(self.include_null)

    @property
    def null_ids(self) -> ConditionIDs | None:
        if not self.include_null:
            return None
        return ConditionIDs(len(self.animals), len(self.objects), len(self.pairs))

    def encode(self, animal: str, object_name: str) -> ConditionIDs:
        try:
            animal_id = self.animal_to_id[animal]
        except KeyError as exc:
            raise ValueError(f"unknown animal label: {animal}") from exc
        try:
            object_id = self.object_to_id[object_name]
        except KeyError as exc:
            raise ValueError(f"unknown object label: {object_name}") from exc
        try:
            pair_id = self.pair_to_id[(animal, object_name)]
        except KeyError as exc:
            raise ValueError(f"unknown animal-object pair: {animal}/{object_name}") from exc
        return ConditionIDs(animal_id, object_id, pair_id)

    def validate_requests(self, requests: Iterable[object]) -> None:
        for request in requests:
            self.encode(_get_label(request, "animal"), _get_label(request, "object"))

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": 1,
            "animals": list(self.animals),
            "objects": list(self.objects),
            "pairs": [[animal, object_name] for animal, object_name in self.pairs],
            "include_null": self.include_null,
            "null_label": NULL_LABEL if self.include_null else None,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "ConditionMappings":
        if metadata.get("version") != 1:
            raise ValueError("unsupported condition mapping metadata version")
        animals = _string_list(metadata.get("animals"), "animals")
        objects = _string_list(metadata.get("objects"), "objects")
        pairs_value = metadata.get("pairs")
        if not isinstance(pairs_value, Sequence):
            raise ValueError("pairs must be a sequence")
        pairs: list[tuple[str, str]] = []
        for pair in pairs_value:
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
            ):
                raise ValueError("each pair must be [animal, object]")
            pairs.append((pair[0], pair[1]))
        return cls(
            animals=tuple(animals),
            objects=tuple(objects),
            pairs=tuple(pairs),
            include_null=bool(metadata.get("include_null", True)),
        )


def build_condition_mappings(
    rows: Iterable[object],
    *,
    include_null: bool = True,
    pair_strategy: str = "cartesian",
) -> ConditionMappings:
    """Build deterministic mappings from row-like objects with animal/object labels."""

    rows_list = list(rows)
    animals = sorted({_get_label(row, "animal") for row in rows_list})
    objects = sorted({_get_label(row, "object") for row in rows_list})
    if not animals or not objects:
        raise ValueError("cannot build mappings from empty labels")
    if pair_strategy == "cartesian":
        pairs = [(animal, object_name) for animal in animals for object_name in objects]
    elif pair_strategy == "observed":
        pairs = sorted({(_get_label(row, "animal"), _get_label(row, "object")) for row in rows_list})
    else:
        raise ValueError("pair_strategy must be 'cartesian' or 'observed'")
    return ConditionMappings(tuple(animals), tuple(objects), tuple(pairs), include_null=include_null)


def condition_batch_from_labels(
    mappings: ConditionMappings,
    animals: Sequence[str],
    objects: Sequence[str],
) -> dict[str, list[int]]:
    if len(animals) != len(objects):
        raise ValueError("animals and objects must have the same length")
    encoded = [mappings.encode(animal, object_name) for animal, object_name in zip(animals, objects)]
    return {
        "animal_id": [item.animal_id for item in encoded],
        "object_id": [item.object_id for item in encoded],
        "pair_id": [item.pair_id for item in encoded],
    }


def _get_label(row: object, field: str) -> str:
    if isinstance(row, Mapping):
        value = row.get(field)
    else:
        value = getattr(row, field, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row missing nonempty {field!r} label")
    return value


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must contain nonempty strings")
        result.append(item)
    return result


def _reject_duplicates(name: str, values: Iterable[object]) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {name} entry: {value}")
        seen.add(value)
