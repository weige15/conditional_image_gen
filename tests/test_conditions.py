from __future__ import annotations

import random

import pytest

from brainrot_diffusion.conditions import (
    ConditionMappings,
    build_condition_mappings,
    condition_batch_from_labels,
)


ROWS = [
    {"id": "000001.png", "animal": "dog", "object": "pizza"},
    {"id": "000002.png", "animal": "cat", "object": "chair"},
]


def test_mappings_are_sorted_and_stable_across_row_order() -> None:
    first = build_condition_mappings(ROWS)
    second = build_condition_mappings(list(reversed(ROWS)))

    assert first.animals == ("cat", "dog")
    assert first.objects == ("chair", "pizza")
    assert first.pairs == second.pairs
    assert first.encode("cat", "chair").as_dict() == second.encode("cat", "chair").as_dict()


def test_mappings_stable_across_deterministic_random_permutations() -> None:
    rows = [
        {"animal": animal, "object": object_name}
        for animal in ("cat", "dog", "frog")
        for object_name in ("chair", "pizza", "drum")
    ]
    expected = build_condition_mappings(rows)
    rng = random.Random(123)

    for _ in range(5):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert build_condition_mappings(shuffled) == expected


def test_metadata_round_trip_preserves_ids() -> None:
    mappings = build_condition_mappings(ROWS)
    restored = ConditionMappings.from_metadata(mappings.to_metadata())

    assert restored == mappings
    assert restored.encode("dog", "pizza") == mappings.encode("dog", "pizza")
    assert restored.null_ids == mappings.null_ids


def test_unknown_labels_fail_before_sampling() -> None:
    mappings = build_condition_mappings(ROWS)

    with pytest.raises(ValueError, match="unknown animal"):
        mappings.encode("frog", "chair")

    with pytest.raises(ValueError, match="unknown object"):
        mappings.encode("cat", "drum")


def test_pair_ids_are_unique_and_batchable() -> None:
    mappings = build_condition_mappings(ROWS)
    pair_ids = [mappings.encode(animal, object_name).pair_id for animal, object_name in mappings.pairs]
    batch = condition_batch_from_labels(mappings, ["cat", "dog"], ["chair", "pizza"])

    assert len(pair_ids) == len(set(pair_ids))
    assert batch["animal_id"] == [0, 1]
    assert batch["object_id"] == [0, 1]


def test_duplicate_or_inconsistent_metadata_fails() -> None:
    with pytest.raises(ValueError, match="duplicate animals"):
        ConditionMappings.from_metadata(
            {
                "version": 1,
                "animals": ["cat", "cat"],
                "objects": ["chair"],
                "pairs": [["cat", "chair"]],
                "include_null": True,
            }
        )

    with pytest.raises(ValueError, match="unknown animal"):
        ConditionMappings.from_metadata(
            {
                "version": 1,
                "animals": ["cat"],
                "objects": ["chair"],
                "pairs": [["dog", "chair"]],
                "include_null": True,
            }
        )
