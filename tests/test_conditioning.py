from __future__ import annotations

import pytest
import torch

from brainrot_diffusion.conditioning import ANIMALS, OBJECTS, ConditionMapper


def test_all_pairs_and_null_dropout() -> None:
    mapper = ConditionMapper()
    seen = set()
    for animal in ANIMALS:
        for obj in OBJECTS:
            animal_id, object_id, pair_id = mapper.encode(animal, obj)
            assert pair_id == animal_id * len(OBJECTS) + object_id
            seen.add(pair_id)
    assert len(seen) == 100
    batch = mapper.encode_many(["cat", "dog"], ["car", "chair"])
    dropped = mapper.apply_dropout(batch, 1.0, generator=torch.Generator().manual_seed(0))
    assert dropped.animal.tolist() == [mapper.null_animal_id, mapper.null_animal_id]
    assert dropped.pair.tolist() == [mapper.null_pair_id, mapper.null_pair_id]


def test_unknown_label_and_mapping_validation() -> None:
    mapper = ConditionMapper()
    with pytest.raises(ValueError, match="unknown animal"):
        mapper.encode("lion", "car")
    with pytest.raises(ValueError, match="vocabulary"):
        ConditionMapper.from_dict({"animals": ["cat"], "objects": OBJECTS})
