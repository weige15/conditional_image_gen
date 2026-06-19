# Conditions

## Goal

Implement stable animal, object, and pair mappings that can be saved in checkpoints and reused for generation without mapping drift.

## Inputs

- `doc/proposal.md`: Stable condition mappings are required for animal, object, and animal-object pair labels.
- `doc/high-level-design.md`: Conditions owns `src/brainrot_diffusion/conditions.py` and generation must use checkpoint-compatible mappings.
- `doc/detailed-design.md`: Conditions must build, serialize, restore, and validate mappings plus optional null-condition metadata.
- `doc/test-plan.md`: Condition tests cover sorted/stable mappings, serialization round trip, unknown labels, and row permutation stability.

## Write Scope

`src/brainrot_diffusion/conditions.py`, condition tests under `tests/`, and any shared test fixtures needed for labels.

## Read Scope

`doc/detailed-design.md`, `doc/test-plan.md`, data-loading records, and checkpoint metadata schema once implemented.

## Dependencies

Data Loading for input row shape. Checkpointing later depends on serialized mapping metadata.

## Tasks

- [x] Implement deterministic mapping construction for animal labels, object labels, and pair labels.
- [x] Implement mapping serialization/deserialization to JSON-compatible metadata.
- [x] Implement generation-label compatibility validation against restored checkpoint mappings.
- [x] Represent null/unconditional condition IDs if classifier-free guidance is enabled.
- [x] Add tests for row permutation stability, round trip, unknown labels, duplicate/inconsistent metadata, and pair ID uniqueness.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_conditions.py`
- [x] Condition mapping property tests run on CPU with deterministic seeds.

## Done When

- [x] Generation cannot rebuild incompatible mappings from `dataset/generate.csv`.
- [x] Unknown generation labels fail before sampling.
- [x] Conditions tests pass.
