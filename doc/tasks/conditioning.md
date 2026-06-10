# Conditioning

## Goal

Create stable animal, object, pair, and null-condition ids for training, classifier-free guidance, checkpointing, and generation.

## Inputs

- `doc/proposal.md`: learned animal, object, and pair embeddings; around 10% condition dropout for classifier-free guidance.
- `doc/detailed-design.md`: conditioning owns assignment vocabularies, deterministic pair ids, null ids, unknown-label errors, and persisted mappings.

## Tasks

- [ ] Define the 10 assignment animals and 10 assignment objects as stable vocabularies.
- [ ] Implement deterministic mapping from animal/object strings to ids and from `(animal_id, object_id)` to pair id.
- [ ] Add null/unconditional ids for classifier-free guidance.
- [ ] Implement configurable condition dropout that replaces condition ids with null ids during training.
- [ ] Add mapping serialization and checkpoint-load validation.
- [ ] Add tests for all 100 pairs, unknown labels, stable pair ids, null ids, and condition dropout shapes.

## Done When

- [ ] The same mappings are used by training, sampling, checkpoint loading, and validation.
- [ ] Conditioning tests pass without image or model dependencies.
