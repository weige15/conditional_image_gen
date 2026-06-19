# Checkpointing

## Goal

Implement checkpoint save/load and metadata validation so training and generation can reproduce model, config, mappings, diffusion state, and seeds.

## Inputs

- `doc/proposal.md`: Checkpoints must save config, EMA, condition mappings, diffusion metadata, architecture metadata, and seed metadata.
- `doc/high-level-design.md`: Checkpointing owns `src/brainrot_diffusion/checkpoint.py` and enables generation from saved model/mappings.
- `doc/detailed-design.md`: Checkpoint schema requires `model`, `config`, `condition_mappings`, `diffusion`, `architecture`, `seed`, and `step`.
- `doc/test-plan.md`: Checkpoint tests cover round trip, required metadata, missing-key rejection, mapping compatibility, and CPU map-location loading.

## Write Scope

`src/brainrot_diffusion/checkpoint.py`, checkpoint tests under `tests/`, and integration hooks in training/sampling/package tasks.

## Read Scope

`src/brainrot_diffusion/config.py`, `src/brainrot_diffusion/conditions.py`, `src/brainrot_diffusion/ema.py`, `doc/detailed-design.md`, and packaging requirements.

## Dependencies

Configuration, Conditions, Generator Model state dicts, Diffusion metadata, and EMA state.

## Tasks

- [x] Implement checkpoint schema validation for required and optional fields.
- [x] Implement CPU-compatible save/load helpers using PyTorch serialization.
- [x] Preserve config, condition mappings, diffusion metadata, architecture metadata, seed metadata, and step counters.
- [x] Add atomic write behavior where practical.
- [x] Add tests for tiny checkpoint round trip, missing-key rejection, mapping metadata preservation, unknown-label compatibility failure, and CPU loading.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_checkpoint.py`
- [x] Tiny checkpoint tests pass without full training.

## Done When

- [x] Sampling can load checkpoint mappings and model metadata before generation.
- [x] Corrupt or incomplete checkpoints fail before sampling.
- [x] Checkpointing tests pass.
