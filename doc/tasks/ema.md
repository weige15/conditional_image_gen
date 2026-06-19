# EMA

## Goal

Implement exponential moving average weight tracking so sampling can use averaged model weights when available.

## Inputs

- `doc/proposal.md`: EMA weights are part of the intended optimized method and checkpoint metadata.
- `doc/high-level-design.md`: EMA owns `src/brainrot_diffusion/ema.py` and provides model-compatible averaged weights.
- `doc/detailed-design.md`: EMA must support initialization, update, state restoration, disabled no-op behavior, and shape validation.
- `doc/test-plan.md`: EMA tests cover one-step math, disabled behavior, checkpoint inclusion, and mismatched state failures.

## Write Scope

`src/brainrot_diffusion/ema.py`, EMA tests under `tests/`, and integration points in training/checkpointing tasks when those modules are implemented.

## Read Scope

`src/brainrot_diffusion/model.py`, `doc/detailed-design.md`, and checkpoint schema requirements.

## Dependencies

Generator Model parameters. Checkpointing will consume EMA state.

## Tasks

- [x] Implement EMA initialization from a PyTorch model.
- [x] Implement update-after-optimizer-step behavior with configurable decay.
- [x] Implement state dict serialization/restoration.
- [x] Implement disabled/no-op path that sampling cannot mistake for real EMA weights.
- [x] Add tests for scalar update math, disabled behavior, state round trip, and shape mismatch.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_ema.py`
- [x] EMA tests pass without training a real model.

## Done When

- [x] EMA can track and restore tiny model parameters.
- [x] Invalid EMA state fails clearly.
- [x] EMA tests pass.
