# Sampling

## Goal

Implement checkpoint-backed reverse diffusion generation that writes or returns CSV-conditioned RGB `64x64` image outputs.

## Inputs

- `doc/proposal.md`: Generation must produce CSV-matching PNGs from a trained checkpoint using DDIM/DDPM sampling.
- `doc/high-level-design.md`: Sampling owns `src/brainrot_diffusion/sample.py` and `scripts/generate.py` writes `generated_images/{id}`.
- `doc/detailed-design.md`: Sampling loads checkpoint mappings, rebuilds model/diffusion, uses EMA when available, validates labels, seeds sampling, batches requests, and writes PNGs.
- `doc/test-plan.md`: Sampling tests cover deterministic tiny outputs, exact filenames, RGB `64x64` files, and refusal to overwrite without permission.

## Write Scope

`src/brainrot_diffusion/sample.py`, `scripts/generate.py` integration, sampling tests under `tests/`, and tiny fake-checkpoint fixtures.

## Read Scope

Configuration, Data Loading, Conditions, Generator Model, Diffusion, EMA, Checkpointing, and Validation contracts.

## Dependencies

Configuration, Data Loading, Conditions, Generator Model, Diffusion, EMA, and Checkpointing.

## Tasks

- [x] Implement generation entry point that loads checkpoint-backed config, mappings, model, diffusion, and EMA weights when available.
- [x] Validate generation labels against checkpoint mappings before sampling.
- [x] Implement DDIM/DDPM sampling path with guidance-scale handling if supported by checkpoint/model.
- [x] Clamp/convert tensors and save RGB `64x64` PNGs with exact CSV filenames.
- [x] Refuse existing outputs unless overwrite is explicit.
- [x] Add tests with a fake/tiny model checkpoint for filenames, image mode/size, deterministic fixed-seed behavior where feasible, and overwrite failure.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_sample.py`
- [x] Tiny generate path can feed Validation tests.

## Done When

- [x] Tiny generation writes exactly the requested PNG filenames.
- [x] Unknown labels and existing outputs fail before partial generation.
- [x] Sampling tests pass.
