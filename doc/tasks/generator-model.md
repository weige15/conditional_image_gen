# Generator Model

## Goal

Implement the from-scratch conditional image generator backbone that predicts diffusion noise for `64x64` RGB image tensors.

## Inputs

- `doc/proposal.md`: Use a small UNet-style pixel-space conditional DDPM model trained from scratch.
- `doc/high-level-design.md`: Generator Model owns `src/brainrot_diffusion/model.py` and outputs model predictions for diffusion.
- `doc/detailed-design.md`: Model contract is `model(x_t, timesteps, conditions) -> predicted_noise` with shape `[B, 3, 64, 64]`.
- `doc/test-plan.md`: Model tests cover random initialization, forward shape, batch sizes, and optional null-condition path.

## Write Scope

`src/brainrot_diffusion/model.py`, model tests under `tests/`, and minimal model config defaults in `configs/default.yaml` if needed.

## Read Scope

`src/brainrot_diffusion/config.py`, `src/brainrot_diffusion/conditions.py`, `doc/detailed-design.md`, and checkpoint architecture metadata requirements.

## Dependencies

Configuration for architecture values. Conditions for embedding sizes and optional null IDs.

## Tasks

- [x] Implement a compact PyTorch UNet-style noise-prediction model initialized from scratch.
- [x] Add timestep embeddings and learned animal/object/pair condition embeddings.
- [x] Support null/unconditional conditions if classifier-free guidance is enabled.
- [x] Return predicted noise with the same shape as input image tensors.
- [x] Add CPU tests for forward shape, batch size 1, mixed batch sizes, invalid condition IDs, and no pretrained weight loading.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_model.py`
- [x] `python -m compileall src scripts tests`

## Done When

- [x] The model forward contract works with tiny CPU tensors.
- [x] Architecture metadata needed for reconstruction is available through config/checkpoint inputs.
- [x] Model tests pass.
