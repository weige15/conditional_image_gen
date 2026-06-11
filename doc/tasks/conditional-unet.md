# Conditional UNet

## Goal

Implement the from-scratch conditional UNet that predicts diffusion noise for 64x64 images.

## Inputs

- `doc/proposal.md`: Conditional UNet architecture and conditioning strategy.
- `doc/detailed-design.md`: Conditional UNet module contract, input/output shapes, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/unet.py` with residual blocks, downsample/upsample path, GroupNorm, SiLU, and output head.
- [x] Implement sinusoidal timestep embeddings and learned animal/object/pair/null condition embeddings.
- [x] Inject combined timestep and condition embeddings into residual blocks with FiLM-style scale/shift or the documented additive fallback.
- [x] Add attention at 16x16 and config-controlled optional attention at 8x8.
- [x] Validate tensor ranks, image channels, and condition id ranges in debug or explicit checks.
- [x] Add tests for forward shape, null-condition path, mixed condition batches, timestep batches, and default parameter count reporting.

## Done When

- [x] `model(x_t, t, conditions)` returns `[B, 3, 64, 64]`.
- [x] The model initializes without pretrained weights.
- [x] UNet tests pass on CPU with tiny channel settings.
