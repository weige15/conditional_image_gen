# Sampling

## Goal

Implement DDPM and DDIM reverse sampling plus tensor-to-image postprocessing.

## Inputs

- `doc/proposal.md`: DDPM for debugging, DDIM for final generation, 50-250 step sweep.
- `doc/detailed-design.md`: Sampling APIs, failure handling, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/sample.py`.
- [x] Implement `sample_ddpm(model, diffusion, conditions, shape, guidance_scale)`.
- [x] Implement `sample_ddim(model, diffusion, conditions, shape, steps, eta, guidance_scale)`.
- [x] Apply classifier-free guidance inside reverse steps using conditional and null-condition model calls.
- [x] Implement clamp/denormalize conversion to uint8-ready image arrays.
- [x] Add tests with a fake model for output shape, tiny-step DDIM CPU sampling, conversion range, invalid sampler settings, and guidance forwarding.

## Done When

- [x] DDIM sampling returns `[B, 3, 64, 64]` image tensors for condition batches.
- [x] Final conversion produces valid uint8 image data.
- [x] Sampling tests pass independently.
