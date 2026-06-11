# Classifier-Free Guidance

## Goal

Implement condition dropout for training and guided epsilon combination for sampling.

## Inputs

- `doc/proposal.md`: 10% condition dropout and guidance scale sweep.
- `doc/detailed-design.md`: Classifier-Free Guidance APIs, contracts, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/guidance.py`.
- [x] Implement `drop_conditions(condition_batch, p, generator=None)`.
- [x] Implement `make_null_condition_batch(batch_size, mappings, device)`.
- [x] Implement `combine_cfg(eps_uncond, eps_cond, guidance_scale)`.
- [x] Validate dropout probability, guidance scale, batch shape, and prediction shape.
- [x] Add tests for `p=0`, `p=1`, shape mismatch, invalid parameters, and guidance-combination math.

## Done When

- [x] Training code can convert a batch to null conditions with configurable probability.
- [x] Sampling code can combine unconditional and conditional predictions.
- [x] Guidance tests pass independently.
