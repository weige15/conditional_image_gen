# Diffusion Objective

## Goal

Implement DDPM noising, epsilon-prediction loss, schedule buffers, and reverse-step helpers shared by DDPM/DDIM sampling.

## Inputs

- `doc/proposal.md`: 1,000-step cosine schedule and epsilon-prediction MSE objective.
- `doc/detailed-design.md`: Diffusion Objective module APIs and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/diffusion.py` with `GaussianDiffusion`.
- [x] Implement cosine beta schedule, alpha products, posterior coefficients, and device-safe schedule buffers.
- [x] Implement `q_sample`, `training_loss`, `predict_x0_from_eps`, `p_sample_ddpm`, and `ddim_step`.
- [x] Validate timestep count, beta range, finite schedule values, and tensor shape compatibility.
- [x] Add tests for schedule construction, `q_sample`, scalar loss with a fake model, reverse-helper shapes, and seeded determinism.

## Done When

- [x] Diffusion loss computes finite scalar MSE for a fake batch.
- [x] DDPM/DDIM helper outputs have expected shapes.
- [x] Diffusion tests pass independently.
