# Diffusion Algorithm

## Goal

Implement the DDPM forward noising process, epsilon-prediction training helpers, and DDIM sampling coefficients.

## Inputs

- `doc/proposal.md`: 1,000 training timesteps, cosine schedule primary, linear schedule fallback, MSE epsilon prediction, DDIM with 50 to 250 final steps.
- `doc/detailed-design.md`: diffusion module owns schedules, `q_sample`, predicted `x_0` conversion, DDIM timestep generation, and shape-safe tensor coefficients.

## Tasks

- [ ] Implement cosine beta/noise schedule creation for 1,000 training timesteps.
- [ ] Add linear schedule creation as a debug fallback behind config.
- [ ] Implement `q_sample(x_0, t, epsilon)` and helper extraction of timestep coefficients for image tensors.
- [ ] Implement conversion from predicted epsilon to predicted clean image.
- [ ] Implement DDIM timestep selection and reverse update coefficients with configurable step count and eta.
- [ ] Add tests for schedule shapes, coefficient monotonicity, noising tensor shapes, timestep validation, and DDIM step generation.

## Done When

- [ ] Training can create noisy images and targets through this module only.
- [ ] Diffusion tests pass using fake tensors and do not instantiate the UNet.
