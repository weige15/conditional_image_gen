# Diffusion

## Goal

Implement diffusion schedule construction, forward noising, epsilon-prediction loss, and reverse-step helpers shared by DDPM/DDIM sampling.

## Inputs

- `doc/proposal.md`: Baseline is a standard noise-prediction DDPM with DDIM available for faster sampling.
- `doc/high-level-design.md`: Diffusion owns `src/brainrot_diffusion/diffusion.py`, schedules, objective, and sampler math.
- `doc/detailed-design.md`: Diffusion must expose noising, scalar loss, reverse-step behavior, and diffusion metadata.
- `doc/test-plan.md`: Diffusion tests cover finite schedules, deterministic noising, scalar finite loss, output shapes, and invalid config.

## Write Scope

`src/brainrot_diffusion/diffusion.py`, diffusion tests under `tests/`, and diffusion-related config defaults.

## Read Scope

`src/brainrot_diffusion/model.py`, `src/brainrot_diffusion/config.py`, `doc/detailed-design.md`, and sampling requirements.

## Dependencies

Configuration for schedule/timestep settings. Generator Model forward contract for loss computation.

## Tasks

- [x] Implement beta/alpha schedule creation and derived coefficient tensors.
- [x] Implement `q_sample`-style forward noising and epsilon-prediction MSE loss.
- [x] Implement reverse-step helpers needed by DDPM/DDIM sampling.
- [x] Store diffusion metadata needed for checkpoints and generation compatibility.
- [x] Add tests for schedule validity, noising shape, scalar finite loss with a fake/tiny model, deterministic fixed-seed behavior, and invalid timestep config.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_diffusion.py`
- [x] Diffusion tests run on CPU without generated image artifacts.

## Done When

- [x] Diffusion loss computes a finite scalar for a tiny batch.
- [x] Reverse helpers return tensors with expected shapes.
- [x] Diffusion tests pass.
