# Training Workflow

## Goal

Create the direct PyTorch training loop that connects data, conditioning, diffusion, model, optimizer, EMA, checkpoints, and sample grids.

## Inputs

- `doc/proposal.md`: AdamW, learning rate around `1e-4` to `2e-4`, gradient accumulation if needed, optional mixed precision, checkpoints, EMA, periodic sample grids.
- `doc/detailed-design.md`: training orchestrator owns optimization, finite-loss checks, checkpoint cadence, logging, and smoke training.

## Tasks

- [ ] Implement the training entrypoint that loads config, builds dataloader, conditioning, diffusion schedule, model, optimizer, and EMA.
- [ ] Implement per-batch timestep sampling, Gaussian noise sampling, `q_sample`, epsilon prediction, MSE loss, and optimizer step.
- [ ] Add gradient accumulation and optional mixed precision behind config flags.
- [ ] Update EMA after optimizer steps and save checkpoints at configured intervals.
- [ ] Add periodic sample-grid generation using EMA weights for representative conditions.
- [ ] Add a smoke test that runs one optimization step on a tiny synthetic dataset and verifies finite loss, parameter update, and checkpoint write.

## Done When

- [ ] A tiny training run can complete one step and save a reloadable checkpoint.
- [ ] Training smoke tests pass without the full dataset.
