# Checkpoints and EMA

## Goal

Implement EMA weight tracking and checkpoint save/load contracts for training resume and sampling-only generation.

## Inputs

- `doc/proposal.md`: checkpoint contents include model weights, EMA weights, optimizer state, scheduler state, config, step count, seed, and label mappings.
- `doc/detailed-design.md`: checkpoints must include architecture config, diffusion config, condition mappings, seed, and compatibility diagnostics.

## Tasks

- [ ] Implement EMA initialization, update, state dict export, and state dict load.
- [ ] Implement checkpoint save with raw model state, EMA state, optimizer state, config, mappings, seed, and progress counters.
- [ ] Implement training-resume load that restores optimizer and progress state.
- [ ] Implement sampling-only load that requires model config, diffusion config, mappings, and EMA weights.
- [ ] Add compatibility checks for architecture metadata and vocabulary metadata before loading weights.
- [ ] Add tests for EMA math, checkpoint round-trip, missing metadata failures, and sampling-only load.

## Done When

- [ ] Training can resume from a checkpoint and sampling can load EMA weights without optimizer state.
- [ ] Checkpoint and EMA tests pass independently of the full training loop.
