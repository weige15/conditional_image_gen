# Training Loop

## Goal

Implement a minimal reproducible training lifecycle that can run a tiny CPU smoke train and later scale to full Brainrot training.

## Inputs

- `doc/proposal.md`: Training must use PyTorch directly, save checkpoints, and start with smoke runs before full GPU training.
- `doc/high-level-design.md`: Training Loop owns `src/brainrot_diffusion/train_loop.py` and orchestrates config, data, conditions, model, diffusion, EMA, and checkpointing.
- `doc/detailed-design.md`: Training step samples timesteps, optionally drops conditions, computes diffusion loss, updates optimizer/EMA, logs, and checkpoints.
- `doc/test-plan.md`: Training tests must run tiny CPU training, produce finite losses, and write checkpoints with metadata.

## Write Scope

`src/brainrot_diffusion/train_loop.py`, `scripts/train.py` integration, training tests under `tests/`, and smoke config fixtures.

## Read Scope

Configuration, Data Loading, Conditions, Generator Model, Diffusion, EMA, Checkpointing, `dataset/train.csv`, and `dataset/trainset/`.

## Dependencies

Configuration, Data Loading, Conditions, Generator Model, Diffusion, EMA, and Checkpointing.

## Tasks

- [x] Implement fresh training entry point using resolved config and assignment dataset paths.
- [x] Seed Python, NumPy, and PyTorch where practical and record seed metadata.
- [x] Build dataloader, mappings, model, diffusion object, optimizer, and optional EMA.
- [x] Implement bounded training loop with finite-loss checks, optimizer steps, EMA updates, logs, and checkpoint cadence.
- [x] Implement resume validation if resume support is included.
- [x] Add tiny CPU smoke test that trains for a few steps and writes a metadata-complete checkpoint.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_train_loop.py`
- [x] Tiny training smoke path runs on CPU without full dataset generation.

## Done When

- [x] A tiny CPU training run produces finite loss and a checkpoint.
- [x] Checkpoint includes config, mappings, diffusion metadata, architecture metadata, seed metadata, and step.
- [x] Training-loop tests pass.
