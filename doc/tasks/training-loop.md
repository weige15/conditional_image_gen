# Training Loop

## Goal

Implement the end-to-end training lifecycle for the conditional DDPM model.

## Inputs

- `doc/proposal.md`: training objective, AdamW, EMA, condition dropout, and full-model experiment path.
- `doc/detailed-design.md`: Training Loop responsibilities, dependencies, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/config.py` and `configs/default.yaml` with paths, model, diffusion, optimizer, EMA, sampling, and logging settings.
- [x] Create `src/brainrot_diffusion/train_loop.py` with `train(config)`.
- [x] Create `scripts/train.py` as a thin CLI around config loading and `train(config)`.
- [x] Wire dataset, mappings, UNet, diffusion, AdamW, condition dropout, EMA, checkpoint cadence, and optional resume.
- [x] Stop clearly on missing paths, invalid config, non-finite loss, or incompatible resume checkpoint.
- [x] Add a CPU smoke test using a tiny model, tiny dataset fixture, `max_steps=2`, checkpoint write, and resume.

## Done When

- [x] `python scripts/train.py --config configs/default.yaml` is the intended full-training entrypoint.
- [x] A tiny smoke training run writes a checkpoint with finite loss and EMA state.
- [x] Training-loop smoke tests pass.
