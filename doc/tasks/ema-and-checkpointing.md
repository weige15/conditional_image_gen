# EMA and Checkpointing

## Goal

Persist training and generation state, including EMA weights and metadata required for reproducibility.

## Inputs

- `doc/proposal.md`: EMA sampling and checkpoint metadata requirements.
- `doc/detailed-design.md`: EMA and Checkpointing schema, failure handling, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/ema.py` with shadow-parameter update and apply/copy helpers.
- [x] Create `src/brainrot_diffusion/checkpoint.py` with save, load, and `model.pth` export helpers.
- [x] Store model, EMA, optimizer, step, epoch, config, condition mappings, diffusion metadata, architecture metadata, and seed metadata.
- [x] Use atomic checkpoint writes through a temporary path and final replace.
- [x] Reject missing checkpoint keys and incompatible condition mappings with clear errors.
- [x] Add tests for EMA update, checkpoint round-trip, missing-key rejection, and mapping round-trip.

## Done When

- [x] A tiny model checkpoint can be saved, loaded, and exported.
- [x] Generation can load EMA weights and saved condition mappings.
- [x] EMA/checkpoint tests pass independently.
