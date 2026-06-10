# Sampling And Submission Generation

## Goal

Generate final submission images from `generate.csv` using EMA weights, DDIM sampling, and classifier-free guidance.

## Inputs

- `doc/proposal.md`: load EMA by default, read `generate.csv`, generate one RGB 64x64 PNG per row, support guidance scale sweeps.
- `doc/detailed-design.md`: sampler owns checkpoint loading, row-to-filename preservation, DDIM reverse steps, CFG combination, PNG writing, and generation manifests.

## Tasks

- [ ] Implement the generation entrypoint that loads config, checkpoint metadata, model, EMA weights, diffusion schedule, and condition mappings.
- [ ] Parse `generate.csv` with required columns `id`, `animal`, `object`, and `prompt`, rejecting duplicate ids.
- [ ] Implement batched DDIM sampling from Gaussian noise while preserving row order and filename mapping.
- [ ] Implement classifier-free guidance by combining conditional and unconditional noise predictions.
- [ ] Convert final tensors to uint8 RGB 64x64 PNGs and save each file as `generated_images/{id}`.
- [ ] Add tests with a tiny mocked model for CSV parsing, duplicate-id failure, deterministic small generation, and PNG output properties.

## Done When

- [ ] A generation command can create one PNG per input CSV row using a checkpoint.
- [ ] Sampling tests pass without needing a trained full-size model.
