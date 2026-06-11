# Generation Script

## Goal

Implement the assignment inference command that reads `dataset/generate.csv` and writes exact PNG outputs.

## Inputs

- `doc/proposal.md`: generation must produce exactly 2,000 RGB 64x64 PNGs with requested filenames.
- `doc/detailed-design.md`: Generation Script lifecycle, overwrite behavior, and tests.

## Tasks

- [x] Create `scripts/generate.py` as a thin CLI.
- [x] Load config, checkpoint, EMA weights by default, saved condition mappings, model, and diffusion scheduler.
- [x] Read generation requests from `dataset/generate.csv` and batch them for sampling.
- [x] Save each sampled image to `generated_images/{id}` as RGB 64x64 PNG.
- [x] Fail before sampling if output files exist without `--overwrite`, labels are unknown, or required paths are missing.
- [x] Add a tiny fixture test that generates two PNGs and checks filenames, RGB mode, size, and overwrite failure.

## Done When

- [x] `python scripts/generate.py --checkpoint <ckpt> --config configs/default.yaml --overwrite` writes CSV-matching PNG files.
- [x] Generation uses checkpoint mappings rather than rebuilding mappings from `generate.csv`.
- [x] Generation-script tests pass.
