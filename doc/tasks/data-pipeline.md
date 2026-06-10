# Data Pipeline

## Goal

Implement dataset loading for Brainrot training records and images, returning normalized 64x64 RGB tensors with condition labels.

## Inputs

- `doc/proposal.md`: read `train.csv`, load RGB images, resize or validate 64x64 images, apply conservative augmentation.
- `doc/detailed-design.md`: data module owns CSV/image loading, image validation, transforms, diagnostics, and dataset fixtures.

## Tasks

- [ ] Implement structured parsing for `train.csv` with required columns `id`, `animal`, and `object`.
- [ ] Resolve image ids under the configured training image directory and load each image as RGB.
- [ ] Implement the configured 64x64 policy: validate size or resize according to config.
- [ ] Normalize image tensors to the diffusion training range and return labels needed by the Conditioning module.
- [ ] Add optional conservative transforms for horizontal flip and mild color jitter behind config flags.
- [ ] Add tests using tiny synthetic CSV/image fixtures for parsing, image loading, RGB conversion, size handling, normalization, and missing-file errors.

## Done When

- [ ] A dataloader can produce batches of `[B, 3, 64, 64]` image tensors plus animal/object labels.
- [ ] Data pipeline tests pass without requiring the full Codabench dataset.
