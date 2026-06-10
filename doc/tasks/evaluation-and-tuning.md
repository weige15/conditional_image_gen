# Evaluation And Tuning

## Goal

Add optional local metric and diagnostic utilities for FID, CLIP-T proxy, and tuning comparisons after structural validation passes.

## Inputs

- `doc/proposal.md`: local FID against `test_mu.npy` and `test_sigma.npy` if available, CLIP-T proxy with OpenAI CLIP ViT-B-32-quickgelu, per-condition grids, tuning sweeps.
- `doc/detailed-design.md`: evaluation must skip optional metrics gracefully when files or dependencies are missing and must not replace Codabench scoring.

## Tasks

- [ ] Implement evaluation entrypoint that runs structural validation before metrics.
- [ ] Add FID computation path that accepts configured `test_mu.npy` and `test_sigma.npy` paths.
- [ ] Add CLIP-T proxy path that pairs generated images with `generate.csv` prompts in row order.
- [ ] Emit metric reports that include dependency versions, checkpoint id, guidance scale, DDIM steps, and output directory.
- [ ] Add per-condition grid or manifest diagnostics for tuning condition coverage and mode collapse.
- [ ] Add tests for missing optional metric files, mocked metric reports, and prompt/image pairing order.

## Done When

- [ ] Evaluation can produce a report when optional files are present and skip clearly when they are absent.
- [ ] Evaluation tests pass without requiring downloaded CLIP weights or Codabench.
