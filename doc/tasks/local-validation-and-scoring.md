# Local Validation and Scoring

## Goal

Validate generated submission files and compute local FID when reference resources are available.

## Inputs

- `doc/proposal.md`: validation requirements and local FID using provided reference stats.
- `doc/detailed-design.md`: Local Validation and Scoring checks, optional proxy CLIP behavior, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/validate.py` with filename, count, PNG, RGB, and 64x64 checks.
- [x] Create `scripts/validate_submission.py` with CLI args for generation CSV, output dir, optional JSON report, and smoke mode.
- [x] Create `src/brainrot_diffusion/evaluate.py` and `scripts/evaluate.py` that run validation before metrics.
- [x] Implement local FID path compatible with `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`, and `scoring_program/score.py` behavior.
- [x] Treat missing metric dependencies or hidden CLIP-T metadata as skipped with explicit report fields.
- [x] Add tests for valid output, missing image, extra image, wrong size, non-RGB image, and skipped optional metric paths.

## Done When

- [x] Validation rejects malformed submissions before scoring or packaging.
- [x] Local FID can run when required reference files and dependencies are present.
- [x] Validation/scoring tests pass independently.
