# Evaluation

## Goal

Implement validation-first local evaluation reporting with FID/evaluator hooks that skip explicitly when resources or dependencies are unavailable.

## Inputs

- `doc/proposal.md`: Local FID/evaluator integration is useful after generated images exist; CLIP proxy checks remain optional.
- `doc/high-level-design.md`: Evaluation owns `src/brainrot_diffusion/evaluate.py` and writes reports without becoming part of the main generator.
- `doc/detailed-design.md`: Evaluation calls Validation first, computes available metrics, and records skipped metric reasons.
- `doc/test-plan.md`: Evaluation tests cover validation-first behavior, missing reference stats, skipped CLIP-T proxy, and lightweight FID plumbing.

## Write Scope

`src/brainrot_diffusion/evaluate.py`, `scripts/evaluate.py`, `scripts/prepare_score_input.py` if needed, evaluation tests under `tests/`, and report fixture tests.

## Read Scope

Validation module, `hw6_reference/`, `scoring_program/score.py`, `requirements.txt`, and `doc/quality-gates.md`.

## Dependencies

Validation. Optional NumPy/SciPy/PyTorch/torchvision metric path. Optional CLIP proxy only if explicitly approved later.

## Tasks

- [x] Implement evaluation entry point that runs Validation first and stops on invalid outputs.
- [x] Implement report structure with validation result, metrics, and explicit skipped metric reasons.
- [x] Implement local FID hook or scorer-input preparation only when assets/dependencies are available.
- [x] Keep CLIP-T proxy skipped unless explicitly enabled and documented.
- [x] Add tests for invalid-output validation stop, missing reference-stat skip, CLIP proxy skip, and mocked/lightweight FID report plumbing.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_evaluate.py`
- [ ] `python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json` once generated images exist.

## Done When

- [x] Evaluation never scores invalid generated outputs.
- [x] Missing optional resources are reported as skipped, not silent success.
- [x] Evaluation tests pass.
