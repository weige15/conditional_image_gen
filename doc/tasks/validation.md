# Validation

## Goal

Implement strict structural validation for generated submissions before evaluation or packaging.

## Inputs

- `doc/proposal.md`: Final acceptance requires exactly 2,000 expected filenames, RGB PNG format, `64x64`, and no missing or extra outputs.
- `doc/high-level-design.md`: Validation owns `src/brainrot_diffusion/validate.py` and compares `generate.csv` with generated outputs.
- `doc/detailed-design.md`: Validation checks count, filename set, PNG format, RGB mode, image size, corrupt files, and JSON-compatible reports.
- `doc/test-plan.md`: Validation tests cover missing, extra, wrong-size, wrong-mode, corrupt, and valid output cases.

## Write Scope

`src/brainrot_diffusion/validate.py`, `scripts/validate_submission.py` integration, validation tests under `tests/`, and tiny generated-image fixtures.

## Read Scope

`dataset/generate.csv`, generated output directories, Data Loading CSV helpers, Pillow behavior, and `doc/test-plan.md` edge cases.

## Dependencies

Data Loading for CSV parsing or a shared CSV helper. Packaging and Evaluation depend on Validation.

## Tasks

- [x] Implement expected filename loading from generation CSV with duplicate-ID rejection.
- [x] Implement actual PNG discovery and exact missing/extra filename comparison.
- [x] Check every expected image opens, is PNG format, RGB mode, and `64x64`.
- [x] Return structured validation findings and optional JSON report data.
- [x] Add tests for valid fixture, missing file, extra file, wrong size, wrong mode, corrupt PNG, and report contents.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_validate.py`
- [ ] `python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json` once generated images exist.

## Done When

- [x] Validation passes only for exact CSV-matching RGB `64x64` PNG outputs.
- [x] Evaluation and Packaging can call Validation first.
- [x] Validation tests pass.
