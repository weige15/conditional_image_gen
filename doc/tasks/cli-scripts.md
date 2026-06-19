# CLI Scripts

## Goal

Create thin command-line wrappers that expose the package workflows without duplicating core logic.

## Inputs

- `doc/proposal.md`: Scripts are required for train, generate, validate, evaluate, scorer input preparation, and packaging.
- `doc/high-level-design.md`: CLI Scripts own `scripts/*.py`, parse arguments, delegate to modules, and return nonzero on invalid input.
- `doc/detailed-design.md`: Known README/quality-gate commands should remain aligned with implemented script arguments.
- `doc/test-plan.md`: CLI tests cover argument parsing, delegation, invalid exits, and README command alignment.

## Write Scope

`scripts/train.py`, `scripts/generate.py`, `scripts/validate_submission.py`, `scripts/evaluate.py`, `scripts/prepare_score_input.py`, `scripts/package_submission.py`, script tests under `tests/`, and README command updates if implementation flags change.

## Read Scope

Package module public interfaces, `README.md`, `doc/quality-gates.md`, and `doc/detailed-design.md`.

## Dependencies

Configuration, Training Loop, Sampling, Validation, Evaluation, Packaging, and Checkpointing as each script is implemented.

## Tasks

- [x] Implement `argparse` wrappers for train, generate, validate, evaluate, scorer input preparation, and packaging.
- [x] Keep script logic limited to argument parsing, config loading, package calls, concise status, and exit code handling.
- [x] Preserve known README command shapes where practical.
- [x] Add nonzero exits for invalid inputs and failed validation.
- [x] Add parser or script-level smoke tests using tiny fixtures after package modules exist.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_scripts.py`
- [x] README documented commands match implemented script arguments.

## Done When

- [x] Scripts delegate to package modules and do not duplicate core logic.
- [x] Known commands from docs are runnable after implementation exists.
- [x] CLI script tests pass.
