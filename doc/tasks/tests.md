# Tests

## Goal

Create the small CPU test suite and fixtures that verify module contracts, integration paths, and regression cases from the test plan.

## Inputs

- `doc/proposal.md`: Correctness strategy requires small CPU tests for parsing, mappings, shapes, checkpoints, filenames, PNG mode, and image size.
- `doc/high-level-design.md`: Tests own `tests/`, tiny CSVs, tiny images, temporary output directories, and pytest checks.
- `doc/detailed-design.md`: Tests are layered unit, integration, and CLI smoke checks without GPU/scorer/model-download requirements.
- `doc/test-plan.md`: Provides module-specific unit tests, golden fixtures, randomized/property tests, edge cases, quality gates, and manual verification.

## Write Scope

`tests/`, tiny fixtures generated inside tests or committed fixture files if needed, and small test helpers.

## Read Scope

All package modules under `src/brainrot_diffusion/`, scripts under `scripts/`, `doc/test-plan.md`, and `doc/quality-gates.md`.

## Dependencies

All implementation modules. Tests can be added incrementally with each module.

## Tasks

- [x] Create shared tiny CSV/image/checkpoint fixtures that do not require full assignment data.
- [x] Add module unit tests for config, data, conditions, model, diffusion, EMA, checkpointing, validation, evaluation, packaging, and scripts.
- [x] Add a tiny integration path for train -> checkpoint -> generate -> validate when core modules exist.
- [x] Add property/randomized checks for mapping stability and exact filename-set validation with deterministic seeds.
- [x] Keep full GPU training, official scoring, and subjective visual review out of automated tests.
- [x] Document manual final checks for generated-image grids, assignment compliance, README reproduction, and final package layout.

## Tests and Quality Gates

- [x] `python -m pytest`
- [x] `python -m compileall src scripts tests`

## Done When

- [x] `python -m pytest` covers every current detailed-design module.
- [x] Tests remain CPU-friendly and do not require generated final outputs.
- [x] Full-project quality gates are ready to run after implementation exists.
