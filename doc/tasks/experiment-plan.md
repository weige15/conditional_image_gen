# Experiment Plan

## Goal

Track experiment runs, configs, generated outputs, metrics, and final checkpoint selection.

## Inputs

- `doc/proposal.md`: experiment order for smoke run, full baseline, guidance sweep, DDIM-step sweep, and optional ablations.
- `doc/detailed-design.md`: Experiment Plan module responsibilities and tests.

## Tasks

- [x] Define run-directory layout for config snapshot, logs, checkpoints, generated samples, validation reports, and metric reports.
- [x] Add helper code or script support for writing resolved config, seed, checkpoint path, generation command, and reports per run.
- [x] Add result-summary logic that reads validation and metric reports across runs.
- [x] Exclude failed or invalid runs from final checkpoint selection.
- [x] Add tests for fake run metadata writing, multi-report summary reading, and invalid-report exclusion.

## Done When

- [x] Each experiment run records enough metadata to reproduce training and generation commands.
- [x] Final selection can identify a valid run with available metrics.
- [x] Experiment-plan tests pass independently.
