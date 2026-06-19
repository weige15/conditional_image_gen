# Configuration

## Goal

Implement config loading and validation so every workflow receives one reproducible runtime configuration before allocating models, loading data, or writing artifacts.

## Inputs

- `doc/proposal.md`: Create `configs/default.yaml` and use config for train/generate/evaluate/package commands.
- `doc/high-level-design.md`: Configuration owns `src/brainrot_diffusion/config.py`, YAML loading, defaults, and validated runtime settings.
- `doc/detailed-design.md`: Configuration public interface must load config, apply overrides, validate sections, and serialize into checkpoints.
- `doc/test-plan.md`: Config tests must load defaults, reject missing keys and invalid values, and verify deterministic overrides.

## Write Scope

`src/brainrot_diffusion/config.py`, `configs/default.yaml`, config-related tests under `tests/`, and script imports needed to consume resolved config.

## Read Scope

`pyproject.toml`, `requirements.txt`, `README.md`, `doc/detailed-design.md`, and any existing config/script files if created earlier.

## Dependencies

None for the first implementation. Downstream modules depend on this task.

## Tasks

- [x] Create `src/brainrot_diffusion/config.py` with config loading, override application, and validation.
- [x] Create `configs/default.yaml` with data, model, diffusion, training, sampling, checkpoint, validation, evaluation, and packaging sections.
- [x] Validate required paths, image size `64`, positive timesteps, positive batch/step values, and supported sampler names.
- [x] Keep resolved config JSON/YAML-serializable for checkpoint metadata.
- [x] Add tiny config fixtures and tests for valid config, missing keys, invalid values, and deterministic overrides.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_config.py`
- [x] `python -m compileall src scripts tests`

## Done When

- [x] Config loading returns one resolved mapping/object usable by scripts and package modules.
- [x] Invalid config fails before expensive work starts.
- [x] Config tests pass.
