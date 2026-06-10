# Configuration and Reproducibility

## Goal

Create the minimal configuration and seed-management layer needed by every training, sampling, validation, and evaluation command.

## Inputs

- `doc/proposal.md`: greenfield PyTorch project, reproducibility-first implementation, saved config and seeds.
- `doc/detailed-design.md`: configuration owns paths, hyperparameters, schedule settings, architecture settings, seeds, and serialized runtime metadata.

## Tasks

- [ ] Create the project configuration module and define typed fields for dataset paths, output paths, seed, optimizer, diffusion, UNet, EMA, sampling, validation, and metric settings.
- [ ] Add config loading from a file and CLI overrides for required workflow paths.
- [ ] Add path validation that reports missing required files or directories with exact paths.
- [ ] Add seed setup for Python, NumPy, and PyTorch, including CUDA seed setup when available.
- [ ] Persist the resolved config and seed metadata in a serializable form for checkpoints and generation manifests.
- [ ] Add tests for config parsing, missing required fields, path validation, and repeatable seed setup.

## Done When

- [ ] Training and generation code can consume one resolved config object instead of hard-coded settings.
- [ ] Config and seed tests pass independently of the dataset and model.
