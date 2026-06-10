# Vibe Coding Implementation Prompt

## Objective

Implement the HW6 Brainrot image generation project end to end as a reproducible from-scratch conditional DDPM/DDIM system in Python/PyTorch. The completed repository must train a conditional generator without pretrained generative weights and generate exactly 2,000 RGB PNG images at 64x64 resolution, one per row in `generate.csv`, into `generated_images/`.

The primary algorithm is a pixel-space conditional DDPM trained with epsilon-prediction MSE, a compact conditional UNet, learned animal/object/pair embeddings, classifier-free guidance, EMA sampling weights, and DDIM generation. Keep the StyleGAN2-ADA fallback isolated and secondary.

## Inputs

Read these files first:

- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/tasks/progress.md`
- All files under `doc/tasks/`
- `findings.md`
- `Brainrot_Image_Gen.pdf` and `HW6.pdf` if PDF reading is available

The current task breakdown is:

- `doc/tasks/configuration-and-reproducibility.md`
- `doc/tasks/data-pipeline.md`
- `doc/tasks/conditioning.md`
- `doc/tasks/diffusion-algorithm.md`
- `doc/tasks/model-architecture.md`
- `doc/tasks/training-workflow.md`
- `doc/tasks/checkpoints-and-ema.md`
- `doc/tasks/sampling-and-submission-generation.md`
- `doc/tasks/validation.md`
- `doc/tasks/evaluation-and-tuning.md`
- `doc/tasks/submission-packaging.md`
- `doc/tasks/fallback-experiment.md`

## Current Implementation

The repository is currently planning-only. Observed top-level files are assignment PDFs, research notes, `findings.md`, `research-log.md`, `research-state.yaml`, and planning docs under `doc/`. There is no `src/`, `tests/`, `pyproject.toml`, package metadata, dataset, training code, or existing implementation to preserve.

Use this greenfield layout unless repository inspection reveals a better local convention:

- `pyproject.toml` for package metadata, pytest, ruff, and build configuration.
- `requirements.txt` for assignment submission compatibility.
- `configs/default.yaml` for default training/generation settings.
- `src/brainrot_diffusion/` for importable modules.
- `scripts/train.py`, `scripts/generate.py`, `scripts/validate_submission.py`, `scripts/evaluate.py`, and `scripts/package_submission.py` for command entrypoints.
- `tests/` for unit and smoke tests with synthetic fixtures.
- `README.md` for environment setup, training, generation, validation, evaluation, and packaging commands.

Respect repository constraints: do not batch-delete files or directories. If cleanup is needed, delete only one explicit file path at a time, or stop and ask the user to delete manually.

## Execution Model

Work autonomously until the implementation, tests, and quality gates are complete. The main agent owns overall progress tracking, task decomposition, integration, final validation, and documentation. Spawn subagents for independent modules where useful, with disjoint write scopes and clear contracts. Subagents are working in the same repository and must not revert concurrent edits; they should adapt to changes made by other agents.

Do not use human-in-the-loop checkpoints for ordinary design choices. Make conservative choices consistent with the docs. Ask the user only when blocked by missing information that cannot be handled with configurable defaults.

## Module Plan

### Workstream A: Project Skeleton, Configuration, and Reproducibility

Write scope:

- `pyproject.toml`
- `requirements.txt`
- `configs/default.yaml`
- `src/brainrot_diffusion/config.py`
- `src/brainrot_diffusion/__init__.py`
- `tests/test_config.py`

Implement typed configuration for paths, seed, optimizer, diffusion schedule, UNet architecture, EMA, sampling, validation, metric settings, and fallback settings. Add config loading from YAML plus CLI overrides for required paths. Add path validation with exact missing paths. Add reproducible seed setup for Python, NumPy, PyTorch, and CUDA when available. Persist resolved config and seed metadata in checkpoint-serializable form.

### Workstream B: Data Pipeline and Conditioning

Write scope:

- `src/brainrot_diffusion/data.py`
- `src/brainrot_diffusion/conditioning.py`
- `tests/test_data.py`
- `tests/test_conditioning.py`

Implement `train.csv` parsing with required columns `id`, `animal`, and `object`. Resolve image ids under a configurable image directory, load as RGB, validate or resize to 64x64 according to config, normalize tensors to `[-1, 1]`, and provide optional conservative horizontal flip and mild color jitter behind config flags.

Implement stable vocabularies:

- animals: `shark`, `crocodile`, `frog`, `cat`, `dog`, `capybara`, `elephant`, `bird`, `fish`, `monkey`
- objects: `sneaker`, `airplane`, `coffee cup`, `banana`, `cactus`, `toilet`, `pizza`, `drum`, `car`, `chair`

Implement deterministic animal ids, object ids, pair ids for all 100 pairs, null/unconditional ids for classifier-free guidance, condition dropout, mapping serialization, and checkpoint-load validation.

### Workstream C: Diffusion Algorithm and Model Architecture

Write scope:

- `src/brainrot_diffusion/diffusion.py`
- `src/brainrot_diffusion/model.py`
- `tests/test_diffusion.py`
- `tests/test_model.py`

Implement DDPM schedule helpers with 1,000 training timesteps by default. Use cosine schedule as the primary schedule and retain linear schedule only as a debug fallback. Implement `q_sample(x_0, t, epsilon)`, coefficient extraction, predicted-epsilon to predicted-`x_0` conversion, DDIM timestep selection, and DDIM reverse update coefficients with configurable step count and eta.

Implement a compact conditional UNet for `[B, 3, 64, 64] -> [B, 3, 64, 64]`: 64 -> 32 -> 16 -> 8 down path, symmetric up path, GroupNorm, SiLU, residual blocks, timestep embeddings, learned animal/object/pair embeddings, one chosen conditioning injection method, and configurable attention at 16x16 and/or 8x8. Keep base channels configurable, with 96 or 128 as normal training candidates and a tiny test config for unit tests.

### Workstream D: Training, EMA, and Checkpoints

Write scope:

- `src/brainrot_diffusion/ema.py`
- `src/brainrot_diffusion/checkpoint.py`
- `src/brainrot_diffusion/training.py`
- `scripts/train.py`
- `tests/test_ema_checkpoint.py`
- `tests/test_training_smoke.py`

Implement a direct PyTorch training loop: load config, build dataloader, conditioning, diffusion schedule, model, AdamW optimizer, optional gradient accumulation, optional mixed precision, per-batch timestep/noise sampling, `q_sample`, epsilon prediction, MSE loss, finite-loss checks, optimizer steps, EMA updates, logs, periodic checkpoints, and periodic sample grids using EMA weights.

Checkpoints must include model state, EMA state, optimizer state for training resume, scheduler/scaler state when used, config, architecture metadata, diffusion metadata, condition mappings, seed, and progress counters. Sampling-only load must require EMA, model config, diffusion config, and mappings but not optimizer state.

### Workstream E: Sampling and Submission Validation

Write scope:

- `src/brainrot_diffusion/sampling.py`
- `src/brainrot_diffusion/validation.py`
- `scripts/generate.py`
- `scripts/validate_submission.py`
- `tests/test_sampling.py`
- `tests/test_validation.py`

Implement generation from `generate.csv` with required columns `id`, `animal`, `object`, and `prompt`. Reject duplicate ids. Load checkpoint metadata, model, EMA weights, diffusion schedule, and mappings. Generate in batches from Gaussian noise with DDIM and classifier-free guidance, preserving row order and filenames. Save every image as `generated_images/{id}` with uint8 RGB 64x64 PNG output.

Validation must compare `generated_images/` against `generate.csv`, check exactly 2,000 images in final mode, detect duplicate ids, missing files, extra PNG files, wrong extension, wrong format, wrong mode, and wrong size, emit human-readable and machine-readable reports, and exit nonzero on failure. Include a smoke mode for tests with fewer than 2,000 rows.

### Workstream F: Evaluation, Packaging, README, and Fallback Isolation

Write scope:

- `src/brainrot_diffusion/evaluation.py`
- `src/brainrot_diffusion/packaging.py`
- `src/brainrot_diffusion/fallback/`
- `scripts/evaluate.py`
- `scripts/package_submission.py`
- `README.md`
- `tests/test_evaluation.py`
- `tests/test_packaging.py`

Implement optional evaluation after structural validation. FID should accept configured `test_mu.npy` and `test_sigma.npy` paths and skip clearly when absent. CLIP-T proxy should pair generated images with `generate.csv` prompts in row order and skip clearly when optional CLIP dependencies or weights are unavailable. Reports must include dependency versions, checkpoint id, guidance scale, DDIM steps, and output directory.

Packaging must require validation first and verify required artifacts: `generated_images/`, scripts/source code, final checkpoint or `model.pth`, `README.md`, and dependency file. Optional zip creation must require configured student id and explicit overwrite policy.

Keep fallback StyleGAN2-ADA work isolated under a separate config namespace and output directory. Do not implement fallback ahead of the primary DDPM unless the primary path is complete and the trigger threshold is documented. Any fallback must train from scratch and reuse validation/evaluation contracts without changing DDPM checkpoint contracts.

## Testing and Quality Gates

Create and run the repository's actual quality gates before finishing. At minimum, add and pass:

- `python -m pytest`
- `python -m ruff check .`
- `python -m ruff format --check .`
- `python -m compileall src scripts tests`

If type checking is configured in `pyproject.toml`, also run the configured type-check command. If a build backend is configured, run the package build or import smoke check. Document every command in `README.md`.

Minimum test coverage:

- Config parsing, path validation, missing fields, and seed reproducibility.
- Data parsing/loading with synthetic CSV and PNG fixtures, RGB conversion, size policy, normalization, and missing file errors.
- Conditioning for all 100 pairs, unknown labels, null ids, dropout shapes, and mapping serialization.
- Diffusion schedule shapes, coefficient monotonicity, noising tensor shapes, timestep validation, and DDIM step generation.
- Model forward pass with fake tensors, output shape, null-condition ids, and attention-enabled tiny config.
- EMA math and checkpoint round-trip, including metadata failure cases and sampling-only load.
- One-step synthetic training smoke test with finite loss, parameter update, and checkpoint write.
- Sampling smoke test with a tiny or mocked model, duplicate-id failure, deterministic small generation, and PNG output properties.
- Validation tests for valid output, missing file, extra file, wrong size, wrong mode, duplicate id, and wrong extension.
- Evaluation tests for optional dependency skipping, mocked metric reports, and prompt/image order.
- Packaging tests for manifest checks, missing artifact failures, and refusal to package invalid images.

## Acceptance Criteria

The implementation is complete when:

- A Python/PyTorch project exists with importable modules under `src/brainrot_diffusion/`.
- Training can run at least one synthetic smoke step and write a reloadable checkpoint.
- The primary DDPM/DDIM path uses no pretrained generative weights, high-level Diffusers pipelines, Stable Diffusion, SDXL, FLUX, ControlNet, IP-Adapter, LoRA, DreamBooth, or external generated images as the submitted generator.
- Checkpoints contain model, EMA, config, seed, progress, architecture, diffusion, and mapping metadata.
- Generation loads EMA by default and writes exactly one RGB 64x64 PNG per `generate.csv` row with filenames preserved exactly.
- Validation can prove the final `generated_images/` contract: exactly 2,000 PNG files, no missing or extra PNG files, RGB mode, 64x64 resolution, and ids matching `generate.csv`.
- Optional FID and CLIP-T proxy evaluation skip cleanly when files or dependencies are absent.
- Packaging refuses invalid artifacts and documents the required HW6 submission structure.
- `README.md` contains setup, training, generation, validation, evaluation, packaging, and quality-gate commands.
- All configured tests, lint, format, compile, type-check, build, or static-analysis gates pass.

## Uncertainty Protocol

Known unresolved inputs are dataset root, training image folder, `train.csv`, `generate.csv`, optional `test_mu.npy`, optional `test_sigma.npy`, GPU/time budget, overwrite policy for existing outputs, final student id for zip naming, and whether optional CLIP conditioning should ever be attempted.

Handle these by making them configurable. Use safe defaults that do not overwrite existing artifacts unless an explicit overwrite flag is set. Add smoke-test modes that use synthetic fixtures and do not require the full dataset. Do not block implementation on missing dataset files.

If a decision cannot be represented safely as configuration and blocks implementation, ask one concise question before coding that part. Otherwise proceed with the conservative choice documented in the design.
