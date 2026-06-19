# High-Level Design

## Overview

This project is a scratch Python/PyTorch implementation for HW6 Brainrot Image Generation. The system trains a from-scratch conditional image generator on the provided Brainrot Dataset and generates exactly 2,000 RGB `64x64` PNG images from `dataset/generate.csv`.

The architecture is a small package under `src/brainrot_diffusion/` with thin CLI scripts under `scripts/`. The core path is:

1. Load configuration, training CSV, training images, and condition mappings.
2. Train a from-scratch conditional diffusion generator.
3. Save checkpoints with reproducibility metadata.
4. Load a checkpoint and generate CSV-matching images.
5. Validate output structure and optionally evaluate local FID.
6. Package the required E3 submission layout.

Source traceability: `doc/proposal.md` Objective, Proposed Approach, Algorithm Strategy, Module Candidates; `doc/problem-brief.md` Required Outputs and Constraints; `doc/repo-map.md` Current Project State; `doc/quality-gates.md` Recommended Minimum Done Criteria.

## Goals

- Produce a reproducible from-scratch conditional image generator.
- Generate exactly 2,000 PNG files named according to `dataset/generate.csv`.
- Keep all final images RGB and `64x64`.
- Avoid pretrained generative weights and high-level generative pipelines.
- Preserve stable condition mappings between training and generation.
- Provide validation, local evaluation hooks, and final packaging.
- Align implementation with the discovered quality gates once source files exist.

## Non-Goals

- Do not implement pretrained Stable Diffusion, Diffusers pipelines, pretrained GANs, or pretrained generative Transformers.
- Do not make pretrained CLIP or VAE part of final-image generation.
- Do not use extra data in the first implementation path.
- Do not optimize for the final leaderboard before a valid reproducible baseline exists.
- Do not treat stale planning docs as proof that implementation files exist.

## Requirements Summary

- Inputs:
  - `dataset/train.csv` with `id,animal,object`.
  - `dataset/trainset/*.png`.
  - `dataset/generate.csv` with `id,animal,object,prompt`.
  - Optional reference assets under `hw6_reference/` for local FID.
- Outputs:
  - `generated_images/{id}` for every row in `dataset/generate.csv`.
  - Checkpoints with model, config, mappings, and reproducibility metadata.
  - Validation and evaluation reports when requested.
  - Final `HW6_{student_id}.zip` package.
- Constraints:
  - Main generator trained from scratch.
  - No pretrained generative weights or high-level generation/training flows.
  - Pretrained CLIP/VAE only if explicitly used as auxiliary support and documented.
  - TA reproduction must be possible from submitted artifacts.
- Acceptance:
  - Output count, filenames, RGB mode, PNG format, and `64x64` size pass validation.
  - Core quality gates pass after implementation exists.

## Proposed Architecture

The architecture separates reusable package logic from command-line orchestration:

- `src/brainrot_diffusion/` owns configuration, data, model, training, sampling, validation, evaluation, and packaging logic.
- `scripts/` owns thin CLI entry points that parse arguments and call package modules.
- `configs/` owns reproducible runtime settings.
- `tests/` owns small CPU fixtures and acceptance checks.
- Assignment assets remain read-only inputs unless explicitly approved.

Primary lifecycle:

```text
dataset/train.csv + dataset/trainset/
  -> data + conditions
  -> train_loop(config, model, diffusion, checkpoint, ema)
  -> checkpoints/*.pt
  -> sample(checkpoint, dataset/generate.csv)
  -> generated_images/
  -> validate/evaluate
  -> package
```

## Modules

### Configuration

- Module: `src/brainrot_diffusion/config.py`
- Responsibility: load and validate project configuration.
- Inputs: YAML config path, CLI overrides.
- Outputs: resolved config object or mapping.
- Owned data: defaults and validated runtime settings.
- Dependencies: PyYAML, standard library paths/types.
- Externally visible behavior: scripts can load one config and receive consistent paths and hyperparameters.
- Source traceability: `doc/proposal.md` Module Candidates; `doc/quality-gates.md` command examples use `configs/default.yaml`.

### Data Loading

- Module: `src/brainrot_diffusion/data.py`
- Responsibility: read training rows, generation requests, and RGB images.
- Inputs: `dataset/train.csv`, `dataset/generate.csv`, `dataset/trainset/`.
- Outputs: training samples and generation requests.
- Owned data: parsed rows and image tensors for current process.
- Dependencies: `conditions.py`, Pillow, PyTorch dataset utilities.
- Externally visible behavior: train/generate flows receive validated samples with labels and IDs.
- Source traceability: `doc/problem-brief.md` Required Inputs; `doc/proposal.md` Proposed Approach.

### Conditions

- Module: `src/brainrot_diffusion/conditions.py`
- Responsibility: create and apply stable animal, object, and pair mappings.
- Inputs: observed labels from training CSV and saved checkpoint mappings.
- Outputs: integer condition IDs and serializable mapping metadata.
- Owned data: vocabulary/mapping tables.
- Dependencies: standard library collections/JSON-compatible structures.
- Externally visible behavior: generation uses checkpoint-compatible mappings rather than rebuilding incompatible mappings.
- Source traceability: `doc/proposal.md` Algorithm Strategy and Module Candidates.

### Generator Model

- Module: `src/brainrot_diffusion/model.py`
- Responsibility: define the from-scratch conditional image generator backbone.
- Inputs: noisy image tensors, timesteps, condition IDs.
- Outputs: model predictions used by the diffusion objective and sampler.
- Owned data: model parameters initialized from scratch.
- Dependencies: PyTorch.
- Externally visible behavior: train and sample flows can construct the model from config and load compatible checkpoints.
- Source traceability: `doc/proposal.md` Algorithm Strategy; `doc/problem-brief.md` Constraints.

### Diffusion

- Module: `src/brainrot_diffusion/diffusion.py`
- Responsibility: own noise schedule, training objective, and sampler math.
- Inputs: clean/noisy tensors, timesteps, model predictions, sampler settings.
- Outputs: training losses and denoised samples.
- Owned data: schedule tensors and diffusion metadata.
- Dependencies: PyTorch, generator model contract.
- Externally visible behavior: training can compute a loss; generation can run a supported sampler.
- Source traceability: `doc/proposal.md` Algorithm Strategy.

### EMA

- Module: `src/brainrot_diffusion/ema.py`
- Responsibility: maintain exponential moving average weights for sampling.
- Inputs: current model parameters and EMA settings.
- Outputs: EMA state and model-compatible averaged weights.
- Owned data: EMA parameter state.
- Dependencies: PyTorch model parameters.
- Externally visible behavior: checkpoint and sampling modules can use EMA weights when available.
- Source traceability: `doc/proposal.md` Algorithm Strategy and Module Candidates.

### Checkpointing

- Module: `src/brainrot_diffusion/checkpoint.py`
- Responsibility: persist and restore training/generation state.
- Inputs: model state, optional EMA state, optimizer state, config, condition mappings, counters, metadata.
- Outputs: checkpoint files and loaded checkpoint objects.
- Owned data: checkpoint schema.
- Dependencies: PyTorch serialization, config and conditions modules.
- Externally visible behavior: generation can reproduce outputs from a saved model and mappings.
- Source traceability: `doc/proposal.md` Proposed Approach and Validation Plan.

### Training Loop

- Module: `src/brainrot_diffusion/train_loop.py`
- Responsibility: orchestrate training lifecycle.
- Inputs: resolved config, training dataset, condition mappings.
- Outputs: checkpoints, logs, optional sample previews.
- Owned data: training run state during execution.
- Dependencies: config, data, conditions, model, diffusion, EMA, checkpointing.
- Externally visible behavior: `scripts/train.py` can start or resume training.
- Source traceability: `doc/proposal.md` Milestones and Algorithm Strategy.

### Sampling

- Module: `src/brainrot_diffusion/sample.py`
- Responsibility: generate images from checkpoint and generation requests.
- Inputs: checkpoint, resolved config, generation CSV requests, sampler settings.
- Outputs: generated image tensors or image files.
- Owned data: generation run metadata and seeds.
- Dependencies: config, data, conditions, model, diffusion, checkpointing.
- Externally visible behavior: `scripts/generate.py` writes `generated_images/{id}`.
- Source traceability: `doc/problem-brief.md` Required Outputs; `doc/proposal.md` Proposed Approach.

### Validation

- Module: `src/brainrot_diffusion/validate.py`
- Responsibility: verify generated submission structure.
- Inputs: `dataset/generate.csv`, output directory.
- Outputs: validation result and optional report data.
- Owned data: validation findings for current run.
- Dependencies: standard library file traversal, Pillow.
- Externally visible behavior: scripts and package flow can fail fast on invalid outputs.
- Source traceability: `doc/problem-brief.md` Required Outputs and Evaluation; `doc/quality-gates.md` Integration Test Commands.

### Evaluation

- Module: `src/brainrot_diffusion/evaluate.py`
- Responsibility: run validation and local metric hooks when resources are available.
- Inputs: generated images, generate CSV, `hw6_reference/`, scorer-compatible resources.
- Outputs: evaluation report with scores or skipped metrics.
- Owned data: report payload for current run.
- Dependencies: validation module; optional NumPy/SciPy/PyTorch/torchvision scorer path.
- Externally visible behavior: `scripts/evaluate.py` writes a report without becoming part of the main generator.
- Source traceability: `doc/problem-brief.md` Evaluation; `doc/quality-gates.md` Benchmark or Evaluator Commands.

### Packaging

- Module: `src/brainrot_diffusion/package.py`
- Responsibility: assemble the final E3 submission archive after validation.
- Inputs: generated images, scripts, model checkpoint, README, requirements, student ID.
- Outputs: `HW6_{student_id}.zip`.
- Owned data: package manifest for current archive.
- Dependencies: validation module, standard library zip/path utilities.
- Externally visible behavior: package command refuses invalid generated outputs.
- Source traceability: `doc/problem-brief.md` Required Deliverables; `doc/proposal.md` Module Candidates.

### CLI Scripts

- Module group: `scripts/*.py`
- Responsibility: expose command-line entry points as thin wrappers.
- Inputs: CLI arguments and paths.
- Outputs: process exit status, reports, checkpoints, images, or packages.
- Owned data: none beyond parsed arguments.
- Dependencies: package modules.
- Externally visible behavior: documented commands in README and quality gates become runnable.
- Source traceability: `doc/proposal.md` Module Candidates; `doc/quality-gates.md` Commands.

### Tests

- Module group: `tests/`
- Responsibility: verify package behavior with small CPU fixtures.
- Inputs: tiny CSVs, tiny images, temporary output directories.
- Outputs: pytest pass/fail results.
- Owned data: test fixtures.
- Dependencies: package modules, pytest, Pillow, PyTorch.
- Externally visible behavior: `python -m pytest` validates core behavior.
- Source traceability: `doc/proposal.md` Validation Plan; `doc/quality-gates.md` Unit Test Commands.

## Module Relationships

| Type | Source | Target | Direction and Contract | Status |
| --- | --- | --- | --- | --- |
| Configuration dependency | CLI scripts | `config.py` | Scripts load config before constructing workflows. | Confirmed by proposal |
| Data flow | `data.py` | `conditions.py` | Training labels produce stable mappings; generation labels are mapped through saved mappings. | Confirmed by proposal |
| Call | `train_loop.py` | `data.py` | Training obtains image tensors and condition IDs. | Confirmed by proposal |
| Call | `train_loop.py` | `model.py` + `diffusion.py` | Training computes diffusion loss through the model. | Confirmed by proposal |
| Persistence dependency | `train_loop.py` | `checkpoint.py` | Training saves model/config/mapping metadata. | Confirmed by proposal |
| Persistence dependency | `checkpoint.py` | `ema.py` | Checkpoints include EMA state when enabled. | Confirmed by proposal |
| Call | `sample.py` | `checkpoint.py` | Generation loads model weights and mappings from checkpoint. | Confirmed by proposal |
| Data flow | `sample.py` | `generated_images/` | Generation writes one PNG per generation row. | Confirmed by problem brief |
| Evaluator/test dependency | `validate.py` | `generated_images/` + `generate.csv` | Validation compares output files against requested rows. | Confirmed by problem brief |
| Call | `evaluate.py` | `validate.py` | Evaluation runs structural validation before metrics. | Confirmed by proposal |
| Evaluator dependency | `evaluate.py` | `hw6_reference/` + `scoring_program/` | Local FID/evaluator hooks use provided reference assets when available. | Confirmed by supporting docs |
| Call | `package.py` | `validate.py` | Packaging validates generated images before writing zip. | Confirmed by proposal |
| Test dependency | `tests/` | package modules | Tests exercise data, model, checkpoint, sampling, validation, and packaging behavior. | Confirmed by proposal |

## Data Flow

Training flow:

```text
config + train.csv + trainset images
  -> condition mappings
  -> training dataset
  -> from-scratch model + diffusion objective
  -> optimizer/EMA updates
  -> checkpoint with metadata
```

Generation flow:

```text
checkpoint + config + generate.csv
  -> load model and saved condition mappings
  -> sample conditioned images
  -> write generated_images/{id}
```

Validation and packaging flow:

```text
generate.csv + generated_images/
  -> structural validation
  -> optional local evaluation report
  -> final package with generated_images/, scripts/, model.pth, README.md, requirements.txt
```

## Interfaces and Contracts

- Config contract:
  - A single resolved config drives model, diffusion, training, sampling, and paths.
  - Config values used for checkpoint compatibility must be saved with the checkpoint.
- Dataset contract:
  - Training CSV columns: `id`, `animal`, `object`.
  - Generation CSV columns: `id`, `animal`, `object`, `prompt`.
  - Training images are RGB image files referenced by training IDs.
- Condition contract:
  - Animal/object/pair mappings must be stable and serializable.
  - Generation must use mappings compatible with the checkpoint.
- Model contract:
  - The generator is initialized from scratch.
  - Inputs and outputs are tensors compatible with diffusion training and sampling.
- Checkpoint contract:
  - Checkpoints include enough metadata for reproducible generation.
  - Generation fails if checkpoint mappings and requested labels are incompatible.
- Generated image contract:
  - Exactly one RGB `64x64` PNG for each row in `dataset/generate.csv`.
  - No extra or missing filenames.
- CLI contract:
  - Scripts are thin wrappers and delegate behavior to package modules.
  - Commands should return nonzero on invalid input or failed validation.
- Package contract:
  - The archive matches the assignment-required layout and includes reproducibility instructions.

## Operational Considerations

- Training may require GPU time; smoke tests should run before full training.
- The scorer path may require CUDA and pretrained evaluation weights; it should remain separate from core unit tests.
- Generated artifacts such as checkpoints, reports, and images should not be overwritten without explicit user approval.
- The first implementation path should use only provided Brainrot data unless the user approves extra data.
- Auxiliary pretrained CLIP or VAE usage must be documented and kept out of final image generation.
- The live repository is missing implementation directories, so initial work must create package, scripts, configs, and tests before gates can run.

## Testing and Quality Gate Alignment

Quality gates from `doc/quality-gates.md` become meaningful after implementation files exist:

```bash
python -m compileall src scripts tests
python -m pytest
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json
```

Test coverage should align with architecture boundaries:

- Data tests: CSV parsing, missing files, duplicate IDs, label mapping stability.
- Model/diffusion tests: tensor shape compatibility and finite loss on tiny fixtures.
- Checkpoint tests: save/load metadata and mapping compatibility.
- Sampling tests: deterministic tiny generation and output image contract.
- Validation tests: missing, extra, wrong mode, wrong size, and valid output cases.
- Packaging tests: refuses invalid outputs and includes required files for valid inputs.

Evaluator-related gates should run only after generated images exist:

```bash
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json
```

## Risks and Tradeoffs

- Diffusion quality depends on GPU time and tuning. The design prioritizes a valid reproducible baseline before score tuning.
- FID and CLIP-T may prefer different sampling/guidance settings. Treat guidance scale and sampling steps as experiment knobs.
- Local CLIP-T may not match Codabench because hidden metadata is unavailable. Use local CLIP checks only as proxy checks if added.
- The package structure is currently absent. The HLD assumes the proposal's `src` and `scripts` layout will be created.
- Existing planning docs may be stale. Future implementation should trust live files and update docs when behavior changes.

## Assumptions

- Python/PyTorch remains the implementation stack.
- The first implementation trains only on the provided Brainrot Dataset.
- The default architecture is pixel-space conditional diffusion.
- `model.pth` in the final package can be produced from a selected checkpoint or exported from one.
- The final student ID will be provided before packaging.
- Remaining architecture details can be refined in detailed design without changing the module boundaries above.

## Open Questions

- What GPU and training-time budget are available for full training?
- What student ID should be used for the final `HW6_{student_id}.zip`?
- Should local CLIP proxy evaluation be implemented or left to Codabench only?
- Should any auxiliary pretrained VAE/CLIP path be included after the baseline, or avoided entirely?
- What is the exact E3 deadline?
