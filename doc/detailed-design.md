# Detailed Design

## Purpose

This document turns `doc/proposal.md`, `doc/high-level-design.md`, and `doc/test-plan.md` into an implementation-ready design for HW6 Brainrot Image Generation. It preserves the HLD module boundaries and describes responsibilities, contracts, algorithms, failure handling, and independently testable work packages without writing production code.

The design target is a reproducible Python/PyTorch project that trains a from-scratch conditional image generator and produces exactly 2,000 RGB `64x64` PNG files named according to `dataset/generate.csv`.

## Source Proposal Summary

The proposal selects a pixel-space conditional diffusion baseline implemented directly in PyTorch. The main generator must be trained from scratch and must not use pretrained generative weights, Diffusers pipelines, or high-level generation/training flows. The first implementation is scoped to the provided Brainrot Dataset, with optional pretrained CLIP or VAE use deferred until explicitly approved.

The proposed package layout is `src/brainrot_diffusion/` plus thin scripts under `scripts/`. The implementation must cover data loading, stable condition mappings, a from-scratch conditional generator, diffusion training and sampling, checkpoint metadata, structural validation, optional local FID/evaluator hooks, and packaging.

## HLD Summary

The HLD defines these modules and module groups:

- Configuration: `src/brainrot_diffusion/config.py`
- Data Loading: `src/brainrot_diffusion/data.py`
- Conditions: `src/brainrot_diffusion/conditions.py`
- Generator Model: `src/brainrot_diffusion/model.py`
- Diffusion: `src/brainrot_diffusion/diffusion.py`
- EMA: `src/brainrot_diffusion/ema.py`
- Checkpointing: `src/brainrot_diffusion/checkpoint.py`
- Training Loop: `src/brainrot_diffusion/train_loop.py`
- Sampling: `src/brainrot_diffusion/sample.py`
- Validation: `src/brainrot_diffusion/validate.py`
- Evaluation: `src/brainrot_diffusion/evaluate.py`
- Packaging: `src/brainrot_diffusion/package.py`
- CLI Scripts: `scripts/*.py`
- Tests: `tests/`

The primary flow is:

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

## Design Goals

- Keep the main generator from-scratch and assignment-compliant.
- Preserve stable condition mappings between training, checkpointing, and generation.
- Make final output validation strict: exact count, filenames, RGB mode, PNG format, and `64x64` size.
- Keep scripts as thin wrappers over package modules.
- Save enough checkpoint metadata for reproducible TA generation.
- Keep tests small, CPU-friendly, and tied to module contracts.
- Treat scorer and metric execution as optional evaluation work, not a core unit-test dependency.

## Non-Goals

- No pretrained generative model, Diffusers pipeline, pretrained GAN, pretrained Transformer generator, Stable Diffusion, or equivalent high-level generation flow.
- No extra training data in the first implementation path.
- No auxiliary pretrained CLIP or VAE in final-image generation.
- No leaderboard tuning before the train/generate/validate/package path is valid.
- No hidden Codabench data assumptions.
- No broad tooling additions such as lint/type-check dependencies until explicitly chosen.

## Architecture Overview

The project remains a small Python package with command-line adapters:

```text
src/brainrot_diffusion/
  config.py
  data.py
  conditions.py
  model.py
  diffusion.py
  ema.py
  checkpoint.py
  train_loop.py
  sample.py
  validate.py
  evaluate.py
  package.py

scripts/
  train.py
  generate.py
  validate_submission.py
  evaluate.py
  prepare_score_input.py
  package_submission.py

configs/
  default.yaml

tests/
```

Classifier-free guidance is not a separate HLD module. Its training-time condition dropout belongs in Training Loop/Conditions, its null-condition model support belongs in Generator Model/Conditions, and its prediction-combination behavior belongs in Sampling/Diffusion.

## Shared Data Contracts

CSV contracts:

- Training CSV columns: `id`, `animal`, `object`.
- Generation CSV columns: `id`, `animal`, `object`, `prompt`.
- Duplicate IDs are invalid.
- Generation prompts are expected to follow `a {animal} and a {object}`; mismatch is at least a validation warning and may be an error if implementation chooses strict prompt validation.

Image contracts:

- Training images are loaded from `dataset/trainset/{id}`.
- Training tensors use shape `[B, 3, 64, 64]`, dtype `float32`, and normalized range `[-1, 1]`.
- Generated tensors are clamped/converted to RGB PNG files.
- Final output files must be PNG format, RGB mode, and `64x64`.

Condition contracts:

- Animal, object, and pair mappings are stable, serializable, and checkpointed.
- Generation uses checkpoint mappings, not rebuilt mappings from `dataset/generate.csv`.
- Unknown generation labels relative to checkpoint mappings are invalid.
- Null/unconditional conditions for classifier-free guidance must be represented in checkpoint-compatible metadata if used.

Checkpoint contract:

- Required metadata: model state, config, condition mappings, diffusion metadata, architecture metadata, seed metadata, and step/progress counters.
- Optional metadata: optimizer state and EMA state.
- Missing required keys make generation fail before sampling.

Report contracts:

- Validation and evaluation reports are JSON-compatible dictionaries when written.
- Metric skips must be explicit, not silent success.
- Commands discovered but not run remain labeled `Known, not run`.

## Module Designs

### Configuration

#### Responsibility

Load, merge, and validate runtime configuration for paths, model settings, diffusion settings, training settings, sampling settings, validation/report paths, and packaging settings.

#### Non-Responsibility

Configuration does not load datasets, construct models, run training, run sampling, write checkpoints, or decide final experiment quality.

#### Inputs and Outputs

Inputs:

- YAML config path, expected default `configs/default.yaml`.
- CLI overrides from scripts.
- Repository-relative default paths.

Outputs:

- One resolved config mapping or object used by package modules.
- Validation errors for missing or invalid keys.

#### Public Interface

Expose a small package-level loading/validation API used by scripts and tests. Exact function/class names may be finalized during implementation, but the interface must support: load config file, apply overrides, validate required sections, and serialize config into checkpoint metadata.

#### Data Structures

The resolved config should be JSON/YAML-serializable. It must include sections for data paths, model shape, diffusion schedule, training loop, sampling, checkpointing, validation/evaluation, and packaging.

#### Internal Design

Use PyYAML for parsing and standard library path handling. Keep defaults in config files rather than scattering path constants through modules. Validate before expensive work such as model allocation or dataloader creation.

#### Algorithm Details

1. Read YAML.
2. Normalize paths relative to repository root or current working directory according to script policy.
3. Apply CLI overrides.
4. Validate required sections and simple ranges.
5. Return resolved config.

#### Dependencies

- PyYAML.
- Python standard library path/type handling.

#### Failure Handling

- Missing config file: `FileNotFoundError`.
- Malformed YAML: parser error with path.
- Missing required key or invalid range: `ValueError` before work starts.

#### Independent Test Plan

- Load a minimal valid config fixture.
- Reject missing required sections.
- Reject invalid numeric ranges such as nonpositive timesteps or image size other than `64`.
- Confirm CLI overrides produce deterministic resolved values.
- Confirm resolved config can be stored in checkpoint metadata.

#### Open Questions

- Exact config key names are not fixed by the planning docs and should be chosen once implementation starts.

### Data Loading

#### Responsibility

Read training rows, generation requests, and RGB image files. Produce validated sample/request records and tensors for downstream modules.

#### Non-Responsibility

Data Loading does not own condition vocabulary persistence, model code, diffusion math, training lifecycle, sampling, final validation reports, or packaging.

#### Inputs and Outputs

Inputs:

- `dataset/train.csv`.
- `dataset/generate.csv`.
- `dataset/trainset/`.
- Image-size and transform settings from config.

Outputs:

- Parsed training rows with `id`, `animal`, and `object`.
- Parsed generation requests with `id`, `animal`, `object`, and `prompt`.
- Training image tensors shaped `[3, 64, 64]` in `[-1, 1]`.
- Batch-ready records that can be combined with condition IDs.

#### Public Interface

Expose dataset/loader behavior for training rows and generation requests. Scripts and tests must be able to load CSV rows without constructing a model.

#### Data Structures

- Training row: filename/id plus animal/object strings.
- Generation request: output filename/id plus animal/object strings and prompt.
- Training sample: image tensor plus row metadata and condition-compatible labels.

#### Internal Design

Use Python `csv.DictReader` for simple CSV parsing. Use Pillow to open images and convert them to RGB. Use PyTorch tensor conversion directly or torchvision transforms if already available. Keep any augmentation minimal and config-driven.

#### Algorithm Details

CSV parsing:

1. Read header.
2. Verify required columns.
3. Reject duplicate IDs.
4. Return ordered rows.

Image loading:

1. Resolve `trainset/{id}`.
2. Open with Pillow.
3. Convert to RGB.
4. Ensure or transform to `64x64` according to config.
5. Convert to tensor and normalize to `[-1, 1]`.

#### Dependencies

- Python standard library `csv` and paths.
- Pillow.
- PyTorch.
- Conditions module only for applying existing mappings when batch records need IDs.

#### Failure Handling

- Missing CSV column: `ValueError`.
- Duplicate ID: `ValueError`.
- Missing image file: `FileNotFoundError`.
- Unreadable image: filename-specific error.
- Unexpected generation prompt: warning or error, to be finalized.

#### Independent Test Plan

- Parse tiny train/generate CSV fixtures.
- Reject missing columns and duplicate IDs.
- Load tiny RGB images and assert tensor shape/range.
- Fail on missing image file.
- Confirm generation request ordering follows CSV ordering.

#### Open Questions

- Whether prompt mismatch is a hard error or warning is not fixed by source docs.

### Conditions

#### Responsibility

Create, serialize, deserialize, and apply stable animal, object, and animal-object pair mappings.

#### Non-Responsibility

Conditions does not load image pixels, train models, save checkpoints, or write generated PNGs.

#### Inputs and Outputs

Inputs:

- Labels from training CSV.
- Saved checkpoint mappings.
- Generation labels to look up.

Outputs:

- Animal IDs, object IDs, pair IDs.
- JSON-compatible mapping metadata.
- Null-condition metadata when classifier-free guidance is enabled.

#### Public Interface

Expose mapping construction from training labels, mapping restoration from checkpoint metadata, label-to-ID lookup, and compatibility validation for generation requests.

#### Data Structures

- `animal_to_id`, `object_to_id`, `pair_to_id`.
- Reverse lookup lists or dictionaries for metadata.
- Optional null IDs for unconditional guidance.

#### Internal Design

Use deterministic mapping creation, preferably sorted unique labels unless an explicit saved mapping is provided. Pair IDs should be deterministic for `(animal, object)` and stable across process runs.

#### Algorithm Details

1. Extract unique animals and objects from training rows.
2. Build stable animal/object IDs.
3. Build pair IDs for observed or configured pairs.
4. Serialize mappings into checkpoint metadata.
5. During generation, validate every requested label against restored mappings before sampling.

#### Dependencies

- Python standard library data structures.

#### Failure Handling

- Unknown label at generation time: `ValueError`.
- Missing mapping in checkpoint: `ValueError`.
- Duplicate or inconsistent reverse mappings: `ValueError`.

#### Independent Test Plan

- Build mappings from tiny CSV rows.
- Confirm stable mapping across row permutations.
- Confirm serialization round trip preserves IDs exactly.
- Reject unknown generation labels.
- Verify pair IDs are stable and unique for tested pairs.

#### Open Questions

- Whether pair IDs cover all 100 possible animal-object pairs or only observed training pairs should be finalized during implementation. Generation must fail clearly for any unsupported pair.

### Generator Model

#### Responsibility

Define the from-scratch conditional image generator backbone that predicts diffusion noise from noisy image tensors, timesteps, and condition IDs.

#### Non-Responsibility

Generator Model does not own CSV parsing, condition mapping construction, diffusion schedule math, optimizer steps, checkpoint schema, sampling loop, or image file writing.

#### Inputs and Outputs

Inputs:

- Noisy image tensor `[B, 3, 64, 64]`.
- Timestep tensor `[B]`.
- Condition IDs for animal, object, and pair.
- Optional null/unconditional condition IDs.

Output:

- Predicted noise tensor `[B, 3, 64, 64]`.

#### Public Interface

Expose a PyTorch `nn.Module` compatible with the model-forward contract:

```text
model(x_t, timesteps, conditions) -> predicted_noise
```

Exact class name is an implementation detail, but checkpoint metadata must record enough architecture config to rebuild it.

#### Data Structures

- Trainable PyTorch parameters initialized from scratch.
- Learned timestep and condition embeddings.
- Architecture metadata in checkpoint config.

#### Internal Design

Use a compact UNet-style pixel-space backbone as proposed. Keep image size fixed at `64x64` for the assignment. Use PyTorch modules directly. If classifier-free guidance is enabled, support null/unconditional condition IDs through learned embeddings or an equivalent checkpointed convention.

#### Algorithm Details

Baseline forward path:

1. Embed timestep.
2. Embed animal/object/pair conditions.
3. Combine time and condition embeddings.
4. Process noisy image through downsample/upsample residual blocks.
5. Return epsilon/noise prediction.

The initial objective is epsilon prediction only. v-prediction and learned variance are not required by the source docs.

#### Dependencies

- PyTorch.
- Conditions metadata for embedding sizes.
- Config for architecture settings.

#### Failure Handling

- Invalid tensor rank/channel count should fail clearly in debug validation or through PyTorch shape errors.
- Out-of-range condition IDs must fail before embedding lookup where practical.
- Config validation rejects unsupported image size or invalid architecture settings.

#### Independent Test Plan

- Instantiate a tiny model on CPU.
- Forward random tensors and assert output shape equals input image shape.
- Test batch size 1 and mixed batch sizes.
- Test null/unconditional condition path if classifier-free guidance is enabled.
- Confirm default construction does not load pretrained weights.

#### Open Questions

- Final channel width, block count, attention placement, and conditioning injection details are training-quality choices and should remain config-driven.

### Diffusion

#### Responsibility

Own diffusion schedule construction, forward noising, training objective math, and reverse-step equations shared by samplers.

#### Non-Responsibility

Diffusion does not own model architecture internals, optimizer updates, data loading, checkpoint writing, or final PNG saving.

#### Inputs and Outputs

Inputs:

- Clean image tensor `x_0` in `[-1, 1]`.
- Timesteps.
- Gaussian noise.
- Model predictions.
- Diffusion/sampler config.

Outputs:

- Noisy image tensor `x_t`.
- Scalar training loss and optional metrics.
- Reverse-step outputs for sampling.
- Diffusion metadata for checkpoints.

#### Public Interface

Expose schedule construction, noising, loss computation, and DDPM/DDIM-compatible reverse-step behavior. Exact function/class names may be finalized during implementation, but Training Loop and Sampling must use the same schedule metadata.

#### Data Structures

- Beta/alpha schedule tensors.
- Cumulative alpha tensors and derived coefficients.
- Diffusion metadata: schedule type, timesteps, prediction target, sampler settings used for generation.

#### Internal Design

Implement DDPM epsilon-prediction loss directly in PyTorch. Use a fixed default timestep count of 1,000 unless config overrides it. DDIM sampling uses the trained DDPM schedule for faster final generation.

#### Algorithm Details

Training objective:

```text
noise = randn_like(x_0)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
pred = model(x_t, t, conditions)
loss = mse(pred, noise)
```

Reverse sampling equations should be deterministic under fixed seed and sampler settings as far as PyTorch/device behavior permits.

#### Dependencies

- PyTorch.
- Generator Model forward contract.
- Config.

#### Failure Handling

- Reject nonpositive timestep counts.
- Reject invalid schedule values.
- Reject unsupported sampler/schedule names.
- Prevent device mismatch by moving schedule tensors to the active batch/model device.

#### Independent Test Plan

- Verify schedule tensors are finite and expected length.
- Verify noising output shape matches input.
- Verify loss is scalar and finite for a fake or tiny model.
- Verify deterministic noising with fixed seed and fixed inputs.
- Verify invalid timestep/sampler config fails.

#### Open Questions

- Exact beta schedule default is not fixed by source docs; cosine or linear are both valid. The proposal leans toward standard DDPM/DDIM behavior, with final default to be chosen in config.

### EMA

#### Responsibility

Maintain exponential moving average model weights for sampling and checkpoint persistence.

#### Non-Responsibility

EMA does not construct models, compute losses, run optimizer steps, decide checkpoint schema, or run sampling.

#### Inputs and Outputs

Inputs:

- Current model parameters.
- Decay setting.
- Existing EMA state when resuming.

Outputs:

- EMA shadow weights.
- EMA state for checkpoints.
- A way for Sampling to evaluate with averaged weights.

#### Public Interface

Expose EMA initialization, update-after-step behavior, state serialization/restoration, and temporary/copy application to a model for sampling.

#### Data Structures

- Shadow parameter mapping keyed consistently with model state dict.
- Decay metadata.

#### Internal Design

Keep EMA independent of optimizer internals. Update only trainable floating-point model parameters after optimizer steps. Disabled EMA should be a no-op path that does not complicate Training Loop tests.

#### Algorithm Details

For each tracked parameter:

```text
ema = decay * ema + (1 - decay) * current
```

#### Dependencies

- PyTorch model parameters/state dicts.

#### Failure Handling

- Decay outside `[0, 1)` is invalid.
- Missing or shape-mismatched EMA state on resume fails clearly.
- Disabled EMA should not create fake checkpoint state that sampling mistakes for real EMA weights.

#### Independent Test Plan

- Verify one-step scalar EMA math.
- Verify disabled EMA no-op behavior.
- Verify EMA state round trips through checkpoint metadata.
- Verify shape mismatch fails.

#### Open Questions

- Exact default EMA decay is not fixed by source docs.

### Checkpointing

#### Responsibility

Persist and restore training/generation state, including model weights, optional EMA, optimizer state, config, condition mappings, counters, and reproducibility metadata.

#### Non-Responsibility

Checkpointing does not train, sample, validate PNGs, or decide final model quality.

#### Inputs and Outputs

Inputs:

- Model state.
- Optional EMA state.
- Optional optimizer state.
- Config.
- Condition mappings.
- Diffusion and architecture metadata.
- Step/epoch/seed metadata.

Outputs:

- Checkpoint files.
- Loaded checkpoint objects or dictionaries.
- `model.pth` export path when packaging needs it.

#### Public Interface

Expose save, load, schema validation, and optional export behavior. Exact names are implementation details, but the schema must be stable enough for Training Loop, Sampling, and Packaging.

#### Data Structures

Required checkpoint fields:

```text
model
config
condition_mappings
diffusion
architecture
seed
step
```

Optional fields:

```text
ema
optimizer
epoch
metrics
```

#### Internal Design

Use PyTorch serialization. Prefer CPU-compatible loading by default, with caller-controlled device transfer. Write checkpoints atomically where feasible to avoid corrupt partial files.

#### Algorithm Details

1. Gather state dictionaries and metadata.
2. Validate required keys.
3. Save to temporary path.
4. Replace final path.
5. On load, validate schema before constructing downstream objects.

#### Dependencies

- PyTorch serialization.
- Config metadata.
- Conditions metadata.

#### Failure Handling

- Missing required checkpoint key: `ValueError`.
- Incompatible architecture/mapping metadata: `ValueError`.
- Missing checkpoint file: `FileNotFoundError`.
- Partial/corrupt checkpoint: load error surfaced with path.

#### Independent Test Plan

- Save/load tiny checkpoint.
- Verify required metadata round trips exactly.
- Reject checkpoint missing mappings/config/diffusion metadata.
- Verify generation compatibility check rejects unknown labels.
- Verify CPU map-location load works.

#### Open Questions

- Exact `model.pth` export contents should be finalized before packaging. It must be enough for TA reproduction.

### Training Loop

#### Responsibility

Orchestrate training: seeding, config use, data loading, mapping creation, model/diffusion/EMA construction, optimization, loss computation, checkpoint cadence, and logs.

#### Non-Responsibility

Training Loop does not define model blocks, parse CSV internals, implement final validation, package submissions, or run Codabench scoring.

#### Inputs and Outputs

Inputs:

- Resolved config.
- Training CSV and image directory.
- Optional resume checkpoint.

Outputs:

- Checkpoints.
- Logs or progress records.
- Optional preview samples if implemented.

#### Public Interface

Expose a package entry callable used by `scripts/train.py`. It must support fresh training and planned resume behavior, with exact argument names finalized in implementation.

#### Data Structures

- Training state: step, epoch, optimizer state, random seed metadata, loss metrics.
- Batch: image tensor plus condition ID tensors.

#### Internal Design

Set deterministic seeds for Python, NumPy, and PyTorch where practical. Build mappings from training data for fresh training and restore mappings from checkpoint for resume. Use AdamW or another directly implemented PyTorch optimizer if chosen in config. Mixed precision is optional and must auto-disable or be bypassed for CPU smoke tests.

#### Algorithm Details

Training step:

1. Get image/condition batch.
2. Sample random timesteps.
3. Optionally drop conditions for classifier-free guidance.
4. Compute diffusion epsilon-prediction loss.
5. Backpropagate.
6. Optimizer step.
7. EMA update if enabled.
8. Log and checkpoint at configured intervals.

#### Dependencies

- Configuration.
- Data Loading.
- Conditions.
- Generator Model.
- Diffusion.
- EMA.
- Checkpointing.

#### Failure Handling

- Config/dataset validation before model allocation.
- Missing dataset paths fail before training.
- Non-finite loss fails the run unless a future config explicitly permits skip/continue.
- Resume checkpoint validates architecture and mappings before optimizer work.

#### Independent Test Plan

- Tiny CPU smoke training for a bounded number of steps.
- Assert finite loss.
- Assert checkpoint written.
- Assert checkpoint contains mappings/config/diffusion metadata.
- Assert resume advances step counter when resume is implemented.

#### Open Questions

- Full training hardware and time budget are unknown.
- Default optimizer, learning rate, batch size, accumulation, and mixed precision settings should be finalized in `configs/default.yaml`.

### Sampling

#### Responsibility

Load generation-time objects and run reverse diffusion to produce image tensors or files conditioned on generation requests.

#### Non-Responsibility

Sampling does not train, build condition mappings from scratch for final generation, validate final output structure, compute FID, or package zip files.

#### Inputs and Outputs

Inputs:

- Checkpoint with model/mapping/diffusion metadata.
- Resolved config.
- Generation requests.
- Sampler settings: sampler name, steps, guidance scale, seed, batch size.

Outputs:

- Generated image tensors or saved RGB PNG files depending on caller path.
- Generation metadata such as seed and sampler settings.

#### Public Interface

Expose generation behavior callable from `scripts/generate.py`. It must accept checkpoint-backed mappings and write or return outputs in generation CSV order.

#### Data Structures

- Condition batch tensors for requested rows.
- Generated image tensor `[B, 3, 64, 64]`.
- Optional generation result records: filename, seed, status.

#### Internal Design

Use EMA weights by default when present. Run DDIM for faster final generation after the DDPM objective is trained; keep DDPM/debug sampling available only if implemented. Apply classifier-free guidance by combining conditional and unconditional predictions when supported by the checkpoint.

#### Algorithm Details

Generation flow:

1. Load checkpoint.
2. Validate generation labels against checkpoint mappings.
3. Rebuild model/diffusion from checkpoint metadata/config.
4. Seed sampling.
5. Batch requests.
6. Run reverse diffusion.
7. Clamp and convert tensors to RGB PNG outputs.

#### Dependencies

- Configuration.
- Data Loading.
- Conditions.
- Generator Model.
- Diffusion.
- EMA.
- Checkpointing.

#### Failure Handling

- Missing checkpoint/generate CSV fails immediately.
- Existing outputs fail unless overwrite is explicit.
- Unsupported sampler or invalid step count fails before sampling.
- Unknown labels fail before sampling.
- Final tensor values are clamped before PNG conversion.

#### Independent Test Plan

- Use a tiny checkpoint/fake model to generate two fixture PNGs.
- Assert deterministic output for fixed seed where feasible.
- Assert filenames match generation CSV exactly.
- Assert image files open as RGB and `64x64`.
- Assert existing outputs fail without overwrite.

#### Open Questions

- Exact default sampler steps and guidance scale are training-quality decisions.

### Validation

#### Responsibility

Verify generated submission structure: count, filenames, PNG format, RGB mode, and image size.

#### Non-Responsibility

Validation does not train, sample, score FID/CLIP-T, or create final archives.

#### Inputs and Outputs

Inputs:

- `dataset/generate.csv` or fixture generation CSV.
- Output directory such as `generated_images/`.
- Optional report path.

Outputs:

- Pass/fail validation result.
- JSON-compatible report if requested.

#### Public Interface

Expose validation callable for scripts, Evaluation, Packaging, and tests. It must return structured findings and support nonzero CLI exit on failure.

#### Data Structures

- Expected filename set from generate CSV.
- Actual PNG filename set from output directory.
- Finding list with severity, filename when relevant, and message.

#### Internal Design

Use standard library file traversal and Pillow. Compare expected and actual filenames exactly. Open every expected PNG and inspect format/mode/size.

#### Algorithm Details

1. Load generation CSV.
2. Build expected filename set and count.
3. Find PNG files in output directory.
4. Compare missing/extra filenames.
5. For each expected file, open image and check PNG/RGB/`64x64`.
6. Return pass only if all checks pass.

#### Dependencies

- Data Loading for CSV parsing or shared CSV helper.
- Pillow.
- Standard library paths.

#### Failure Handling

- Missing output directory: validation failure.
- Missing/extra files: validation failure.
- Corrupt image: validation failure with filename.
- Wrong mode/size/format: validation failure with filename.

#### Independent Test Plan

- Valid tiny two-image fixture passes.
- Missing expected file fails.
- Extra file fails.
- Wrong size fails.
- Wrong mode fails.
- Corrupt PNG fails.
- JSON report includes enough detail for debugging.

#### Open Questions

- Whether nested output directories are allowed should be fixed during implementation. The assignment output implies flat `generated_images/`.

### Evaluation

#### Responsibility

Run structural validation first, then run local FID/evaluator hooks when assets and dependencies are available. Report metric results or explicit skips.

#### Non-Responsibility

Evaluation does not affect the main generator training path, replace Codabench, or make local CLIP-T official.

#### Inputs and Outputs

Inputs:

- Generated image directory.
- Generation CSV.
- `hw6_reference/` assets.
- Optional scorer-compatible input layout.
- Optional report path.

Outputs:

- Evaluation report with validation result, FID score if run, and skipped metric reasons.

#### Public Interface

Expose evaluation callable for `scripts/evaluate.py` and a helper path for scorer-input preparation if implemented in package logic.

#### Data Structures

- Report dictionary with `validation`, `metrics`, and `skipped` sections.
- Optional FID statistics arrays.

#### Internal Design

Call Validation first. If validation fails, stop before scoring. Local FID may use provided reference stats and torchvision/Inception-style features where available. CLIP-T remains skipped or proxy-only unless explicitly approved and documented.

#### Algorithm Details

1. Validate structure.
2. If invalid, write/report validation failure and stop.
3. Check reference stats and metric dependencies.
4. Compute available metrics or record skip reasons.
5. Write report.

#### Dependencies

- Validation.
- NumPy/SciPy/PyTorch/torchvision for local FID when used.
- Provided `scoring_program/score.py` behavior as reference.
- Optional CLIP dependency only if later approved.

#### Failure Handling

- Validation failure prevents metric calculation.
- Missing reference stats: metric skipped.
- Missing optional dependency: metric skipped.
- CUDA-only scorer path is not assumed available.

#### Independent Test Plan

- Validation-first behavior with invalid output.
- Report skip when reference stats are missing.
- Report skip when CLIP proxy is disabled.
- Mock or tiny-test FID report plumbing without requiring heavy model downloads.

#### Open Questions

- Whether local CLIP proxy evaluation should be implemented is undecided.

### Packaging

#### Responsibility

Assemble the final E3 submission archive after validating generated outputs and required artifacts.

#### Non-Responsibility

Packaging does not train models, generate images, compute metrics, or infer the student's ID.

#### Inputs and Outputs

Inputs:

- `generated_images/`.
- `scripts/`.
- `src/brainrot_diffusion/`.
- `configs/` if required for reproducibility.
- `model.pth`.
- `README.md`.
- `requirements.txt`.
- Student ID.
- Output path/overwrite flag.

Outputs:

- `HW6_{student_id}.zip` with assignment-required contents.

#### Public Interface

Expose packaging callable used by `scripts/package_submission.py`.

#### Data Structures

- Package manifest.
- Validation result.
- Zip entry list.

#### Internal Design

Run Validation before writing the archive. Require a real student ID rather than a placeholder. Include implementation files needed for reproduction even though the PDF minimum lists `scripts/`, `model.pth`, `README.md`, and `requirements.txt`.

#### Algorithm Details

1. Validate generated images.
2. Verify required files/directories exist.
3. Build manifest.
4. Refuse existing zip unless overwrite is explicit.
5. Write zip with deterministic relative paths where practical.

#### Dependencies

- Validation.
- Standard library `zipfile` and paths.

#### Failure Handling

- Invalid images fail before archive write.
- Missing artifact fails with path list.
- Placeholder/missing student ID fails.
- Existing zip fails unless overwrite is explicit.

#### Independent Test Plan

- Package tiny valid fixture.
- Inspect zip entries.
- Invalid image fixture prevents packaging.
- Missing `model.pth` prevents packaging.
- Placeholder student ID fails.

#### Open Questions

- Final student ID is unknown.
- Final `model.pth` export schema should be confirmed before submission.

### CLI Scripts

#### Responsibility

Provide thin command-line wrappers for training, generation, validation, evaluation, scorer input preparation, and packaging.

#### Non-Responsibility

Scripts do not own core algorithms or duplicate package logic.

#### Inputs and Outputs

Inputs:

- CLI arguments and paths.

Outputs:

- Process exit status.
- Checkpoints, images, reports, scorer input directories, or packages through package module calls.

#### Public Interface

Known planned commands from README/quality gates:

```bash
python scripts/train.py --config configs/default.yaml
python scripts/generate.py --checkpoint checkpoints/checkpoint_step_1000.pt --config configs/default.yaml --overwrite
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json
python scripts/prepare_score_input.py --generate-csv dataset/generate.csv --generated-images generated_images --score-input-dir score_input --test-mu hw6_reference/test_mu.npy --test-sigma hw6_reference/test_sigma.npy --scores fid --overwrite
python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite
```

#### Data Structures

- Parsed `argparse` namespace or equivalent.
- Package module return values converted to exit status and messages.

#### Internal Design

Use `argparse` and call package modules. Keep scripts short. Any behavior that needs tests belongs in `src/brainrot_diffusion/`, not only in scripts.

#### Algorithm Details

1. Parse arguments.
2. Load config if needed.
3. Call package module.
4. Print concise status.
5. Return nonzero on validation or runtime errors.

#### Dependencies

- Package modules.
- Python standard library `argparse`.

#### Failure Handling

- Missing required arguments fail through argument parser.
- Package exceptions result in nonzero exits.
- Mutating commands require explicit overwrite/resume flags where applicable.

#### Independent Test Plan

- Unit-test parser behavior where useful.
- Script-level smoke tests on tiny fixtures after implementation exists.
- Confirm thin scripts do not duplicate core validation/generation logic.
- Confirm invalid inputs return nonzero.

#### Open Questions

- Exact optional flags may change during implementation, but known README commands should remain aligned.

### Tests

#### Responsibility

Provide small CPU tests and fixtures that verify module contracts, integration paths, and regression cases from `doc/test-plan.md`.

#### Non-Responsibility

Tests do not perform full GPU training, official Codabench scoring, or subjective image-quality evaluation.

#### Inputs and Outputs

Inputs:

- Tiny CSV fixtures.
- Tiny generated/training image fixtures.
- Temporary directories.
- Package modules.

Outputs:

- Pytest pass/fail results.

#### Public Interface

`python -m pytest` is the discovered test command once tests exist.

#### Data Structures

- Fixture CSVs matching assignment columns.
- Tiny RGB images.
- Fake/tiny checkpoints.
- Expected validation reports.

#### Internal Design

Keep tests independent by module. Use shared tiny fixtures only where they reduce duplication without coupling unrelated modules. Avoid heavy GPU/scorer/model-download requirements.

#### Algorithm Details

Layered tests:

1. Unit tests for config, data, conditions, model shapes, diffusion math, EMA, checkpointing, validation, evaluation skips, and packaging.
2. Integration tests for tiny train -> checkpoint -> generate -> validate.
3. CLI smoke tests for representative script paths when implementation exists.

#### Dependencies

- Pytest.
- Pillow.
- PyTorch.
- Package modules.

#### Failure Handling

- Tests should fail on invalid shape, mapping drift, missing metadata, output format mistakes, and accidental dependence on unavailable GPU/scorer resources.

#### Independent Test Plan

- This module owns the test suite itself. Its independent verification is that `python -m pytest` discovers and runs tests after implementation.
- `python -m compileall src scripts tests` verifies import/syntax once files exist.

#### Open Questions

- Exact fixture layout is not fixed. Keep it minimal.

## Cross-Module Contracts

- Configuration feeds every workflow and is stored in checkpoints.
- Data Loading owns row parsing; Conditions owns ID mapping.
- Training Loop builds or restores mappings before constructing condition-aware datasets and model embeddings.
- Generator Model consumes only tensors and condition IDs; it never reads files.
- Diffusion calls Generator Model through the forward contract and never owns optimizer state.
- EMA reads model parameters after optimizer steps and is checkpointed separately from raw model weights.
- Checkpointing is the only persistence boundary for model/config/mapping/diffusion metadata.
- Sampling must use Checkpointing and Conditions metadata before writing images.
- Validation is the gate before Evaluation and Packaging.
- Evaluation is optional and reports skips explicitly.
- CLI Scripts are wrappers; core behavior belongs in package modules.
- Tests verify module contracts and at least one end-to-end tiny workflow.

## End-to-End Workflow

Training workflow:

```text
scripts/train.py
  -> Configuration
  -> Data Loading
  -> Conditions
  -> Generator Model
  -> Diffusion
  -> Training Loop
  -> EMA
  -> Checkpointing
```

Generation workflow:

```text
scripts/generate.py
  -> Configuration
  -> Checkpointing
  -> Conditions compatibility validation
  -> Generator Model
  -> Diffusion
  -> EMA weights when available
  -> Sampling
  -> generated_images/{id}
```

Validation/evaluation workflow:

```text
scripts/validate_submission.py
  -> Validation

scripts/evaluate.py
  -> Validation
  -> Evaluation metrics or explicit skips
```

Packaging workflow:

```text
scripts/package_submission.py
  -> Validation
  -> Packaging
  -> HW6_{student_id}.zip
```

## Test Strategy Mapping

| Test-plan requirement | Design coverage |
| --- | --- |
| Config loading and validation | Configuration module tests and CLI script parser checks |
| CSV parsing | Data Loading unit tests |
| RGB image loading and tensor shapes | Data Loading unit tests |
| Stable animal/object/pair mappings | Conditions unit and property tests |
| Checkpoint-compatible generation mappings | Conditions, Checkpointing, Sampling integration tests |
| From-scratch model initialization and output shape | Generator Model unit tests |
| Diffusion schedule, noising, finite loss | Diffusion unit tests |
| EMA update and checkpoint inclusion | EMA and Checkpointing tests |
| Tiny CPU training smoke path | Training Loop integration test |
| Deterministic tiny sampling where feasible | Sampling tests |
| Final filename/count/PNG/RGB/size validation | Validation unit and integration tests |
| Evaluation validation-first behavior and metric skips | Evaluation tests |
| Packaging refusal on invalid outputs | Packaging tests |
| CLI argument parsing and nonzero invalid exits | CLI Scripts tests |
| README command alignment | Tests or manual verification after implementation |
| Golden tiny train/generate CSV cases | Data Loading, Conditions, Sampling, Validation tests |
| Randomized exact filename-set validation | Validation property tests |
| Wrong mode/size/corrupt PNG edge cases | Validation tests |
| Missing checkpoint metadata | Checkpointing tests |
| Unknown labels at generation time | Conditions/Sampling tests |
| Existing outputs without overwrite | Sampling and CLI tests |
| Manual visual review | Manual Verification before final submission |

## Quality Gates

Known, not run:

```bash
python -m compileall src scripts tests
```

Known, not run:

```bash
python -m pytest
```

Known, not run:

```bash
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json
```

Known, not run:

```bash
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json
```

Known, not run:

```bash
python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite
```

Known, not run:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

Notes:

- `src/`, `scripts/`, `configs/`, and `tests/` are not currently present, so the planned gates are not yet runnable as documented.
- The Codabench-style scorer path expects a scorer input layout, writes `scores.json`, uses `cuda:0`, and may require pretrained evaluation weights.
- No command above is verified until explicitly run.

## Risks and Mitigations

- Risk: implementation files are absent while docs describe planned behavior. Mitigation: treat live files as truth and start with scaffold plus tests.
- Risk: output format mistakes cause penalties. Mitigation: implement Validation early and use it before evaluation/packaging.
- Risk: mapping drift breaks conditional generation. Mitigation: checkpoint mappings and reject incompatible generation labels before sampling.
- Risk: full training may exceed available hardware/time. Mitigation: tiny CPU smoke tests first; keep batch size, model width, precision, accumulation, DDIM steps, and guidance scale configurable.
- Risk: local metrics differ from official Codabench. Mitigation: label local FID/CLIP checks as local/proxy and preserve official submission as final authority.
- Risk: pretrained auxiliary modules could violate the assignment if misused. Mitigation: do not add CLIP/VAE generation paths without explicit approval and documentation.
- Risk: optional scorer dependencies may require network/GPU. Mitigation: keep scorer execution out of core tests and report metric skips explicitly.

## Assumptions

- Python 3.10+, PyTorch, and the current `requirements.txt` remain the implementation stack.
- The first implementation uses only the provided Brainrot Dataset.
- Pixel-space conditional diffusion is the selected baseline.
- Generated images are written to a flat `generated_images/` directory.
- TA reproduction should be possible from submitted package artifacts and checkpoint metadata.
- The user has implicitly authorized updating `doc/detailed-design.md` by requesting this skill command; no production code or evaluator command is included.

## Open Questions

- What GPU and training-time budget are available?
- What student ID should be used for `HW6_{student_id}.zip`?
- What is the exact E3 deadline?
- Should local CLIP proxy evaluation be implemented, or should CLIP-T be left to Codabench only?
- Should any auxiliary pretrained CLIP or VAE path be used later, or avoided entirely?
- Should generation prompt mismatch be a warning or a hard validation error?
- Should pair mappings include all 100 possible animal-object pairs or only observed training pairs?
- What default diffusion schedule, model width, optimizer settings, EMA decay, DDIM step count, and guidance scale should be used in `configs/default.yaml`?
- What exact `model.pth` export schema should final packaging use?
