# Detailed Design

## Purpose
Design the implementation for the HW6 Brainrot Image Generation assignment described in `Brainrot_Image_Gen.pdf`, using `doc/proposal.md` as the source proposal. The solution is a from-scratch conditional DDPM/DDIM image generator that trains on the provided Brainrot Dataset and produces exactly 2,000 RGB PNG images at 64x64 resolution from `dataset/generate.csv`.

## Source Proposal Summary
The proposal selects a pixel-space conditional diffusion model as the primary algorithm. The model uses a compact residual UNet trained with epsilon-prediction MSE, learned animal/object/pair condition embeddings, condition dropout for classifier-free guidance, EMA weights for sampling, and DDIM sampling for final generation.

The repository currently contains the assignment PDF, research notes, dataset files, reference FID statistics, and the Codabench scorer. The implementation files described by `README.md` are not yet present, so this design treats the codebase as a greenfield Python/PyTorch project inside the existing assignment bundle.

## Design Goals
- Train the main generator from scratch without pretrained generative weights.
- Avoid Diffusers pipelines and high-level generation/training flows.
- Produce valid submission output: exactly 2,000 RGB PNG files, 64x64, named according to `dataset/generate.csv`.
- Keep model, diffusion, training, sampling, validation, and packaging modules independently testable.
- Save enough metadata for reproducible training and generation.
- Optimize for assignment metrics: lower FID and higher CLIP-T.

## Non-Goals
- Do not implement Stable Diffusion, FLUX, pretrained GANs, ControlNet, LoRA, DreamBooth, or other pretrained generation paths.
- Do not use external generated images in the first implementation.
- Do not make a pretrained CLIP or VAE the main generator.
- Do not implement the optional StyleGAN2-ADA fallback in the first detailed design. It remains a future experiment.
- Do not depend on hidden Codabench test images or hidden CLIP-T metadata for local correctness.

## Architecture Overview
The project will be organized around a small Python package plus command-line scripts:

```text
src/brainrot_diffusion/
  config.py
  data.py
  conditions.py
  unet.py
  diffusion.py
  guidance.py
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
  package_submission.py
  prepare_score_input.py

configs/
  default.yaml

tests/
```

The package owns reusable logic. Scripts are thin CLI adapters that parse paths and config, call package functions, and return clear exit codes.

Primary workflow:

1. `scripts/train.py` loads config, dataset, condition mappings, UNet, diffusion scheduler, optimizer, and EMA.
2. The training loop samples timesteps and noise, computes DDPM epsilon-prediction loss, updates model and EMA, and writes checkpoints.
3. `scripts/generate.py` loads a checkpoint, uses EMA weights by default, reads `dataset/generate.csv`, samples images with DDIM and classifier-free guidance, and writes `generated_images/{id}`.
4. `scripts/validate_submission.py` checks output shape, mode, count, filenames, and PNG validity.
5. `scripts/evaluate.py` runs structural validation and local FID when reference stats are available.
6. `scripts/package_submission.py` packages required files for E3 submission.

## Module Designs

### Data and Labels

#### Responsibility
Own reading CSV files, loading images, converting animal/object labels into stable ids, and returning tensors and condition ids for training and generation. It does not own model architecture, diffusion math, sampling, checkpoint persistence, or evaluation.

#### Inputs and Outputs
Inputs:

- `dataset/train.csv` with columns `id,animal,object`.
- `dataset/generate.csv` with columns `id,animal,object,prompt`.
- `dataset/trainset/{id}` PNG images.
- Config values for image size, transform behavior, and optional augmentation.

Outputs:

- Training samples: image tensor `[3, 64, 64]` normalized to `[-1, 1]`, `animal_id`, `object_id`, `pair_id`, and source filename.
- Generation requests: output filename, animal/object labels, prompt, and mapped condition ids.
- Stable condition mappings saved to checkpoints and reused for generation.

#### Internal Design
`conditions.py` defines the canonical animal and object vocabularies from observed CSV labels. It creates deterministic mappings by sorting labels or by preserving a saved mapping from checkpoint metadata. `pair_id` is derived from `(animal_id, object_id)` and supports all 100 possible combinations.

`data.py` defines:

- `BrainrotTrainDataset`
- `GenerationRequestDataset`
- `build_condition_mappings`
- `load_generation_requests`

Image handling:

- Open images with PIL and convert to RGB.
- Resize or center-crop defensively to 64x64 if image dimensions differ.
- Convert to tensor and normalize to `[-1, 1]`.
- Random horizontal flip is controlled by config and disabled by default until visual inspection confirms it is harmless.

#### Dependencies
- Python CSV or pandas-like parsing. Prefer Python stdlib `csv` unless tabular operations become complex.
- PIL for image loading.
- PyTorch and torchvision transforms for tensor conversion.

#### Failure Handling
- Missing CSV columns raise a clear `ValueError`.
- Missing image files raise `FileNotFoundError` with the image id.
- Unknown generation label not present in saved mappings raises `ValueError`.
- Invalid image mode or unreadable image raises a filename-specific error.
- Duplicate ids in `generate.csv` are rejected.

#### Independent Test Plan
- Build a tiny temporary dataset with two PNG files and a small CSV.
- Verify image tensors have shape `[3, 64, 64]` and range approximately `[-1, 1]`.
- Verify stable mappings are deterministic across repeated loads.
- Verify `pair_id` is stable and unique for animal/object pairs.
- Verify missing columns, missing files, duplicate ids, and unknown labels fail cleanly.

#### Open Questions
- None for the initial design.

### Conditional UNet

#### Responsibility
Own the neural network that predicts diffusion noise from a noisy image, timestep, and condition ids. It does not own diffusion schedule construction, loss sampling, optimizer updates, checkpoint IO, or image file writing.

#### Inputs and Outputs
Inputs:

- `x_t`: float tensor `[B, 3, 64, 64]`.
- `t`: integer tensor `[B]`.
- Condition ids: `animal_id`, `object_id`, `pair_id`, each `[B]`.
- Optional condition-dropout mask or null-condition ids for classifier-free guidance.

Output:

- Predicted noise tensor `[B, 3, 64, 64]`.

#### Internal Design
The UNet follows the proposal:

- Resolution path: `64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64`.
- Base channels: configurable, default candidate 96 or 128.
- Residual blocks: 2 per resolution by default.
- Normalization and activation: GroupNorm and SiLU.
- Attention: self-attention at 16x16, with optional 8x8 attention controlled by config.
- Timestep embedding: sinusoidal embedding followed by an MLP.
- Condition embeddings: learned animal, object, pair, and null embeddings.
- Conditioning injection: combine time and condition embeddings and inject into residual blocks through FiLM-style scale/shift when implemented, or additive embedding projection as the simpler fallback.

The initial output head predicts epsilon only. v-prediction and learned variance are not part of the initial model contract.

#### Dependencies
- PyTorch `nn.Module`.
- Condition mappings for embedding table sizes.
- Config values for channels, blocks, attention resolutions, dropout, and embedding dimension.

#### Failure Handling
- Validate tensor ranks and channel count in debug mode.
- Reject condition ids outside embedding-table ranges.
- Config validation prevents invalid attention resolutions or unsupported image sizes.

#### Independent Test Plan
- Instantiate the model with tiny channel counts for CPU tests.
- Pass random tensors through the model and assert output shape equals input image shape.
- Test unconditional/null condition path.
- Test mixed batch condition ids and timestep ids.
- Count parameters for the default config and record the value for reproducibility.

#### Open Questions
- The final base channel width is a training-performance decision. The design keeps it configurable with 96 and 128 as supported defaults.

### Diffusion Objective

#### Responsibility
Own beta/alpha schedules, timestep sampling, noising equation, training loss, and reverse-step math shared by DDPM and DDIM. It does not own the UNet internals, optimizer, image dataset, or CLI parsing.

#### Inputs and Outputs
Inputs:

- Clean image tensor `x_0` in `[-1, 1]`.
- Timestep tensor `t`.
- Gaussian noise tensor `epsilon`.
- Model-predicted noise tensor.

Outputs:

- Noisy image `x_t`.
- Training loss scalar and optional logging metrics.
- Reverse-process coefficients used by samplers.

#### Internal Design
`diffusion.py` defines a `GaussianDiffusion` object with:

- `num_timesteps = 1000` by default.
- Cosine beta schedule by default.
- Precomputed buffers for betas, alphas, cumulative alphas, square roots, posterior coefficients, and DDIM step schedules.
- `q_sample(x_0, t, noise)`.
- `training_loss(model, x_0, t, condition)`.
- `predict_x0_from_eps(x_t, t, eps)`.
- `p_sample_ddpm(...)`.
- `ddim_step(...)`.

Training loss:

```text
noise = normal_like(x_0)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
pred = model(x_t, t, condition)
loss = mse(pred, noise)
```

#### Dependencies
- PyTorch tensor operations.
- Conditional UNet interface.
- Config for schedule type and timestep count.

#### Failure Handling
- Config validation rejects nonpositive timesteps.
- Schedule construction checks beta values are finite and within valid range.
- Tensor-device mismatch errors are prevented by registering schedule arrays as module buffers or moving them explicitly to the batch device.

#### Independent Test Plan
- Verify schedule tensors have length 1,000 and finite values.
- Verify `q_sample(x_0, t=0)` is close to `x_0` under the schedule's first alpha value.
- Verify noising and denoising helper output shapes.
- Verify loss returns a scalar for a fake model.
- Verify deterministic behavior with fixed seeds for fixed inputs.

#### Open Questions
- None for the initial epsilon-prediction design.

### Classifier-Free Guidance

#### Responsibility
Own condition dropout during training and conditional/unconditional prediction combination during sampling. It does not own condition vocabulary construction or the sampler loop.

#### Inputs and Outputs
Inputs:

- Condition ids for animal, object, and pair.
- Condition dropout probability, default 0.10.
- Conditional and unconditional model predictions during sampling.
- Guidance scale, configured at generation time.

Outputs:

- Possibly dropped condition ids for training.
- Guided epsilon prediction during sampling.

#### Internal Design
`guidance.py` provides:

- `drop_conditions(condition_batch, p, generator=None)`.
- `make_null_condition_batch(batch_size, mappings, device)`.
- `combine_cfg(eps_uncond, eps_cond, guidance_scale)`.

Null condition support is implemented with dedicated null embedding ids in the UNet or a separate learned null embedding branch. The checkpoint records which convention is used so generation is compatible with training.

#### Dependencies
- Condition mapping metadata.
- Conditional UNet supports null/unconditional inputs.
- Sampling code calls conditional and unconditional forward passes.

#### Failure Handling
- Reject guidance scale below 0.
- Reject condition dropout probability outside `[0, 1]`.
- Ensure unconditional batch shape matches conditional batch shape before combining predictions.

#### Independent Test Plan
- Verify `p=0` leaves all conditions unchanged.
- Verify `p=1` converts all conditions to null ids.
- Verify guided prediction matches conditional prediction when guidance scale is 1 and formula inputs are consistent.
- Verify shape mismatch raises an error.

#### Open Questions
- None for the initial design.

### EMA and Checkpointing

#### Responsibility
Own moving-average model weights and persistent training/generation state. It does not own model architecture code, training-step computation, or sampling decisions.

#### Inputs and Outputs
Inputs:

- Model state dict.
- Optimizer state dict.
- EMA state dict.
- Config and metadata.
- Current step and epoch counters.

Outputs:

- Checkpoint files under a configured checkpoint directory.
- A final `model.pth` for packaging.
- Loaded model, EMA state, condition mappings, diffusion metadata, and config.

#### Internal Design
`ema.py` defines an `EMA` helper:

- Tracks shadow parameters.
- Updates after optimizer steps.
- Can copy EMA weights into a model for sampling.
- Stores decay value in checkpoint metadata.

`checkpoint.py` defines:

- `save_checkpoint(path, state)`.
- `load_checkpoint(path, map_location)`.
- `export_model_pth(checkpoint_path, output_path)`.

Checkpoint schema:

```text
{
  "model": state_dict,
  "ema": state_dict,
  "optimizer": state_dict,
  "step": int,
  "epoch": int,
  "config": dict,
  "condition_mappings": dict,
  "diffusion": dict,
  "architecture": dict,
  "seed": dict
}
```

#### Dependencies
- PyTorch serialization.
- Config module.
- Condition mapping module.
- Model and optimizer constructors for reload.

#### Failure Handling
- Missing required checkpoint keys raise a clear `ValueError`.
- Incompatible condition mappings raise an error before generation.
- Loading defaults to CPU map location first, then caller moves model to target device.
- Atomic save pattern writes to a temporary path before replacing the final checkpoint path.

#### Independent Test Plan
- Save and load a tiny model checkpoint.
- Verify loaded parameters match saved parameters.
- Verify EMA update changes shadow parameters after a fake optimizer step.
- Verify missing checkpoint keys are rejected.
- Verify condition mappings round-trip exactly.

#### Open Questions
- None for the initial design.

### Training Loop

#### Responsibility
Own the end-to-end training lifecycle: config loading, seeding, dataset/dataloader creation, model construction, optimizer, diffusion loss calls, backpropagation, EMA updates, logging, sample previews, and checkpoint cadence. It does not own individual model block internals or file-level submission validation.

#### Inputs and Outputs
Inputs:

- Config file, default `configs/default.yaml`.
- `dataset/train.csv`.
- `dataset/trainset/`.
- Optional resume checkpoint.

Outputs:

- Periodic checkpoints.
- Training logs.
- Optional preview sample grids.
- Final checkpoint or exported `model.pth`.

#### Internal Design
`train_loop.py` exposes `train(config)`.

Default training behavior:

- Set random seeds for Python, NumPy, and PyTorch.
- Build condition mappings from training CSV.
- Construct dataset and dataloader.
- Construct UNet and `GaussianDiffusion`.
- Use AdamW with learning rate from config.
- For each batch, sample random timesteps, apply condition dropout, compute diffusion loss, update model, update EMA, and log loss.
- Save checkpoints at configured step intervals.

Batch size and gradient accumulation are config-driven because GPU memory is unknown. Mixed precision can be config-controlled if available, but CPU smoke tests must work without it.

#### Dependencies
- Data and Labels.
- Conditional UNet.
- Diffusion Objective.
- Classifier-Free Guidance.
- EMA and Checkpointing.
- Config.

#### Failure Handling
- Config validation happens before allocating large models.
- Missing dataset paths fail before training starts.
- Non-finite loss triggers checkpoint/log message and stops unless config says to continue.
- Resume checkpoint validates architecture and mapping compatibility.

#### Independent Test Plan
- Run a tiny CPU smoke train with a tiny model, tiny image fixture, and `max_steps=2`.
- Assert a checkpoint is written.
- Assert loss is finite.
- Assert EMA state exists.
- Assert resume from the tiny checkpoint advances the step counter.

#### Open Questions
- Full training budget and target GPU are not fixed. The design keeps batch size, channels, precision, and accumulation configurable.

### Sampling

#### Responsibility
Own reverse diffusion sampling and image tensor postprocessing. It does not own generation CSV parsing, checkpoint schema, or submission validation.

#### Inputs and Outputs
Inputs:

- Loaded model, preferably with EMA weights.
- Diffusion scheduler.
- Condition ids.
- Sampling config: DDIM/DDPM mode, number of steps, guidance scale, seed, batch size.

Outputs:

- Image tensors in `[0, 1]` or uint8-ready format with shape `[B, 3, 64, 64]`.

#### Internal Design
`sample.py` provides:

- `sample_ddpm(model, diffusion, conditions, shape, guidance_scale)`.
- `sample_ddim(model, diffusion, conditions, shape, steps, eta, guidance_scale)`.
- `denormalize_to_uint8(images)`.

DDIM is the default for final generation because the assignment requires 2,000 images and the proposal prioritizes faster sampling. DDPM remains available for debugging.

Classifier-free guidance is applied inside each reverse step by evaluating the model with null and real conditions, then combining epsilon predictions.

#### Dependencies
- Conditional UNet.
- Diffusion Objective.
- Classifier-Free Guidance.
- Condition batch structures.

#### Failure Handling
- Reject unsupported sampler names.
- Reject DDIM step counts larger than training timesteps or below 1.
- Clamp final decoded images to valid range before saving.
- Make seeded sampling deterministic for fixed model, config, and device where PyTorch permits.

#### Independent Test Plan
- Use a fake model returning zeros and verify sampler output shape.
- Run DDIM for a tiny number of steps on CPU.
- Verify output conversion produces uint8-like values in `[0, 255]`.
- Verify guidance scale parameter is passed through and affects fake-model combination as expected.

#### Open Questions
- None for the initial design.

### Generation Script

#### Responsibility
Own final assignment inference: read `dataset/generate.csv`, load checkpoint, sample images in batches, and write files to `generated_images/` with the exact requested filenames. It does not own training or scoring.

#### Inputs and Outputs
Inputs:

- Checkpoint path.
- Config path.
- `dataset/generate.csv`.
- Output directory, default `generated_images/`.
- Sampling overrides: sampler, steps, guidance scale, seed, batch size.

Outputs:

- Exactly one RGB 64x64 PNG for each row in `dataset/generate.csv`.

#### Internal Design
`scripts/generate.py` calls package logic:

1. Load config and checkpoint.
2. Reconstruct condition mappings from checkpoint.
3. Load generation requests.
4. Build model and diffusion scheduler.
5. Load EMA weights by default.
6. Iterate requests in batches.
7. Sample images using DDIM by default.
8. Save PNGs to `generated_images/{id}`.

Overwrite behavior is explicit. If output files already exist and `--overwrite` is not set, the script fails before partial generation.

#### Dependencies
- Data and Labels.
- EMA and Checkpointing.
- Conditional UNet.
- Diffusion Objective.
- Sampling.

#### Failure Handling
- Missing checkpoint or generate CSV fails immediately.
- Existing output directory without overwrite fails before sampling.
- Unknown labels relative to checkpoint mappings fail before sampling.
- Partial generation writes can be resumed only if an explicit resume mode is later added; initial design keeps overwrite behavior simple.

#### Independent Test Plan
- Use a tiny checkpoint and tiny `generate.csv` fixture.
- Generate two images into a temporary output directory.
- Assert filenames match the CSV exactly.
- Assert images open as RGB and are 64x64.
- Assert existing files fail without `--overwrite`.

#### Open Questions
- None for the initial design.

### Local Validation and Scoring

#### Responsibility
Own structural validation of generated submissions and local metric execution when reference resources are available. It does not own model training or image generation.

#### Inputs and Outputs
Inputs:

- `dataset/generate.csv`.
- Generated image directory.
- Optional `hw6_reference/test_mu.npy` and `hw6_reference/test_sigma.npy`.
- Optional scorer resources and open_clip dependency for CLIP-style proxy checks.

Outputs:

- Human-readable validation result.
- Optional JSON report.
- Optional FID score.
- Optional CLIP-T proxy score if local resources allow it.

#### Internal Design
`validate.py` checks:

- PNG count equals CSV row count.
- Expected filenames exactly match output filenames.
- No missing or extra files.
- Each image opens with PIL.
- Each image mode is RGB.
- Each image size is 64x64.

`evaluate.py` wraps validation and then computes local FID using the same style as `scoring_program/score.py` when reference stats are present. Official CLIP-T cannot be fully reproduced without hidden test metadata, so CLIP-based evaluation remains optional and clearly labeled as proxy-only.

`prepare_score_input.py` creates a Codabench-like directory layout for running the provided scorer on FID.

#### Dependencies
- PIL.
- NumPy/SciPy/Torch/Torchvision for FID if used.
- Existing `scoring_program/score.py` for course-compatible scoring behavior.
- Optional `open_clip` only for proxy CLIP checks.

#### Failure Handling
- Validation failure prevents scoring and packaging.
- Metric dependency absence is reported as skipped, not as a silent zero.
- Reference stats absence is reported as skipped.
- Image-size mismatch is reported with filename.

#### Independent Test Plan
- Validate a temporary directory with exact CSV-matching images.
- Test missing, extra, wrong-size, and non-RGB files.
- Test FID path with mocked feature arrays or a tiny scorer fixture when full Inception execution is too heavy.
- Test that optional CLIP proxy reports skipped when dependency or metadata is missing.

#### Open Questions
- Whether to install and support an `open_clip` proxy locally remains optional. The official assignment score comes from Codabench.

### Packaging

#### Responsibility
Own creation of the final E3 submission zip layout. It does not own generation, training, or model scoring.

#### Inputs and Outputs
Inputs:

- `generated_images/`.
- `scripts/`.
- `src/brainrot_diffusion/`.
- `model.pth`.
- `README.md`.
- `requirements.txt`.
- Output zip path.

Outputs:

- `HW6_{student_id}.zip` or configured zip path with the assignment-required structure.

#### Internal Design
`package.py` verifies required artifacts exist, runs validation, and writes a zip. It refuses packaging if generated images are invalid.

The package includes:

- `generated_images/`
- `scripts/`
- `model.pth`
- `README.md`
- `requirements.txt`
- Source/config files needed for reproducibility

The exact top-level folder name is configurable because the PDF requires `HW6_{student_id}` and the student id is not present in the repo.

#### Dependencies
- Local Validation and Scoring for structural checks.
- Python `zipfile`.
- Config or CLI argument for student id/output path.

#### Failure Handling
- Missing required artifact fails with a clear path list.
- Invalid generated images fail before zip creation.
- Existing zip requires explicit overwrite.

#### Independent Test Plan
- Package a tiny valid fixture.
- Inspect zip entries for expected layout.
- Verify invalid generated image fixture prevents packaging.
- Verify missing `model.pth` prevents packaging.

#### Open Questions
- Student id is required for final zip naming and must be supplied at packaging time.

### Experiment Plan

#### Responsibility
Own experiment sequencing, run naming, config snapshots, and result summaries. It does not own training math or validation logic.

#### Inputs and Outputs
Inputs:

- Config files.
- Training checkpoints.
- Generated image directories.
- FID and validation reports.
- Visual sample grids.

Outputs:

- Run directories with config, logs, checkpoints, generated samples, and reports.
- A selected final checkpoint/config for submission.

#### Internal Design
Experiments run in the proposal order:

1. Tiny smoke run.
2. Full DDPM baseline with animal + object + pair embeddings.
3. Guidance sweep over `1.0`, `1.5`, `2.0`, `3.0`.
4. DDIM step sweep over 50, 100, 200, 250.
5. Conditioning ablation only if time allows.

Each run stores resolved config, seed, checkpoint path, generation command, validation report, and available metrics.

#### Dependencies
- Training Loop.
- Generation Script.
- Local Validation and Scoring.
- Checkpointing.

#### Failure Handling
- Failed runs keep logs but are not eligible for final packaging.
- Metric absence is reported as unavailable rather than as success.
- Final selection requires valid generated output.

#### Independent Test Plan
- Test experiment metadata writing with a fake run.
- Test that a result summary can read multiple report JSON files.
- Test that invalid validation reports are excluded from final selection.

#### Open Questions
- Full run selection criteria depend on available Codabench attempts and training time.

## Cross-Module Contracts

### Condition Batch Contract
All modules pass condition ids in a structure equivalent to:

```text
{
  "animal_id": LongTensor[B],
  "object_id": LongTensor[B],
  "pair_id": LongTensor[B]
}
```

Null/unconditional conditions use the same keys and batch dimension.

### Image Tensor Contract
Training image tensors use:

```text
shape: [B, 3, 64, 64]
dtype: float32
range: [-1, 1]
```

Generated image tensors are converted to RGB PNG after clamping and denormalizing.

### Model Forward Contract
The UNet forward interface is:

```text
model(x_t, t, conditions) -> epsilon_pred
```

Input and output image tensors have identical shape.

### Checkpoint Contract
Generation requires checkpoint metadata for:

- Model weights or EMA weights.
- Architecture config.
- Diffusion config.
- Condition mappings.
- Seed/config metadata.

Generation must not rebuild mappings from `generate.csv`; it must use checkpoint mappings from training.

### CLI Contract
Scripts expose stable assignment-oriented commands:

```bash
python scripts/train.py --config configs/default.yaml
python scripts/generate.py --checkpoint checkpoints/best.pt --config configs/default.yaml --overwrite
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images
python scripts/package_submission.py --checkpoint model.pth --zip-path HW6_STUDENT_ID.zip
```

Exact flags may grow, but these commands should remain valid.

## Test Strategy
Testing is layered:

1. Unit tests for conditions, datasets, diffusion math, model shape, guidance, EMA, checkpoints, validation, and packaging.
2. CPU smoke tests with tiny channel counts and tiny image fixtures.
3. Script-level smoke tests for train, generate, validate, and package using temporary directories.
4. Local FID execution after real generation when reference stats and dependencies are available.
5. Visual sample grids for semantic alignment and collapse checks.

Suggested quality commands after implementation:

```bash
python -m pytest
python -m compileall src scripts tests
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images
```

`ruff` can be added if the project chooses to include it in `requirements.txt` or development tooling.

## Risks and Mitigations
- Risk: diffusion training may be slow on limited hardware. Mitigation: DDIM sampling, configurable model width, mixed precision, gradient accumulation, and smoke tests before full runs.
- Risk: high guidance improves CLIP-T but worsens FID. Mitigation: sweep guidance scales and select with local FID plus visual review.
- Risk: condition mappings drift between train and generate. Mitigation: save mappings in checkpoints and require generation to load them.
- Risk: output format mistakes cause score loss. Mitigation: validation gates before scoring and packaging.
- Risk: local CLIP-T cannot exactly match Codabench without hidden metadata. Mitigation: treat local CLIP checks as proxy-only and rely on official submissions for final CLIP-T.
- Risk: README currently describes files not yet implemented. Mitigation: update README after implementation to match actual commands and artifacts.

## Open Questions
- What GPU and training-time budget are available for full runs?
- What student id should packaging use for the final `HW6_{student_id}.zip` filename?
- Should optional `open_clip` proxy scoring be included in the first implementation dependencies, or kept out until needed?
- Should random horizontal flip be enabled after visual inspection, or kept disabled for the first full training run?
