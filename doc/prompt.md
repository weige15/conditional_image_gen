# Vibe Coding Implementation Prompt

## Objective

Implement the HW6 Brainrot Image Generation project end to end as a from-scratch conditional DDPM/DDIM generator. The finished repository must train a pixel-space conditional diffusion model on the provided Brainrot dataset and provide scripts to generate exactly 2,000 RGB 64x64 PNG files from `dataset/generate.csv`.

The main generator must be trained from scratch. Do not use pretrained UNet, Transformer, diffusion, GAN, or other pretrained generative weights. Do not use Diffusers pipelines or high-level pretrained generation/training flows. Pretrained Inception/CLIP-style models may be used only for evaluation/proxy scoring, consistent with the assignment.

## Inputs

Read these files first:

- `Brainrot_Image_Gen.pdf`: assignment rules, scoring, output format, and submission structure.
- `doc/proposal.md`: selected algorithm and project assumptions.
- `doc/detailed-design.md`: module boundaries, contracts, failure handling, and test strategy.
- `doc/tasks/progress.md`: overall module checklist.
- `doc/tasks/*.md`: concrete implementation tasks per module.
- `README.md`: intended command surface, but verify against actual implementation because the described code does not yet exist.
- `scoring_program/score.py`: local course-compatible FID/CLIP scorer behavior.
- `hw6_reference/config.json`, `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`: local FID resources.
- `dataset/train.csv`, `dataset/generate.csv`, and `dataset/trainset/`: assignment data.

## Current Implementation

The current repository is mostly an assignment bundle plus planning docs. It contains:

- Dataset files under `dataset/`.
- Reference FID statistics under `hw6_reference/`.
- Codabench-style scorer under `scoring_program/score.py`.
- Planning docs under `doc/`.
- Research notes: `research-log.md`, `Brainrot_Image_Gen_implementation_survey.md`, and `findings.md`.
- A README that describes the intended future project layout.

The implementation does not currently contain:

- `src/brainrot_diffusion/`
- `scripts/`
- `configs/`
- `tests/`
- `requirements.txt`
- `pyproject.toml`

Create the missing implementation. Prefer a simple pip/`requirements.txt` project because the assignment requires `requirements.txt`. Do not introduce `uv` as a hard dependency unless there is a clear reason.

## Execution Model

Work autonomously to completion. The main agent owns the whole implementation, tracks progress in `doc/tasks/progress.md`, decomposes work by module, spawns subagents for independent modules where useful, integrates their results, resolves conflicts, and finishes with the actual repository quality gates passing.

Do not pause for human-in-the-loop checkpoints unless blocked by missing external information that cannot be safely assumed. Make conservative implementation decisions consistent with `doc/proposal.md` and `doc/detailed-design.md`.

When using subagents:

- Give each subagent a disjoint write scope.
- Tell each subagent to read the relevant `doc/tasks/<module>.md` and the contracts in `doc/detailed-design.md`.
- Tell subagents not to revert unrelated changes and to adapt to concurrent edits.
- Integrate subagent work through the main agent and run the full test suite after integration.

Update each module task file and `doc/tasks/progress.md` only when the corresponding checklist items are actually complete.

## Module Plan

### Workstream 1: Project Scaffold and Data

Ownership:

- `requirements.txt`
- `src/brainrot_diffusion/__init__.py`
- `src/brainrot_diffusion/config.py`
- `src/brainrot_diffusion/conditions.py`
- `src/brainrot_diffusion/data.py`
- Data-related tests under `tests/`

Tasks:

- Implement config loading and validation for paths, model, diffusion, optimizer, EMA, sampling, and logging settings.
- Implement deterministic animal/object/pair/null mappings.
- Implement `BrainrotTrainDataset`, `GenerationRequestDataset`, and generation-request loading.
- Load images as RGB 64x64 tensors normalized to `[-1, 1]`.
- Add isolated tests for mappings, image loading, missing files, missing columns, duplicate generation ids, and unknown labels.

Task files:

- `doc/tasks/data-and-labels.md`
- config task from `doc/tasks/training-loop.md`

### Workstream 2: Model, Diffusion, and Guidance

Ownership:

- `src/brainrot_diffusion/unet.py`
- `src/brainrot_diffusion/diffusion.py`
- `src/brainrot_diffusion/guidance.py`
- Model/diffusion/guidance tests under `tests/`

Tasks:

- Implement a from-scratch residual conditional UNet for `[B, 3, 64, 64] -> [B, 3, 64, 64]`.
- Include sinusoidal timestep embeddings and learned animal/object/pair/null condition embeddings.
- Inject conditioning into residual blocks using FiLM-style scale/shift or the documented additive fallback.
- Implement cosine DDPM schedule, epsilon-prediction MSE loss, `q_sample`, DDPM reverse helper, and DDIM step helper.
- Implement classifier-free condition dropout and guided epsilon combination.
- Add CPU tests with tiny channel/model settings.

Task files:

- `doc/tasks/conditional-unet.md`
- `doc/tasks/diffusion-objective.md`
- `doc/tasks/classifier-free-guidance.md`

### Workstream 3: Checkpointing and Training

Ownership:

- `src/brainrot_diffusion/ema.py`
- `src/brainrot_diffusion/checkpoint.py`
- `src/brainrot_diffusion/train_loop.py`
- `configs/default.yaml`
- `scripts/train.py`
- EMA/checkpoint/training tests under `tests/`

Tasks:

- Implement EMA shadow-parameter tracking and application.
- Implement checkpoint save/load/export with model, EMA, optimizer, step, epoch, config, condition mappings, diffusion metadata, architecture metadata, and seed metadata.
- Implement `train(config)` with seeding, dataloader, UNet, diffusion, AdamW, condition dropout, EMA updates, checkpoint cadence, optional resume, and finite-loss checks.
- Implement `scripts/train.py` as a thin CLI.
- Add a CPU smoke training test with a tiny image fixture and `max_steps=2`.

Task files:

- `doc/tasks/ema-and-checkpointing.md`
- `doc/tasks/training-loop.md`

### Workstream 4: Sampling and Generation

Ownership:

- `src/brainrot_diffusion/sample.py`
- `scripts/generate.py`
- Sampling/generation tests under `tests/`

Tasks:

- Implement DDPM and DDIM samplers.
- Apply classifier-free guidance inside reverse steps.
- Convert generated tensors to valid RGB PNG data.
- Implement `scripts/generate.py` to load config/checkpoint/EMA/mappings, read `dataset/generate.csv`, and write `generated_images/{id}`.
- Fail before sampling when output files exist without `--overwrite`.
- Add tiny fixture tests that generate two PNGs and verify exact filenames, RGB mode, size, and overwrite behavior.

Task files:

- `doc/tasks/sampling.md`
- `doc/tasks/generation-script.md`

### Workstream 5: Validation, Evaluation, Packaging, and Experiments

Ownership:

- `src/brainrot_diffusion/validate.py`
- `src/brainrot_diffusion/evaluate.py`
- `src/brainrot_diffusion/package.py`
- optional experiment metadata helpers
- `scripts/validate_submission.py`
- `scripts/evaluate.py`
- `scripts/prepare_score_input.py`
- `scripts/package_submission.py`
- validation/evaluation/packaging/experiment tests under `tests/`

Tasks:

- Implement strict submission validation: exact count, exact filenames, PNG readability, RGB mode, and 64x64 size.
- Implement local FID path compatible with `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`, and `scoring_program/score.py`.
- Treat missing optional metric dependencies or hidden CLIP-T metadata as skipped with explicit report fields.
- Implement Codabench-like score input preparation for FID.
- Implement packaging that validates first and then creates the assignment zip with required artifacts.
- Implement experiment metadata helpers or minimal scripts that record config, seed, checkpoint, generation command, validation report, and metric report per run.

Task files:

- `doc/tasks/local-validation-and-scoring.md`
- `doc/tasks/packaging.md`
- `doc/tasks/experiment-plan.md`

### Workstream 6: Documentation and Integration

Ownership:

- `README.md`
- `doc/tasks/*.md`
- integration tests and final cleanup

Tasks:

- Update `README.md` to match the actual implemented commands and paths.
- Keep `doc/tasks/*.md` and `doc/tasks/progress.md` synchronized with completed work.
- Ensure CLI commands match the contracts from `doc/detailed-design.md`.
- Ensure all scripts return useful errors for missing paths or invalid inputs.

## Required Public Interfaces

Implement these package/script surfaces unless a better local implementation requires a small naming adjustment:

```text
src/brainrot_diffusion/conditions.py
  build_condition_mappings(...)
  condition ids: animal_id, object_id, pair_id

src/brainrot_diffusion/data.py
  BrainrotTrainDataset
  GenerationRequestDataset
  load_generation_requests(...)

src/brainrot_diffusion/unet.py
  model(x_t, t, conditions) -> epsilon_pred

src/brainrot_diffusion/diffusion.py
  GaussianDiffusion
  q_sample(...)
  training_loss(...)
  p_sample_ddpm(...)
  ddim_step(...)

src/brainrot_diffusion/guidance.py
  drop_conditions(...)
  make_null_condition_batch(...)
  combine_cfg(...)

src/brainrot_diffusion/ema.py
  EMA helper

src/brainrot_diffusion/checkpoint.py
  save_checkpoint(...)
  load_checkpoint(...)
  export_model_pth(...)

scripts/train.py
scripts/generate.py
scripts/validate_submission.py
scripts/evaluate.py
scripts/prepare_score_input.py
scripts/package_submission.py
```

Stable command targets:

```bash
python scripts/train.py --config configs/default.yaml
python scripts/generate.py --checkpoint checkpoints/best.pt --config configs/default.yaml --overwrite
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images
python scripts/package_submission.py --checkpoint model.pth --zip-path HW6_STUDENT_ID.zip
```

## Testing and Quality Gates

Add comprehensive unit tests and smoke tests. Keep tests fast enough for CPU execution by using tiny model settings and tiny temporary image fixtures.

Minimum required checks before completion:

```bash
python -m compileall src scripts tests
python -m pytest
```

Also run script-level smoke checks where feasible:

```bash
python scripts/validate_submission.py --generate-csv <tiny_generate_csv> --output-dir <tiny_generated_dir> --smoke
```

If you add Ruff or other configured tooling, run the corresponding repository-native checks, for example:

```bash
python -m ruff check .
python -m ruff format --check .
```

Do not invent a lint/type gate and then skip it. If a tool is configured or included as a development dependency, run it and fix issues. If a heavy local FID run is impractical during implementation, keep it behind a script and cover metric plumbing with tests or a tiny mocked path.

## Acceptance Criteria

The implementation is complete when:

- `src/brainrot_diffusion/`, `scripts/`, `configs/`, `tests/`, and `requirements.txt` exist.
- The main generator is implemented from scratch and does not load pretrained generative weights.
- `scripts/train.py` can run a tiny CPU smoke training job and write a checkpoint with EMA and condition mappings.
- `scripts/generate.py` can load a tiny checkpoint and write valid RGB 64x64 PNGs matching a generation CSV.
- `scripts/validate_submission.py` validates exact filename/count/mode/size requirements.
- `scripts/evaluate.py` runs validation first and computes or explicitly skips optional metrics.
- `scripts/package_submission.py` refuses invalid artifacts and can package a valid fixture.
- Unit tests cover every module task file.
- `python -m compileall src scripts tests` passes.
- `python -m pytest` passes.
- README reflects actual setup, train, generate, validate, evaluate, and package commands.
- `doc/tasks/progress.md` marks a module complete only when all checklist items in that module task file are complete.

Full training a competitive model and producing final 2,000 scored images may require GPU time and Codabench attempts. Implement the full path, but use tiny smoke runs for automated verification unless GPU/time is available.

## Constraints and Safety Rules

- No pretrained generative model weights.
- No Diffusers pipelines or high-level pretrained generation/training flows.
- Use only the provided Brainrot Dataset for the first implementation path.
- Keep optional CLIP/open_clip support evaluation-only and clearly labeled as proxy if official hidden metadata is unavailable.
- Generation must use checkpoint-saved condition mappings, not mappings rebuilt from `generate.csv`.
- Do not revert unrelated user changes.
- Do not remove dataset, reference stats, scorer, planning docs, or research notes.
- Keep large generated outputs, checkpoints, reports, and zips out of git unless explicitly requested.

## Uncertainty Protocol

Make these conservative defaults unless blocked:

- Use pip/`requirements.txt`, not uv-only tooling.
- Disable random horizontal flip by default until visual inspection justifies enabling it.
- Keep local CLIP-T proxy optional; official CLIP-T comes from Codabench.
- Use DDIM as the default final sampler and DDPM as a debugging sampler.
- Use config-driven batch size, gradient accumulation, mixed precision, channel width, and sampling steps because GPU budget is unknown.
- Require student id or explicit zip path at packaging time rather than guessing the final `HW6_{student_id}.zip` name.

If genuinely blocked, ask a concise question naming the file/module and the exact decision needed. Otherwise proceed with the conservative assumption, document it in config/README where relevant, and keep coding.
