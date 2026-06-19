# Implementation Prompt

## Objective

Implement the HW6 Brainrot Image Generation repository end to end. The finished project must train a from-scratch conditional image generation model for animal-object Brainrot images and generate exactly 2,000 RGB `64x64` PNG files from `dataset/generate.csv`.

Use the current design direction: a Python 3.10+/PyTorch pixel-space conditional diffusion baseline under `src/brainrot_diffusion/`, with thin CLI scripts under `scripts/`, reproducible config under `configs/`, and CPU-friendly tests under `tests/`.

## Inputs to Read First

Read these before coding, in this order:

- `AGENTS.md`: repository rules, permission rules, forbidden commands, and assignment constraints.
- `doc/problem-brief.md` (present): assignment objective, inputs, outputs, constraints, scoring, deliverables, and open questions.
- `doc/repo-map.md` (present): current repository state and missing implementation directories.
- `doc/quality-gates.md` (present): discovered commands, missing gates, and command status.
- `doc/proposal.md`: selected from-scratch conditional diffusion approach, milestones, validation plan, risks, and assumptions.
- `doc/high-level-design.md`: module boundaries, data flow, contracts, and operational constraints.
- `doc/test-plan.md`: required verification strategy, unit/integration scope, golden cases, edge cases, and known commands.
- `doc/detailed-design.md`: module responsibilities, public contracts, algorithms, failure handling, and test strategy mapping.
- `doc/tasks/progress.md`: current module checklist.
- `doc/tasks/configuration.md`
- `doc/tasks/data-loading.md`
- `doc/tasks/conditions.md`
- `doc/tasks/generator-model.md`
- `doc/tasks/diffusion.md`
- `doc/tasks/ema.md`
- `doc/tasks/checkpointing.md`
- `doc/tasks/training-loop.md`
- `doc/tasks/sampling.md`
- `doc/tasks/validation.md`
- `doc/tasks/evaluation.md`
- `doc/tasks/packaging.md`
- `doc/tasks/cli-scripts.md`
- `doc/tasks/tests.md`
- `README.md`: planned command surface; treat it as intent until implementation exists.
- `pyproject.toml`: `setuptools` project metadata, package discovery under `src`.
- `requirements.txt`: current dependency list.
- `scoring_program/score.py` and `scoring_program/metadata`: provided evaluator behavior.

## Current Implementation

Current repo root: `/home/kuotzuwei15/GenAI/hw6`.

Present:

- Assignment assets: `Brainrot_Image_Gen.pdf`, `dataset/train.csv`, `dataset/generate.csv`, `dataset/trainset/`.
- Reference/evaluator assets: `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`, `hw6_reference/config.json`, `scoring_program/score.py`, `scoring_program/metadata`.
- Packaging metadata: `pyproject.toml`, `requirements.txt`.
- Planning docs: `doc/problem-brief.md`, `doc/repo-map.md`, `doc/quality-gates.md`, `doc/proposal.md`, `doc/high-level-design.md`, `doc/test-plan.md`, `doc/detailed-design.md`, `doc/tasks/*.md`.
- README with planned setup/train/generate/validate/evaluate/package commands.

Missing and must be created:

- `src/brainrot_diffusion/`
- `scripts/`
- `configs/`
- `tests/`
- `checkpoints/`, `generated_images/`, `reports/`, and `score_input/` only when commands legitimately generate them.

No main package modules, CLI scripts, config files, or tests currently exist. Do not trust older planning notes or stale task checkboxes over live files. `doc/tasks/progress.md` currently marks all current module tasks incomplete.

Current dependency files:

- `requirements.txt`: `torch`, `torchvision`, `numpy`, `Pillow`, `PyYAML`, `scipy`, `tqdm`, `pytest`.
- `pyproject.toml`: project name `brainrot-diffusion`, Python `>=3.10`, package discovery under `src`.

## Hard Constraints

- Main generator must be trained from scratch.
- Do not use pretrained UNet, Transformer, diffusion, GAN, or other pretrained generative weights for the main generator.
- Do not use Diffusers pipelines or high-level pretrained generation/training flows.
- Use PyTorch directly for model, loss, scheduler, training loop, and sampling loop.
- Pretrained CLIP or VAE may be used only as auxiliary evaluation, feature extraction, or latent representation modules. Ask the user before adding either.
- First implementation path uses only the provided Brainrot Dataset.
- Final generated output must contain exactly 2,000 PNG files.
- Final generated images must be RGB, `64x64`, and named exactly according to `dataset/generate.csv`.
- Generation must use checkpoint-saved condition mappings, not mappings rebuilt from `dataset/generate.csv`.
- Checkpoints must preserve enough metadata to reproduce generation: config, step, EMA state if used, condition mappings, diffusion metadata, architecture metadata, and seed metadata.
- Preserve assignment assets: `Brainrot_Image_Gen.pdf`, `dataset/`, `hw6_reference/`, and `scoring_program/`.
- Do not overwrite generated images, checkpoints, reports, submissions, or package zips unless explicitly requested or an explicit overwrite flag is used.
- Do not commit or push unless the user asks.
- Follow `AGENTS.md`: ask before installs, tests, build commands, training, generation, packaging, scoring, file deletion, Git state changes, and other mutating/expensive commands.

## Non-Goals

- Do not optimize for leaderboard score before the train/generate/validate/package path works.
- Do not add extra data workflows in the first pass.
- Do not make CLIP/VAE part of final-image generation.
- Do not implement a separate `guidance.py` module unless the implementation clearly needs it; the current detailed design assigns guidance behavior across Conditions, Generator Model, Training Loop, Sampling, and Diffusion.
- Do not add lint/format/type-check tools unless explicitly chosen. Current docs mark those gates as missing.
- Do not run official scorer paths as unit tests; `scoring_program/score.py` uses CUDA and pretrained evaluation models.

## Execution Model

- Start by running read-only discovery: `git status --short --branch`, `rg --files`, and targeted `sed` reads of the docs above.
- Maintain `doc/tasks/progress.md` throughout the implementation. Add timestamped notes or checkpoint entries when a module starts, completes, is blocked, or is verified.
- Implement one module or small workstream at a time.
- After each module, run the smallest relevant module-specific check if the user has approved test/build execution in this session.
- Run full quality gates at the end after implementation exists and permission is approved.
- Summarize command output as evidence; the user does not see raw tool output.
- Keep write scopes disjoint when using subagents.
- Never revert edits made by the user or other agents unless explicitly asked.
- Stop and ask only when truly blocked by missing requirements, destructive choices, credentials, external services, or an assignment-rule ambiguity that cannot be handled conservatively.

## Module Workstreams

### Workstream A: Scaffold, Config, Data, Conditions

Owns:

- `src/brainrot_diffusion/__init__.py`
- `src/brainrot_diffusion/config.py`
- `src/brainrot_diffusion/data.py`
- `src/brainrot_diffusion/conditions.py`
- `configs/default.yaml`
- `tests/test_config.py`
- `tests/test_data.py`
- `tests/test_conditions.py`
- `doc/tasks/configuration.md`
- `doc/tasks/data-loading.md`
- `doc/tasks/conditions.md`

Implement:

- Config loading, CLI override application, validation, and serializable resolved config.
- `configs/default.yaml` with sections for data, model, diffusion, training, sampling, checkpointing, validation, evaluation, and packaging.
- Train/generate CSV parsing with required-column and duplicate-ID validation.
- RGB image loading to `[3, 64, 64]` tensors normalized to `[-1, 1]`.
- Ordered generation request records.
- Stable animal/object/pair mappings, checkpoint-compatible serialization, unknown-label rejection, and optional null-condition IDs.

Verify:

- `python -m pytest tests/test_config.py tests/test_data.py tests/test_conditions.py`

### Workstream B: Model, Diffusion, EMA, Checkpointing

Owns:

- `src/brainrot_diffusion/model.py`
- `src/brainrot_diffusion/diffusion.py`
- `src/brainrot_diffusion/ema.py`
- `src/brainrot_diffusion/checkpoint.py`
- `tests/test_model.py`
- `tests/test_diffusion.py`
- `tests/test_ema.py`
- `tests/test_checkpoint.py`
- `doc/tasks/generator-model.md`
- `doc/tasks/diffusion.md`
- `doc/tasks/ema.md`
- `doc/tasks/checkpointing.md`

Implement:

- From-scratch compact UNet-style PyTorch model with timestep and animal/object/pair condition embeddings.
- Forward contract: `model(x_t, timesteps, conditions) -> predicted_noise`, output shape `[B, 3, 64, 64]`.
- Diffusion schedules, `q_sample`-style noising, epsilon-prediction MSE loss, and DDPM/DDIM reverse helpers.
- EMA helper with update, disabled no-op path, state serialization, and state validation.
- Checkpoint save/load/schema validation with required metadata: `model`, `config`, `condition_mappings`, `diffusion`, `architecture`, `seed`, and `step`; optional `ema`, `optimizer`, `epoch`, `metrics`.

Verify:

- `python -m pytest tests/test_model.py tests/test_diffusion.py tests/test_ema.py tests/test_checkpoint.py`

### Workstream C: Training And Sampling

Owns:

- `src/brainrot_diffusion/train_loop.py`
- `src/brainrot_diffusion/sample.py`
- training/sampling pieces of `scripts/train.py` and `scripts/generate.py`
- `tests/test_train_loop.py`
- `tests/test_sample.py`
- `doc/tasks/training-loop.md`
- `doc/tasks/sampling.md`

Implement:

- Fresh training entry point using resolved config and assignment dataset paths.
- Seeds for Python, NumPy, and PyTorch where practical; record seed metadata.
- Dataloader, model, diffusion, optimizer, EMA, finite-loss checks, logging, and checkpoint cadence.
- Optional resume validation when included.
- Checkpoint-backed generation that loads config/mappings/model/diffusion/EMA, validates generation labels, seeds sampling, batches requests, and writes exact CSV filenames.
- DDIM/DDPM sampling path with guidance scale handling if model/checkpoint supports classifier-free guidance.
- Refusal to overwrite existing outputs without explicit `--overwrite`.

Verify:

- `python -m pytest tests/test_train_loop.py tests/test_sample.py`

### Workstream D: Validation, Evaluation, Packaging

Owns:

- `src/brainrot_diffusion/validate.py`
- `src/brainrot_diffusion/evaluate.py`
- `src/brainrot_diffusion/package.py`
- validation/evaluation/packaging pieces of `scripts/validate_submission.py`, `scripts/evaluate.py`, `scripts/prepare_score_input.py`, `scripts/package_submission.py`
- `tests/test_validate.py`
- `tests/test_evaluate.py`
- `tests/test_package.py`
- `doc/tasks/validation.md`
- `doc/tasks/evaluation.md`
- `doc/tasks/packaging.md`

Implement:

- Strict validation: exact expected filename set, no extras/missing files, PNG readability, PNG format, RGB mode, and `64x64`.
- JSON-compatible validation report data.
- Evaluation that runs validation first and then computes available local FID or records explicit skip reasons.
- Scorer input preparation if needed by the planned CLI.
- Packaging that validates first, rejects placeholder student ID, verifies required artifacts, refuses overwrite unless explicit, and writes `HW6_{student_id}.zip`.

Verify:

- `python -m pytest tests/test_validate.py tests/test_evaluate.py tests/test_package.py`

### Workstream E: CLI Scripts, Tests, Docs Integration

Owns:

- `scripts/train.py`
- `scripts/generate.py`
- `scripts/validate_submission.py`
- `scripts/evaluate.py`
- `scripts/prepare_score_input.py`
- `scripts/package_submission.py`
- `tests/test_scripts.py`
- shared test fixtures/helpers under `tests/`
- `README.md`
- `doc/tasks/cli-scripts.md`
- `doc/tasks/tests.md`
- `doc/tasks/progress.md`

Implement:

- Thin `argparse` scripts that call package modules and return nonzero on invalid input or failed validation.
- CPU-friendly shared test fixtures for tiny CSVs, images, checkpoints, and output directories.
- A tiny integration path for train -> checkpoint -> generate -> validate.
- README updates so documented commands match implemented flags and paths.
- Progress tracker updates only when work is actually complete and verified.

Verify:

- `python -m pytest tests/test_scripts.py`
- `python -m pytest`
- `python -m compileall src scripts tests`

## Subagent Plan

Use subagents only after the main agent has created the package scaffold and agreed on shared contracts for config, condition batch structure, checkpoint schema, and test fixtures.

Good subagent candidates:

- Data/conditions/config subagent: may edit only Workstream A files.
- Model/diffusion/EMA/checkpoint subagent: may edit only Workstream B files.
- Validation/evaluation/packaging subagent: may edit only Workstream D files.
- Tests/docs subagent: may edit only `tests/`, `README.md`, and `doc/tasks/progress.md` after implementation interfaces settle.

Main-agent-only or integration-sensitive work:

- Initial scaffold and public contracts.
- Training Loop and Sampling integration.
- CLI script final wiring.
- Full test suite fixes.
- README final command reconciliation.
- Final progress tracker status.

Shared files:

- `configs/default.yaml`, `tests/` fixtures, and `doc/tasks/progress.md` can create conflicts. If subagents touch them, the main agent must merge deliberately and rerun relevant checks.

Subagent rules:

- Give each subagent one task file and its write scope.
- Tell subagents to read `doc/detailed-design.md`, relevant `doc/tasks/<module>.md`, and current files before editing.
- Require subagents to report changed files and checks run.
- Do not accept subagent changes that violate assignment constraints or bypass tests.

## Implementation Order

1. Read required docs and check current tree.
   - Local check: `git status --short --branch`, `rg --files`.
2. Scaffold package/test directories and config.
   - Files: `src/brainrot_diffusion/__init__.py`, `configs/default.yaml`, `tests/`.
   - Check: `python -m compileall src` after code exists.
3. Implement Configuration.
   - Check: `python -m pytest tests/test_config.py`.
4. Implement Data Loading and Conditions.
   - Check: `python -m pytest tests/test_data.py tests/test_conditions.py`.
5. Implement Generator Model and Diffusion.
   - Check: `python -m pytest tests/test_model.py tests/test_diffusion.py`.
6. Implement EMA and Checkpointing.
   - Check: `python -m pytest tests/test_ema.py tests/test_checkpoint.py`.
7. Implement Training Loop with tiny CPU smoke path.
   - Check: `python -m pytest tests/test_train_loop.py`.
8. Implement Sampling/generation logic.
   - Check: `python -m pytest tests/test_sample.py`.
9. Implement Validation.
   - Check: `python -m pytest tests/test_validate.py`.
10. Implement Evaluation and Packaging.
   - Check: `python -m pytest tests/test_evaluate.py tests/test_package.py`.
11. Implement CLI scripts and script tests.
   - Check: `python -m pytest tests/test_scripts.py`.
12. Add/verify tiny integration path: train -> checkpoint -> generate -> validate.
   - Check: `python -m pytest`.
13. Update README and task progress.
   - Check: README commands match implemented scripts.
14. Run final quality gates after approval.
   - Check: `python -m compileall src scripts tests`, `python -m pytest`, and applicable validation/evaluation/package commands when outputs exist.

## Testing and Quality Gates

Known setup commands, not verified:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Known core gates after implementation files exist:

```bash
python -m compileall src scripts tests
python -m pytest
```

Known structural validation command after generated images exist:

```bash
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json
```

Known local evaluation command after generated images exist:

```bash
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json
```

Known scorer input command after generated images exist:

```bash
python scripts/prepare_score_input.py --generate-csv dataset/generate.csv --generated-images generated_images --score-input-dir score_input --test-mu hw6_reference/test_mu.npy --test-sigma hw6_reference/test_sigma.npy --scores fid --overwrite
```

Known packaging command before final submission, with real student ID:

```bash
python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite
```

Known Codabench-style scorer command, not a unit-test gate:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

Notes:

- `scoring_program/score.py` hardcodes `cuda:0` and uses pretrained evaluation models, so keep it separate from core tests.
- No lint, format, or type-check command is configured. Do not claim those gates passed unless tooling is added and run.
- Any command that installs dependencies, runs tests, trains, generates, writes reports, packages, scores, or mutates files requires permission under `AGENTS.md`.

## Progress Tracking

Maintain `doc/tasks/progress.md` throughout implementation.

Required progress behavior:

- Mark a module started when coding begins.
- Mark tasks complete only after the corresponding code and tests are in place.
- Record blocked items with the exact missing decision, command failure, or external requirement.
- Record verification with command names and short output summaries.
- Update progress after each module or small workstream.
- Do not mark full-project gates complete until those commands actually run and pass.

If a module task file needs adjustment because implementation revealed a better boundary, update the task file and explain the reason in the final response.

## Commit or Checkpoint Strategy

Do not commit unless the user asks. If commits are requested, commit in logical groups:

1. Scaffold/config/data/conditions.
2. Model/diffusion/EMA/checkpointing.
3. Training/sampling.
4. Validation/evaluation/packaging.
5. CLI/tests/docs.

Without commits, keep the final diff grouped by workstream in the final response. Do not use destructive Git commands. Do not revert unrelated user changes.

Generated artifacts, checkpoints, reports, final images, scorer inputs, and zip submissions should stay out of Git unless the user explicitly wants them tracked.

## Acceptance Criteria

Implementation is complete when:

- `src/brainrot_diffusion/` exists with modules from `doc/detailed-design.md`.
- `configs/default.yaml` exists and is used by train/generate flows.
- `scripts/` contains train, generate, validate, evaluate, scorer-input preparation, and packaging CLIs.
- `tests/` contains CPU-friendly unit and integration tests covering every current module.
- Main generator is trained from scratch and does not load pretrained generative weights.
- Tiny CPU training writes a checkpoint with required metadata.
- Tiny checkpoint-backed generation writes exact CSV-matching RGB `64x64` PNGs.
- Validation rejects missing, extra, corrupt, wrong-mode, wrong-format, and wrong-size outputs.
- Evaluation validates first and reports metric skips explicitly when resources are unavailable.
- Packaging validates first and rejects placeholder student ID or missing artifacts.
- README matches actual implemented commands.
- `doc/tasks/progress.md` reflects completed and verified work.
- `python -m compileall src scripts tests` passes.
- `python -m pytest` passes.
- Structural validation passes for final `generated_images/` before scoring/packaging.
- No unrelated files are changed.

Final submission readiness additionally requires:

- Exactly 2,000 generated PNG files matching `dataset/generate.csv`.
- All final images are RGB and `64x64`.
- `model.pth` or selected checkpoint can reproduce generation.
- Final `HW6_{student_id}.zip` uses a real student ID and contains required artifacts.
- Any auxiliary pretrained evaluation usage is documented and does not affect final-image generation.

## Uncertainty Protocol

Make conservative, documented assumptions when safe:

- Use pip/`requirements.txt`; do not introduce another package manager as a requirement.
- Keep pair mappings stable and fail clearly for unsupported generation pairs.
- Treat generation prompt mismatch as a warning unless strict validation is chosen and documented.
- Keep local CLIP-T proxy disabled/skipped unless the user explicitly approves adding CLIP support.
- Use DDIM as the default final sampler after the DDPM objective works; keep DDPM useful for debugging if implemented.
- Keep model width, batch size, gradient accumulation, mixed precision, EMA decay, sampling steps, and guidance scale config-driven because hardware budget is unknown.
- Require real student ID at packaging time; do not guess it.

Ask the user only when blocked by:

- A risky assignment-rule interpretation.
- Adding pretrained CLIP/VAE or extra data.
- Running expensive training/generation/scoring.
- Installing dependencies or needing network access.
- Overwriting generated images, checkpoints, reports, or submissions.
- Missing credentials, external services, or student ID.
- A destructive Git or filesystem action.

## Final Response Requirements

When implementation is finished, respond concisely with:

- Implementation summary grouped by workstream.
- Changed files grouped by module/workstream.
- Tests and quality gates run, with command output summaries.
- Commands not run and why.
- Generated artifacts created, if any.
- Known limitations and open questions.
- Follow-up required before full training, Codabench upload, or final E3 packaging.

Do not claim a command passed unless it was actually run. Do not claim final submission readiness unless the 2,000 generated images and packaging gates have been validated.
