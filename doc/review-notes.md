# Review Notes

## Review Scope

- Full implementation review for HW6 Brainrot Image Generation after all task modules were marked complete.
- Reviewed requirements, design docs, task checklists, package modules, CLI scripts, tests, dataset counts, and evaluator-facing paths.
- Performed scoped repair only for small, clear issues in generation overwrite handling and scorer-input preparation.

## Files Reviewed

- Requirements/design: `doc/problem-brief.md`, `doc/proposal.md`, `doc/high-level-design.md`, `doc/test-plan.md`, `doc/detailed-design.md`, `doc/tasks/*.md`, `README.md`, `AGENTS.md`.
- Config/data/conditions: `configs/default.yaml`, `src/brainrot_diffusion/config.py`, `src/brainrot_diffusion/data.py`, `src/brainrot_diffusion/conditions.py`.
- Model/training/generation: `src/brainrot_diffusion/model.py`, `src/brainrot_diffusion/diffusion.py`, `src/brainrot_diffusion/ema.py`, `src/brainrot_diffusion/checkpoint.py`, `src/brainrot_diffusion/train_loop.py`, `src/brainrot_diffusion/sample.py`.
- Validation/evaluation/package: `src/brainrot_diffusion/validate.py`, `src/brainrot_diffusion/evaluate.py`, `src/brainrot_diffusion/package.py`, `scoring_program/score.py`.
- CLI/tests: `scripts/*.py`, `tests/*.py`, `pyproject.toml`, `requirements.txt`.

## Commands Run

- `git status --short --branch`: pass; repo was clean before repair, now shows reviewed repair files plus this notes file.
- `git diff --stat`: pass; no pre-existing unstaged diff before repair.
- `git diff --cached --stat`: pass; no staged diff before repair.
- `rg --files`: pass; implementation, scripts, configs, tests, docs, dataset, scoring assets found.
- `wc -l dataset/generate.csv dataset/train.csv`: pass; `dataset/generate.csv` has 2,001 lines including header, `dataset/train.csv` has 4,800 lines including header.
- `find dataset/trainset -maxdepth 1 -type f | wc -l`: pass; 4,799 training image files.
- `rg -n "open_clip|inception_v3|from_pretrained|diffusers|DiffusionPipeline|StableDiffusion|transformers|AutoModel|pipeline\(" src scripts configs requirements.txt pyproject.toml README.md`: pass; no forbidden pretrained generation or Diffusers usage in main implementation.
- `python -m pytest tests/test_sample.py tests/test_evaluate.py tests/test_scripts.py`: pass; 11 passed.
- `python -m pytest`: pass; 53 passed.
- `python -m compileall src scripts tests`: pass; compile completed.
- `git diff --check`: pass; no whitespace errors.

## Summary

- The implementation matches the planned direct PyTorch, from-scratch conditional DDPM/DDIM baseline at the code and test level.
- Fixed three localized issues found during review.
- The code is ready for real training/generation attempts, but the final submission is not ready because `generated_images/` and `model.pth` are not present.

## Requirement Match

- Main generator is implemented in PyTorch under `src/brainrot_diffusion/model.py` and does not load pretrained generative weights.
- Data loading validates assignment CSV columns, duplicate IDs, filenames without directories, image RGB conversion, and `64x64` tensors.
- Condition mappings are stable, checkpointed, restored during generation, and unknown labels fail before sampling.
- Checkpoints preserve model, config, condition mappings, diffusion metadata, architecture metadata, seed metadata, and step.
- Validation enforces exact CSV filename set, no extras, PNG readability/format, RGB mode, and `64x64` size.
- Packaging validates generated outputs first and includes `generated_images/`, `scripts/`, `src/brainrot_diffusion/`, `configs/`, `model.pth`, `README.md`, and `requirements.txt`.

## Module Boundary Check

- Repairs stayed within existing module ownership: sampling preflight in `sample.py`, scorer-input preflight in `evaluate.py`, CLI parsing in `scripts/generate.py`, and focused tests.
- Scripts remain thin wrappers around package modules.
- No dependencies, model architecture, checkpoint schema, or assignment assets were changed.

## Test Coverage Check

- Unit and smoke tests cover config, data, conditions, model shape, diffusion math, EMA, checkpointing, training smoke, sampling, validation, evaluation skips, packaging, and CLI flow.
- Added regression coverage for non-empty generation output dirs, scorer-input overwrite preflight, and generate parser overwrite default.
- Full suite is CPU-friendly and passed locally.
- Full GPU training quality, official Codabench scoring, and visual inspection remain manual/final-run checks.

## Edge Cases

- Covered: missing/duplicate/empty CSV rows, bad IDs, missing/unreadable images, strict prompt mismatch, unknown labels, invalid condition IDs, corrupt PNGs, wrong image size/mode, missing/extra generated files, invalid package artifacts, placeholder student ID.
- Fixed: generation now refuses unrelated leftover files in the output directory unless overwrite is explicit.
- Fixed: scorer-input preparation now checks reference files before deleting an existing output directory.

## Performance Concerns

- Default training config is GPU-oriented (`batch_size: 256`, `num_workers: 16`, `max_steps: 100000`) and may be too heavy for CPU.
- Final DDIM generation of 2,000 images with 100 steps should be run on GPU if possible.
- Local evaluator wrapper does not compute real FID/CLIP-T; official scorer/Codabench remains the metric authority.

## Bugs Found

- fixed: `src/brainrot_diffusion/evaluate.py:67` deleted an existing scorer-input directory on `overwrite=True` before confirming scorer reference files existed.
- fixed: `src/brainrot_diffusion/sample.py:155` allowed generation into a non-empty output directory if existing files were not requested filenames, leaving invalid extra files.
- fixed: `scripts/generate.py:24` parsed omitted `--overwrite` as `False`, overriding `sampling.overwrite` from config instead of preserving the config default.

## Repairs Made

- fixed: Added scorer reference preflight in `src/brainrot_diffusion/evaluate.py:66` before overwrite deletion; verified by `tests/test_evaluate.py:78`.
- fixed: Added output path/non-empty directory checks in `src/brainrot_diffusion/sample.py:158`; verified by `tests/test_sample.py:99`.
- fixed: Changed generate CLI `--overwrite` default to `None` in `scripts/generate.py:24`; verified by `tests/test_scripts.py:167`.

## Remaining Issues

- unresolved: Final `generated_images/` output is absent, so assignment validation/evaluation/package commands for the real 2,000-image submission were not run.
- unresolved: `model.pth` is absent, so final E3 packaging cannot be completed yet.
- unresolved: Real model quality is unknown until full training, visual review, and Codabench scoring are performed.
- unresolved: Some planning docs still include pre-implementation wording that `src/`, `scripts/`, `configs/`, and `tests/` are missing; live files and `doc/tasks/progress.md` reflect the implemented state.
- needs-user-decision: Real student ID is required before final packaging.
- needs-user-decision: GPU/training budget and final checkpoint selection are still open.

## Recommended Next Steps

- Train a real checkpoint with an approved GPU run.
- Generate `generated_images/` from the selected checkpoint and validate with `python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json --expected-count 2000`.
- Export or copy the selected checkpoint to `model.pth`.
- Prepare scorer input or upload to Codabench for actual metrics.
- Package with the real student ID after generated images and `model.pth` exist.

## Final Readiness

- Not ready.
- Implementation checks pass, but final trained artifacts and the 2,000-image validated submission are not present.
