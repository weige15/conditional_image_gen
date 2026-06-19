# Repository Map

## Repository Summary

- Repository root: `/home/kuotzuwei15/GenAI/hw6`.
- Current project: HW6 Brainrot Image Generation assignment.
- Current implementation state: partial/scratch repository. Documentation, assignment assets, dataset files, reference scoring assets, package metadata, and scorer files are present; no main package source directory was discovered.
- User intent recorded in `AGENTS.md`: start the implementation from scratch.

## Directory Structure

- `.`: top-level metadata and assignment/research docs.
- `doc/`: planning and task documents.
- `doc/tasks/`: task breakdown documents.
- `dataset/`: assignment data files, discovered by `find`.
- `dataset/trainset/`: PNG training images, discovered by `find`.
- `hw6_reference/`: reference FID assets, discovered by `find`.
- `scoring_program/`: provided scorer files.
- `.git/`: Git repository metadata.
- `.pytest_cache/`, `.ruff_cache/`: local tool caches.
- Not discovered in the live tree: `src/`, `scripts/`, `configs/`, `tests/`, `checkpoints/`, `generated_images/`, `reports/`, `score_input/`.

## Main Source Files

- No main implementation source files were discovered.
- `scoring_program/score.py` exists and appears to be a provided/evaluator-side script, not the main project implementation.
- `pyproject.toml` is configured to find packages under `src`, but `src/` is not present.

## Existing Tests

- No `tests/` directory or test files were discovered.
- `.pytest_cache/` exists, but it is a local cache artifact and does not imply tests are present.
- `README.md` documents planned `python -m pytest` usage, but the test suite is not present in the live tree.

## Build System

- `pyproject.toml` uses `setuptools>=68` with `setuptools.build_meta`.
- Project metadata:
  - name: `brainrot-diffusion`
  - version: `0.1.0`
  - Python requirement: `>=3.10`
  - package discovery root: `src`
- `requirements.txt` is present.
- Because `src/` is missing, editable install/build behavior is not yet verifiable.

## Runtime or CLI Entry Points

- No executable project CLI entry points were discovered in the live tree.
- `README.md` documents planned scripts:
  - `scripts/train.py`
  - `scripts/generate.py`
  - `scripts/validate_submission.py`
  - `scripts/evaluate.py`
  - `scripts/prepare_score_input.py`
  - `scripts/package_submission.py`
- The documented `scripts/` directory does not currently exist.
- `scoring_program/score.py` is the only discovered Python script-like file.

## Data and Assets

- Assignment PDF: `Brainrot_Image_Gen.pdf`.
- Dataset files discovered:
  - `dataset/train.csv`
  - `dataset/generate.csv`
  - `dataset/trainset/*.png`
- Reference assets discovered:
  - `hw6_reference/test_mu.npy`
  - `hw6_reference/test_sigma.npy`
  - `hw6_reference/config.json`
- Scoring assets discovered:
  - `scoring_program/score.py`
  - `scoring_program/metadata`
- `dataset/`, `Brainrot_Image_Gen.pdf`, and `hw6_reference/` should be preserved as assignment assets.

## Existing Documentation

- `README.md`: describes a planned from-scratch DDPM/DDIM workflow and commands.
- `AGENTS.md`: repository guardrails created for safe future Codex work.
- `doc/problem-brief.md`: source-grounded assignment brief.
- `doc/proposal.md`: prior proposal for a diffusion-based approach.
- `doc/detailed-design.md`: prior detailed design.
- `doc/prompt.md`: prior implementation prompt.
- `doc/tasks/*.md`: task breakdown documents.
- `doc/tasks/progress.md`: all listed task docs are marked complete, but the corresponding implementation files are absent.
- Other research/support docs:
  - `research-log.md`
  - `research-state.yaml`
  - `Brainrot_Image_Gen_implementation_survey.md`
  - `findings.md`

## Detected Dependencies

From `requirements.txt`:

- `torch>=2.0`
- `torchvision>=0.15`
- `numpy>=1.24`
- `Pillow>=9.0`
- `PyYAML>=6.0`
- `scipy>=1.10`
- `tqdm>=4.65`
- `pytest>=8.0`

From `pyproject.toml`:

- Build backend: `setuptools.build_meta`
- Build requirement: `setuptools>=68`

## Important Scripts

- `scoring_program/score.py`: provided scorer/evaluator script.
- Planned but missing scripts documented by `README.md`:
  - `scripts/train.py`
  - `scripts/generate.py`
  - `scripts/validate_submission.py`
  - `scripts/evaluate.py`
  - `scripts/prepare_score_input.py`
  - `scripts/package_submission.py`

## Current Git State

- Branch: `main`.
- Tracking: `main...origin/main`.
- Worktree path: `/home/kuotzuwei15/GenAI/hw6`.
- Worktree commit shown by `git worktree list`: `d05f0a8`.
- Untracked files at discovery time:
  - `AGENTS.md`
  - `doc/problem-brief.md`
- This generated map, `doc/repo-map.md`, will also appear as untracked after creation unless added to Git.

## Missing or Ambiguous Areas

- Main source implementation is missing: no `src/` package was discovered.
- Documented CLIs are missing: no `scripts/` directory was discovered.
- Config files are missing: no `configs/` directory was discovered.
- Tests are missing: no `tests/` directory was discovered.
- README and planning docs describe an intended implementation, but the live tree does not contain that implementation.
- The scoring script was discovered but not deeply analyzed during this reconnaissance.
- Dataset row counts and image validity were not verified during this reconnaissance.
- The final student ID, deadline, training hardware, and training-time budget remain unknown.

## Notes for Future Skills

- Treat README and planning docs as intended direction, not proof of implemented code.
- Start implementation from the live facts: package metadata and assets exist; source, scripts, configs, and tests must be created.
- Preserve assignment assets and avoid overwriting generated outputs or checkpoints without approval.
- Future proposal/design/implementation work should use `doc/problem-brief.md`, `AGENTS.md`, and this repo map as grounding documents.
- Before running install, build, test, train, generation, or packaging commands, ask for approval as required by `AGENTS.md`.
