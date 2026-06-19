# Quality Gates

## Environment Summary

- Repository root: `/home/kuotzuwei15/GenAI/hw6`.
- Project type: Python project metadata with `setuptools`.
- Python requirement from `pyproject.toml`: `>=3.10`.
- Package discovery root from `pyproject.toml`: `src`.
- Dependency file: `requirements.txt`.
- Current live tree is scratch/partial: `src/`, `scripts/`, `configs/`, and `tests/` were not discovered.
- Assignment/evaluator assets discovered: `scoring_program/score.py`, `scoring_program/metadata`, `hw6_reference/config.json`, `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`.
- No build, test, lint, format, type-check, static-analysis, benchmark, or evaluator command was run in this session.

## Build Commands

- Discovered, not verified, setup/install command from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python -m pip install -r requirements.txt`
  - Notes: package-manager command; may require network and writes to the active Python environment.
- Discovered, not verified, editable install command from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python -m pip install -e .`
  - Notes: package-manager/build setup command; writes to the active Python environment. `pyproject.toml` points package discovery at missing `src/`, so current behavior is unverified.
- Missing: no explicit package artifact build command was discovered.

## Unit Test Commands

- Discovered, not verified, from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python -m pytest`
  - Notes: `tests/` was not discovered, so this is documented but not currently backed by visible tests.

## Integration Test Commands

- Discovered, not verified, structural submission validation from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json`
  - Notes: writes a report; requires generated images. `scripts/validate_submission.py` and `generated_images/` were not discovered.
- Discovered, not verified, packaging validation path from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite`
  - Notes: writes a submission zip and may overwrite package output. Requires generated images, model weights, and the missing script.

## Lint Commands

- Missing: no lint command was discovered in `pyproject.toml`, README, CI files, task runners, or scripts.
- Note: `.ruff_cache/` exists, but no Ruff config or repo-defined Ruff command was discovered.

## Format Commands

- Missing: no format command was discovered in `pyproject.toml`, README, CI files, task runners, or scripts.

## Type-Check Commands

- Missing: no type-check command was discovered in `pyproject.toml`, README, CI files, task runners, or scripts.

## Static Analysis Commands

- Discovered, not verified, syntax/import bytecode compilation gate from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python -m compileall src scripts tests`
  - Notes: `src/`, `scripts/`, and `tests/` were not discovered, so this command is planned but not currently runnable as documented.

## Benchmark or Evaluator Commands

- Discovered, not verified, local evaluation wrapper from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json`
  - Notes: writes an evaluation report; requires generated images and missing `scripts/evaluate.py`.
- Discovered, not verified, scorer input preparation from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python scripts/prepare_score_input.py --generate-csv dataset/generate.csv --generated-images generated_images --score-input-dir score_input --test-mu hw6_reference/test_mu.npy --test-sigma hw6_reference/test_sigma.npy --scores fid --overwrite`
  - Notes: writes `score_input/`; requires generated images and missing `scripts/prepare_score_input.py`.
- Discovered, not verified, Codabench-style scorer command from `scoring_program/metadata`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6/scoring_program`
  - Command: `python3 score.py --input_dir $input --output_dir $output --config config.json`
  - Notes: writes `scores.json` under `$output`; expects Codabench-style `$input/ref` and `$input/res` directories. `score.py` hardcodes `cuda:0`, uses pretrained Inception, and may require GPU, generated images, hidden/reference resources, and cached or downloadable model weights.

## Smoke Test Commands

- Discovered, not verified, from `README.md`.
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Command: `python scripts/validate_submission.py --generate-csv tiny_generate.csv --output-dir tiny_generated --smoke`
  - Notes: requires missing script and undiscovered tiny fixture files.

## Verified Commands

- None. No build, test, lint, format, type-check, static-analysis, smoke-test, benchmark, or evaluator command was run during this session.

## Commands Not Run

- `python -m pip install -r requirements.txt`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: package-manager command; may require network and mutates the active Python environment.
- `python -m pip install -e .`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: package-manager/build setup command; mutates the active Python environment, and `src/` is currently missing.
- `python -m compileall src scripts tests`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; documented paths are currently missing.
- `python -m pytest`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; no `tests/` directory was discovered.
- `python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; writes a report; missing script and generated images.
- `python scripts/validate_submission.py --generate-csv tiny_generate.csv --output-dir tiny_generated --smoke`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; missing script and tiny fixture files.
- `python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; writes a report; missing script and generated images.
- `python scripts/prepare_score_input.py --generate-csv dataset/generate.csv --generated-images generated_images --score-input-dir score_input --test-mu hw6_reference/test_mu.npy --test-sigma hw6_reference/test_sigma.npy --scores fid --overwrite`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; writes `score_input/`; missing script and generated images.
- `python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6`
  - Reason: requires approval; writes package output; missing script, generated images, checkpoint, and real student ID.
- `python3 score.py --input_dir $input --output_dir $output --config config.json`
  - Working directory: `/home/kuotzuwei15/GenAI/hw6/scoring_program`
  - Reason: requires approval; writes scorer output; requires Codabench-style input layout, GPU because `score.py` uses `cuda:0`, and may require cached/downloadable pretrained model weights.

## Missing Quality Gates

- Build: no explicit artifact build command was discovered.
- Unit tests: command is documented, but no test files were discovered.
- Integration tests: commands are documented, but their scripts and outputs are missing.
- Lint: no discovered lint command.
- Format: no discovered format command.
- Type-check: no discovered type-check command.
- Static analysis: `compileall` is documented, but its target directories are missing.
- Benchmark/evaluator: scorer and evaluator commands are documented/discovered, but required generated outputs and helper scripts are missing.
- Smoke tests: smoke command is documented, but its script and tiny fixtures are missing.

## Recommended Minimum Done Criteria

- Before implementation exists, there is no runnable minimum quality gate set beyond read-only discovery.
- After creating `src/`, `scripts/`, `configs/`, and `tests/`, use the discovered gates as the minimum acceptance set:
  - `python -m compileall src scripts tests`
  - `python -m pytest`
  - `python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json`
- For scoring-related work, add the discovered local evaluation path once generated images exist:
  - `python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json`
- Before final packaging, run the discovered packaging command with the real student ID and final checkpoint:
  - `python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite`
- Consider adding explicit lint, format-check, and type-check commands after implementation tooling is chosen; none are currently discovered.
