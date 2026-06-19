# Test Plan

## Purpose

This plan defines observable pass/fail checks for the proposed HW6 Brainrot Image Generation implementation before coding proceeds. Verification is done when the planned package can train a from-scratch conditional generator, generate exactly 2,000 CSV-matching RGB `64x64` PNG files, validate the output structure, and preserve enough checkpoint metadata for TA reproduction.

No build, test, training, generation, packaging, or evaluator command has been run for this plan.

## Source Requirements

Sources read:

- `doc/proposal.md`: objective, constraints, proposed diffusion baseline, module candidates, validation plan, risks, and open questions.
- `doc/high-level-design.md`: module boundaries, data flow, contracts, quality-gate alignment, and non-goals.
- `doc/problem-brief.md`: assignment requirements, categories, deliverables, scoring rules, penalties, and open questions.
- `doc/repo-map.md`: live repository state and missing implementation directories.
- `doc/quality-gates.md`: discovered commands and command statuses.
- `README.md`: documented intended CLI workflow and quality gates.
- `requirements.txt` and `pyproject.toml`: Python packaging and dependency context.
- `scoring_program/score.py`: provided evaluator behavior, including PNG discovery, image-size checks, FID, and CLIP scoring paths.

Requirements extracted:

- Main generator must be trained from scratch; pretrained generative weights and high-level generative pipelines are forbidden.
- Inputs are `dataset/train.csv`, `dataset/trainset/*.png`, and `dataset/generate.csv`.
- Final output must contain exactly 2,000 PNG images, one per `dataset/generate.csv` row.
- Each final image must be RGB and `64x64`.
- Generation must use checkpoint-compatible condition mappings for animal/object labels.
- Checkpoints must preserve config, model state, optional EMA state, diffusion metadata, condition mappings, and seed/progress metadata needed for reproducible generation.
- Final package must include generated images, scripts, `model.pth`, `README.md`, and `requirements.txt`.
- Local FID/evaluator work is separate from core model training and may require CUDA and pretrained evaluation models.

Missing or ambiguous source material:

- `src/`, `scripts/`, `configs/`, and `tests/` are currently missing.
- Full training hardware and training-time budget are unknown.
- Student ID for `HW6_{student_id}.zip` is unknown.
- Exact E3 deadline is unknown.
- Local CLIP proxy evaluation is undecided.
- Auxiliary pretrained CLIP/VAE usage is not approved and is out of the first implementation path.

## Test Scope

Covered by this plan:

- Config loading and validation for the planned YAML config.
- CSV parsing for train and generation data.
- Image loading, RGB conversion, normalization, and tensor shape contracts.
- Stable condition vocabulary construction and checkpoint-compatible lookup.
- From-scratch model initialization and tensor shape behavior.
- Diffusion schedule, noising objective, loss finiteness, and sampler shape/range behavior.
- EMA state updates and checkpoint inclusion when enabled.
- Checkpoint save/load metadata and mapping compatibility checks.
- Training-loop smoke execution on tiny CPU fixtures.
- Sampling/generation output filenames, PNG mode, size, count, and overwrite behavior.
- Structural validation for missing, extra, wrong-size, wrong-mode, and valid submissions.
- Evaluation wrapper behavior, including validation-first execution and graceful metric skipping when resources are absent.
- Packaging refusal on invalid outputs and required archive contents on valid fixtures.
- CLI argument parsing and nonzero exits for invalid inputs.
- README command alignment with real scripts after implementation exists.

## Non-Tested Scope

Out of scope for this plan:

- Full competitive model quality or leaderboard placement.
- Codabench hidden-test scoring reproducibility.
- Long GPU training stability beyond smoke and benchmark gates.
- Human aesthetic judgment of generated images.
- Extra training data workflows.
- Any auxiliary pretrained CLIP or VAE path unless explicitly approved later.
- Direct execution of `scoring_program/score.py` during unit tests, because it uses CUDA and pretrained evaluation models.

## Smoke Tests

Planned smoke checks:

- `python -m compileall src scripts tests` succeeds after `src/`, `scripts/`, and `tests/` exist.
- `python -m pytest` succeeds on tiny CPU fixtures after tests exist.
- `scripts/train.py` can run a tiny CPU smoke configuration for a few steps and write one checkpoint with config, model state, diffusion metadata, mappings, and seed metadata.
- `scripts/generate.py` can load the tiny checkpoint and write one RGB `64x64` PNG per row in a tiny generation CSV.
- `scripts/validate_submission.py` accepts a tiny valid output directory and rejects missing, extra, wrong-size, and wrong-mode outputs.

Current status: Missing. The implementation directories and smoke fixtures do not exist yet.

## Unit Tests by Module

| HLD Module | Status | Verification Method |
| --- | --- | --- |
| `config.py` | Planned | Load `configs/default.yaml`, reject missing required keys, reject invalid paths/types, and confirm CLI overrides resolve to deterministic values. |
| `data.py` | Planned | Parse train/generate CSV fixtures, reject missing columns and duplicate IDs, load images as RGB tensors, and fail on missing image files. |
| `conditions.py` | Planned | Build stable sorted animal/object/pair mappings, serialize/deserialize mappings, and reject generation labels absent from checkpoint mappings. |
| `model.py` | Planned | Construct model from config with random initialization only, pass tiny tensors through it, and assert output shape equals noise target shape. |
| `diffusion.py` | Planned | Build valid schedule tensors, add noise deterministically for fixed seeds, compute finite loss, and return sampler tensors in image range after clamping. |
| `ema.py` | Planned | Update EMA from a tiny module, verify decay math on known weights, save/load EMA state, and handle disabled EMA as a no-op. |
| `checkpoint.py` | Planned | Save/load tiny checkpoint, preserve config/mappings/step/seed metadata, and fail clearly when required checkpoint fields are missing. |
| `train_loop.py` | Planned | Run a tiny CPU training loop for a bounded number of steps, produce finite losses, and write a checkpoint without requiring generated final outputs. |
| `sample.py` | Planned | Generate deterministic tiny outputs from a fixed seed/checkpoint, preserve requested filenames, and refuse overwrite unless explicitly requested. |
| `validate.py` | Planned | Validate exact filename set, PNG format, RGB mode, `64x64` size, count, and JSON report contents. |
| `evaluate.py` | Planned | Run validation first, compute or skip local FID based on available assets, and report skipped CLIP-T unless a proxy is explicitly enabled. |
| `package.py` | Planned | Refuse invalid generated outputs, include required files for a valid fixture package, and require a non-placeholder student ID. |
| `scripts/*.py` | Planned | Parse required arguments, delegate to package modules, return nonzero on invalid input, and avoid embedding core logic in scripts. |
| `tests/` | Missing | No existing tests were discovered. Add tiny fixtures covering the planned package behavior. |

## Integration Tests

Planned integration checks:

- Data-to-training path: tiny `train.csv` plus tiny RGB images produce batches with condition IDs and tensors shaped for the model.
- Training-to-checkpoint path: tiny CPU training writes a checkpoint that can be loaded by generation without rebuilding incompatible mappings.
- Checkpoint-to-generation path: tiny generation CSV produces exactly the requested PNG filenames using saved condition mappings.
- Generation-to-validation path: generated fixture directory passes structural validation only when count, filenames, PNG mode, and image size match.
- Validation-to-packaging path: packaging runs validation first and refuses to create a final archive when generated outputs are invalid.
- Evaluation path: `scripts/evaluate.py` writes a report that distinguishes structural validation results from FID/CLIP metric availability.

Failure examples:

- A generated image named `000001.jpg` fails because the extension/format is not PNG.
- A PNG saved as grayscale or RGBA fails because final outputs must be RGB.
- A `65x64` image fails even when the filename is correct.
- A generation label not present in checkpoint mappings fails before sampling.
- A package command with `STUDENT_ID` placeholder fails.

## Golden Test Cases

Planned hand-checkable fixtures:

- Tiny train CSV with two rows:

```text
id,animal,object
000001.png,cat,chair
000002.png,dog,pizza
```

- Tiny generate CSV with two rows:

```text
id,animal,object,prompt
000101.png,cat,chair,a cat and a chair
000102.png,dog,pizza,a dog and a pizza
```

Expected behavior:

- Condition mappings are stable and serializable, for example `cat` and `dog` map deterministically regardless of CSV row order if the implementation uses sorted vocabularies.
- Validation passes only when `000101.png` and `000102.png` both exist, are PNG files, are RGB, and are `64x64`.
- Validation fails if `000103.png` is present as an extra file or if either expected filename is missing.
- Checkpoint load for the tiny generation CSV fails if the checkpoint only contains a `cat/chair` mapping and the CSV requests `dog/pizza`.

## Oracle or Reference Implementation Strategy

Use simple oracles for structural and deterministic behavior:

- CSV oracle: Python `csv.DictReader` on tiny fixtures with exact expected rows and column names.
- Condition oracle: sorted unique label lists and dictionary lookup expected values.
- Validation oracle: direct Pillow checks for filename, format, mode, and size.
- EMA oracle: closed-form one-step update on scalar parameters.
- Diffusion oracle: deterministic shape/range/finite-value checks for fixed seeds rather than visual quality.
- Packaging oracle: inspect zip names with `zipfile` and compare them to the required manifest.

No oracle can prove final generative quality. FID and CLIP-T are metric checks, not unit-test oracles.

## Randomized or Property Tests

Planned property checks with deterministic seeds:

- Generate random small CSV row sets and assert condition mappings are stable across row permutations.
- Generate random expected filename sets and output-directory contents; validation passes only for exact set equality.
- Generate random RGB/RGBA/L/grayscale tiny images and assert validation accepts only RGB PNG images of the configured size.
- Sample random batch sizes and assert model/diffusion outputs preserve batch, channel, height, and width dimensions.
- Use fixed seeds for tiny sampling and assert repeated generation produces identical bytes or identical tensors before PNG encoding, depending on implementation.

Bounds:

- Keep randomized tests CPU-only.
- Use tiny images or tensors where possible.
- Do not run long training in property tests.

## Edge Cases

Planned edge-case coverage:

- Empty CSV files.
- Missing required CSV columns.
- Duplicate `id` rows.
- Missing training images referenced by CSV.
- Non-PNG training image paths if the implementation accepts arbitrary files.
- Generation CSV prompt text that does not match `a {animal} and a {object}`.
- Unknown animal/object labels at generation time.
- Reordered CSV rows.
- Output directory already containing files when `--overwrite` is false.
- Extra output files in nested directories.
- Wrong image size, including `64x65`, `65x64`, and larger resized images.
- Wrong image mode, including `L`, `P`, and `RGBA`.
- Corrupt PNG files.
- Checkpoint missing condition mappings, config, model state, or diffusion metadata.
- Disabled EMA and enabled EMA checkpoint paths.
- CPU-only environment with mixed precision disabled or auto-skipped.
- Reports directory missing before validation/evaluation writes reports.

## Performance Benchmarks

Known benchmark-like checks, not yet runnable:

- Tiny CPU smoke training should complete quickly enough for `pytest`; exact threshold is Unknown until implementation exists.
- Full training requires GPU time and user approval; target hardware and training-time budget are Unknown.
- Generation of exactly 2,000 final PNGs should be timed after a real checkpoint exists; acceptable wall-clock threshold is Unknown.
- Local FID evaluation may require CUDA and pretrained Inception weights through `torchvision`; runtime threshold is Unknown.

Performance metrics to record when available:

- Training steps per second.
- Peak GPU memory.
- Generation images per second for chosen sampler steps.
- Validation runtime for 2,000 images.
- Local FID score when scorer-compatible assets and dependencies are available.

## Evaluator or Grading Commands

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
python scripts/prepare_score_input.py --generate-csv dataset/generate.csv --generated-images generated_images --score-input-dir score_input --test-mu hw6_reference/test_mu.npy --test-sigma hw6_reference/test_sigma.npy --scores fid --overwrite
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

- The discovered Codabench-style scorer command expects Codabench-style `$input/ref` and `$input/res` directories and writes `scores.json`.
- `scoring_program/score.py` uses `cuda:0`, so local execution requires suitable GPU setup.
- Unknown commands are not treated as verified.

## Regression Tests

Planned regression checks:

- Validation rejects extra files, not just missing files.
- Validation checks real image mode and size with Pillow, not only filename extensions.
- Generation uses checkpoint mappings instead of rebuilding mappings from `generate.csv`.
- Checkpoint metadata includes config and seed information needed for reproduction.
- Packaging refuses invalid generated images before writing the archive.
- Evaluation reports metric skips explicitly instead of silently succeeding without scores.
- README documented commands match implemented script arguments.

## Manual Verification

Manual checks required before final submission:

- Inspect a grid of generated images for obvious corruption, blank outputs, or repeated single-color images.
- Confirm generated samples visually reflect requested animal/object pairs well enough to justify a Codabench upload.
- Confirm final README explains training, generation, environment setup, and any auxiliary pretrained/evaluation-only modules used.
- Confirm no forbidden pretrained generative weights, Diffusers pipelines, or high-level training flows are used by the main model.
- Confirm final package layout matches the assignment PDF.
- Confirm the selected `model.pth` is the checkpoint used to generate the submitted images or is an equivalent reproducible export.

## Minimum Done Criteria

Detailed implementation is ready to proceed only when all of the following are true:

- `doc/test-plan.md` exists and maps every HLD module to at least one verification method or an explicit missing entry.
- The planned tests cover final output count, filenames, PNG format, RGB mode, and `64x64` size.
- The planned tests cover stable condition mappings and checkpoint-compatible generation.
- The planned smoke path covers train, checkpoint, generate, and validate on tiny CPU fixtures.
- Evaluator and grading commands are labeled `Known, not run` unless they have actually been executed.
- Unknown hardware, deadline, student ID, and optional CLIP/VAE decisions remain documented as open questions.

Implementation is ready for final submission only when all of the following pass:

- `python -m compileall src scripts tests`
- `python -m pytest`
- Structural validation reports exactly 2,000 expected RGB `64x64` PNG outputs with no missing or extra files.
- A checkpoint or `model.pth` plus README instructions can reproduce generation from `dataset/generate.csv`.
- Packaging creates the required `HW6_{student_id}.zip` with a real student ID and validated generated images.
- Any local evaluation scores are reported as local proxies only unless they come from the official platform.

## Open Questions

- What GPU and training-time budget are available?
- What student ID should be used for `HW6_{student_id}.zip`?
- What is the exact E3 deadline?
- Should local CLIP proxy evaluation be implemented, or should CLIP-T be left to Codabench only?
- Should any auxiliary pretrained CLIP or VAE path be used later, or avoided entirely?
- What wall-clock thresholds are acceptable for full generation and local FID evaluation?
- Should extra data be considered after the baseline works, or should the final solution stay limited to the provided Brainrot Dataset?
