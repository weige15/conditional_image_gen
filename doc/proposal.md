# Proposal: From-Scratch Conditional Brainrot Image Generation

## Objective

Build a reproducible from-scratch conditional image generation project for HW6 Brainrot Image Generation. The system must train a main generator without pretrained generative weights and produce exactly 2,000 RGB `64x64` PNG images named and conditioned according to `dataset/generate.csv`.

The proposed implementation direction is a Python/PyTorch project centered on a pixel-space conditional diffusion generator. This is a proposal choice, not an assignment requirement: the PDF permits different condition designs and does not mandate a specific generator family.

## Source Inputs

- `Brainrot_Image_Gen.pdf`: assignment objective, data format, restrictions, grading, and deliverables.
- `doc/problem-brief.md`: source-grounded summary of the assignment.
- `doc/repo-map.md`: current repository facts.
- `doc/quality-gates.md`: discovered and missing quality gates.
- `README.md`: documented intended Python/PyTorch DDPM/DDIM workflow, but not proof of implemented files.
- User instruction: start from scratch.

## Current Project State

- Present: assignment PDF, `dataset/`, `hw6_reference/`, `scoring_program/`, `README.md`, `requirements.txt`, `pyproject.toml`, and planning docs.
- Present data assets include `dataset/train.csv`, `dataset/generate.csv`, `dataset/trainset/*.png`, `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`, and `hw6_reference/config.json`.
- Missing in the live tree: `src/`, `scripts/`, `configs/`, `tests/`, `generated_images/`, `checkpoints/`, `reports/`, and `score_input/`.
- `pyproject.toml` already declares a `src` package layout, but no package currently exists.
- `requirements.txt` includes PyTorch, torchvision, NumPy, Pillow, PyYAML, SciPy, tqdm, and pytest.
- No quality gate has been verified yet because the implementation and tests are absent.

## Problem Summary

The assignment requires training a conditional image generator for 100 animal-object pairs. The train CSV maps image IDs to `animal` and `object`; the generation CSV maps output IDs to `animal`, `object`, and a prompt of the form `a {animal} and a {object}`.

Final generated images are scored on Codabench with FID and CLIP-T, each worth 50% of the grade. Output validity matters: incorrect count or format can affect scoring and incorrect file format has an explicit penalty.

## Constraints

- The main generator must be trained from scratch.
- Do not use pretrained UNet, Transformer, diffusion, GAN, or other generative model weights.
- Do not use Diffusers or other high-level generative pipelines/training flows as the main solution.
- Pretrained CLIP or VAE may be used only as an auxiliary module and must not directly generate final images.
- The result must be reproducible by the TAs.
- Final Codabench output must be 2,000 RGB `64x64` PNG files under `generated_images/`.
- Final E3 package must include generated images, scripts, `model.pth`, `README.md`, and `requirements.txt`.
- Codabench uploads are limited to 3 per day.

## Proposed Approach

Create a small Python package under `src/brainrot_diffusion/` plus thin scripts under `scripts/`. Keep the first implementation focused on a valid, reproducible baseline before tuning for score.

The project should provide:

- Data loading for `train.csv`, `generate.csv`, and `dataset/trainset/`.
- Stable condition mappings for animal, object, and animal-object pair labels.
- A from-scratch conditional generator.
- Training, checkpointing, and reproducible generation.
- Structural validation for the exact submission format.
- Local FID/evaluator integration when required files and dependencies are available.
- Packaging for the required E3 layout.

## Algorithm Strategy

Baseline method:

- Implement a minimal pixel-space conditional DDPM trained from scratch on `64x64` RGB images.
- Condition on animal and object labels with learned class embeddings.
- Use a small UNet-style backbone, a standard noise-prediction objective, and a simple sampler.
- Acceptance target for the baseline is validity and reproducibility: it trains, saves a checkpoint, generates CSV-matching PNGs, and passes structural validation.

Intended optimized method:

- Extend the baseline with pair conditioning, EMA weights, classifier-free guidance, and DDIM sampling.
- Sweep guidance scale and DDIM step count using local FID and visual condition checks.
- Keep optional CLIP-based checks separate from the main generator and document any auxiliary pretrained usage.

Correctness strategy:

- Verify CSV parsing, label mapping stability, image normalization, model input/output shapes, checkpoint metadata, generation filenames, PNG mode, and image size with small CPU tests.
- Use deterministic seeds for smoke runs and log enough metadata to reproduce generation.

Performance strategy:

- Use DDIM for faster generation after the DDPM training objective is working.
- Keep model width, batch size, mixed precision, and gradient accumulation configurable for the available GPU.
- Start with small smoke runs before spending GPU time on full training.

## Alternatives Considered

- From-scratch conditional GAN: potentially faster sampling, but usually less stable on small conditional datasets and harder to tune for both FID and CLIP-T.
- Transformer-based image generator: allowed if trained from scratch, but heavier to implement and train well for this deadline.
- Pretrained VAE latent-space generator: assignment allows pretrained VAE only as auxiliary support, but it adds ambiguity and reproducibility risk; reserve for a later explicitly approved experiment.
- Extra training data: allowed by the PDF, but adds data curation and reproducibility risk. Keep the first proposal scoped to the provided Brainrot Dataset.

## Module Candidates

- `src/brainrot_diffusion/config.py`: load and validate YAML config.
- `src/brainrot_diffusion/data.py`: CSV parsing, image loading, generation requests.
- `src/brainrot_diffusion/conditions.py`: stable condition vocabularies and mappings.
- `src/brainrot_diffusion/model.py`: from-scratch conditional image generator.
- `src/brainrot_diffusion/diffusion.py`: schedule, noising objective, sampling math.
- `src/brainrot_diffusion/ema.py`: EMA state management.
- `src/brainrot_diffusion/checkpoint.py`: checkpoint save/load and metadata.
- `src/brainrot_diffusion/train_loop.py`: training lifecycle.
- `src/brainrot_diffusion/sample.py`: generation from checkpoint and conditions.
- `src/brainrot_diffusion/validate.py`: output count, filenames, RGB mode, PNG format, and size checks.
- `src/brainrot_diffusion/evaluate.py`: local FID/evaluator helpers where available.
- `src/brainrot_diffusion/package.py`: final E3 package creation.
- `scripts/*.py`: thin CLI wrappers for train, generate, validate, evaluate, scorer input preparation, and packaging.
- `tests/`: small CPU tests and fixtures for the above behavior.

## Milestones

1. Create the missing project scaffold: `src/brainrot_diffusion/`, `scripts/`, `configs/`, and `tests/`.
2. Implement data loading, condition mappings, and submission validation first.
3. Implement the minimal conditional DDPM baseline and tiny CPU smoke training.
4. Add checkpointing, EMA, DDIM sampling, and generation from `dataset/generate.csv`.
5. Add local evaluation/reporting and package creation.
6. Run the discovered quality gates after implementation exists.
7. Train full checkpoints, generate 2,000 images, validate output, evaluate local FID when possible, and package the final submission.

## Validation Plan

Minimum implementation gates from `doc/quality-gates.md` after files exist:

```bash
python -m compileall src scripts tests
python -m pytest
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json
```

Scoring-related validation after generated images exist:

```bash
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json
```

Final acceptance should require:

- exactly 2,000 expected filenames;
- RGB PNG format;
- `64x64` size;
- no missing or extra output files;
- a checkpoint that can reproduce generation;
- README instructions that let a TA regenerate images from `model.pth`;
- package structure matching the assignment PDF.

## Risks and Tradeoffs

- Training quality may be limited by GPU time. Mitigation: smoke-test early, then tune model size and sampling steps only after the pipeline works.
- CLIP-T and FID can pull in different directions. Mitigation: treat guidance scale and sampling steps as explicit experiment knobs.
- Local CLIP-T may not match Codabench without hidden metadata. Mitigation: rely on official Codabench for final CLIP-T and use local checks only as proxies.
- The provided scorer uses pretrained evaluation models and appears to require CUDA. Mitigation: keep scorer use separate from core training tests and document requirements.
- Existing planning docs mark tasks complete while implementation files are absent. Mitigation: future work should trust live files over stale progress markers.

## Assumptions

- Python/PyTorch is acceptable because the repo already contains `pyproject.toml`, `requirements.txt`, and PyTorch dependencies.
- The first implementation will use only the provided Brainrot Dataset.
- The main generator will not use any pretrained generative weights or high-level generative pipeline.
- Auxiliary pretrained CLIP/VAE usage, if later desired, will be explicitly approved and documented.
- The user will provide the student ID before final packaging.

## Open Questions

- What GPU and training-time budget are available?
- What is the final student ID for `HW6_{student_id}.zip`?
- Should the first scored run target only the provided Brainrot Dataset, or should extra data be considered after the baseline works?
- Should local CLIP proxy evaluation be added, or should CLIP-T be left to Codabench only?
- What is the exact E3 deadline? The PDF references assignment instructions but does not include a date.
