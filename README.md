# HW6 Brainrot DDPM/DDIM

This repository implements a from-scratch conditional diffusion generator for the HW6 Brainrot image task. The primary path is a pixel-space conditional DDPM trained with epsilon-prediction MSE, learned animal/object/pair embeddings, classifier-free guidance, EMA weights, and DDIM sampling. It does not use pretrained generative weights or Diffusers pipelines.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Default paths assume the local assignment layout:

- `train.csv`
- `generate.csv`
- `trainset/`
- optional `test_mu.npy` and `test_sigma.npy`

## Train

```bash
python scripts/train.py --config configs/default.yaml
```

Useful overrides:

```bash
python scripts/train.py --train-csv train.csv --train-image-dir trainset --checkpoint-dir checkpoints --max-steps 1000
```

Checkpoints include raw model weights, EMA weights, optimizer state, resolved config, seed metadata, architecture metadata, diffusion metadata, progress counters, and stable condition mappings.

## Generate

```bash
python scripts/generate.py --checkpoint checkpoints/checkpoint_step_1000.pt --config configs/default.yaml --overwrite
```

The script reads `generate.csv` and writes one RGB `64x64` PNG per row into `generated_images/{id}` using EMA weights by default.

## Validate

```bash
python scripts/validate_submission.py --generate-csv generate.csv --output-dir generated_images --report-json reports/validation.json
```

For synthetic or partial smoke runs:

```bash
python scripts/validate_submission.py --generate-csv tiny_generate.csv --output-dir tiny_generated --smoke
```

## Evaluate

```bash
python scripts/evaluate.py --generate-csv generate.csv --output-dir generated_images --report-path reports/evaluation.json
```

Structural validation always runs first. FID and CLIP-T proxy entries skip clearly when local feature extractors or optional dependencies are not configured.

## Course Scorer

The downloaded `score.py` expects a Codabench-style directory with `input/ref` and `input/res`. Prepare that layout after generation:

```bash
python scripts/prepare_score_input.py \
  --generate-csv generate.csv \
  --generated-images generated_images \
  --score-input-dir score_input \
  --test-mu test_mu.npy \
  --test-sigma test_sigma.npy \
  --scores fid \
  --overwrite
```

Then run the scorer from the folder that contains `score.py`:

```bash
mkdir -p score_output
python score.py --input_dir score_input --output_dir score_output --config config.json
```

`clip_t` and `clip_i` also require the scorer's hidden `test.json` and reference test images under `input/ref`; those files are not included in the local homework folder.

## Package

```bash
python scripts/package_submission.py --checkpoint model.pth --zip-path submission.zip
```

Packaging refuses to proceed unless validation passes and required artifacts are present: `generated_images/`, `src/brainrot_diffusion/`, `scripts/`, `README.md`, `requirements.txt`, and the configured checkpoint.

## Quality Gates

Run before submission:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall src scripts tests
```

## Fallback

`src/brainrot_diffusion/fallback/` is intentionally isolated for a possible from-scratch conditional GAN experiment. The fallback path is disabled by default and does not change DDPM checkpoint or validation contracts.
