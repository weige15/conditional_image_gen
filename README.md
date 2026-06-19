# HW6 Brainrot DDPM/DDIM

From-scratch conditional diffusion generator for the HW6 Brainrot image task. The main model is a pixel-space DDPM/DDIM UNet trained from scratch with animal/object/pair embeddings, classifier-free guidance, EMA weights, and DDIM sampling. It does not use pretrained generative weights or Diffusers pipelines.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Default paths assume the provided assignment layout under `dataset/` and `hw6_reference/`.

## Train

```bash
python scripts/train.py --config configs/default.yaml
```

Useful overrides:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --train-csv dataset/train.csv \
  --train-image-dir dataset/trainset \
  --checkpoint-dir checkpoints \
  --max-steps 1000 \
  --mixed-precision
```

`training.mixed_precision` is kept in the config for future GPU tuning; the current training loop runs the same CPU-friendly full-precision path unless extended.

Checkpoints contain raw model weights, EMA state, optimizer state, config, diffusion metadata, architecture metadata, seed metadata, progress counters, and stable condition mappings.

## Generate

```bash
python scripts/generate.py \
  --checkpoint checkpoints/checkpoint_step_1000.pt \
  --config configs/default.yaml \
  --overwrite
```

The script reads `dataset/generate.csv`, uses the condition mappings saved in the checkpoint, and writes RGB `64x64` PNGs to `generated_images/{id}`.

## Validate

```bash
python scripts/validate_submission.py \
  --generate-csv dataset/generate.csv \
  --output-dir generated_images \
  --report-json reports/validation.json
```

For small fixtures or smoke runs, pass the matching small CSV and output directory:

```bash
python scripts/validate_submission.py --generate-csv tiny_generate.csv --output-dir tiny_generated --smoke
```

## Evaluate

```bash
python scripts/evaluate.py \
  --generate-csv dataset/generate.csv \
  --output-dir generated_images \
  --reference-dir hw6_reference \
  --report-path reports/evaluation.json
```

Structural validation always runs first. FID and CLIP-T are reported as explicit skips in this wrapper; use `prepare_score_input.py` and the provided course scorer, or Codabench, for metric calculation.

## Course Scorer Input

```bash
python scripts/prepare_score_input.py \
  --generate-csv dataset/generate.csv \
  --generated-images generated_images \
  --score-input-dir score_input \
  --test-mu hw6_reference/test_mu.npy \
  --test-sigma hw6_reference/test_sigma.npy \
  --scores fid \
  --overwrite
```

Then run the provided scorer from the directory containing `score.py` as directed by the assignment.

## Package

```bash
python scripts/package_submission.py \
  --generate-csv dataset/generate.csv \
  --generated-images generated_images \
  --checkpoint model.pth \
  --student-id STUDENT_ID \
  --overwrite
```

Packaging validates generated images first and creates `HW6_{student_id}.zip` with `generated_images/`, `scripts/`, `src/brainrot_diffusion/`, `configs/`, `model.pth`, `README.md`, and `requirements.txt`.

## Quality Gates

```bash
python -m compileall src scripts tests
python -m pytest
```

Full training a competitive checkpoint requires GPU time. The test suite uses tiny CPU fixtures to verify data loading, training, checkpointing, sampling, generation, validation, evaluation skips, and packaging.
