# Proposal: From-Scratch Conditional DDPM/DDIM for Brainrot Image Generation

## Objective

Build a reproducible conditional image generation system for HW6 that trains the main generator from scratch and produces exactly 2,000 RGB PNG images at 64x64 resolution for the animal-object conditions listed in `generate.csv`.

The proposed algorithm is a class-conditional DDPM trained in pixel space with a compact UNet backbone, EMA sampling weights, classifier-free guidance, and DDIM sampling for faster final generation. This is chosen because the assignment target resolution is small, diffusion training is comparatively stable, and the scoring combines image distribution quality through FID with condition alignment through CLIP-T.

## Current Project State

Observed files in `/home/kuotzuwei15/GenAI/hw6`:

- `Brainrot_Image_Gen.pdf`: detailed homework specification.
- `HW6.pdf`: lecture slide overview.
- `Brainrot_Image_Gen_implementation_survey.md`: implementation survey and recommended direction.
- `research-log.md`: research activity log.
- `findings.md`: summary of constraints and best direction.
- `research-state.yaml`: structured research state.

No local dataset files, baseline implementation, `pyproject.toml`, `uv.lock`, `src/`, or `tests/` files were observed during inspection. The proposal therefore assumes a greenfield Python/PyTorch project will be added around the assignment files.

## Assumptions

- The main generative model must be trained from scratch.
- Pretrained generative weights, pretrained UNet/Transformer/diffusion checkpoints, and high-level Diffusers-style pipelines are disallowed for the submitted generation path.
- Pretrained CLIP may be used only as an auxiliary module for evaluation, analysis, or optional condition feature support, not as the generator.
- The final submission must include code, model weights, a README, requirements, and `generated_images/` with exactly 2,000 PNG files matching `generate.csv`.
- The initial implementation should optimize for reliability and reproducibility before attempting more experimental alternatives.

## Proposed Approach

### Algorithm

Implement a from-scratch conditional DDPM with optional DDIM sampling:

- Train the model to predict Gaussian noise `epsilon` from noisy images `x_t`, diffusion timestep `t`, and condition labels.
- Use 1,000 diffusion timesteps for training.
- Use a cosine beta/noise schedule as the primary schedule, with a linear schedule as a simple fallback if debugging requires it.
- Use MSE loss between sampled noise and predicted noise.
- Maintain an exponential moving average copy of the model weights for sampling.
- Use DDIM sampling with 50 to 250 steps for final image generation.

### Conditioning

Represent each prompt condition with learned embeddings:

- animal id: 10 classes
- object id: 10 classes
- pair id: 100 classes

Combine the animal, object, and pair embeddings with the timestep embedding, then inject the result into UNet residual blocks using FiLM or scale-shift normalization. Include condition dropout during training, around 10%, so the same model can support classifier-free guidance at sampling time.

This design directly targets CLIP-T by making animal/object information available at every denoising stage, while pair embeddings help the model learn the characteristic composition of each known animal-object combination.

### Model Architecture

Use a compact UNet appropriate for 64x64 images:

- Input/output shape: `[B, 3, 64, 64]`.
- Base channels: start with 96 or 128 depending on available GPU memory.
- Resolution path: 64 -> 32 -> 16 -> 8, then symmetric upsampling.
- Residual blocks: 2 to 3 per resolution level.
- Normalization and activation: GroupNorm and SiLU.
- Attention: self-attention at 16x16 and/or 8x8.
- Conditioning: scale-shift modulation from timestep plus condition embedding.

The target model size should be in the same practical range as the homework baselines, roughly tens of millions of parameters, while still allowing experiments on the available hardware.

### Data Pipeline

Add a dataset loader that:

- Reads `train.csv`.
- Loads training images as RGB.
- Resizes or validates images to 64x64.
- Maps animal/object strings to stable integer ids.
- Creates a stable pair id from `(animal, object)`.
- Applies conservative augmentations such as horizontal flip and mild color jitter only after checking that they do not damage label semantics.

Avoid heavy augmentations that could change object identity, reduce CLIP-T, or create a mismatch with the hidden test distribution.

### Training Workflow

Implement the training loop directly in PyTorch:

- Optimizer: AdamW.
- Initial learning rate: `1e-4` to `2e-4`.
- Batch size: as large as the GPU allows, with gradient accumulation if needed.
- Mixed precision: allowed for training efficiency if it does not change reproducibility guarantees.
- Checkpoint contents: model weights, EMA weights, optimizer state, scheduler state, config, step count, seed, and label mappings.
- Periodically save sample grids across representative animal-object pairs.

The first milestone should prioritize a working end-to-end training and sampling path over advanced tuning.

### Sampling And Submission Generation

Implement a generation script that:

- Loads the EMA model weights by default.
- Reads `generate.csv`.
- Generates exactly one image per requested row.
- Uses each row's animal/object condition and filename.
- Saves RGB 64x64 PNG files into `generated_images/`.
- Supports a guidance scale sweep, initially `1.0`, `1.5`, `2.0`, and `3.0`.
- Validates file count, filenames, image mode, and image size before packaging.

Use the local validation score and visual grids to choose the final DDIM step count and guidance scale. Higher guidance should improve condition alignment but may hurt FID if overused.

### Evaluation And Tuning

Add local evaluation utilities for:

- Submission structure validation.
- FID against the provided `test_mu.npy` and `test_sigma.npy` if available.
- CLIP-T proxy using the same or closest available OpenAI CLIP ViT-B-32-quickgelu setup.
- Per-condition sample grids to catch mode collapse or ignored conditions.

Tune in this order:

1. Confirm the model learns recognizable 64x64 samples.
2. Enable EMA sampling and compare against raw weights.
3. Sweep DDIM step count for speed versus quality.
4. Sweep classifier-free guidance scale for FID versus CLIP-T.
5. Compare condition embedding variants: animal/object only, pair only, and animal/object/pair.

### Fallback Experiment

If the diffusion model is too slow or fails to reach acceptable FID, implement a conditional StyleGAN2-ADA model trained from scratch as a fallback experiment. This may help on a 4,799-image dataset, but it is riskier for CLIP-T because condition coverage and mode collapse must be controlled carefully.

## Milestones

1. Build the project skeleton, dataset loader, label mapping, image validation, and direct PyTorch DDPM training loop.
2. Implement the conditional UNet, cosine schedule, epsilon-prediction loss, checkpointing, EMA, and sample grid generation.
3. Implement DDIM sampling, classifier-free guidance, `generate.csv`-driven output, and submission validation.
4. Add local FID/CLIP-T proxy evaluation and run tuning sweeps for guidance scale, DDIM steps, and condition embedding variants.
5. Finalize reproducibility artifacts: README commands, requirements, config files, seeds, trained weights, and checked `generated_images/`.

## Open Questions

- Where will the Codabench dataset files be placed locally, and what exact paths should the scripts assume by default?
- What GPU and training time budget are available for the final run?
- Are the provided `test_mu.npy` and `test_sigma.npy` already available locally, or should evaluation utilities accept a path supplied by the user?
- Should optional CLIP conditioning be attempted, or should the first implementation use only learned class embeddings for maximum rule clarity?

## Validation Plan

Validate the eventual implementation with:

- A smoke test on a tiny training subset that runs one training step and one sampling step.
- Unit tests for label mapping, pair id creation, CSV parsing, and output filename matching.
- Image checks confirming exactly 2,000 PNG files, all RGB, all 64x64, and all names matching `generate.csv`.
- Reproducibility checks confirming a saved checkpoint can regenerate images from the documented command.
- Local FID and CLIP-T proxy evaluation before Codabench submission.
- Manual visual grids covering all 100 animal-object pairs to check condition use and diversity.
