# Proposal: From-Scratch Conditional DDPM/DDIM for Brainrot Image Generation

## Objective
Build a reproducible from-scratch conditional image generation system for the HW6 Brainrot Image Generation assignment. The system will train on the provided Brainrot Dataset and generate exactly 2,000 RGB PNG images at 64x64 resolution according to `dataset/generate.csv`.

The primary algorithm will be a pixel-space conditional diffusion model: a compact UNet trained with the DDPM noise-prediction objective, animal/object/pair conditioning, classifier-free guidance, EMA weights, and DDIM sampling for final image generation.

This direction is selected because the assignment requires a self-trained generator, forbids pretrained generative weights and high-level Diffusers-style pipelines, and scores submissions using both FID and CLIP-T. A conditional diffusion model is the lowest-risk path for balancing image quality, diversity, and semantic alignment on 64x64 images.

## Current Project State
Observed assignment and project files:

- `Brainrot_Image_Gen.pdf` defines the HW6 task: train a conditional image generator from scratch and generate 2,000 64x64 PNG files.
- `research-log.md` and `Brainrot_Image_Gen_implementation_survey.md` already recommend a conditional DDPM/DDIM with a compact UNet, EMA, cosine schedule, classifier-free guidance, and DDIM sampling.
- `dataset/train.csv` contains 4,799 training rows plus header, with image ids and animal/object labels.
- `dataset/generate.csv` contains 2,000 generation rows plus header, with image ids, animal/object labels, and prompts of the form `a {animal} and a {object}`.
- `dataset/trainset/` contains the training images.
- `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`, and `hw6_reference/config.json` are available for local FID scoring.
- `scoring_program/score.py` contains the Codabench-style scorer, including FID and CLIP-T logic.
- `README.md` describes a future DDPM/DDIM project layout, but the corresponding implementation directories and files are not currently present in the repo. In particular, no `src/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`, or `pyproject.toml` were observed.

## Assumptions
The proposal assumes:

- The implementation will be written as a local Python/PyTorch project.
- The main generator will not load pretrained UNet, Transformer, diffusion, GAN, or other generative weights.
- Diffusers pipelines, pretrained generation modules, and high-level training loops will not be used.
- Pretrained CLIP/Inception may be used only for evaluation or auxiliary analysis, consistent with the assignment rules.
- The first target is a complete reproducible baseline that can generate valid submissions; later tuning can optimize FID and CLIP-T.
- Training will primarily use the provided Brainrot Dataset. Extra data is allowed by the PDF, but should be treated as a later experiment because it adds reproducibility and distribution-matching risk.

## Proposed Approach
Implement a from-scratch conditional DDPM/DDIM pipeline with explicit dataset loading, model definition, scheduler, training loop, sampling loop, validation, and packaging scripts.

### 1. Data and Labels
Create a dataset module that reads `dataset/train.csv`, loads RGB images from `dataset/trainset/`, and maps labels into stable integer ids:

- `animal_id`: 10 classes.
- `object_id`: 10 classes.
- `pair_id`: 100 possible animal-object combinations.

Use `dataset/generate.csv` for inference only. The generation script should preserve every requested output filename and condition exactly.

Recommended transforms:

- Resize or center-crop defensively to 64x64 if needed.
- Convert to RGB.
- Normalize pixels to `[-1, 1]`.
- Use random horizontal flip only after visual inspection confirms it does not systematically damage object semantics.
- Avoid aggressive augmentation at first because CLIP-T depends on recognizable animals and objects.

### 2. Conditional UNet
Build a compact residual UNet for 64x64 pixel-space diffusion:

- Input: noisy image `x_t` with shape `[B, 3, 64, 64]`.
- Output: predicted noise `epsilon` with shape `[B, 3, 64, 64]`.
- Resolution path: `64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64`.
- Base width: start with 96 or 128 channels.
- Blocks: 2 residual blocks per level, GroupNorm, SiLU, dropout around 0.1.
- Attention: self-attention at 16x16 and optionally 8x8.
- Time embedding: sinusoidal timestep embedding followed by an MLP.
- Condition embedding: learned embeddings for animal, object, and pair id.
- Conditioning injection: add the condition embedding to the timestep embedding and inject it into residual blocks with FiLM-style scale/shift or additive bias.

The pair embedding should be included from the start. Animal and object embeddings capture compositional structure, while pair embeddings let the model learn frequent composition patterns specific to the 100 condition pairs.

### 3. Diffusion Objective
Use the standard DDPM epsilon-prediction objective:

1. Sample image `x_0`, timestep `t`, and Gaussian noise `epsilon`.
2. Use a cosine beta schedule over 1,000 training timesteps.
3. Construct `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`.
4. Train the UNet to predict `epsilon` from `(x_t, t, condition)`.
5. Minimize MSE between predicted and true noise.

Keep the first implementation simple and stable before adding learned variance or v-prediction. These can be later ablations if the baseline trains successfully but leaves scoring headroom.

### 4. Classifier-Free Guidance
Train with condition dropout, initially 10%. When dropped, animal/object/pair ids should be replaced with a learned null condition.

At sampling time, run conditional and unconditional predictions and combine them:

```text
epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```

Initial guidance sweep:

- `1.0`: no extra guidance, likely best FID baseline.
- `1.5`: balanced first candidate.
- `2.0`: stronger CLIP-T candidate.
- `3.0`: high-alignment candidate that may hurt FID.

Select the final scale using local FID plus visual condition accuracy.

### 5. EMA and Checkpointing
Maintain an exponential moving average of model weights for sampling. Save checkpoints containing:

- Raw model weights.
- EMA model weights.
- Optimizer state.
- Training step and epoch.
- Resolved config.
- Stable animal/object/pair mappings.
- Diffusion schedule metadata.
- Random seed metadata.

Use EMA weights by default for validation and final generation.

### 6. Sampling
Implement both DDPM and DDIM sampling:

- DDPM sampler for correctness and debugging.
- DDIM sampler for faster final generation.

Use DDIM with 50 to 250 steps for generating 2,000 images. Start with 100 steps for iteration speed, then test 200 or 250 steps if FID improves enough to justify the runtime.

Generation must:

- Read every row in `dataset/generate.csv`.
- Use the row's animal/object condition.
- Write `generated_images/{id}`.
- Ensure every output is RGB PNG and exactly 64x64.
- Use deterministic seeds or logged per-image seeds for reproducibility.

### 7. Local Validation and Scoring
Add a validation script that checks:

- Exactly 2,000 PNG files are produced.
- Every filename matches `dataset/generate.csv`.
- No extra or missing filenames exist.
- Every image is RGB and 64x64.

Use the provided scorer files to compute local FID against `hw6_reference/test_mu.npy` and `hw6_reference/test_sigma.npy`. CLIP-T can be computed locally only if the needed CLIP package and reference metadata are available; otherwise, use visual grids and optional open_clip-based proxy checks.

### 8. Experiment Plan
Run experiments in this order:

1. Small smoke run over a subset of images to validate data loading, forward pass, loss decrease, checkpointing, sampling, and file output.
2. Full DDPM baseline with animal + object + pair embeddings, cosine schedule, EMA, 1,000 diffusion steps, and DDIM sampling.
3. Guidance sweep over `1.0`, `1.5`, `2.0`, and `3.0`.
4. Sampling-step sweep over 50, 100, 200, and 250 DDIM steps.
5. Conditioning ablation if time allows: animal+object only, pair only, and animal+object+pair.
6. Optional fallback experiment: from-scratch conditional StyleGAN2-ADA-style GAN only if diffusion is too slow or FID remains poor.

The fallback should not replace the primary DDPM/DDIM implementation unless it clearly improves local FID and maintains semantic alignment.

## Milestones
1. Create the project scaffold: `src/brainrot_diffusion/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`, and a runnable default config.
2. Implement dataset loading, stable label mappings, image transforms, and submission validation.
3. Implement the conditional UNet, cosine diffusion scheduler, DDPM training loss, EMA, and checkpoint format.
4. Implement DDPM/DDIM samplers and a generation script that writes valid `generated_images/` outputs from `dataset/generate.csv`.
5. Run smoke tests and generate sample grids for several animal-object pairs.
6. Train the full model, sweep guidance and DDIM step counts, and select the checkpoint/config with the best FID and visual condition alignment.
7. Package `generated_images/`, `scripts/`, `model.pth`, `README.md`, and `requirements.txt` in the required HW6 zip structure.

## Open Questions
- What GPU and time budget are available for the full training run?
- Should the implementation target plain `pip`/`requirements.txt`, `uv`, or both? The assignment requires `requirements.txt`, while the proposal-generator skill is oriented toward uv-managed Python projects.
- Should extra data be avoided entirely for reproducibility, or reserved as an optional late-stage experiment?
- Is a local CLIP-T proxy needed in this repo, or will Codabench CLIP-T be the only semantic-alignment measurement?

## Validation Plan
Use layered validation:

1. Unit-level checks for label mapping, scheduler math, UNet input/output shapes, EMA updates, and sampler tensor shapes.
2. Smoke training on a tiny subset to confirm loss decreases and sample generation completes.
3. Submission validation to verify exactly 2,000 matching RGB 64x64 PNGs.
4. Local FID using `hw6_reference/test_mu.npy`, `hw6_reference/test_sigma.npy`, and `scoring_program/score.py`.
5. Visual grids grouped by animal-object pair to catch condition failures and mode collapse.
6. Reproducibility check from a clean command sequence: install dependencies, train or load checkpoint, generate images, validate outputs, and compute available scores.
