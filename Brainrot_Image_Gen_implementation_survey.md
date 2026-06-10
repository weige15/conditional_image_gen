# Brainrot Image Gen: Implementation Survey

Date: 2026-06-11

## Assignment Summary From PDF

The homework asks for a from-scratch conditional image generation model for 64x64 "Brainrot" images. Each image is conditioned on an animal-object pair, e.g. `shark + sneaker` or `dog + banana`.

Key constraints:

- Train the main generative model from scratch.
- Do not use pretrained UNet, Transformer, diffusion model, or other pretrained generative weights.
- Do not use high-level pipelines or training loops from packages such as Diffusers.
- Pretrained VAE or CLIP may be used only as auxiliary modules, not as the main image generator.
- Generate 2,000 PNG images at 64x64, using `generate.csv`.
- Evaluation is 50% FID and 50% CLIP-T.
- Dataset: 4,799 training images, 10 animals x 10 objects = 100 condition combinations.

## Best Implementation Direction

Use a conditional DDPM/DDIM model trained from scratch with a compact UNet backbone.

Why this is the best fit:

- The target resolution is only 64x64, so pixel-space diffusion is feasible.
- The assignment baseline has about 62.68M parameters, which strongly suggests a moderate UNet-style diffusion baseline.
- Diffusion training is more stable than GAN training for a one-shot homework setting.
- Classifier-free guidance can improve condition alignment without a separate pretrained classifier.
- DDIM sampling lets you generate 2,000 images faster after training.

## Recommended Model

### Conditional UNet DDPM

Input:

- Noisy image `x_t`: shape `[B, 3, 64, 64]`
- Timestep `t`
- Condition:
  - animal id: 10 classes
  - object id: 10 classes
  - optional pair id: 100 classes

Conditioning:

- Learn embeddings for animal, object, and optionally pair id.
- Sum or concatenate condition embeddings with sinusoidal timestep embedding.
- Inject into ResBlocks through FiLM / scale-shift normalization.
- Use condition dropout, e.g. 10%, for classifier-free guidance.

Backbone:

- UNet channel base 96 or 128.
- Downsample levels for 64x64: 64 -> 32 -> 16 -> 8.
- 2-3 residual blocks per level.
- Self-attention at 16x16 and/or 8x8.
- GroupNorm + SiLU.
- Predict epsilon or v; epsilon is simplest, v-pred can be tried later.
- EMA weights for sampling.

Diffusion:

- 1,000 training timesteps.
- Cosine noise schedule or linear schedule baseline.
- MSE loss between true noise and predicted noise.
- DDIM sampler with 50-250 steps for final generation.
- Classifier-free guidance scale around 1.5-3.0; tune for FID vs CLIP-T.

Training:

- AdamW, learning rate around 1e-4 to 2e-4.
- Batch size as large as GPU allows; use gradient accumulation if needed.
- Random horizontal flip only if it does not damage object semantics.
- Mild color jitter / random crop may help FID, but avoid aggressive transforms that change labels.
- Train long enough to see stable validation samples per condition.

## Strong Alternatives

### StyleGAN2-ADA Conditional

StyleGAN2-ADA is attractive because the dataset has only 4,799 images, and ADA was designed to stabilize GANs in limited-data regimes. It may get strong FID. The risk is CLIP-T: conditioning 100 animal-object combinations must be implemented carefully, and GAN mode collapse can hurt coverage across all pairs.

Use as a second experiment if DDPM is too slow or if FID lags.

### Class-Conditional Improved Diffusion / Guided Diffusion Reimplementation

OpenAI's `improved-diffusion` and `guided-diffusion` repos are strong references for architecture flags, cosine schedules, EMA sampling, learned sigmas, and DDIM respacing. Because the assignment forbids high-level training flows and pretrained weights, treat these as references, not something to submit unchanged.

## What Not To Use

- FLUX, Stable Diffusion, SDXL, DALL-E, Midjourney, pretrained diffusion checkpoints.
- Diffusers pipelines or training scripts as the main implementation.
- Pretrained ControlNet, IP-Adapter, LoRA, DreamBooth, or Textual Inversion as generation components.
- External generated images unless the rules or TA confirm they are acceptable and reproducible.

## Ranked Papers

1. DDPM: Denoising Diffusion Probabilistic Models
   - Core training objective for denoising image generation.
   - Best foundation for a from-scratch homework implementation.

2. Improved DDPM
   - Useful improvements: cosine schedule, learned variance, better likelihood/sample tradeoffs, EMA sampling practice.

3. Classifier-Free Diffusion Guidance
   - Best method for improving conditional alignment without training a separate classifier.

4. DDIM
   - Faster sampling from a DDPM-trained model, useful for generating 2,000 final images.

5. Diffusion Models Beat GANs
   - Architecture and guidance lessons for class-conditional image synthesis.

6. StyleGAN2-ADA
   - Useful fallback if limited-data GAN training gives better FID than diffusion.

7. EDM
   - Advanced diffusion design-space reference. Good for later optimization, but less necessary for the first working model.

## Ranked Repositories To Study

1. `openai/improved-diffusion`
   - Best reference for 64x64 diffusion training flags, cosine schedule, class conditioning, EMA, and DDIM sampling.

2. `openai/guided-diffusion`
   - Best reference for stronger UNet architecture and class-conditional guided diffusion.

3. `lucidrains/denoising-diffusion-pytorch`
   - Clean PyTorch DDPM implementation, useful for understanding a compact code structure.

4. `NVlabs/stylegan2-ada-pytorch`
   - Best GAN fallback for limited-data image generation.

## Suggested Experiment Plan

### H1: Conditional DDPM Baseline

Prediction: A 50-70M parameter conditional UNet trained from scratch can pass the first baselines if trained long enough and sampled with EMA.

Implementation:

- Dataset loader reads `train.csv` and images.
- Encode animal/object ids and pair id.
- Train epsilon-prediction DDPM.
- Sample using EMA + DDIM 100-250 steps.

### H2: Better Conditioning

Prediction: animal + object + pair embeddings outperform animal + object alone, because each pair has a distinct visual composition.

Implementation:

- Compare condition embedding choices using fixed training budget:
  - animal + object
  - animal + object + pair
  - pair only

### H3: Classifier-Free Guidance Sweep

Prediction: guidance improves CLIP-T but can worsen FID when too high.

Implementation:

- Train with 10% condition dropout.
- Generate validation grids with guidance scales 1.0, 1.5, 2.0, 3.0.
- Select scale using local FID estimate plus visual condition accuracy.

### H4: StyleGAN2-ADA Fallback

Prediction: StyleGAN2-ADA may improve FID on 4,799 images but could underperform CLIP-T if conditioning is weak.

Implementation:

- Train from scratch with conditional labels.
- Compare FID/CLIP-T proxy against DDPM samples.

## Practical Submission Checklist

- `generated_images/` contains exactly 2,000 PNG files.
- Every filename matches `generate.csv`.
- Every image is RGB and 64x64.
- README includes exact train and generate commands.
- `model.pth` contains only your trained weights.
- No pretrained generative weights are loaded.
- Seeds/configs are saved for reproducibility.
- Zip structure matches the PDF requirement.

## Sources

- DDPM: https://arxiv.org/abs/2006.11239
- Improved DDPM: https://arxiv.org/abs/2102.09672
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- Guided Diffusion: https://arxiv.org/abs/2105.05233
- EDM: https://arxiv.org/abs/2206.00364
- OpenAI improved-diffusion: https://github.com/openai/improved-diffusion
- OpenAI guided-diffusion: https://github.com/openai/guided-diffusion
- lucidrains DDPM PyTorch: https://github.com/lucidrains/denoising-diffusion-pytorch
- StyleGAN2-ADA PyTorch: https://github.com/NVlabs/stylegan2-ada-pytorch
