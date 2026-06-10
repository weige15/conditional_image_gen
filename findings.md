# Findings

## Current Understanding

The project is a constrained homework assignment: train a conditional image generator from scratch for 64x64 animal-object "Brainrot" images. The final system must generate 2,000 images from `generate.csv`, and scoring is split between FID and CLIP-T.

The important correction is that pretrained FLUX/Stable Diffusion/SDXL-style solutions are disallowed. Pretrained VAE or CLIP may be auxiliary only and cannot replace the generator.

## Best Direction

The best primary implementation is a from-scratch conditional DDPM:

- UNet backbone around the assignment baseline scale.
- Learned animal/object/pair embeddings.
- FiLM or scale-shift conditioning inside residual blocks.
- 1,000 diffusion steps for training.
- Cosine schedule.
- EMA model for sampling.
- DDIM sampling for faster generation.
- Classifier-free guidance to tune CLIP-T vs FID.

## Patterns And Insights

FID rewards realistic distribution matching, while CLIP-T rewards semantic alignment to `a {animal} and a {object}`. The model therefore needs both high sample quality and strong condition use. Pair embeddings are likely helpful because the dataset contains only 100 known combinations and each pair may have a characteristic visual style.

## Lessons And Constraints

- Do not use Diffusers pipelines, pretrained diffusion checkpoints, ControlNet, IP-Adapter, LoRA, DreamBooth, FLUX, or SDXL as the main generator.
- Open-source repositories can be studied, but the submitted training/generation path should be owned, reproducible, and from scratch.
- StyleGAN2-ADA is a credible fallback because the dataset is small, but diffusion is safer for stable conditional generation.

## Open Questions

- Local dataset files and baseline code still need to be inspected.
- Available GPU and time budget will determine whether to train a 50-70M parameter UNet or a smaller prototype first.
- Need local FID/CLIP-T scripts or faithful replicas of Codabench metrics for tuning.
