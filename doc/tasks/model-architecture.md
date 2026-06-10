# Model Architecture

## Goal

Build a compact conditional UNet that predicts noise for 64x64 RGB images from noisy images, timesteps, and condition ids.

## Inputs

- `doc/proposal.md`: compact UNet, base channels 96 or 128, 64 -> 32 -> 16 -> 8 path, GroupNorm, SiLU, attention at 16x16 and/or 8x8, FiLM or scale-shift conditioning.
- `doc/detailed-design.md`: model module owns the forward contract `[B, 3, 64, 64] -> [B, 3, 64, 64]` and must keep architecture metadata checkpoint-compatible.

## Tasks

- [ ] Implement timestep embeddings and learned animal, object, and pair embeddings.
- [ ] Implement conditional residual blocks with GroupNorm, SiLU, and one selected conditioning injection method.
- [ ] Implement downsampling and upsampling blocks for the 64 -> 32 -> 16 -> 8 UNet path.
- [ ] Add configurable self-attention at 16x16 and/or 8x8 resolutions.
- [ ] Implement the UNet forward method with shape checks for image, timestep, and condition batches.
- [ ] Add tests for a small model forward pass, output shape, null-condition ids, and attention-enabled configuration.

## Done When

- [ ] The model returns a predicted noise tensor with the same shape as the input image tensor.
- [ ] Model architecture tests pass with fake tensors and no dataset dependency.
