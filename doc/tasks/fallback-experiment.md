# Fallback Experiment

## Goal

Keep a secondary from-scratch conditional StyleGAN2-ADA experiment available without interfering with the primary DDPM/DDIM path.

## Inputs

- `doc/proposal.md`: fallback only if diffusion is too slow or fails to reach acceptable FID; condition collapse is the main risk.
- `doc/detailed-design.md`: fallback reuses data, conditioning, validation, and evaluation contracts but must not alter DDPM checkpoints.

## Tasks

- [ ] Define a separate fallback config namespace and output directory so fallback runs do not overwrite DDPM artifacts.
- [ ] Reuse Data Pipeline and Conditioning outputs for GAN training inputs.
- [ ] Implement or integrate a from-scratch conditional GAN training entrypoint only after the trigger threshold is chosen.
- [ ] Generate fallback comparison samples through the same Validation and Evaluation modules.
- [ ] Add a minimal smoke test that verifies fallback output can satisfy the same generated image contract if the module is implemented.

## Done When

- [ ] Fallback work is isolated from the primary DDPM/DDIM training and checkpoint contracts.
- [ ] Any implemented fallback outputs can be validated with the same submission validator.
