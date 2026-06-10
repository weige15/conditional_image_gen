# Research Log

## 2026-06-11

- Started autoresearch-style implementation survey for Brainrot Image Gen.
- Initial shell commands failed because the configured shell and `wsl.exe` launcher were unavailable in sandboxed execution.
- Used an approved read-only WSL command to extract `Brainrot_Image_Gen.pdf`.
- PDF confirms the task is a from-scratch conditional 64x64 image generation homework, not a pretrained image-generator app.
- Key rule: no pretrained generative model weights and no high-level Diffusers-style pipelines/training loops for the main model.
- Surveyed suitable papers and repositories: DDPM, Improved DDPM, Classifier-Free Guidance, Guided Diffusion, DDIM, EDM, StyleGAN2-ADA, OpenAI improved/guided diffusion repos, lucidrains DDPM PyTorch.
- Current recommendation: implement a conditional DDPM/DDIM with a compact UNet, condition embeddings for animal/object/pair, EMA, cosine schedule, classifier-free guidance, and DDIM sampling.
