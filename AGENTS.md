# AGENTS.md

## Project Context

- Project: HW6 Brainrot Image Generation.
- Source assignment: `Brainrot_Image_Gen.pdf`.
- User intent: start the implementation from scratch inside this repository.
- Goal: train a from-scratch conditional image generation model for animal-object Brainrot images and generate exactly 2,000 RGB `64x64` PNG files from `dataset/generate.csv`.
- Language/toolchain: Python 3.10+, PyTorch, `setuptools`.
- Package manager: `pip` with `requirements.txt`.
- Main assignment constraint: do not use pretrained generative weights, pretrained UNet/Transformer/diffusion/GAN generation modules, Diffusers pipelines, or high-level generation/training flows as the main model.
- Allowed by assignment: pretrained CLIP or VAE only as auxiliary modules for evaluation, feature extraction, or latent representation. Ask the user before adding either.

## Repository Rules

- Treat this as a scratch implementation unless a file is verified in the current tree.
- Do not trust stale planning notes over live files. Check the tree first.
- Keep changes small and directly tied to the current task.
- Preserve assignment assets: `Brainrot_Image_Gen.pdf`, `dataset/`, `hw6_reference/`, and `scoring_program/`.
- Never write secrets, tokens, private credentials, or copied environment values into files.
- Do not overwrite generated images, checkpoints, reports, or submissions unless the user explicitly asks.

## Read-Only Discovery Commands

These are safe to run without asking:

```bash
pwd
git status --short --branch
git branch --show-current
git worktree list
rg --files
sed -n '1,220p' README.md
sed -n '1,220p' pyproject.toml
sed -n '1,220p' requirements.txt
pdftotext Brainrot_Image_Gen.pdf -
```

## Commands Requiring Permission

Ask before running commands that install, build, test, train, generate, format, mutate files, or change Git state.

Examples requiring permission:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m compileall src scripts tests
python -m pytest
python scripts/train.py --config configs/default.yaml
python scripts/generate.py --checkpoint checkpoints/checkpoint_step_1000.pt --config configs/default.yaml --overwrite
python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images
python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json
python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite
```

Also ask before:

- Creating, editing, moving, renaming, or deleting files.
- Creating branches, switching branches, committing, rebasing, merging, stashing, or pushing.
- Running training, generation, packaging, scoring, or commands that write checkpoints/reports.

## Forbidden Commands

Do not run these unless the user explicitly requests the exact operation:

```bash
rm -rf
git reset --hard
git clean -fd
git checkout -- .
git restore .
git push --force
git push --force-with-lease
chmod -R
chown -R
sudo
```

Do not add these as default workflow commands in docs or scripts.

## Build, Test, and Quality Gates

- Verified dependency files: `requirements.txt`, `pyproject.toml`.
- Documented setup commands:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

- Documented quality gates:

```bash
python -m compileall src scripts tests
python -m pytest
```

- Current implementation status: `src/`, `scripts/`, `configs/`, and `tests/` were not present during setup. Treat commands that reference them as planned until the scratch implementation creates them.
- Full training requires GPU time and user approval.
- Submission validation must check exactly 2,000 PNG files, filename match to `dataset/generate.csv`, RGB mode, and `64x64` size.

## Documentation Rules

- Keep README commands aligned with actual scripts after implementation.
- Mark unverified facts as `Unknown` or `Planned`.
- If using extra data, pretrained CLIP, pretrained VAE, or any cloud-hosted artifact, document it clearly for reproducibility.
- Do not create large workflow docs unless the workflow becomes too long for this file.

## Coding Rules

- Prefer the simplest working implementation that satisfies the assignment.
- Use Python stdlib for simple CSV/path/config glue when adequate.
- Use PyTorch directly for the model, loss, scheduler, training loop, and sampling loop.
- Do not use Diffusers pipelines or pretrained generative components.
- Save enough checkpoint metadata to reproduce generation: config, step, EMA state if used, condition mappings, diffusion metadata, and seed metadata.
- Avoid broad refactors while the scratch implementation is still being built.

## Git and Commit Rules

- Check `git status --short --branch` before editing.
- Do not revert or overwrite user changes.
- Do not commit unless the user asks.
- Do not push unless the user asks.
- Keep generated artifacts out of Git unless the user explicitly wants them tracked.

## Uncertainty Protocol

- If a repo fact is not visible in files, say it is unknown.
- If an assignment rule is ambiguous, prefer the conservative interpretation and ask before implementing the risky path.
- If a command may be slow, expensive, networked, or state-mutating, ask first.
- If generated outputs or checkpoints already exist, ask before replacing them.
