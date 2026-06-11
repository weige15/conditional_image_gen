# Packaging

## Goal

Create the final assignment zip only after required artifacts and generated images are valid.

## Inputs

- `doc/proposal.md`: required HW6 zip structure.
- `doc/detailed-design.md`: Packaging module inputs, failure handling, and tests.

## Tasks

- [x] Create `src/brainrot_diffusion/package.py`.
- [x] Create `scripts/package_submission.py` with args for checkpoint/model path, generated images, student id or zip path, and overwrite.
- [x] Verify required artifacts: `generated_images/`, `scripts/`, `src/brainrot_diffusion/`, `model.pth`, `README.md`, and `requirements.txt`.
- [x] Run submission validation before creating the zip.
- [x] Write the zip with the assignment-required top-level structure and reject existing zip without explicit overwrite.
- [x] Add tests for valid fixture packaging, expected zip entries, missing `model.pth`, invalid generated image, and existing zip behavior.

## Done When

- [x] Packaging refuses invalid or incomplete submissions.
- [x] A valid fixture produces an inspectable zip with expected entries.
- [x] Packaging tests pass independently.
