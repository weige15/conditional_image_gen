# Packaging

## Goal

Implement final E3 zip creation after generated outputs and required artifacts pass validation.

## Inputs

- `doc/proposal.md`: Final E3 package must include generated images, scripts, `model.pth`, `README.md`, and `requirements.txt`.
- `doc/high-level-design.md`: Packaging owns `src/brainrot_diffusion/package.py` and refuses invalid outputs.
- `doc/detailed-design.md`: Packaging validates images, requires real student ID, verifies artifacts, builds a manifest, and writes `HW6_{student_id}.zip`.
- `doc/test-plan.md`: Packaging tests cover invalid-output refusal, required files, zip entries, and placeholder student ID rejection.

## Write Scope

`src/brainrot_diffusion/package.py`, `scripts/package_submission.py`, packaging tests under `tests/`, and package manifest/report helpers.

## Read Scope

Validation module, `generated_images/`, `scripts/`, `src/brainrot_diffusion/`, `configs/`, `model.pth`, `README.md`, `requirements.txt`, and assignment package requirements.

## Dependencies

Validation, CLI Scripts, generated images, final checkpoint or `model.pth`, and real student ID.

## Tasks

- [x] Implement packaging entry point that runs Validation before writing any zip.
- [x] Verify required artifacts and reject missing paths with a clear list.
- [x] Reject placeholder or missing student ID.
- [x] Write zip with required assignment layout and reproducibility files.
- [x] Refuse existing zip unless overwrite is explicit.
- [x] Add tests for tiny valid package, zip entry inspection, invalid generated output, missing `model.pth`, and placeholder student ID.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_package.py`
- [ ] `python scripts/package_submission.py --generate-csv dataset/generate.csv --generated-images generated_images --checkpoint model.pth --student-id STUDENT_ID --overwrite` with real student ID before final submission.

## Done When

- [x] Packaging creates `HW6_{student_id}.zip` only after valid generated outputs.
- [x] Required files are present in the archive.
- [x] Packaging tests pass.
