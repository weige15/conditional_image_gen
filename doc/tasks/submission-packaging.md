# Submission Packaging

## Goal

Assemble validated generated images and reproducibility artifacts into the assignment-required submission structure.

## Inputs

- `doc/proposal.md`: final submission includes code, model weights, README, requirements, and `generated_images/` with exactly 2,000 PNG files.
- `doc/detailed-design.md`: packaging depends on successful validation and must avoid batch deletion or artifact cleanup.

## Tasks

- [ ] Define the required artifact manifest: `generated_images/`, scripts/source code, final checkpoint or `model.pth`, `README.md`, and dependency file.
- [ ] Implement packaging checks that refuse to proceed unless validation passes.
- [ ] Implement artifact presence checks with clear errors for missing weights, README, source, or dependency file.
- [ ] Add README generation or README checklist content for environment setup, training, checkpoint use, and generation command.
- [ ] Add optional zip creation only after the student id and overwrite policy are configured.
- [ ] Add tests for manifest creation, missing artifact failures, and refusal to package invalid generated images.

## Done When

- [ ] Packaging produces or verifies a submission folder matching the HW6 structure.
- [ ] Packaging tests pass without touching real generated images or deleting files.
