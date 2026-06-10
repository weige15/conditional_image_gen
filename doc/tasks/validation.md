# Validation

## Goal

Validate generated submission structure before metric evaluation, packaging, or upload.

## Inputs

- `doc/proposal.md`: validate file count, filenames, image mode, image size, and exact match to `generate.csv`.
- `doc/detailed-design.md`: validation owns pass/fail reports for missing, extra, malformed, duplicate, wrong-mode, and wrong-size images.

## Tasks

- [ ] Implement a validation entrypoint that reads expected ids from `generate.csv`.
- [ ] Check duplicate ids, missing files, extra PNG files, and final expected count of exactly 2,000 images.
- [ ] Open each image and verify PNG format, RGB mode, and 64x64 resolution.
- [ ] Emit both human-readable and machine-readable validation reports.
- [ ] Return a failing process status when validation fails.
- [ ] Add tests for valid outputs, missing file, extra file, wrong size, wrong mode, duplicate id, and wrong extension.

## Done When

- [ ] Final validation can prove `generated_images/` satisfies the assignment image contract.
- [ ] Validation tests pass using small fixture directories.
