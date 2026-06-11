# Data and Labels

## Goal

Implement CSV parsing, image loading, and stable animal/object/pair label mapping for training and generation.

## Inputs

- `doc/proposal.md`: Data and Labels, generation must preserve `dataset/generate.csv` ids and conditions.
- `doc/detailed-design.md`: Data and Labels module contracts, failure handling, and independent tests.

## Tasks

- [x] Create `src/brainrot_diffusion/conditions.py` with deterministic animal, object, pair, and null-condition mappings.
- [x] Create `src/brainrot_diffusion/data.py` with `BrainrotTrainDataset`, `GenerationRequestDataset`, and generation-request loading.
- [x] Load images as RGB, resize or center-crop defensively to 64x64, convert to tensors, and normalize to `[-1, 1]`.
- [x] Reject missing CSV columns, missing images, duplicate generation ids, invalid images, and unknown labels with clear errors.
- [x] Add isolated tests with tiny temporary CSV/image fixtures for mappings, tensor shape/range, duplicate ids, and missing files.

## Done When

- [x] Training samples return image tensor, `animal_id`, `object_id`, `pair_id`, and source filename.
- [x] Generation requests preserve output filenames and map conditions through saved mappings.
- [x] Data and label tests pass independently.
