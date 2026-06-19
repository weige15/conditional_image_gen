# Data Loading

## Goal

Implement assignment CSV parsing and RGB image loading so training and generation workflows receive validated rows, requests, and tensors.

## Inputs

- `doc/proposal.md`: Load `train.csv`, `generate.csv`, and `dataset/trainset/`; preserve output filenames from generation CSV.
- `doc/high-level-design.md`: Data Loading owns `src/brainrot_diffusion/data.py` and outputs training samples and generation requests.
- `doc/detailed-design.md`: Data Loading uses `csv.DictReader`, Pillow RGB conversion, tensor normalization, and ordered request records.
- `doc/test-plan.md`: Data tests cover CSV columns, duplicates, missing images, RGB tensor shape/range, and generation request ordering.

## Write Scope

`src/brainrot_diffusion/data.py`, data-loading tests under `tests/`, and tiny CSV/image fixtures inside tests.

## Read Scope

`dataset/train.csv`, `dataset/generate.csv`, `dataset/trainset/`, `doc/detailed-design.md`, and `src/brainrot_diffusion/config.py` after it exists.

## Dependencies

Configuration for image size/path settings. Conditions for mapping labels to IDs when batches need condition IDs.

## Tasks

- [x] Implement train CSV and generation CSV readers with required-column and duplicate-ID validation.
- [x] Implement RGB image loading from `dataset/trainset/{id}` with `64x64` handling and `[-1, 1]` tensor normalization.
- [x] Return ordered generation request records with `id`, `animal`, `object`, and `prompt`.
- [x] Fail clearly for missing images, unreadable images, empty CSVs, and malformed rows.
- [x] Add tests with tiny CSVs and tiny RGB images for valid and invalid inputs.

## Tests and Quality Gates

- [x] `python -m pytest tests/test_data.py`
- [x] Data portion of `python -m pytest` passes without GPU.

## Done When

- [x] Training samples return image tensors and row metadata.
- [x] Generation requests preserve CSV filenames and order.
- [x] Data-loading tests pass.
