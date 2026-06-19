# Performance Diagnosis

## Purpose

- Diagnose likely runtime, memory, and quality bottlenecks before making optimization changes.
- Produce one small, measurable first optimization recommendation for the next implementation loop.
- Avoid blind architecture or hyperparameter changes before baseline training/evaluation numbers exist.

## Diagnosis Scope

- Project: HW6 Brainrot Image Generation.
- Target path: train from scratch, generate 2,000 RGB `64x64` PNGs, validate structure, and evaluate with FID/CLIP-T.
- Scope includes configuration, model size, training loop, sampling loop, evaluator/scorer commands, and recent correctness evidence.
- Scope excludes source-code changes, long GPU training, generated-image creation, official Codabench submission, and metric-score claims.

## Source Documents Read

- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `doc/review-notes.md`
- `configs/default.yaml`
- `src/brainrot_diffusion/train_loop.py`
- `src/brainrot_diffusion/model.py`
- `src/brainrot_diffusion/diffusion.py`
- `src/brainrot_diffusion/sample.py`
- `src/brainrot_diffusion/evaluate.py`
- `scoring_program/score.py`

## Current Correctness Status

- Recent verified correctness from `doc/review-notes.md`: `python -m pytest` passed with 53 tests, `python -m compileall src scripts tests` passed, and `git diff --check` passed.
- Current task progress marks all implementation modules complete.
- Structural validation on the real final output has not been run because `generated_images/` is absent.
- Final package validation has not been run because `generated_images/` and `model.pth` are absent.
- Official or local scorer metrics have not been run because no generated outputs exist.
- Current worktree includes uncommitted review repairs and docs; these are not known correctness failures.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | Missing FID and CLIP-T | `python scripts/evaluate.py --generate-csv dataset/generate.csv --output-dir generated_images --reference-dir hw6_reference --report-path reports/evaluation.json`; Codabench scorer after generated outputs exist | No; no generated outputs or score report exist |
| Runtime | Missing full-train and full-generation runtime | `python scripts/train.py --config configs/default.yaml`; `python scripts/generate.py --checkpoint <checkpoint> --config configs/default.yaml --overwrite` | No; long GPU commands not run in this diagnosis |
| Memory | Missing peak GPU memory | Lab-server run with GPU memory monitoring during training/generation | No; GPU run not available in this session |

Additional read-only facts:

- Dataset count: `dataset/train.csv` has 4,799 training rows; `dataset/generate.csv` has 2,000 generation rows.
- Default model parameter count: 5,101,891 trainable parameters, measured with `PYTHONPATH=src python -c ...`.
- Assignment PDF baseline note in `doc/problem-brief.md`: baseline models listed around 62.68M parameters on RTX 4070 12GB, but methods are not disclosed.
- No `checkpoints/`, `generated_images/`, `reports/`, `score_input/`, or `model.pth` files were found locally during diagnosis.

## Expected Performance Target

- Structural target: exactly 2,000 generated PNG files, RGB mode, `64x64`, filenames matching `dataset/generate.csv`.
- FID grading target: lower is better; partial-credit threshold starts at `<= 90.0142`, stronger thresholds at `<= 75.0642`, `<= 58.0755`, and `<= 49.2545`.
- CLIP-T grading target: higher is better; partial-credit threshold starts at `>= 0.2170`, stronger thresholds at `>= 0.2536`, `>= 0.2618`, and `>= 0.2703`.
- Runtime target is not specified by the assignment; practical goal is to fit lab-server GPU memory and complete training/generation before submission.

## Gap Analysis

- The main gap is missing measurement: there is no trained checkpoint, generated output, FID, CLIP-T, generation runtime, or GPU memory baseline.
- The implementation is structurally ready to benchmark, but not submission-ready.
- Runtime/memory risk is visible before benchmarking: `training.mixed_precision: true` exists in config, but the training loop does not use AMP.
- Quality risk is also visible: the default model is about 5.1M parameters, much smaller than the assignment's disclosed baseline size, and has no attention or augmentation.
- Local `evaluate.py` intentionally skips real FID/CLIP-T; official scoring requires generated outputs and the scorer/Codabench path.

## Benchmark or Evaluator Details

- Correctness gates:
  - `python -m pytest`
  - `python -m compileall src scripts tests`
  - `python scripts/validate_submission.py --generate-csv dataset/generate.csv --output-dir generated_images --report-json reports/validation.json --expected-count 2000`
- Local scorer-input preparation:
  - `python scripts/prepare_score_input.py --generate-csv dataset/generate.csv --generated-images generated_images --score-input-dir score_input --test-mu hw6_reference/test_mu.npy --test-sigma hw6_reference/test_sigma.npy --scores fid --overwrite`
- Provided scorer:
  - `python3 score.py --input_dir $input --output_dir $output --config config.json`
  - Must run under `scoring_program/` with Codabench-style `ref/` and `res/` layout.
  - Uses CUDA device `cuda:0` and pretrained evaluation models; it is evaluation-only, not part of the main generator.
- Official metric authority remains Codabench, especially for CLIP-T.

## Observed Bottlenecks

- No measured runtime or score bottleneck exists yet; the first required step is a lab-server baseline.
- Training throughput bottleneck likely: `configs/default.yaml` enables `training.mixed_precision: true`, but `src/brainrot_diffusion/train_loop.py` performs full-precision forward/backward/optimizer steps.
- Memory bottleneck likely: default batch size is 256 with a pixel-space UNet and 64x64 RGB tensors; full precision may force reducing batch size on the lab GPU.
- Quality bottleneck likely: default model is compact and may underfit or generate weak condition fidelity without larger capacity, attention, augmentation, or tuning.
- Evaluation bottleneck known: local `evaluate.py` reports metric skips instead of computing FID/CLIP-T, so scorer/Codabench setup is required for real metrics.

## Likely Causes

### Cause 1: Unused Mixed Precision Configuration

- Type: Implementation inefficiency.
- Evidence: `configs/default.yaml` sets `training.mixed_precision: true`, but `train_loop.py` does not use `torch.autocast`, `GradScaler`, or any AMP path.
- Affected modules: `src/brainrot_diffusion/train_loop.py`, `configs/default.yaml`, training tests.
- Risk: Low to medium; AMP changes numeric behavior, but it is standard for CUDA training and can be disabled on CPU.
- Confidence: High.

### Cause 2: Conservative Model Capacity

- Type: Algorithmic limitation / parameter tuning issue.
- Evidence: Default model has 5,101,891 trainable parameters; assignment notes a baseline around 62.68M parameters. The model has two downsampling levels, residual blocks, and no attention.
- Affected modules: `src/brainrot_diffusion/model.py`, `configs/default.yaml`, checkpoint compatibility, training runtime.
- Risk: Medium to high; increasing capacity improves quality potential but can break GPU memory or slow iteration.
- Confidence: Medium.

### Cause 3: Missing Real Metric Feedback Loop

- Type: Benchmark/evaluator mismatch / missing measurement.
- Evidence: Local `evaluate.py` explicitly skips FID when reference stats exist and tells the user to prepare scorer input or run Codabench; CLIP-T proxy is not implemented.
- Affected modules: `src/brainrot_diffusion/evaluate.py`, `scripts/prepare_score_input.py`, `scoring_program/score.py`, lab-server workflow.
- Risk: Medium; optimizing against loss alone can improve denoising while not improving FID/CLIP-T.
- Confidence: High.

### Cause 4: Minimal Data Augmentation

- Type: Algorithmic limitation / overfitting risk.
- Evidence: `data.py` only converts/resizes images; default config has no random flips, color jitter, or other augmentation. Dataset has only 4,799 images across 100 pairs.
- Affected modules: `src/brainrot_diffusion/data.py`, `configs/default.yaml`, data tests.
- Risk: Medium; augmentation may improve FID but can hurt object/animal fidelity if too aggressive.
- Confidence: Medium.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Implement CUDA AMP honoring `training.mixed_precision` in the training loop | Higher steps/sec and lower memory, enabling larger batch or more steps on lab GPU | Low/medium numeric risk; disable on CPU or unsupported devices | `src/brainrot_diffusion/train_loop.py`, `tests/test_train_loop.py`, maybe `README.md` | `python -m pytest tests/test_train_loop.py`; lab-server before/after train smoke with `steps_per_sec` and peak GPU memory |
| 2 | Add checkpoint-time sample-grid or lightweight metric loop for fixed validation prompts | Faster visual/metric feedback while training; helps avoid wasting long runs | Low if kept optional; writes artifacts if enabled | `train_loop.py`, `sample.py`, config/docs | Unit smoke plus lab-server sample generation from fixed checkpoint |
| 3 | Increase model capacity after AMP baseline, e.g. higher `base_channels` or attention block | Better FID/CLIP-T potential if current model underfits | Medium/high memory and checkpoint compatibility risk | `configs/default.yaml`, `model.py`, model tests | Train same budget, compare validation images/FID/Codabench; rollback if OOM or score worsens |
| 4 | Add conservative horizontal-flip augmentation if semantically safe | Better data coverage and FID potential | Medium; may hurt asymmetric object semantics less than color jitter would | `data.py`, config, data tests | Train same budget, compare visual/FID; rollback if condition fidelity degrades |

## Recommended First Optimization

- Implement CUDA AMP in `src/brainrot_diffusion/train_loop.py`, controlled by existing `training.mixed_precision`.
- Keep CPU behavior unchanged and keep AMP disabled automatically when CUDA is unavailable.
- Use `torch.amp.autocast` and `GradScaler` for forward/loss/backward/optimizer steps.
- Measure before/after on the lab server with the same config, seed, batch size, and short fixed step count.
- Keep the change only if correctness tests pass and lab-server throughput or memory improves without non-finite losses.

## Exact Prompt for Implementation Loop

```text
Use [$implementation-loop-manager] in optimization mode.

Read:
- doc/performance-diagnosis.md
- doc/performance-log.md if present
- doc/test-plan.md
- doc/quality-gates.md
- src/brainrot_diffusion/train_loop.py
- tests/test_train_loop.py
- configs/default.yaml

Implement only the first recommended optimization hypothesis: add CUDA AMP support to the training loop honoring training.mixed_precision.

Rules:
- Do not change unrelated modules.
- Preserve correctness.
- Add or update tests only if needed.
- Run correctness tests first.
- Run the benchmark or evaluator after the change.
- Compare before/after numbers.
- Keep the change only if the target metric improves without breaking correctness.
- If the result is worse, revert the optimization and record why.
- Update doc/performance-log.md.
- Report changed files, commands run, before/after metrics, and whether the change was kept.
```

## Stop Conditions

- Stop optimization if `python -m pytest` or the focused training tests fail.
- Stop optimization if AMP causes non-finite loss, broken checkpoint metadata, or generation incompatibility.
- Stop optimization if the lab GPU cannot run the baseline short training command, because there is no valid before/after comparison.
- Stop quality tuning until at least one validated generated set or scorer/Codabench result exists.

## Risks and Warnings

- AMP can change numerical behavior; verify finite loss and checkpoint generation before long training.
- Larger models are tempting but should wait until AMP and baseline measurements exist.
- Local training loss is not the grading metric; FID/CLIP-T or visual samples must guide later quality changes.
- Local FID/CLIP-T setup may require CUDA and pretrained evaluator weights; keep it separate from the assignment's scratch-generator constraint.
- Current docs contain some stale pre-implementation wording; trust live files and recent review notes over old status text.
- Generated artifacts and checkpoints should not be overwritten without explicit flags.

## Open Questions

- Which lab-server GPU model and VRAM are available?
- What batch size fits before AMP, and what batch size fits after AMP?
- What training duration is available before the deadline?
- What is the first lab-server baseline: steps/sec, peak memory, sample quality, and any scorer result?
- Will the final run use only the provided Brainrot Dataset, or will extra data be considered later?
- What student ID should be used for final packaging?
