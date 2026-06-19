"""Validation-first local evaluation reporting and scorer input preparation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping

from .validate import ValidationResult, validate_submission, write_json


def evaluate_outputs(
    *,
    generate_csv: str | Path,
    output_dir: str | Path,
    reference_dir: str | Path,
    report_path: str | Path | None = None,
    run_fid: bool = True,
    run_clip_proxy: bool = False,
) -> dict[str, object]:
    validation = validate_submission(generate_csv, output_dir)
    metrics: dict[str, object] = {}
    skipped: dict[str, str] = {}
    if not validation.passed:
        skipped["fid"] = "validation failed"
        skipped["clip_t"] = "validation failed"
    else:
        reference = Path(reference_dir)
        if not run_fid:
            skipped["fid"] = "disabled"
        elif not (reference / "test_mu.npy").exists() or not (reference / "test_sigma.npy").exists():
            skipped["fid"] = "missing reference stats"
        else:
            skipped["fid"] = "not run locally; prepare scorer input or run Codabench scorer"

        if run_clip_proxy:
            skipped["clip_t"] = "local CLIP proxy is not implemented without explicit approval"
        else:
            skipped["clip_t"] = "disabled; official CLIP-T is computed on Codabench"

    report = {
        "validation": validation.as_dict(),
        "metrics": metrics,
        "skipped": skipped,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def prepare_score_input(
    *,
    generate_csv: str | Path,
    generated_images: str | Path,
    score_input_dir: str | Path,
    test_mu: str | Path,
    test_sigma: str | Path,
    scores: Iterable[str] = ("fid",),
    overwrite: bool = False,
    config_json: str | Path | None = None,
) -> dict[str, object]:
    validation = validate_submission(generate_csv, generated_images)
    if not validation.passed:
        raise ValueError("generated images are invalid; refusing to prepare scorer input")
    root = Path(score_input_dir)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"score input directory already exists: {root}")
        shutil.rmtree(root)
    ref_dir = root / "ref"
    res_dir = root / "res"
    ref_dir.mkdir(parents=True)
    res_dir.mkdir(parents=True)

    shutil.copy2(test_mu, ref_dir / "test_mu.npy")
    shutil.copy2(test_sigma, ref_dir / "test_sigma.npy")
    if config_json is not None:
        shutil.copy2(config_json, ref_dir / "config.json")
    else:
        (ref_dir / "config.json").write_text(
            json.dumps(
                {
                    "scores": list(scores),
                    "image_size": 64,
                    "num_images": validation.expected_count,
                    "ref_mu": "test_mu.npy",
                    "ref_sigma": "test_sigma.npy",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    for image_path in Path(generated_images).iterdir():
        if image_path.is_file():
            shutil.copy2(image_path, res_dir / image_path.name)
    return {
        "input_dir": str(root),
        "ref_dir": str(ref_dir),
        "res_dir": str(res_dir),
        "validation": validation.as_dict(),
    }


def validation_failed(report: Mapping[str, object]) -> bool:
    validation = report.get("validation")
    return isinstance(validation, Mapping) and validation.get("passed") is False

