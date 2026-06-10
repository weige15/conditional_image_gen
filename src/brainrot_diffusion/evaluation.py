from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .data import read_generate_records
from .validation import validate_submission


def dependency_versions() -> dict[str, str]:
    names = ["numpy", "Pillow", "torch"]
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def prompt_image_pairs(generate_csv: str | Path, output_dir: str | Path) -> list[tuple[str, str]]:
    return [
        (str(Path(output_dir) / record.id), record.prompt or "")
        for record in read_generate_records(generate_csv)
    ]


def evaluate_submission(
    *,
    generate_csv: str | Path,
    output_dir: str | Path,
    test_mu: str | Path | None = None,
    test_sigma: str | Path | None = None,
    strict_count: bool = True,
    expected_count: int = 2000,
    checkpoint_id: str | None = None,
    guidance_scale: float | None = None,
    ddim_steps: int | None = None,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    validation = validate_submission(
        generate_csv, output_dir, strict_count=strict_count, expected_count=expected_count
    )
    report: dict[str, object] = {
        "validation": validation.to_dict(),
        "dependencies": dependency_versions(),
        "checkpoint_id": checkpoint_id,
        "guidance_scale": guidance_scale,
        "ddim_steps": ddim_steps,
        "output_dir": str(output_dir),
        "fid": {"status": "skipped", "reason": "test statistics absent"},
        "clip_t": {"status": "skipped", "reason": "optional CLIP dependency not configured"},
    }
    if (
        validation.ok
        and test_mu
        and test_sigma
        and Path(test_mu).exists()
        and Path(test_sigma).exists()
    ):
        mu = np.load(test_mu)
        sigma = np.load(test_sigma)
        report["fid"] = {
            "status": "skipped",
            "reason": "reference stats loaded, but no local feature extractor is configured",
            "reference_mu_shape": list(mu.shape),
            "reference_sigma_shape": list(sigma.shape),
        }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
