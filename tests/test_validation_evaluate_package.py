from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from brainrot_diffusion.evaluate import evaluate_submission
from brainrot_diffusion.experiment import summarize_runs, write_run_metadata
from brainrot_diffusion.package import package_submission
from brainrot_diffusion.validate import validate_submission
from conftest import write_png


def test_validation_failures_and_evaluation_skip(tiny_dataset, tmp_path: Path):
    out = tmp_path / "generated"
    write_png(out / "000001.png")
    write_png(out / "000002.png")
    report = validate_submission(tiny_dataset["generate_csv"], out)
    assert report["valid"]
    write_png(out / "extra.png")
    with pytest.raises(ValueError, match="extra"):
        validate_submission(tiny_dataset["generate_csv"], out)
    os.remove(out / "extra.png")
    write_png(out / "000002.png", size=32)
    with pytest.raises(ValueError, match="size"):
        validate_submission(tiny_dataset["generate_csv"], out)
    write_png(out / "000002.png")
    report = evaluate_submission(tiny_dataset["generate_csv"], out, reference_dir=tmp_path / "missing", run_fid=True)
    assert report["metrics"]["fid"]["status"] == "skipped"


def test_package_and_experiment_helpers(tiny_dataset, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    out = tmp_path / "generated"
    write_png(out / "000001.png")
    write_png(out / "000002.png")
    model = tmp_path / "model.pth"
    model.write_bytes(b"weights")
    zip_path = tmp_path / "HW6_TEST.zip"
    package_submission(tiny_dataset["generate_csv"], out, model, zip_path, overwrite=True)
    with zipfile.ZipFile(zip_path) as zipf:
        names = set(zipf.namelist())
    assert "HW6_TEST/generated/000001.png" not in names
    assert "HW6_TEST/generated_images/000001.png" in names
    assert "HW6_TEST/model.pth" in names
    with pytest.raises(FileExistsError):
        package_submission(tiny_dataset["generate_csv"], out, model, zip_path, overwrite=False)

    run_dir = tmp_path / "runs" / "run1"
    write_run_metadata(run_dir, {"a": 1}, seed=3, checkpoint_path=model, generation_command="generate")
    (run_dir / "validation.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(json.dumps({"metrics": {"fid": 1.0}}), encoding="utf-8")
    assert len(summarize_runs(tmp_path / "runs")) == 1
