from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from brainrot_diffusion.evaluate import evaluate_outputs, prepare_score_input


def _generate_csv(path: Path) -> Path:
    path.write_text(
        "id,animal,object,prompt\n000101.png,cat,chair,a cat and a chair\n",
        encoding="utf-8",
    )
    return path


def _valid_outputs(path: Path) -> Path:
    path.mkdir()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(path / "000101.png")
    return path


def test_evaluation_stops_on_invalid_outputs(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = tmp_path / "missing"

    report = evaluate_outputs(generate_csv=csv_path, output_dir=output, reference_dir=tmp_path / "ref")

    assert report["validation"]["passed"] is False
    assert report["skipped"]["fid"] == "validation failed"


def test_evaluation_reports_missing_reference_and_clip_skip(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = _valid_outputs(tmp_path / "generated")

    report = evaluate_outputs(generate_csv=csv_path, output_dir=output, reference_dir=tmp_path / "ref")

    assert report["validation"]["passed"] is True
    assert report["skipped"]["fid"] == "missing reference stats"
    assert "official CLIP-T" in report["skipped"]["clip_t"]


def test_prepare_score_input_validates_and_copies_layout(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = _valid_outputs(tmp_path / "generated")
    ref = tmp_path / "ref"
    ref.mkdir()
    np.save(ref / "test_mu.npy", np.zeros(2))
    np.save(ref / "test_sigma.npy", np.eye(2))

    result = prepare_score_input(
        generate_csv=csv_path,
        generated_images=output,
        score_input_dir=tmp_path / "score_input",
        test_mu=ref / "test_mu.npy",
        test_sigma=ref / "test_sigma.npy",
        overwrite=False,
    )

    assert Path(result["ref_dir"], "test_mu.npy").exists()
    assert Path(result["res_dir"], "000101.png").exists()

    with pytest.raises(FileExistsError):
        prepare_score_input(
            generate_csv=csv_path,
            generated_images=output,
            score_input_dir=tmp_path / "score_input",
            test_mu=ref / "test_mu.npy",
            test_sigma=ref / "test_sigma.npy",
            overwrite=False,
        )


def test_prepare_score_input_preflights_reference_files_before_overwrite(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = _valid_outputs(tmp_path / "generated")
    ref = tmp_path / "ref"
    ref.mkdir()
    np.save(ref / "test_mu.npy", np.zeros(2))
    score_input = tmp_path / "score_input"
    score_input.mkdir()
    sentinel = score_input / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing scorer reference"):
        prepare_score_input(
            generate_csv=csv_path,
            generated_images=output,
            score_input_dir=score_input,
            test_mu=ref / "test_mu.npy",
            test_sigma=ref / "missing_sigma.npy",
            overwrite=True,
        )

    assert sentinel.exists()
