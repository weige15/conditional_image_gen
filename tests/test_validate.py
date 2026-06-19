from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from brainrot_diffusion.validate import validate_submission


def _generate_csv(path: Path) -> Path:
    path.write_text(
        "id,animal,object,prompt\n"
        "000101.png,cat,chair,a cat and a chair\n"
        "000102.png,dog,pizza,a dog and a pizza\n",
        encoding="utf-8",
    )
    return path


def _valid_outputs(path: Path) -> Path:
    path.mkdir()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(path / "000101.png")
    Image.new("RGB", (64, 64), (0, 255, 0)).save(path / "000102.png")
    return path


def test_valid_fixture_passes_and_writes_report(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = _valid_outputs(tmp_path / "generated")
    report_path = tmp_path / "report.json"

    result = validate_submission(csv_path, output, report_json=report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.passed
    assert report["passed"] is True
    assert report["expected_count"] == 2


def test_missing_and_extra_files_fail(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = tmp_path / "generated"
    output.mkdir()
    Image.new("RGB", (64, 64)).save(output / "000101.png")
    Image.new("RGB", (64, 64)).save(output / "extra.png")

    result = validate_submission(csv_path, output)
    codes = {finding.code for finding in result.findings}

    assert not result.passed
    assert {"missing", "extra"} <= codes


def test_wrong_size_mode_and_corrupt_png_fail(tmp_path: Path) -> None:
    csv_path = tmp_path / "generate.csv"
    csv_path.write_text(
        "id,animal,object,prompt\n"
        "wrong_size.png,cat,chair,a cat and a chair\n"
        "wrong_mode.png,cat,chair,a cat and a chair\n"
        "corrupt.png,cat,chair,a cat and a chair\n",
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    output.mkdir()
    Image.new("RGB", (65, 64)).save(output / "wrong_size.png")
    Image.new("L", (64, 64)).save(output / "wrong_mode.png")
    (output / "corrupt.png").write_bytes(b"not a png")

    result = validate_submission(csv_path, output)
    codes = {finding.code for finding in result.findings}

    assert not result.passed
    assert {"size", "mode", "corrupt"} <= codes


def test_expected_count_mismatch_fails(tmp_path: Path) -> None:
    csv_path = _generate_csv(tmp_path / "generate.csv")
    output = _valid_outputs(tmp_path / "generated")

    result = validate_submission(csv_path, output, expected_count=2000)

    assert not result.passed
    assert result.findings[0].code == "expected-count"


def test_exact_filename_set_with_deterministic_random_names(tmp_path: Path) -> None:
    rng = random.Random(42)
    names = [f"{rng.randrange(100000):06d}.png" for _ in range(5)]
    csv_path = tmp_path / "generate.csv"
    csv_path.write_text(
        "id,animal,object,prompt\n"
        + "".join(f"{name},cat,chair,a cat and a chair\n" for name in names),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    output.mkdir()
    for name in names:
        Image.new("RGB", (64, 64)).save(output / name)

    assert validate_submission(csv_path, output).passed
    Image.new("RGB", (64, 64)).save(output / "extra.png")
    assert not validate_submission(csv_path, output).passed
