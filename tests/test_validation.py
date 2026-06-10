from __future__ import annotations

from PIL import Image

from brainrot_diffusion.validation import validate_submission

from .conftest import write_csv, write_png


def make_generate(path):
    write_csv(
        path,
        [
            {"id": "a.png", "animal": "cat", "object": "car", "prompt": "a cat and a car"},
            {"id": "b.png", "animal": "dog", "object": "chair", "prompt": "a dog and a chair"},
        ],
    )


def test_valid_and_invalid_outputs(tmp_path) -> None:
    csv_path = tmp_path / "generate.csv"
    make_generate(csv_path)
    out = tmp_path / "out"
    out.mkdir()
    write_png(out / "a.png")
    write_png(out / "b.png")
    assert validate_submission(csv_path, out, strict_count=False).ok
    (out / "extra.png").write_bytes((out / "a.png").read_bytes())
    report = validate_submission(csv_path, out, strict_count=False)
    assert not report.ok
    assert any("extra PNG" in error for error in report.errors)


def test_wrong_size_mode_missing_and_extension(tmp_path) -> None:
    csv_path = tmp_path / "generate.csv"
    make_generate(csv_path)
    out = tmp_path / "out"
    out.mkdir()
    write_png(out / "a.png", size=(32, 32))
    Image.new("L", (64, 64), color=0).save(out / "b.png", format="PNG")
    (out / "note.txt").write_text("bad", encoding="utf-8")
    report = validate_submission(csv_path, out, strict_count=False)
    assert not report.ok
    assert any("size" in error for error in report.errors)
    assert any("mode" in error for error in report.errors)
    assert any("wrong extension" in error for error in report.errors)
