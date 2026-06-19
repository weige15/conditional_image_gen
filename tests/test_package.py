from __future__ import annotations

from pathlib import Path
import zipfile

from PIL import Image
import pytest

from brainrot_diffusion.package import package_submission


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


def _project_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "scripts").mkdir()
    (path / "scripts" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (path / "src" / "brainrot_diffusion").mkdir(parents=True)
    (path / "src" / "brainrot_diffusion" / "__init__.py").write_text("", encoding="utf-8")
    (path / "configs").mkdir()
    (path / "configs" / "default.yaml").write_text("data: {}\n", encoding="utf-8")
    (path / "README.md").write_text("readme\n", encoding="utf-8")
    (path / "requirements.txt").write_text("torch\n", encoding="utf-8")
    return path


def test_package_valid_fixture_contains_required_entries(tmp_path: Path) -> None:
    root = _project_root(tmp_path / "project")
    csv_path = _generate_csv(tmp_path / "generate.csv")
    generated = _valid_outputs(tmp_path / "generated")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    zip_path = tmp_path / "HW6_A12345678.zip"

    result = package_submission(
        generate_csv=csv_path,
        generated_images=generated,
        checkpoint=checkpoint,
        student_id="A12345678",
        output_zip=zip_path,
        project_root=root,
    )

    assert Path(result["zip_path"]).exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "HW6_A12345678/generated_images/000101.png" in names
    assert "HW6_A12345678/model.pth" in names
    assert "HW6_A12345678/README.md" in names
    assert "HW6_A12345678/requirements.txt" in names


def test_package_rejects_invalid_output_missing_checkpoint_and_placeholder_id(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _project_root(root)
    csv_path = _generate_csv(tmp_path / "generate.csv")
    generated = tmp_path / "generated"
    generated.mkdir()
    Image.new("L", (64, 64)).save(generated / "000101.png")

    with pytest.raises(ValueError, match="student_id"):
        package_submission(
            generate_csv=csv_path,
            generated_images=generated,
            checkpoint=tmp_path / "model.pth",
            student_id="STUDENT_ID",
            project_root=root,
        )

    with pytest.raises(ValueError, match="invalid"):
        package_submission(
            generate_csv=csv_path,
            generated_images=generated,
            checkpoint=tmp_path / "model.pth",
            student_id="A12345678",
            project_root=root,
        )

    valid = _valid_outputs(tmp_path / "valid_generated")
    with pytest.raises(FileNotFoundError, match="model.pth"):
        package_submission(
            generate_csv=csv_path,
            generated_images=valid,
            checkpoint=tmp_path / "model.pth",
            student_id="A12345678",
            project_root=root,
        )
