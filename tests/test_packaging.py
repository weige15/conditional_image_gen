from __future__ import annotations

import pytest

from brainrot_diffusion.packaging import package_submission

from .conftest import write_csv, write_png


def test_packaging_refuses_invalid_images(tmp_path) -> None:
    csv_path = tmp_path / "generate.csv"
    write_csv(csv_path, [{"id": "a.png", "animal": "cat", "object": "car", "prompt": "x"}])
    out = tmp_path / "generated_images"
    out.mkdir()
    with pytest.raises(ValueError, match="invalid"):
        package_submission(root=tmp_path, generate_csv=csv_path, output_dir=out, strict_count=False)


def test_packaging_checks_manifest(tmp_path) -> None:
    csv_path = tmp_path / "generate.csv"
    write_csv(csv_path, [{"id": "a.png", "animal": "cat", "object": "car", "prompt": "x"}])
    out = tmp_path / "generated_images"
    out.mkdir()
    write_png(out / "a.png")
    with pytest.raises(FileNotFoundError, match="README"):
        package_submission(root=tmp_path, generate_csv=csv_path, output_dir=out, strict_count=False)
