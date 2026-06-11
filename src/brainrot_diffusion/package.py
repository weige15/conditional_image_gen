from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

from .validate import validate_submission


def _add_path(zipf: zipfile.ZipFile, path: Path, arc_root: Path, root_name: str) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                zipf.write(child, Path(root_name) / child.relative_to(arc_root))
    else:
        zipf.write(path, Path(root_name) / path.name)


def required_artifacts(generated_images: Path, scripts_dir: Path, source_dir: Path, checkpoint: Path) -> list[Path]:
    return [generated_images, scripts_dir, source_dir, checkpoint, Path("README.md"), Path("requirements.txt")]


def package_submission(
    generate_csv: str | Path,
    generated_images: str | Path,
    checkpoint: str | Path,
    zip_path: str | Path,
    overwrite: bool = False,
    root_name: str | None = None,
    include_configs: bool = True,
) -> Path:
    generated_images = Path(generated_images)
    checkpoint = Path(checkpoint)
    zip_path = Path(zip_path)
    scripts_dir = Path("scripts")
    source_dir = Path("src/brainrot_diffusion")
    missing = [path for path in required_artifacts(generated_images, scripts_dir, source_dir, checkpoint) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {[str(path) for path in missing]}")
    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"Zip already exists: {zip_path}")
    validate_submission(generate_csv, generated_images)
    root_name = root_name or zip_path.with_suffix("").name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for child in sorted(generated_images.glob("*.png")):
            zipf.write(child, Path(root_name) / "generated_images" / child.name)
        _add_path(zipf, scripts_dir, Path("."), root_name)
        _add_path(zipf, source_dir, Path("."), root_name)
        if include_configs and Path("configs").exists():
            _add_path(zipf, Path("configs"), Path("."), root_name)
        zipf.write(checkpoint, Path(root_name) / "model.pth")
        zipf.write("README.md", Path(root_name) / "README.md")
        zipf.write("requirements.txt", Path(root_name) / "requirements.txt")
    return zip_path
