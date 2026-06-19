"""E3 package archive creation."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .validate import validate_submission


def package_submission(
    *,
    generate_csv: str | Path,
    generated_images: str | Path,
    checkpoint: str | Path,
    student_id: str,
    output_zip: str | Path | None = None,
    project_root: str | Path = ".",
    overwrite: bool = False,
) -> dict[str, object]:
    _validate_student_id(student_id)
    validation = validate_submission(generate_csv, generated_images)
    if not validation.passed:
        raise ValueError("generated images are invalid; refusing to package")

    root = Path(project_root)
    required = [
        root / "README.md",
        root / "requirements.txt",
        root / "scripts",
        root / "src" / "brainrot_diffusion",
        root / "configs",
        Path(checkpoint),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required artifact(s): " + ", ".join(missing))

    package_root = f"HW6_{student_id}"
    zip_path = Path(output_zip) if output_zip is not None else root / f"{package_root}.zip"
    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"package already exists: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        entries.extend(_write_tree(archive, Path(generated_images), f"{package_root}/generated_images"))
        entries.extend(_write_tree(archive, root / "scripts", f"{package_root}/scripts"))
        entries.extend(_write_tree(archive, root / "src" / "brainrot_diffusion", f"{package_root}/src/brainrot_diffusion"))
        entries.extend(_write_tree(archive, root / "configs", f"{package_root}/configs"))
        entries.append(_write_file(archive, Path(checkpoint), f"{package_root}/model.pth"))
        entries.append(_write_file(archive, root / "README.md", f"{package_root}/README.md"))
        entries.append(_write_file(archive, root / "requirements.txt", f"{package_root}/requirements.txt"))
    return {"zip_path": str(zip_path), "entries": entries, "validation": validation.as_dict()}


def _validate_student_id(student_id: str) -> None:
    if not student_id or student_id == "STUDENT_ID" or "placeholder" in student_id.lower():
        raise ValueError("student_id must be a real student ID, not a placeholder")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", student_id):
        raise ValueError("student_id may only contain letters, numbers, underscores, or hyphens")


def _write_tree(archive: zipfile.ZipFile, source: Path, prefix: str) -> list[str]:
    entries: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            arcname = f"{prefix}/{path.relative_to(source).as_posix()}"
            archive.write(path, arcname)
            entries.append(arcname)
    return entries


def _write_file(archive: zipfile.ZipFile, source: Path, arcname: str) -> str:
    archive.write(source, arcname)
    return arcname

