from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .validation import validate_submission


def required_artifacts(root: str | Path = ".") -> list[Path]:
    root = Path(root)
    return [
        root / "generated_images",
        root / "src" / "brainrot_diffusion",
        root / "scripts",
        root / "README.md",
        root / "requirements.txt",
    ]


def check_manifest(root: str | Path = ".", *, checkpoint: str | Path = "model.pth") -> list[str]:
    missing = [str(path) for path in required_artifacts(root) if not path.exists()]
    checkpoint_path = Path(root) / checkpoint
    if not checkpoint_path.exists():
        missing.append(str(checkpoint_path))
    return missing


def package_submission(
    *,
    root: str | Path = ".",
    generate_csv: str | Path = "generate.csv",
    output_dir: str | Path = "generated_images",
    checkpoint: str | Path = "model.pth",
    strict_count: bool = True,
    expected_count: int = 2000,
    zip_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    validation = validate_submission(
        generate_csv, output_dir, strict_count=strict_count, expected_count=expected_count
    )
    if not validation.ok:
        raise ValueError(
            "refusing to package invalid generated images: " + "; ".join(validation.errors)
        )
    missing = check_manifest(root, checkpoint=checkpoint)
    if missing:
        raise FileNotFoundError("missing required artifact(s): " + ", ".join(missing))
    manifest = {
        "artifacts": [str(path) for path in required_artifacts(root)],
        "checkpoint": str(Path(root) / checkpoint),
        "validation": validation.to_dict(),
    }
    if zip_path:
        zip_path = Path(zip_path)
        if zip_path.exists() and not overwrite:
            raise FileExistsError(f"zip already exists: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in required_artifacts(root):
                if artifact.is_dir():
                    for child in artifact.rglob("*"):
                        if child.is_file():
                            archive.write(child, child.relative_to(root))
                else:
                    archive.write(artifact, artifact.relative_to(root))
            archive.write(Path(root) / checkpoint, Path(checkpoint))
        manifest["zip_path"] = str(zip_path)
    Path(root, "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
