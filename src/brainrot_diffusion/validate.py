from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .data import read_csv_rows


def expected_filenames(generate_csv: str | Path) -> list[str]:
    rows = read_csv_rows(generate_csv, {"id"})
    names = [row["id"] for row in rows]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate ids in generation CSV: {duplicates[:5]}")
    return names


def validate_submission(
    generate_csv: str | Path,
    output_dir: str | Path,
    image_size: int = 64,
    smoke: bool = False,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    expected = expected_filenames(generate_csv)
    actual = sorted(path.name for path in output_dir.glob("*.png")) if output_dir.exists() else []
    missing = sorted(set(expected).difference(actual))
    extra = sorted(set(actual).difference(expected))
    errors: list[str] = []
    if missing:
        errors.append(f"Missing {len(missing)} PNG files")
    if extra:
        errors.append(f"Found {len(extra)} extra PNG files")
    if len(actual) != len(expected):
        errors.append(f"Expected {len(expected)} PNG files, found {len(actual)}")
    checked = 0
    for name in expected:
        path = output_dir / name
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.mode != "RGB":
                    errors.append(f"{name} has mode {image.mode}, expected RGB")
                if image.size != (image_size, image_size):
                    errors.append(f"{name} has size {image.size}, expected {(image_size, image_size)}")
        except (UnidentifiedImageError, OSError) as exc:
            errors.append(f"{name} is not a readable PNG: {exc}")
        checked += 1
        if smoke and checked >= len(expected):
            break
    report = {
        "valid": not errors,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "extra": extra,
        "errors": errors,
        "image_size": image_size,
    }
    if report_json:
        path = Path(report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError("; ".join(errors[:5]))
    return report
