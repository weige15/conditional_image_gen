from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .data import read_generate_records


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_images: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_images": self.checked_images,
        }


def validate_submission(
    generate_csv: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int = 2000,
    strict_count: bool = True,
    report_json: str | Path | None = None,
) -> ValidationReport:
    errors: list[str] = []
    try:
        records = read_generate_records(generate_csv)
    except Exception as exc:
        return ValidationReport(False, [str(exc)], [], 0)
    expected_ids = [record.id for record in records]
    expected_set = set(expected_ids)
    if any(not image_id.lower().endswith(".png") for image_id in expected_ids):
        errors.append("all generate.csv ids must end with .png")
    if strict_count and len(expected_ids) != expected_count:
        errors.append(f"generate.csv row count is {len(expected_ids)}, expected {expected_count}")

    out_dir = Path(output_dir)
    if not out_dir.exists():
        errors.append(f"output directory does not exist: {out_dir}")
        return ValidationReport(False, errors, [], 0)
    files = [path for path in out_dir.iterdir() if path.is_file()]
    png_names = {path.name for path in files if path.suffix.lower() == ".png"}
    non_png = sorted(path.name for path in files if path.suffix.lower() not in {".png", ".json"})
    missing = sorted(expected_set - png_names)
    extra = sorted(png_names - expected_set)
    if non_png:
        errors.append("wrong extension file(s): " + ", ".join(non_png[:10]))
    if missing:
        errors.append("missing generated image(s): " + ", ".join(missing[:10]))
    if extra:
        errors.append("extra PNG file(s): " + ", ".join(extra[:10]))
    if strict_count and len(png_names & expected_set) != expected_count:
        errors.append(
            f"matched PNG count is {len(png_names & expected_set)}, expected {expected_count}"
        )

    checked = 0
    for image_id in expected_ids:
        path = out_dir / image_id
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    errors.append(f"{image_id} is not PNG format")
                if image.mode != "RGB":
                    errors.append(f"{image_id} has mode {image.mode}, expected RGB")
                if image.size != (64, 64):
                    errors.append(f"{image_id} has size {image.size}, expected (64, 64)")
                checked += 1
        except Exception as exc:
            errors.append(f"{image_id} cannot be opened: {exc}")
    report = ValidationReport(ok=not errors, errors=errors, checked_images=checked)
    if report_json:
        Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(report_json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
