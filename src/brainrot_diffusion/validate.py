"""Strict generated-image structure validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .data import read_generation_csv


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    expected_count: int
    actual_count: int
    findings: tuple[ValidationFinding, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def validate_submission(
    generate_csv: str | Path,
    output_dir: str | Path,
    *,
    image_size: int = 64,
    expected_count: int | None = None,
    report_json: str | Path | None = None,
) -> ValidationResult:
    requests = read_generation_csv(generate_csv)
    expected = [request.id for request in requests]
    expected_set = set(expected)
    findings: list[ValidationFinding] = []
    if expected_count is not None and len(expected) != expected_count:
        findings.append(
            ValidationFinding(
                "expected-count",
                f"generate CSV contains {len(expected)} rows, expected {expected_count}",
                str(generate_csv),
            )
        )

    root = Path(output_dir)
    actual = _actual_files(root)
    actual_set = set(actual)
    for filename in sorted(expected_set - actual_set):
        findings.append(ValidationFinding("missing", "expected file is missing", filename))
    for filename in sorted(actual_set - expected_set):
        findings.append(ValidationFinding("extra", "unexpected file is present", filename))

    for filename in expected:
        path = root / filename
        if filename not in actual_set:
            continue
        findings.extend(_check_image(path, image_size))

    result = ValidationResult(
        passed=not findings,
        expected_count=len(expected),
        actual_count=len(actual),
        findings=tuple(findings),
    )
    if report_json is not None:
        write_json(report_json, result.as_dict())
    return result


def write_json(path: str | Path, data: dict[str, object]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _actual_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    if not root.is_dir():
        return [root.name]
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _check_image(path: Path, image_size: int) -> Iterable[ValidationFinding]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                yield ValidationFinding("format", f"image format is {image.format}, expected PNG", str(path))
            if image.mode != "RGB":
                yield ValidationFinding("mode", f"image mode is {image.mode}, expected RGB", str(path))
            if image.size != (image_size, image_size):
                yield ValidationFinding(
                    "size",
                    f"image size is {image.size}, expected {(image_size, image_size)}",
                    str(path),
                )
    except Exception as exc:
        yield ValidationFinding("corrupt", f"could not open image: {exc}", str(path))

