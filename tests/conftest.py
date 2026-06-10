from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_png(path: Path, size: tuple[int, int] = (64, 64), mode: str = "RGB") -> None:
    image = Image.new(mode, size, color=(128, 64, 32) if mode == "RGB" else 128)
    image.save(path, format="PNG")
