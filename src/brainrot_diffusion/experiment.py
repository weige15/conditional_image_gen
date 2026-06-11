from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_metadata(
    run_dir: str | Path,
    config: dict[str, Any],
    seed: int,
    checkpoint_path: str | Path | None = None,
    generation_command: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "generation_command": generation_command,
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def summarize_runs(root_dir: str | Path) -> list[dict[str, Any]]:
    root_dir = Path(root_dir)
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        validation_path = run_dir / "validation.json"
        metrics_path = run_dir / "evaluation.json"
        if not validation_path.exists():
            continue
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("valid", False):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        summaries.append({"run_dir": str(run_dir), "validation": validation, "metrics": metrics})
    return summaries
