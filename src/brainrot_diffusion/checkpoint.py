from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "ema",
    "optimizer",
    "step",
    "epoch",
    "config",
    "condition_mappings",
    "diffusion",
    "architecture",
    "seed",
}


def validate_checkpoint_state(state: dict[str, Any]) -> None:
    missing = REQUIRED_CHECKPOINT_KEYS.difference(state)
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")
    mappings = state["condition_mappings"]
    for key in ["animal_to_id", "object_to_id", "pair_to_id"]:
        if key not in mappings:
            raise ValueError(f"Checkpoint condition_mappings missing key {key!r}")


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    validate_checkpoint_state(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} did not contain a dictionary")
    validate_checkpoint_state(state)
    return state


def export_model_pth(checkpoint_path: str | Path, output_path: str | Path) -> None:
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state["model"],
            "ema": state["ema"],
            "optimizer": state["optimizer"],
            "config": state["config"],
            "condition_mappings": state["condition_mappings"],
            "diffusion": state["diffusion"],
            "architecture": state["architecture"],
            "seed": state["seed"],
            "step": state["step"],
            "epoch": state["epoch"],
        },
        output_path,
    )
