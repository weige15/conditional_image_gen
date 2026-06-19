"""Checkpoint schema helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .conditions import ConditionMappings


REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "config",
    "condition_mappings",
    "diffusion",
    "architecture",
    "seed",
    "step",
}

OPTIONAL_CHECKPOINT_KEYS = {"ema", "optimizer", "epoch", "metrics"}


def save_checkpoint(
    path: str | Path,
    *,
    model_state: Mapping[str, Any],
    config: Mapping[str, Any],
    condition_mappings: ConditionMappings | Mapping[str, Any],
    diffusion: Mapping[str, Any],
    architecture: Mapping[str, Any],
    seed: Mapping[str, Any],
    step: int,
    ema: Mapping[str, Any] | None = None,
    optimizer: Mapping[str, Any] | None = None,
    epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": dict(model_state),
        "config": dict(config),
        "condition_mappings": (
            condition_mappings.to_metadata()
            if isinstance(condition_mappings, ConditionMappings)
            else dict(condition_mappings)
        ),
        "diffusion": dict(diffusion),
        "architecture": dict(architecture),
        "seed": dict(seed),
        "step": int(step),
    }
    if ema is not None:
        payload["ema"] = dict(ema)
    if optimizer is not None:
        payload["optimizer"] = dict(optimizer)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    validate_checkpoint(payload)

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return payload


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dict: {checkpoint_path}")
    validate_checkpoint(payload)
    return payload


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload.keys()))
    if missing:
        raise ValueError(f"checkpoint missing required key(s): {', '.join(missing)}")
    if not isinstance(payload["model"], Mapping):
        raise ValueError("checkpoint model must be a mapping")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if not isinstance(payload["condition_mappings"], Mapping):
        raise ValueError("checkpoint condition_mappings must be a mapping")
    if not isinstance(payload["diffusion"], Mapping):
        raise ValueError("checkpoint diffusion must be a mapping")
    if not isinstance(payload["architecture"], Mapping):
        raise ValueError("checkpoint architecture must be a mapping")
    if not isinstance(payload["seed"], Mapping):
        raise ValueError("checkpoint seed must be a mapping")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    ConditionMappings.from_metadata(payload["condition_mappings"])


def checkpoint_mappings(payload: Mapping[str, Any]) -> ConditionMappings:
    validate_checkpoint(payload)
    return ConditionMappings.from_metadata(payload["condition_mappings"])


def validate_generation_compatibility(payload: Mapping[str, Any], requests: Iterable[object]) -> None:
    mappings = checkpoint_mappings(payload)
    mappings.validate_requests(requests)

