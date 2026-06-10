from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .conditioning import ConditionMapper

REQUIRED_SAMPLING_KEYS = {"model", "ema", "config", "mappings", "architecture", "diffusion", "seed"}


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_state: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    mapper: ConditionMapper,
    step: int,
    epoch: int,
    seed_metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "ema": ema_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "mappings": mapper.to_dict(),
        "architecture": getattr(model, "metadata", {}),
        "diffusion": config.get("diffusion", {}),
        "seed": seed_metadata,
        "progress": {"step": step, "epoch": epoch},
    }
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def validate_sampling_checkpoint(payload: dict[str, Any]) -> None:
    missing = REQUIRED_SAMPLING_KEYS - set(payload)
    if missing:
        raise ValueError(
            "checkpoint missing required sampling key(s): " + ", ".join(sorted(missing))
        )
    ConditionMapper.from_dict(payload["mappings"])
    if "model" not in payload["ema"]:
        raise ValueError("checkpoint EMA state missing model weights")
