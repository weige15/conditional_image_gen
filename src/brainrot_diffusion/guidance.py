from __future__ import annotations

from typing import Any

import torch

from .conditions import CONDITION_KEYS, null_condition_ids


def _resolve_null_ids(condition_batch: dict[str, torch.Tensor], null_ids: dict[str, int] | None) -> dict[str, int]:
    if null_ids is not None:
        return null_ids
    return {key: int(condition_batch[key].max().item()) + 1 for key in CONDITION_KEYS}


def drop_conditions(
    condition_batch: dict[str, torch.Tensor],
    p: float,
    generator: torch.Generator | None = None,
    null_ids: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= p <= 1.0:
        raise ValueError("Condition dropout probability p must be in [0, 1]")
    if any(key not in condition_batch for key in CONDITION_KEYS):
        raise ValueError(f"condition_batch must contain {CONDITION_KEYS}")
    batch_size = condition_batch["animal_id"].shape[0]
    if any(condition_batch[key].shape[0] != batch_size for key in CONDITION_KEYS):
        raise ValueError("All condition tensors must have the same batch size")
    resolved_null_ids = _resolve_null_ids(condition_batch, null_ids)
    if p == 0.0:
        return {key: value.clone() for key, value in condition_batch.items()}
    mask = torch.rand(batch_size, device=condition_batch["animal_id"].device, generator=generator) < p
    result = {key: value.clone() for key, value in condition_batch.items()}
    for key in CONDITION_KEYS:
        result[key][mask] = int(resolved_null_ids[key])
    return result


def make_null_condition_batch(
    batch_size: int,
    mappings: dict[str, Any],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ids = null_condition_ids(mappings)
    return {
        key: torch.full((batch_size,), value, dtype=torch.long, device=device)
        for key, value in ids.items()
    }


def combine_cfg(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    if eps_uncond.shape != eps_cond.shape:
        raise ValueError(f"CFG prediction shapes differ: {eps_uncond.shape} vs {eps_cond.shape}")
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
