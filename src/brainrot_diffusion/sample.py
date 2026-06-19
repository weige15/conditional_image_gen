"""Checkpoint-backed image generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .checkpoint import checkpoint_mappings, load_checkpoint, validate_generation_compatibility
from .conditions import ConditionMappings
from .data import GenerationRequest, read_generation_csv
from .diffusion import GaussianDiffusion
from .ema import EMA
from .model import build_model_from_config


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    files: tuple[Path, ...]
    seed: int


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: Mapping[str, object] | None = None,
    generate_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
    device: str = "auto",
) -> GenerationResult:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    runtime_config = config if config is not None else payload["config"]
    data_config = _section(runtime_config, "data")
    sampling_config = _section(runtime_config, "sampling")
    mappings = checkpoint_mappings(payload)
    csv_path = Path(str(generate_csv or data_config["generate_csv"]))
    requests = read_generation_csv(csv_path, strict_prompt=bool(data_config.get("strict_prompt", False)))
    validate_generation_compatibility(payload, requests)

    out_dir = Path(str(output_dir or sampling_config["output_dir"]))
    allow_overwrite = bool(sampling_config.get("overwrite", False) if overwrite is None else overwrite)
    _check_outputs(requests, out_dir, allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_device = _device(device)
    model = build_model_from_config({"model": payload["architecture"]}, mappings).to(active_device)
    model.load_state_dict(payload["model"])
    if payload.get("ema") is not None:
        ema = EMA(model, enabled=True)
        ema.load_state_dict(payload["ema"], model)
        ema.copy_to(model)
    model.eval()
    diffusion = GaussianDiffusion.from_metadata(payload["diffusion"])

    sampler = str(sampling_config.get("sampler", "ddim"))
    steps = int(sampling_config.get("steps", diffusion.timesteps))
    if steps <= 0 or steps > diffusion.timesteps:
        raise ValueError("sampling steps must be in [1, diffusion.timesteps]")
    batch_size = int(sampling_config.get("batch_size", 64))
    seed = int(sampling_config.get("seed", 0))
    guidance_scale = float(sampling_config.get("guidance_scale", 1.0))
    eta = float(sampling_config.get("eta", 0.0))
    generator = torch.Generator(device=active_device)
    generator.manual_seed(seed)

    written: list[Path] = []
    with torch.no_grad():
        for batch in _batches(requests, batch_size):
            conditions = _condition_tensors(mappings, batch, active_device)
            sample = _sample_batch(
                model,
                diffusion,
                conditions,
                mappings,
                batch_size=len(batch),
                sampler=sampler,
                steps=steps,
                guidance_scale=guidance_scale,
                eta=eta,
                generator=generator,
                device=active_device,
            )
            for tensor, request in zip(sample.cpu(), batch):
                path = out_dir / request.id
                tensor_to_image(tensor).save(path, format="PNG")
                written.append(path)
    return GenerationResult(output_dir=out_dir, files=tuple(written), seed=seed)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")


def _sample_batch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    conditions: Mapping[str, torch.Tensor],
    mappings: ConditionMappings,
    *,
    batch_size: int,
    sampler: str,
    steps: int,
    guidance_scale: float,
    eta: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError("sampler must be 'ddpm' or 'ddim'")
    x = torch.randn((batch_size, 3, 64, 64), device=device, generator=generator)
    schedule = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).round().long()
    schedule = torch.unique_consecutive(schedule)
    for index, timestep in enumerate(schedule):
        t = torch.full((batch_size,), int(timestep.item()), dtype=torch.long, device=device)
        predicted = model(x, t, conditions)
        null_ids = mappings.null_ids
        if guidance_scale != 1.0 and null_ids is not None:
            null_conditions = {
                "animal_id": torch.full((batch_size,), null_ids.animal_id, dtype=torch.long, device=device),
                "object_id": torch.full((batch_size,), null_ids.object_id, dtype=torch.long, device=device),
                "pair_id": torch.full((batch_size,), null_ids.pair_id, dtype=torch.long, device=device),
            }
            uncond = model(x, t, null_conditions)
            predicted = uncond + guidance_scale * (predicted - uncond)
        if sampler == "ddpm":
            x = diffusion.ddpm_step(x, t, predicted, generator=generator)
        else:
            prev_timestep = -1 if index == len(schedule) - 1 else int(schedule[index + 1].item())
            prev = torch.full((batch_size,), prev_timestep, dtype=torch.long, device=device)
            x = diffusion.ddim_step(x, t, prev, predicted, eta=eta, generator=generator)
    return x.clamp(-1.0, 1.0)


def _condition_tensors(
    mappings: ConditionMappings,
    requests: Sequence[GenerationRequest],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = [mappings.encode(request.animal, request.object) for request in requests]
    return {
        "animal_id": torch.tensor([item.animal_id for item in encoded], dtype=torch.long, device=device),
        "object_id": torch.tensor([item.object_id for item in encoded], dtype=torch.long, device=device),
        "pair_id": torch.tensor([item.pair_id for item in encoded], dtype=torch.long, device=device),
    }


def _check_outputs(requests: Sequence[GenerationRequest], output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [output_dir / request.id for request in requests if (output_dir / request.id).exists()]
    if existing:
        raise FileExistsError(f"output file already exists: {existing[0]}")


def _batches(items: Sequence[GenerationRequest], batch_size: int) -> list[Sequence[GenerationRequest]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _section(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
