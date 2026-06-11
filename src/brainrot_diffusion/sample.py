from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from .checkpoint import load_checkpoint
from .conditions import batch_conditions
from .data import GenerationRequest, load_generation_requests
from .diffusion import GaussianDiffusion
from .guidance import combine_cfg, make_null_condition_batch
from .unet import build_unet_from_config


def _to_device_conditions(conditions: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.long) for key, value in conditions.items()}


def _predict_eps(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    conditions: dict[str, torch.Tensor],
    guidance_scale: float,
    mappings: dict[str, Any] | None,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return model(x, t, conditions)
    if mappings is None:
        mappings = {
            "null_animal_id": int(conditions["animal_id"].max().item()) + 1,
            "null_object_id": int(conditions["object_id"].max().item()) + 1,
            "null_pair_id": int(conditions["pair_id"].max().item()) + 1,
        }
    null_conditions = make_null_condition_batch(x.shape[0], mappings, x.device)
    eps_uncond = model(x, t, null_conditions)
    eps_cond = model(x, t, conditions)
    return combine_cfg(eps_uncond, eps_cond, guidance_scale)


@torch.no_grad()
def sample_ddpm(
    model,
    diffusion: GaussianDiffusion,
    conditions: dict[str, torch.Tensor],
    shape: tuple[int, int, int, int],
    guidance_scale: float = 1.0,
    mappings: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = torch.randn(shape, device=device, generator=generator)
    conditions = _to_device_conditions(conditions, device)
    model.eval()
    for timestep in reversed(range(diffusion.timesteps)):
        t = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
        eps = _predict_eps(model, x, t, conditions, guidance_scale, mappings)
        x = diffusion.p_sample_ddpm(x, t, eps, generator=generator)
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def sample_ddim(
    model,
    diffusion: GaussianDiffusion,
    conditions: dict[str, torch.Tensor],
    shape: tuple[int, int, int, int],
    steps: int = 100,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    mappings: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if steps <= 0 or steps > diffusion.timesteps:
        raise ValueError("DDIM steps must be in [1, diffusion.timesteps]")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = torch.randn(shape, device=device, generator=generator)
    conditions = _to_device_conditions(conditions, device)
    sequence = torch.linspace(diffusion.timesteps - 1, 0, steps, dtype=torch.long).tolist()
    sequence = [int(v) for v in sequence]
    model.eval()
    for index, timestep in enumerate(sequence):
        prev_timestep = sequence[index + 1] if index + 1 < len(sequence) else -1
        t = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
        prev_t = torch.full((shape[0],), prev_timestep, device=device, dtype=torch.long)
        eps = _predict_eps(model, x, t, conditions, guidance_scale, mappings)
        x = diffusion.ddim_step(x, t, prev_t, eps, eta=eta, generator=generator)
    return x.clamp(-1.0, 1.0)


def denormalize_to_uint8(images: torch.Tensor) -> np.ndarray:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B, 3, H, W]")
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = (images + 1.0) * 127.5
    return images.round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def save_image_batch(images: torch.Tensor, requests: Iterable[GenerationRequest], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    arrays = denormalize_to_uint8(images)
    for array, request in zip(arrays, requests):
        Image.fromarray(array, mode="RGB").save(output_dir / request.image_id)


@torch.no_grad()
def generate_from_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    generate_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    mappings = state["condition_mappings"]
    generate_csv = Path(generate_csv or config["paths"]["generate_csv"])
    output_dir = Path(output_dir or config["paths"]["output_dir"])
    requests = load_generation_requests(generate_csv, mappings)
    existing = [output_dir / request.image_id for request in requests if (output_dir / request.image_id).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} output files already exist; pass --overwrite to replace them")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_unet_from_config(state.get("config", config), mappings).to(device)
    if bool(config["sampling"].get("use_ema", True)) and "shadow" in state["ema"]:
        model.load_state_dict(state["model"])
        model_state = model.state_dict()
        for name, value in state["ema"]["shadow"].items():
            if name in model_state:
                model_state[name].copy_(value.to(device=device, dtype=model_state[name].dtype))
    else:
        model.load_state_dict(state["model"])
    model.eval()
    diffusion = GaussianDiffusion(**state.get("diffusion", config["diffusion"]))

    batch_size = int(config["sampling"].get("batch_size", 64))
    sampler = config["sampling"].get("sampler", "ddim")
    generator = torch.Generator(device=device).manual_seed(int(config["sampling"].get("seed", 1234)))
    written: list[Path] = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        conditions = batch_conditions([request.conditions for request in batch], device=device)
        shape = (len(batch), 3, int(config["data"]["image_size"]), int(config["data"]["image_size"]))
        if sampler == "ddim":
            images = sample_ddim(
                model,
                diffusion,
                conditions,
                shape,
                steps=int(config["sampling"]["steps"]),
                eta=float(config["sampling"].get("eta", 0.0)),
                guidance_scale=float(config["sampling"].get("guidance_scale", 1.0)),
                mappings=mappings,
                generator=generator,
                device=device,
            )
        elif sampler == "ddpm":
            images = sample_ddpm(
                model,
                diffusion,
                conditions,
                shape,
                guidance_scale=float(config["sampling"].get("guidance_scale", 1.0)),
                mappings=mappings,
                generator=generator,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported sampler {sampler!r}")
        save_image_batch(images, batch, output_dir)
        written.extend(output_dir / request.image_id for request in batch)
    return written
