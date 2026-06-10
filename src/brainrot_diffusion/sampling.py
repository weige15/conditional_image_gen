from __future__ import annotations

import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, validate_sampling_checkpoint
from .conditioning import ConditionBatch, ConditionMapper
from .config import Config, config_from_dict, setup_seed
from .data import read_generate_records, tensor_to_uint8_image
from .diffusion import DiffusionSchedule
from .model import build_model


@torch.no_grad()
def ddim_sample_batch(
    model: torch.nn.Module,
    diffusion: DiffusionSchedule,
    mapper: ConditionMapper,
    conditions: ConditionBatch,
    *,
    image_size: int,
    steps: int,
    eta: float,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    batch_size = conditions.animal.shape[0]
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    times = diffusion.ddim_timesteps(steps).to(device)
    null_conditions = mapper.null_batch(batch_size, device=device)
    conditions = conditions.to(device)
    for index, timestep_value in enumerate(times):
        t = torch.full((batch_size,), int(timestep_value.item()), device=device, dtype=torch.long)
        prev_value = int(times[index + 1].item()) if index + 1 < len(times) else -1
        prev_t = torch.full((batch_size,), prev_value, device=device, dtype=torch.long)
        eps_cond = model(x, t, conditions)
        if guidance_scale == 1.0:
            eps = eps_cond
        else:
            eps_uncond = model(x, t, null_conditions)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x = diffusion.ddim_step(x, t, prev_t, eps, eta=eta)
    return x.clamp(-1, 1)


def load_model_for_sampling(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[Config, ConditionMapper, torch.nn.Module, DiffusionSchedule]:
    payload = load_checkpoint(checkpoint_path, map_location=device)
    validate_sampling_checkpoint(payload)
    config = config_from_dict(payload["config"])
    mapper = ConditionMapper.from_dict(payload["mappings"])
    model = build_model(config, mapper).to(device)
    model.load_state_dict(payload["ema"]["model"])
    model.eval()
    diffusion = DiffusionSchedule(config.diffusion.timesteps, config.diffusion.schedule).to(device)
    return config, mapper, model, diffusion


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    config: Config | None = None,
    *,
    device: str | None = None,
) -> dict[str, object]:
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_config, mapper, model, diffusion = load_model_for_sampling(
        checkpoint_path, device=run_device
    )
    config = config or ckpt_config
    setup_seed(config.seed)
    records = read_generate_records(config.paths.generate_csv)
    out_dir = Path(config.paths.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not config.sampling.overwrite:
        raise FileExistsError(
            f"{out_dir} is not empty; set sampling.overwrite=true to write into it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    batch_size = config.sampling.batch_size
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        conditions = mapper.encode_many([r.animal for r in chunk], [r.object for r in chunk])
        images = ddim_sample_batch(
            model,
            diffusion,
            mapper,
            conditions,
            image_size=config.image_size,
            steps=config.sampling.ddim_steps,
            eta=config.sampling.eta,
            guidance_scale=config.sampling.guidance_scale,
            device=run_device,
        )
        for record, tensor in zip(chunk, images, strict=True):
            path = out_dir / record.id
            if path.suffix.lower() != ".png":
                raise ValueError(f"output id must use .png extension: {record.id}")
            tensor_to_uint8_image(tensor).save(path, format="PNG")
            written.append(record.id)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "count": len(written),
        "output_dir": str(out_dir),
        "guidance_scale": config.sampling.guidance_scale,
        "ddim_steps": config.sampling.ddim_steps,
    }
    (out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
