from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .conditioning import ConditionMapper
from .config import Config, setup_seed, validate_paths
from .data import BrainrotDataset, batch_conditions
from .diffusion import DiffusionSchedule
from .ema import EMA
from .model import build_model


def train(
    config: Config, *, max_steps: int | None = None, device: str | None = None
) -> dict[str, float | int | str]:
    validate_paths(config, mode="train")
    seed_metadata = setup_seed(config.seed)
    mapper = ConditionMapper()
    dataset = BrainrotDataset(
        config.paths.train_csv,
        config.paths.train_image_dir,
        mapper,
        image_size=config.image_size,
        resize_policy=config.data.resize_policy,
        horizontal_flip=config.data.horizontal_flip,
        flip_prob=config.data.flip_prob,
        color_jitter=config.data.color_jitter,
        color_jitter_strength=config.data.color_jitter_strength,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=False,
    )
    if len(loader) == 0:
        raise ValueError("training dataset is empty")
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    diffusion = DiffusionSchedule(config.diffusion.timesteps, config.diffusion.schedule).to(
        run_device
    )
    model = build_model(config, mapper).to(run_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    ema = EMA(model, config.ema.decay)
    ema.model.to(run_device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.training.mixed_precision and run_device.type == "cuda"
    )
    target_steps = max_steps or config.training.max_steps
    if target_steps is None:
        target_steps = max(1, config.training.epochs * len(loader))

    step = 0
    last_loss = 0.0
    started_at = time.time()
    print(
        {
            "event": "training_started",
            "device": str(run_device),
            "dataset_size": len(dataset),
            "batch_size": config.training.batch_size,
            "target_steps": target_steps,
            "checkpoint_dir": config.paths.checkpoint_dir,
        },
        flush=True,
    )
    model.train()
    while step < target_steps:
        for batch in loader:
            images = batch["image"].to(run_device)
            conditions = batch_conditions(batch).to(run_device)
            conditions = mapper.apply_dropout(conditions, config.conditioning.dropout_prob)
            timesteps = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=run_device)
            noise = torch.randn_like(images)
            noisy = diffusion.q_sample(images, timesteps, noise)
            with torch.amp.autocast(
                "cuda", enabled=config.training.mixed_precision and run_device.type == "cuda"
            ):
                pred = model(noisy, timesteps, conditions)
                loss = F.mse_loss(pred, noise) / config.training.grad_accum_steps
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward()
            if (step + 1) % config.training.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if step >= config.ema.update_after_step and step % config.ema.update_every == 0:
                    ema.update(model)
            last_loss = float(loss.detach().cpu() * config.training.grad_accum_steps)
            step += 1
            if step % config.training.checkpoint_every == 0 or step >= target_steps:
                checkpoint_path = Path(config.paths.checkpoint_dir) / f"checkpoint_step_{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    ema_state=ema.state_dict(),
                    optimizer=optimizer,
                    config=config.to_dict(),
                    mapper=mapper,
                    step=step,
                    epoch=0,
                    seed_metadata=seed_metadata,
                )
                print(
                    {"event": "checkpoint_saved", "step": step, "path": str(checkpoint_path)},
                    flush=True,
                )
            if step == 1 or step % config.training.log_every == 0 or step >= target_steps:
                elapsed = time.time() - started_at
                steps_per_second = step / elapsed if elapsed > 0 else 0.0
                print(
                    {
                        "event": "train_step",
                        "step": step,
                        "target_steps": target_steps,
                        "loss": round(last_loss, 6),
                        "elapsed_sec": round(elapsed, 1),
                        "steps_per_sec": round(steps_per_second, 4),
                    },
                    flush=True,
                )
            if step >= target_steps:
                break
    return {"step": step, "loss": last_loss, "checkpoint_dir": config.paths.checkpoint_dir}
