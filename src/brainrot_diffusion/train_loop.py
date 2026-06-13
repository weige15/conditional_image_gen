from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .conditions import null_condition_ids
from .data import BrainrotTrainDataset
from .diffusion import GaussianDiffusion
from .ema import EMA
from .guidance import drop_conditions
from .unet import build_unet_from_config


@dataclass
class TrainResult:
    checkpoint_path: Path
    step: int
    epoch: int
    last_loss: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _condition_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "animal_id": batch["animal_id"].to(device=device, dtype=torch.long),
        "object_id": batch["object_id"].to(device=device, dtype=torch.long),
        "pair_id": batch["pair_id"].to(device=device, dtype=torch.long),
    }


def _checkpoint_state(
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler,
    step: int,
    epoch: int,
    config: dict[str, Any],
    mappings: dict[str, Any],
    diffusion: GaussianDiffusion,
    seed: int,
    mixed_precision_enabled: bool,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
        "condition_mappings": mappings,
        "diffusion": diffusion.metadata(),
        "architecture": getattr(model, "architecture", {}),
        "seed": {"seed": seed, "torch_initial_seed": torch.initial_seed()},
        "mixed_precision": {"enabled": mixed_precision_enabled},
    }


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def train(config: dict[str, Any]) -> TrainResult:
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = Path(config["paths"]["train_csv"])
    train_image_dir = Path(config["paths"]["train_image_dir"])
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing training CSV: {train_csv}")
    if not train_image_dir.exists():
        raise FileNotFoundError(f"Missing training image directory: {train_image_dir}")

    dataset = BrainrotTrainDataset(
        train_csv,
        train_image_dir,
        image_size=int(config["data"]["image_size"]),
        random_horizontal_flip=bool(config["data"].get("random_horizontal_flip", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["optimizer"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        drop_last=False,
    )
    if len(loader) == 0:
        raise ValueError("Training dataloader is empty")

    model = build_unet_from_config(config, dataset.mappings).to(device)
    diffusion = GaussianDiffusion(**config["diffusion"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    mixed_precision_enabled = bool(config["optimizer"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(mixed_precision_enabled)
    ema = EMA(model, decay=float(config["ema"]["decay"]))
    start_step = 0
    start_epoch = 0
    resume = config["training"].get("resume")
    if resume:
        state = load_checkpoint(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        if "amp_scaler" in state:
            scaler.load_state_dict(state["amp_scaler"])
        start_step = int(state["step"])
        start_epoch = int(state["epoch"])

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(config["optimizer"]["max_steps"])
    save_every = int(config["training"].get("save_every", 1000))
    log_every = int(config["training"].get("log_every", 50))
    grad_accum = int(config["optimizer"].get("gradient_accumulation_steps", 1))
    dropout_p = float(config["training"].get("condition_dropout", 0.1))
    null_ids = null_condition_ids(dataset.mappings)

    step = start_step
    epoch = start_epoch
    last_loss = float("nan")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step < max_steps:
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            conditions = _condition_batch(batch, device)
            conditions = drop_conditions(conditions, dropout_p, null_ids=null_ids)
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device, dtype=torch.long)
            with _cuda_autocast(mixed_precision_enabled):
                loss = diffusion.training_loss(model, images, t, conditions) / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            last_loss = float(loss.item() * grad_accum)
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            step += 1
            if log_every > 0 and step % log_every == 0:
                print(f"step={step} loss={last_loss:.6f}", flush=True)
            if save_every > 0 and step % save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_step_{step}.pt",
                    _checkpoint_state(
                        model,
                        ema,
                        optimizer,
                        scaler,
                        step,
                        epoch,
                        config,
                        dataset.mappings,
                        diffusion,
                        seed,
                        mixed_precision_enabled,
                    ),
                )
            if step >= max_steps:
                break
        epoch += 1

    final_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    save_checkpoint(
        final_path,
        _checkpoint_state(
            model,
            ema,
            optimizer,
            scaler,
            step,
            epoch,
            config,
            dataset.mappings,
            diffusion,
            seed,
            mixed_precision_enabled,
        ),
    )
    return TrainResult(final_path, step, epoch, last_loss)
