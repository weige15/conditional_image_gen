"""Training loop orchestration."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .conditions import ConditionMappings, build_condition_mappings
from .config import validate_config
from .data import BrainrotTrainDataset, read_train_csv
from .diffusion import GaussianDiffusion
from .ema import EMA
from .model import build_model_from_config


@dataclass(frozen=True)
class TrainingResult:
    final_checkpoint: Path
    step: int
    last_loss: float
    seed: dict[str, int]


def train(config: Mapping[str, object]) -> TrainingResult:
    validate_config(config)
    data_config = _section(config, "data")
    training_config = _section(config, "training")
    checkpoint_config = _section(config, "checkpointing")

    seed = int(training_config.get("seed", 0))
    seed_metadata = set_seed(seed)
    device = _device(str(training_config.get("device", "auto")))

    resume_path = str(training_config.get("resume_checkpoint", "") or "")
    start_step = 0
    optimizer_state = None
    if resume_path:
        resume = load_checkpoint(resume_path, map_location=device)
        mappings = ConditionMappings.from_metadata(resume["condition_mappings"])
        diffusion = GaussianDiffusion.from_metadata(resume["diffusion"])
    else:
        rows = read_train_csv(str(data_config["train_csv"]))
        mappings = build_condition_mappings(rows, include_null=True)
        diffusion = GaussianDiffusion.from_config(config)

    dataset = BrainrotTrainDataset(
        str(data_config["train_csv"]),
        str(data_config["train_image_dir"]),
        mappings=mappings,
        image_size=int(data_config.get("image_size", 64)),
        resize=bool(data_config.get("resize_train_images", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        collate_fn=_collate_batch,
    )

    model = build_model_from_config(config, mappings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    if resume_path:
        model.load_state_dict(resume["model"])
        start_step = int(resume["step"])
        optimizer_state = resume.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    use_ema = bool(checkpoint_config.get("use_ema", True))
    ema = EMA(model, decay=float(checkpoint_config.get("ema_decay", 0.999)), enabled=use_ema)
    if resume_path and use_ema and resume.get("ema") is not None:
        ema.load_state_dict(resume["ema"], model)

    max_steps = int(training_config["max_steps"])
    save_every = int(training_config["save_every"])
    log_every = int(training_config.get("log_every", 0) or 0)
    checkpoint_dir = Path(str(checkpoint_config["checkpoint_dir"]))
    last_loss = float("nan")
    final_checkpoint: Path | None = None

    print(
        "training "
        f"device={device} "
        f"samples={len(dataset)} "
        f"batch_size={int(training_config['batch_size'])} "
        f"steps={start_step + 1}-{max_steps} "
        f"save_every={save_every} "
        f"log_every={log_every}",
        flush=True,
    )
    started_at = time.monotonic()
    model.train()
    for step, batch in zip(range(start_step + 1, max_steps + 1), cycle(loader)):
        images = batch["image"].to(device)
        conditions = {key: value.to(device) for key, value in batch["conditions"].items()}
        conditions = drop_conditions(
            conditions,
            mappings,
            dropout=float(training_config.get("condition_dropout", 0.0)),
        )
        timesteps = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)
        loss = diffusion.training_loss(model, images, timesteps, conditions)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)
        last_loss = float(loss.detach().cpu())

        if log_every > 0 and (step == start_step + 1 or step % log_every == 0 or step == max_steps):
            elapsed = max(time.monotonic() - started_at, 1e-9)
            trained_steps = step - start_step
            print(
                f"step={step}/{max_steps} "
                f"loss={last_loss:.6f} "
                f"steps_per_sec={trained_steps / elapsed:.3f}",
                flush=True,
            )

        if step % save_every == 0 or step == max_steps:
            final_checkpoint = checkpoint_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(
                final_checkpoint,
                model_state=model.state_dict(),
                config=config,
                condition_mappings=mappings,
                diffusion=diffusion.to_metadata(),
                architecture=model.metadata.as_dict(),
                seed=seed_metadata,
                step=step,
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                metrics={"loss": last_loss},
            )
            print(f"saved checkpoint: {final_checkpoint}", flush=True)

    if final_checkpoint is None:
        final_checkpoint = checkpoint_dir / f"checkpoint_step_{start_step}.pt"
        save_checkpoint(
            final_checkpoint,
            model_state=model.state_dict(),
            config=config,
            condition_mappings=mappings,
            diffusion=diffusion.to_metadata(),
            architecture=model.metadata.as_dict(),
            seed=seed_metadata,
            step=start_step,
            ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            metrics={"loss": last_loss},
        )
    return TrainingResult(final_checkpoint=final_checkpoint, step=max_steps, last_loss=last_loss, seed=seed_metadata)


def set_seed(seed: int) -> dict[str, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"python": seed, "numpy": seed, "torch": seed}


def drop_conditions(
    conditions: Mapping[str, torch.Tensor],
    mappings: ConditionMappings,
    *,
    dropout: float,
) -> dict[str, torch.Tensor]:
    if dropout <= 0:
        return {key: value.clone() for key, value in conditions.items()}
    if dropout >= 1:
        mask = torch.ones_like(next(iter(conditions.values())), dtype=torch.bool)
    else:
        first = next(iter(conditions.values()))
        mask = torch.rand(first.shape, device=first.device) < dropout
    null_ids = mappings.null_ids
    if null_ids is None:
        return {key: value.clone() for key, value in conditions.items()}
    dropped = {key: value.clone() for key, value in conditions.items()}
    dropped["animal_id"][mask] = null_ids.animal_id
    dropped["object_id"][mask] = null_ids.object_id
    dropped["pair_id"][mask] = null_ids.pair_id
    return dropped


def _collate_batch(samples: list[dict[str, object]]) -> dict[str, object]:
    images = torch.stack([sample["image"] for sample in samples])
    return {
        "image": images,
        "id": [sample["id"] for sample in samples],
        "animal": [sample["animal"] for sample in samples],
        "object": [sample["object"] for sample in samples],
        "conditions": {
            "animal_id": torch.tensor([sample["conditions"]["animal_id"] for sample in samples], dtype=torch.long),
            "object_id": torch.tensor([sample["conditions"]["object_id"] for sample in samples], dtype=torch.long),
            "pair_id": torch.tensor([sample["conditions"]["pair_id"] for sample in samples], dtype=torch.long),
        },
    }


def _section(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
