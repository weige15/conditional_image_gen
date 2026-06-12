from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "train_csv": "dataset/train.csv",
        "train_image_dir": "dataset/trainset",
        "generate_csv": "dataset/generate.csv",
        "output_dir": "generated_images",
        "checkpoint_dir": "checkpoints",
        "reference_dir": "hw6_reference",
    },
    "data": {
        "image_size": 64,
        "random_horizontal_flip": True,
        "num_workers": 2,
    },
    "model": {
        "image_channels": 3,
        "base_channels": 96,
        "channel_mults": [1, 2, 2, 4],
        "blocks_per_level": 2,
        "dropout": 0.0,
        "attention_resolutions": [16],
    },
    "diffusion": {"timesteps": 1000, "beta_schedule": "cosine"},
    "optimizer": {
        "lr": 2e-4,
        "weight_decay": 0.0,
        "batch_size": 64,
        "gradient_accumulation_steps": 1,
        "max_steps": 200000,
        "mixed_precision": False,
    },
    "ema": {"decay": 0.9999},
    "training": {
        "seed": 1234,
        "condition_dropout": 0.05,
        "log_every": 50,
        "save_every": 1000,
        "resume": None,
    },
    "sampling": {
        "sampler": "ddim",
        "batch_size": 64,
        "steps": 250,
        "eta": 0.0,
        "guidance_scale": 1.0,
        "seed": 1234,
        "use_ema": True,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                loaded = json.load(f)
            else:
                loaded = yaml.safe_load(f) or {}
        config = deep_update(config, loaded)
    if overrides:
        config = deep_update(config, overrides)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ["paths", "data", "model", "diffusion", "optimizer", "ema", "training", "sampling"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    image_size = int(config["data"]["image_size"])
    if image_size != 64:
        raise ValueError("Only 64x64 images are supported for this assignment implementation")
    if int(config["model"]["image_channels"]) != 3:
        raise ValueError("Only RGB image_channels=3 is supported")

    timesteps = int(config["diffusion"]["timesteps"])
    if timesteps <= 0:
        raise ValueError("diffusion.timesteps must be positive")
    if config["diffusion"].get("beta_schedule") != "cosine":
        raise ValueError("Only the cosine beta schedule is currently supported")

    for key in ["base_channels", "blocks_per_level"]:
        if int(config["model"][key]) <= 0:
            raise ValueError(f"model.{key} must be positive")
    if not config["model"]["channel_mults"]:
        raise ValueError("model.channel_mults must not be empty")
    if any(int(mult) <= 0 for mult in config["model"]["channel_mults"]):
        raise ValueError("model.channel_mults must contain positive integers")

    if not 0.0 <= float(config["model"]["dropout"]) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if not 0.0 <= float(config["training"]["condition_dropout"]) <= 1.0:
        raise ValueError("training.condition_dropout must be in [0, 1]")
    if float(config["sampling"]["guidance_scale"]) < 0.0:
        raise ValueError("sampling.guidance_scale must be non-negative")
    if int(config["sampling"]["steps"]) <= 0:
        raise ValueError("sampling.steps must be positive")
    if int(config["sampling"]["steps"]) > timesteps:
        raise ValueError("sampling.steps cannot exceed diffusion.timesteps")
    if int(config["optimizer"]["batch_size"]) <= 0:
        raise ValueError("optimizer.batch_size must be positive")
    if int(config["optimizer"]["gradient_accumulation_steps"]) <= 0:
        raise ValueError("optimizer.gradient_accumulation_steps must be positive")
    if int(config["optimizer"]["max_steps"]) <= 0:
        raise ValueError("optimizer.max_steps must be positive")


def apply_cli_overrides(config_path: str | None, args: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {"paths": {}, "optimizer": {}, "training": {}, "sampling": {}}
    for attr, section, key in [
        ("train_csv", "paths", "train_csv"),
        ("train_image_dir", "paths", "train_image_dir"),
        ("checkpoint_dir", "paths", "checkpoint_dir"),
        ("generate_csv", "paths", "generate_csv"),
        ("output_dir", "paths", "output_dir"),
        ("max_steps", "optimizer", "max_steps"),
        ("batch_size", "optimizer", "batch_size"),
        ("resume", "training", "resume"),
        ("sampler", "sampling", "sampler"),
        ("steps", "sampling", "steps"),
        ("guidance_scale", "sampling", "guidance_scale"),
        ("seed", "sampling", "seed"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            overrides[section][key] = value
    overrides = {section: values for section, values in overrides.items() if values}
    return load_config(config_path, overrides)
