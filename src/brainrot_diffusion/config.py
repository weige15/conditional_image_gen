"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_SECTIONS = (
    "data",
    "model",
    "diffusion",
    "training",
    "sampling",
    "checkpointing",
    "validation",
    "evaluation",
    "packaging",
)

SUPPORTED_SAMPLERS = {"ddpm", "ddim"}
SUPPORTED_SCHEDULES = {"linear", "cosine"}


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load, override, and validate a YAML config."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {config_path}")

    config = _plain_data(loaded)
    if overrides:
        config = apply_overrides(config, overrides)
    validate_config(config)
    return config


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a config copy with dotted-key or nested mapping overrides applied."""

    result = deepcopy(dict(config))
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and "." not in key:
            existing = result.get(key, {})
            if not isinstance(existing, Mapping):
                result[key] = _plain_data(value)
            else:
                result[key] = _merge_dicts(existing, value)
        else:
            _set_dotted(result, key, _plain_data(value))
    validate_config(result)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate config sections and simple assignment-critical values."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"missing config section(s): {', '.join(missing)}")

    data = _section(config, "data")
    model = _section(config, "model")
    diffusion = _section(config, "diffusion")
    training = _section(config, "training")
    sampling = _section(config, "sampling")
    checkpointing = _section(config, "checkpointing")
    validation = _section(config, "validation")
    evaluation = _section(config, "evaluation")
    packaging = _section(config, "packaging")

    _require_str(data, "train_csv")
    _require_str(data, "generate_csv")
    _require_str(data, "train_image_dir")
    _require_str(model, "name")
    _require_str(checkpointing, "checkpoint_dir")
    _require_str(validation, "output_dir")
    _require_str(evaluation, "reference_dir")
    _require_str(packaging, "student_id")

    if int(data.get("image_size", 0)) != 64:
        raise ValueError("data.image_size must be 64")
    if int(model.get("image_size", 0)) != 64:
        raise ValueError("model.image_size must be 64")
    _require_positive_int(diffusion, "timesteps")
    _require_supported(diffusion, "schedule", SUPPORTED_SCHEDULES)
    _require_positive_int(training, "batch_size")
    _require_positive_int(training, "max_steps")
    _require_positive_int(training, "save_every")
    _require_positive_int(sampling, "batch_size")
    _require_positive_int(sampling, "steps")
    _require_supported(sampling, "sampler", SUPPORTED_SAMPLERS)

    if int(sampling["steps"]) > int(diffusion["timesteps"]):
        raise ValueError("sampling.steps cannot exceed diffusion.timesteps")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(sampling.get("guidance_scale", 0.0)) < 0:
        raise ValueError("sampling.guidance_scale must be nonnegative")

    _ensure_serializable(config)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _require_str(section: Mapping[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")


def _require_positive_int(section: Mapping[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _require_supported(section: Mapping[str, Any], key: str, supported: set[str]) -> None:
    value = section.get(key)
    if value not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"{key} must be one of: {choices}")


def _merge_dicts(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = _plain_data(value)
    return result


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid override key: {key!r}")
    current = config
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set nested override under non-mapping key: {part}")
        current = next_value
    current[parts[-1]] = value


def _plain_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    return value


def _ensure_serializable(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _ensure_serializable(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("config keys must be strings")
            _ensure_serializable(item)
        return
    raise ValueError(f"config value is not JSON/YAML serializable: {type(value).__name__}")

