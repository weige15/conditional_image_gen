from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class PathsConfig:
    train_csv: str = "train.csv"
    generate_csv: str = "generate.csv"
    train_image_dir: str = "trainset"
    output_dir: str = "generated_images"
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"
    metrics_dir: str = "reports"
    test_mu: str | None = "test_mu.npy"
    test_sigma: str | None = "test_sigma.npy"


@dataclass
class DataConfig:
    resize_policy: str = "resize"
    horizontal_flip: bool = True
    flip_prob: float = 0.5
    color_jitter: bool = False
    color_jitter_strength: float = 0.08


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 100
    max_steps: int | None = None
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    mixed_precision: bool = False
    log_every: int = 50
    checkpoint_every: int = 1000
    sample_every: int = 1000


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "cosine"


@dataclass
class ModelConfig:
    base_channels: int = 96
    channel_mults: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    attention_resolutions: list[int] = field(default_factory=lambda: [16, 8])
    num_res_blocks: int = 1
    dropout: float = 0.0
    emb_dim: int = 384


@dataclass
class EMAConfig:
    decay: float = 0.9999
    update_after_step: int = 0
    update_every: int = 1


@dataclass
class ConditioningConfig:
    dropout_prob: float = 0.1


@dataclass
class SamplingConfig:
    batch_size: int = 64
    ddim_steps: int = 100
    eta: float = 0.0
    guidance_scale: float = 2.0
    use_ema: bool = True
    overwrite: bool = False


@dataclass
class ValidationConfig:
    final_expected_count: int = 2000
    strict_count: bool = True


@dataclass
class EvaluationConfig:
    enable_fid: bool = True
    enable_clip_t: bool = False


@dataclass
class PackagingConfig:
    student_id: str | None = None
    overwrite: bool = False
    zip_path: str | None = None


@dataclass
class FallbackConfig:
    enabled: bool = False
    output_dir: str = "fallback_outputs"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 1337
    image_size: int = 64
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTION_TYPES = {
    "paths": PathsConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "diffusion": DiffusionConfig,
    "model": ModelConfig,
    "ema": EMAConfig,
    "conditioning": ConditioningConfig,
    "sampling": SamplingConfig,
    "validation": ValidationConfig,
    "evaluation": EvaluationConfig,
    "packaging": PackagingConfig,
    "fallback": FallbackConfig,
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def config_from_dict(data: dict[str, Any]) -> Config:
    defaults = Config().to_dict()
    merged = _merge_dict(defaults, data)
    kwargs: dict[str, Any] = {}
    for key, value in merged.items():
        if key in _SECTION_TYPES:
            kwargs[key] = _SECTION_TYPES[key](**value)
        else:
            kwargs[key] = value
    return Config(**kwargs)


def load_config(
    path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None
) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if overrides:
        raw = _merge_dict(raw, overrides)
    return config_from_dict(raw)


def apply_cli_overrides(config: Config, **kwargs: Any) -> Config:
    raw = config.to_dict()
    for key, value in kwargs.items():
        if value is None:
            continue
        parts = key.split("__")
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return config_from_dict(raw)


def validate_paths(config: Config, *, mode: str) -> None:
    required: list[str | None] = []
    if mode == "train":
        required = [config.paths.train_csv, config.paths.train_image_dir]
    elif mode == "generate":
        required = [config.paths.generate_csv]
    elif mode in {"validate", "evaluate", "package"}:
        required = [config.paths.generate_csv, config.paths.output_dir]
    else:
        raise ValueError(f"unknown validation mode: {mode}")

    missing = [str(Path(p)) for p in required if p and not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def setup_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "deterministic": deterministic,
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
    }
