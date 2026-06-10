from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from brainrot_diffusion.config import config_from_dict, setup_seed, validate_paths


def test_config_parsing_and_seed_reproducibility() -> None:
    config = config_from_dict(
        {"seed": 7, "training": {"batch_size": 2}, "model": {"base_channels": 8}}
    )
    assert config.seed == 7
    assert config.training.batch_size == 2
    assert config.model.base_channels == 8
    setup_seed(123)
    values = (random.random(), np.random.rand(), torch.rand(1).item())
    setup_seed(123)
    assert values == (random.random(), np.random.rand(), torch.rand(1).item())


def test_validate_paths_reports_missing(tmp_path) -> None:
    config = config_from_dict(
        {
            "paths": {
                "train_csv": str(tmp_path / "missing.csv"),
                "train_image_dir": str(tmp_path / "images"),
            }
        }
    )
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        validate_paths(config, mode="train")
