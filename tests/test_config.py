from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from brainrot_diffusion.config import apply_overrides, load_config, validate_config


def test_load_default_config_is_valid_and_serializable() -> None:
    config = load_config("configs/default.yaml")

    assert config["data"]["image_size"] == 64
    assert config["sampling"]["sampler"] == "ddim"
    yaml.safe_dump(config)


def test_missing_required_section_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing config section"):
        load_config(path)


def test_invalid_values_fail() -> None:
    config = load_config("configs/default.yaml")
    config["diffusion"]["timesteps"] = 0

    with pytest.raises(ValueError, match="timesteps"):
        validate_config(config)


def test_invalid_architecture_values_fail() -> None:
    config = load_config("configs/default.yaml")
    config["model"]["attention_resolutions"] = [8]

    with pytest.raises(ValueError, match="attention_resolutions"):
        validate_config(config)


def test_dotted_overrides_are_deterministic() -> None:
    config = load_config("configs/default.yaml")
    updated = apply_overrides(
        config,
        {
            "training.batch_size": 4,
            "sampling.sampler": "ddpm",
            "sampling.steps": 10,
        },
    )

    assert updated["training"]["batch_size"] == 4
    assert updated["sampling"]["sampler"] == "ddpm"
    assert config["training"]["batch_size"] != 4
    assert updated == apply_overrides(config, {"training.batch_size": 4, "sampling.sampler": "ddpm", "sampling.steps": 10})


def test_nested_overrides_keep_other_values() -> None:
    config = load_config("configs/default.yaml")
    original = deepcopy(config["training"])
    updated = apply_overrides(config, {"training": {"batch_size": 8}})

    assert updated["training"]["batch_size"] == 8
    assert updated["training"]["learning_rate"] == original["learning_rate"]
