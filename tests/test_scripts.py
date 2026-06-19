from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from brainrot_diffusion.config import apply_overrides, load_config
from scripts.generate import parse_args as parse_generate_args


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_tiny_config(tmp_path: Path) -> Path:
    image_dir = tmp_path / "trainset"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(image_dir / "000001.png")
    Image.new("RGB", (64, 64), (0, 255, 0)).save(image_dir / "000002.png")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("id,animal,object\n000001.png,cat,chair\n000002.png,dog,pizza\n", encoding="utf-8")
    generate_csv = tmp_path / "generate.csv"
    generate_csv.write_text(
        "id,animal,object,prompt\n"
        "000101.png,cat,chair,a cat and a chair\n"
        "000102.png,dog,pizza,a dog and a pizza\n",
        encoding="utf-8",
    )
    config = apply_overrides(
        load_config(ROOT / "configs/default.yaml"),
        {
            "data.train_csv": str(train_csv),
            "data.generate_csv": str(generate_csv),
            "data.train_image_dir": str(image_dir),
            "model.base_channels": 4,
            "model.embedding_dim": 8,
            "model.dropout": 0.0,
            "diffusion.timesteps": 4,
            "diffusion.schedule": "linear",
            "training.batch_size": 1,
            "training.max_steps": 1,
            "training.save_every": 1,
            "training.learning_rate": 0.001,
            "training.num_workers": 0,
            "training.condition_dropout": 0.0,
            "sampling.sampler": "ddim",
            "sampling.steps": 1,
            "sampling.batch_size": 2,
            "sampling.guidance_scale": 1.0,
            "sampling.output_dir": str(tmp_path / "generated"),
            "checkpointing.checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_script_tiny_train_generate_validate_evaluate_prepare_and_package(tmp_path: Path) -> None:
    config_path = _write_tiny_config(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "checkpoint_step_1.pt"
    generated = tmp_path / "generated"
    generate_csv = tmp_path / "generate.csv"

    train = _run(["scripts/train.py", "--config", str(config_path)])
    assert train.returncode == 0, train.stderr
    assert checkpoint.exists()

    generate = _run(
        [
            "scripts/generate.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--overwrite",
        ]
    )
    assert generate.returncode == 0, generate.stderr

    validate = _run(
        [
            "scripts/validate_submission.py",
            "--generate-csv",
            str(generate_csv),
            "--output-dir",
            str(generated),
            "--report-json",
            str(tmp_path / "validation.json"),
        ]
    )
    assert validate.returncode == 0, validate.stderr

    evaluate = _run(
        [
            "scripts/evaluate.py",
            "--generate-csv",
            str(generate_csv),
            "--output-dir",
            str(generated),
            "--reference-dir",
            str(tmp_path / "missing_ref"),
            "--report-path",
            str(tmp_path / "evaluation.json"),
        ]
    )
    assert evaluate.returncode == 0, evaluate.stderr

    ref = tmp_path / "ref"
    ref.mkdir()
    np.save(ref / "test_mu.npy", np.zeros(2))
    np.save(ref / "test_sigma.npy", np.eye(2))
    score_input = _run(
        [
            "scripts/prepare_score_input.py",
            "--generate-csv",
            str(generate_csv),
            "--generated-images",
            str(generated),
            "--score-input-dir",
            str(tmp_path / "score_input"),
            "--test-mu",
            str(ref / "test_mu.npy"),
            "--test-sigma",
            str(ref / "test_sigma.npy"),
            "--scores",
            "fid",
            "--overwrite",
        ]
    )
    assert score_input.returncode == 0, score_input.stderr

    package = _run(
        [
            "scripts/package_submission.py",
            "--generate-csv",
            str(generate_csv),
            "--generated-images",
            str(generated),
            "--checkpoint",
            str(checkpoint),
            "--student-id",
            "A12345678",
            "--output-zip",
            str(tmp_path / "HW6_A12345678.zip"),
            "--overwrite",
        ]
    )
    assert package.returncode == 0, package.stderr
    assert (tmp_path / "HW6_A12345678.zip").exists()


def test_generate_parser_preserves_config_overwrite_default() -> None:
    default_args = parse_generate_args(["--checkpoint", "checkpoint.pt"])
    overwrite_args = parse_generate_args(["--checkpoint", "checkpoint.pt", "--overwrite"])

    assert default_args.overwrite is None
    assert overwrite_args.overwrite is True


def test_validate_script_returns_nonzero_for_invalid_output(tmp_path: Path) -> None:
    generate_csv = tmp_path / "generate.csv"
    generate_csv.write_text(
        "id,animal,object,prompt\n000101.png,cat,chair,a cat and a chair\n",
        encoding="utf-8",
    )

    result = _run(
        [
            "scripts/validate_submission.py",
            "--generate-csv",
            str(generate_csv),
            "--output-dir",
            str(tmp_path / "missing"),
        ]
    )

    assert result.returncode == 1
    assert "validation failed" in result.stderr
