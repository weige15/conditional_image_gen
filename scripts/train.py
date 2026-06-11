#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.config import apply_cli_overrides
from brainrot_diffusion.train_loop import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HW6 Brainrot conditional DDPM")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-csv")
    parser.add_argument("--train-image-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = apply_cli_overrides(args.config, args)
    result = train(config)
    print(f"wrote checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
