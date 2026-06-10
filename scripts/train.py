from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.config import apply_cli_overrides, load_config  # noqa: E402
from brainrot_diffusion.training import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the from-scratch conditional DDPM.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-csv")
    parser.add_argument("--train-image-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        paths__train_csv=args.train_csv,
        paths__train_image_dir=args.train_image_dir,
        paths__checkpoint_dir=args.checkpoint_dir,
        training__max_steps=args.max_steps,
    )
    result = train(config, max_steps=args.max_steps, device=args.device)
    print(result)


if __name__ == "__main__":
    main()
