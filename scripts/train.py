from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.config import apply_overrides, load_config
from brainrot_diffusion.train_loop import train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Brainrot diffusion model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-csv")
    parser.add_argument("--train-image-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mixed-precision", action="store_true", default=None)
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "data.train_csv": args.train_csv,
        "data.train_image_dir": args.train_image_dir,
        "checkpointing.checkpoint_dir": args.checkpoint_dir,
        "training.max_steps": args.max_steps,
        "training.batch_size": args.batch_size,
        "training.learning_rate": args.learning_rate,
        "training.seed": args.seed,
        "training.mixed_precision": args.mixed_precision,
    }
    try:
        config = apply_overrides(load_config(args.config), overrides)
        result = train(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote checkpoint: {result.final_checkpoint} step={result.step} loss={result.last_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

