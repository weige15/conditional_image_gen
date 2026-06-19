from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.config import apply_overrides, load_config
from brainrot_diffusion.sample import generate_from_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Brainrot images from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generate-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "data.generate_csv": args.generate_csv,
        "sampling.output_dir": args.output_dir,
        "sampling.sampler": args.sampler,
        "sampling.steps": args.steps,
        "sampling.batch_size": args.batch_size,
        "sampling.seed": args.seed,
        "sampling.guidance_scale": args.guidance_scale,
        "sampling.overwrite": args.overwrite,
    }
    try:
        config = apply_overrides(load_config(args.config), overrides)
        result = generate_from_checkpoint(
            args.checkpoint,
            config=config,
            generate_csv=args.generate_csv,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(result.files)} images to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
