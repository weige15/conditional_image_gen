#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.config import apply_cli_overrides
from brainrot_diffusion.sample import generate_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HW6 Brainrot PNGs from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generate-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(args.config, args)
    written = generate_from_checkpoint(
        args.checkpoint,
        config,
        generate_csv=args.generate_csv,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"wrote {len(written)} images to {Path(config['paths']['output_dir']) if args.output_dir is None else args.output_dir}")


if __name__ == "__main__":
    main()
