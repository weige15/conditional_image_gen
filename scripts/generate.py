from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.config import apply_cli_overrides, load_config  # noqa: E402
from brainrot_diffusion.sampling import generate_from_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNGs from generate.csv using EMA DDIM sampling."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generate-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        paths__generate_csv=args.generate_csv,
        paths__output_dir=args.output_dir,
        sampling__batch_size=args.batch_size,
        sampling__ddim_steps=args.ddim_steps,
        sampling__guidance_scale=args.guidance_scale,
        sampling__overwrite=True if args.overwrite else None,
    )
    print(generate_from_checkpoint(args.checkpoint, config, device=args.device))


if __name__ == "__main__":
    main()
