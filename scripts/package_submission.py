from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.packaging import package_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and optionally zip HW6 submission artifacts."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--generate-csv", default="generate.csv")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip-path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        package_submission(
            root=args.root,
            generate_csv=args.generate_csv,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            strict_count=not args.smoke,
            expected_count=args.expected_count,
            zip_path=args.zip_path,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
