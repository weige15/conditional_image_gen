#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.package import package_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the HW6 E3 submission zip")
    parser.add_argument("--generate-csv", default="dataset/generate.csv")
    parser.add_argument("--generated-images", default="generated_images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zip-path")
    parser.add_argument("--student-id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.zip_path and not args.student_id:
        raise ValueError("Provide --zip-path or --student-id")
    zip_path = args.zip_path or f"HW6_{args.student_id}.zip"
    root_name = f"HW6_{args.student_id}" if args.student_id else None
    path = package_submission(
        args.generate_csv,
        args.generated_images,
        args.checkpoint,
        zip_path,
        overwrite=args.overwrite,
        root_name=root_name,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
