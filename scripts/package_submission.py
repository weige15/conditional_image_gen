from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.package import package_submission


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package HW6 E3 submission")
    parser.add_argument("--generate-csv", required=True)
    parser.add_argument("--generated-images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--output-zip")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = package_submission(
            generate_csv=args.generate_csv,
            generated_images=args.generated_images,
            checkpoint=args.checkpoint,
            student_id=args.student_id,
            output_zip=args.output_zip,
            project_root=args.project_root,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote package: {result['zip_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

