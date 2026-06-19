from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.validate import validate_submission


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated Brainrot images")
    parser.add_argument("--generate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--smoke", action="store_true", help="Accepted for tiny fixture compatibility")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_submission(
            args.generate_csv,
            args.output_dir,
            image_size=args.image_size,
            expected_count=args.expected_count,
            report_json=args.report_json,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not result.passed:
        print(f"validation failed with {len(result.findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"validation passed: {result.actual_count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

