from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.validation import validate_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated Brainrot submission images.")
    parser.add_argument("--generate-csv", default="generate.csv")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true", help="Do not require exactly 2,000 rows.")
    parser.add_argument("--report-json")
    args = parser.parse_args()
    report = validate_submission(
        args.generate_csv,
        args.output_dir,
        expected_count=args.expected_count,
        strict_count=not args.smoke,
        report_json=args.report_json,
    )
    if report.ok:
        print(f"validation passed: checked {report.checked_images} images")
    else:
        print("validation failed:")
        for error in report.errors:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
