#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.validate import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated HW6 PNG files")
    parser.add_argument("--generate-csv", default="dataset/generate.csv")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--report-json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = validate_submission(args.generate_csv, args.output_dir, smoke=args.smoke, report_json=args.report_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
