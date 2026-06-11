#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.evaluate import evaluate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and locally evaluate generated HW6 PNG files")
    parser.add_argument("--generate-csv", default="dataset/generate.csv")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--reference-dir", default="hw6_reference")
    parser.add_argument("--report-path")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--clip-proxy", action="store_true")
    args = parser.parse_args()
    report = evaluate_submission(
        args.generate_csv,
        args.output_dir,
        reference_dir=args.reference_dir,
        report_path=args.report_path,
        run_fid=not args.skip_fid,
        run_clip=args.clip_proxy,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
