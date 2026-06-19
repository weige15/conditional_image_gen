from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.evaluate import evaluate_outputs, validation_failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write validation-first evaluation report")
    parser.add_argument("--generate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", default="hw6_reference")
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--clip-proxy", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate_outputs(
            generate_csv=args.generate_csv,
            output_dir=args.output_dir,
            reference_dir=args.reference_dir,
            report_path=args.report_path,
            run_fid=not args.no_fid,
            run_clip_proxy=args.clip_proxy,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if validation_failed(report):
        print(f"evaluation stopped after validation failure: {args.report_path}", file=sys.stderr)
        return 1
    print(f"wrote evaluation report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

