from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.evaluation import evaluate_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run structural validation and optional local metrics."
    )
    parser.add_argument("--generate-csv", default="generate.csv")
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--test-mu", default="test_mu.npy")
    parser.add_argument("--test-sigma", default="test_sigma.npy")
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--report-path", default="reports/evaluation.json")
    args = parser.parse_args()
    report = evaluate_submission(
        generate_csv=args.generate_csv,
        output_dir=args.output_dir,
        test_mu=args.test_mu,
        test_sigma=args.test_sigma,
        strict_count=not args.smoke,
        expected_count=args.expected_count,
        checkpoint_id=args.checkpoint_id,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        report_path=args.report_path,
    )
    print(report)
    if not report["validation"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
