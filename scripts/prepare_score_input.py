from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.evaluate import prepare_score_input


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Codabench-style scorer input")
    parser.add_argument("--generate-csv", required=True)
    parser.add_argument("--generated-images", required=True)
    parser.add_argument("--score-input-dir", required=True)
    parser.add_argument("--test-mu", required=True)
    parser.add_argument("--test-sigma", required=True)
    parser.add_argument("--scores", nargs="+", default=["fid"])
    parser.add_argument("--config-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_score_input(
            generate_csv=args.generate_csv,
            generated_images=args.generated_images,
            score_input_dir=args.score_input_dir,
            test_mu=args.test_mu,
            test_sigma=args.test_sigma,
            scores=args.scores,
            overwrite=args.overwrite,
            config_json=args.config_json,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"prepared scorer input: {result['input_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

