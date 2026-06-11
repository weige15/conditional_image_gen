#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.validate import expected_filenames, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Codabench-style input/ref and input/res folders")
    parser.add_argument("--generate-csv", default="dataset/generate.csv")
    parser.add_argument("--generated-images", default="generated_images")
    parser.add_argument("--score-input-dir", required=True)
    parser.add_argument("--test-mu", default="hw6_reference/test_mu.npy")
    parser.add_argument("--test-sigma", default="hw6_reference/test_sigma.npy")
    parser.add_argument("--scores", nargs="+", default=["fid"], choices=["fid", "clip_i", "clip_t"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_submission(args.generate_csv, args.generated_images)
    root = Path(args.score_input_dir)
    if root.exists() and not args.overwrite:
        raise FileExistsError(f"Score input dir exists: {root}")
    if root.exists():
        shutil.rmtree(root)
    ref = root / "ref"
    res = root / "res"
    ref.mkdir(parents=True)
    res.mkdir(parents=True)
    shutil.copy2(args.test_mu, ref / "test_mu.npy")
    shutil.copy2(args.test_sigma, ref / "test_sigma.npy")
    for name in expected_filenames(args.generate_csv):
        shutil.copy2(Path(args.generated_images) / name, res / name)
    (ref / "config.json").write_text(
        json.dumps({"scores": args.scores, "image_size": 64, "num_images": len(expected_filenames(args.generate_csv))}, indent=2),
        encoding="utf-8",
    )
    print(f"prepared {root}")


if __name__ == "__main__":
    main()
