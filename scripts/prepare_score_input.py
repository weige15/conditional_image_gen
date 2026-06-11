from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainrot_diffusion.validation import validate_submission  # noqa: E402


def copy_pngs(source_dir: Path, destination_dir: Path, *, overwrite: bool) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source_dir.glob("*.png")):
        target = destination_dir / path.name
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite")
        shutil.copy2(path, target)
        count += 1
    return count


def copy_if_present(path: Path | None, destination: Path, *, overwrite: bool) -> bool:
    if path is None or not path.exists():
        return False
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Codabench-style input/ref and input/res folders for score.py."
    )
    parser.add_argument("--generate-csv", required=True)
    parser.add_argument("--generated-images", default="generated_images")
    parser.add_argument("--score-input-dir", default="score_input")
    parser.add_argument("--test-mu")
    parser.add_argument("--test-sigma")
    parser.add_argument("--test-json")
    parser.add_argument("--test-image-root")
    parser.add_argument("--scores", nargs="+", choices=["fid", "clip_t", "clip_i"], default=["fid"])
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generated_images = Path(args.generated_images)
    score_input = Path(args.score_input_dir)
    ref_dir = score_input / "ref"
    res_dir = score_input / "res"

    report = validate_submission(
        args.generate_csv,
        generated_images,
        expected_count=args.expected_count,
        strict_count=not args.smoke,
    )
    if not report.ok:
        raise SystemExit("generated image validation failed: " + "; ".join(report.errors))

    image_count = copy_pngs(generated_images, res_dir, overwrite=args.overwrite)
    copied_mu = copy_if_present(
        Path(args.test_mu) if args.test_mu else None,
        ref_dir / "test_mu.npy",
        overwrite=args.overwrite,
    )
    copied_sigma = copy_if_present(
        Path(args.test_sigma) if args.test_sigma else None,
        ref_dir / "test_sigma.npy",
        overwrite=args.overwrite,
    )
    copied_json = copy_if_present(
        Path(args.test_json) if args.test_json else None,
        ref_dir / "test.json",
        overwrite=args.overwrite,
    )
    if args.test_image_root:
        source_root = Path(args.test_image_root)
        target_root = ref_dir / "test"
        if target_root.exists() and not args.overwrite:
            raise FileExistsError(f"{target_root} already exists; pass --overwrite")
        if source_root.exists():
            if target_root.exists():
                shutil.rmtree(target_root)
            shutil.copytree(source_root, target_root)

    scores = list(args.scores)
    if any(score in {"clip_t", "clip_i"} for score in scores) and not copied_json:
        raise SystemExit("clip_t/clip_i scoring requires --test-json and usually --test-image-root")
    if "fid" in scores and not (copied_mu and copied_sigma):
        raise SystemExit("fid scoring requires --test-mu and --test-sigma")

    config = {
        "image_size": 64,
        "num_images": image_count,
        "scores": scores,
        "ref_mu": "test_mu.npy",
        "ref_sigma": "test_sigma.npy",
        "batch_size": 32,
        "num_workers": 4,
        "verbose": True,
    }
    (ref_dir / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
    print(
        {
            "score_input_dir": str(score_input),
            "res_dir": str(res_dir),
            "ref_dir": str(ref_dir),
            "config": str(ref_dir / "config.json"),
            "num_images": image_count,
            "scores": scores,
        }
    )


if __name__ == "__main__":
    main()
