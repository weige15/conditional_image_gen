from __future__ import annotations

from brainrot_diffusion.evaluation import evaluate_submission, prompt_image_pairs

from .conftest import write_csv, write_png


def test_evaluation_skips_optional_metrics_and_preserves_prompt_order(tmp_path) -> None:
    csv_path = tmp_path / "generate.csv"
    write_csv(
        csv_path,
        [
            {"id": "a.png", "animal": "cat", "object": "car", "prompt": "first"},
            {"id": "b.png", "animal": "dog", "object": "chair", "prompt": "second"},
        ],
    )
    out = tmp_path / "out"
    out.mkdir()
    write_png(out / "a.png")
    write_png(out / "b.png")
    report = evaluate_submission(generate_csv=csv_path, output_dir=out, strict_count=False)
    assert report["validation"]["ok"]
    assert report["fid"]["status"] == "skipped"
    assert [prompt for _, prompt in prompt_image_pairs(csv_path, out)] == ["first", "second"]
