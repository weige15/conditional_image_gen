from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg

from .validate import validate_submission


def calculate_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_local_fid(
    output_dir: str | Path,
    ref_mu: str | Path,
    ref_sigma: str | Path,
    image_size: int = 64,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.models import Inception_V3_Weights, inception_v3
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency-specific
        return {"status": "skipped", "reason": f"FID dependencies unavailable: {exc}"}

    output_dir = Path(output_dir)
    paths = sorted(output_dir.glob("*.png"))
    if not paths:
        return {"status": "skipped", "reason": "No generated PNG files found"}

    class FIDDataset(Dataset):
        def __init__(self, image_paths: list[Path]) -> None:
            self.image_paths = image_paths
            self.transform = transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if image.size != (image_size, image_size):
                raise ValueError(f"{self.image_paths[idx]} has size {image.size}")
            return self.transform(image)

    try:
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = nn.Identity()
        model.eval().to(torch_device)
        loader = DataLoader(FIDDataset(paths), batch_size=batch_size, shuffle=False, num_workers=0)
        features = []
        with torch.no_grad():
            for batch in loader:
                features.append(model(batch.to(torch_device)).cpu().numpy())
        fake = np.concatenate(features, axis=0)
        real_mu = np.load(ref_mu)
        real_sigma = np.load(ref_sigma)
        fake_mu = fake.mean(axis=0)
        fake_sigma = np.cov(fake, rowvar=False)
    except Exception as exc:  # pragma: no cover - depends on local model cache/device
        return {"status": "skipped", "reason": f"FID execution unavailable: {exc}"}
    return {"status": "computed", "FID": calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)}


def evaluate_submission(
    generate_csv: str | Path,
    output_dir: str | Path,
    reference_dir: str | Path = "hw6_reference",
    report_path: str | Path | None = None,
    run_fid: bool = True,
    run_clip: bool = False,
) -> dict[str, Any]:
    validation = validate_submission(generate_csv, output_dir)
    reference_dir = Path(reference_dir)
    metrics: dict[str, Any] = {}
    ref_mu = reference_dir / "test_mu.npy"
    ref_sigma = reference_dir / "test_sigma.npy"
    if run_fid and ref_mu.exists() and ref_sigma.exists():
        metrics["fid"] = compute_local_fid(output_dir, ref_mu, ref_sigma)
    else:
        metrics["fid"] = {"status": "skipped", "reason": "Reference FID stats unavailable or disabled"}
    if run_clip:
        test_json = reference_dir / "test.json"
        if not test_json.exists():
            metrics["clip_t"] = {"status": "skipped", "reason": "Hidden CLIP-T test metadata unavailable locally"}
        else:
            metrics["clip_t"] = {"status": "skipped", "reason": "CLIP proxy is not enabled by default"}
    else:
        metrics["clip_t"] = {"status": "skipped", "reason": "CLIP-T proxy disabled; official score is Codabench"}
    report = {"validation": validation, "metrics": metrics}
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
