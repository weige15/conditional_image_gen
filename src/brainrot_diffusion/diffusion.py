"""DDPM/DDIM diffusion math implemented directly in PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class DiffusionMetadata:
    timesteps: int
    schedule: str
    prediction_type: str
    beta_start: float
    beta_end: float

    def as_dict(self) -> dict[str, object]:
        return {
            "timesteps": self.timesteps,
            "schedule": self.schedule,
            "prediction_type": self.prediction_type,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }


class GaussianDiffusion:
    def __init__(
        self,
        *,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if schedule not in {"linear", "cosine"}:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        if beta_start <= 0 or beta_end <= 0 or beta_start >= 1 or beta_end >= 1:
            raise ValueError("beta_start and beta_end must be in (0, 1)")
        self.metadata = DiffusionMetadata(timesteps, schedule, "epsilon", beta_start, beta_end)
        self.betas = _make_betas(timesteps, schedule, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        if not torch.isfinite(self.betas).all() or torch.any(self.betas <= 0) or torch.any(self.betas >= 1):
            raise ValueError("invalid diffusion beta schedule")

    @property
    def timesteps(self) -> int:
        return self.metadata.timesteps

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def training_loss(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: Mapping[str, object],
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps, noise)
        predicted = model(x_noisy, timesteps, conditions)
        return torch.nn.functional.mse_loss(predicted, noise)

    def predict_x0_from_eps(self, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha

    def ddpm_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x0 = self.predict_x0_from_eps(x_t, timesteps, predicted_noise).clamp(-1.0, 1.0)
        mean = _extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0
        mean = mean + _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        variance = _extract(self.posterior_variance, timesteps, x_t.shape)
        noise = _randn_like(x_t, generator)
        nonzero = (timesteps != 0).float().view(-1, *((1,) * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance.clamp(min=1e-20)) * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        prev_timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
        *,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if eta < 0:
            raise ValueError("eta must be nonnegative")
        alpha_t = _extract(self.alphas_cumprod, timesteps, x_t.shape)
        alpha_prev = _extract_prev_alpha(self.alphas_cumprod, prev_timesteps, x_t.shape)
        x0 = ((x_t - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)).clamp(-1.0, 1.0)
        sigma = eta * torch.sqrt(((1.0 - alpha_prev) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_prev))
        direction = torch.sqrt((1.0 - alpha_prev - sigma**2).clamp(min=0.0)) * predicted_noise
        noise = _randn_like(x_t, generator)
        return torch.sqrt(alpha_prev) * x0 + direction + sigma * noise

    def to_metadata(self) -> dict[str, object]:
        return self.metadata.as_dict()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "GaussianDiffusion":
        diffusion_config = config.get("diffusion") if "diffusion" in config else config
        if not isinstance(diffusion_config, Mapping):
            raise ValueError("diffusion config must be a mapping")
        return cls(
            timesteps=int(diffusion_config.get("timesteps", 1000)),
            schedule=str(diffusion_config.get("schedule", "cosine")),
            beta_start=float(diffusion_config.get("beta_start", 0.0001)),
            beta_end=float(diffusion_config.get("beta_end", 0.02)),
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "GaussianDiffusion":
        return cls(
            timesteps=int(metadata["timesteps"]),
            schedule=str(metadata["schedule"]),
            beta_start=float(metadata.get("beta_start", 0.0001)),
            beta_end=float(metadata.get("beta_end", 0.02)),
        )


def _make_betas(timesteps: int, schedule: str, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.000001, 0.999).float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    device = timesteps.device
    values = values.to(device=device)
    timesteps = timesteps.to(dtype=torch.long, device=device)
    if torch.any(timesteps < 0) or torch.any(timesteps >= values.shape[0]):
        raise ValueError("timesteps out of range")
    gathered = values.gather(0, timesteps)
    return gathered.view(-1, *((1,) * (len(shape) - 1)))


def _extract_prev_alpha(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    device = timesteps.device
    values = values.to(device=device)
    result = torch.ones((timesteps.shape[0],), device=device, dtype=values.dtype)
    mask = timesteps >= 0
    if torch.any(mask):
        result[mask] = values.gather(0, timesteps[mask].long())
    return result.view(-1, *((1,) * (len(shape) - 1)))


def _randn_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
