from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.to(timesteps.device).gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine") -> None:
        if beta_schedule != "cosine":
            raise ValueError("Only cosine beta_schedule is supported")
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.betas = cosine_beta_schedule(self.timesteps)
        if not torch.isfinite(self.betas).all() or not ((self.betas > 0) & (self.betas < 1)).all():
            raise ValueError("Invalid beta schedule")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def metadata(self) -> dict[str, object]:
        return {"timesteps": self.timesteps, "beta_schedule": self.beta_schedule}

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        if t.shape[0] != x_start.shape[0]:
            raise ValueError("t batch size must match x_start batch size")
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def training_loss(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        conditions: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = model(x_noisy, t, conditions)
        if predicted.shape != noise.shape:
            raise ValueError(f"Model predicted shape {predicted.shape}, expected {noise.shape}")
        return F.mse_loss(predicted, noise)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.predict_x0_from_eps(x_t, t, eps).clamp(-1.0, 1.0)
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = _extract(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def p_sample_ddpm(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mean, variance = self.p_mean_variance(x_t, t, eps)
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        nonzero_mask = (t != 0).float().reshape(t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prev_t: torch.Tensor,
        eps: torch.Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if eta < 0:
            raise ValueError("eta must be non-negative")
        alpha_t = _extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = torch.where(
            (prev_t >= 0).reshape(-1, *((1,) * (x_t.ndim - 1))),
            _extract(self.alphas_cumprod, prev_t.clamp(min=0), x_t.shape),
            torch.ones_like(alpha_t),
        )
        pred_x0 = ((x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)).clamp(-1.0, 1.0)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).clamp(min=0)
        direction = torch.sqrt((1 - alpha_prev - sigma**2).clamp(min=0)) * eps
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return torch.sqrt(alpha_prev) * pred_x0 + direction + sigma * noise
