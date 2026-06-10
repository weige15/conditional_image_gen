from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(
    values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size
) -> torch.Tensor:
    if timesteps.min() < 0 or timesteps.max() >= values.shape[0]:
        raise ValueError("timestep out of range")
    out = values.to(timesteps.device).gather(0, timesteps.long())
    return out.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine") -> None:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown diffusion schedule: {schedule}")
        self.timesteps = timesteps
        self.schedule = schedule
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: torch.device | str) -> DiffusionSchedule:
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def q_sample(
        self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_x0_from_epsilon(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_one_minus * epsilon) / sqrt_alpha

    def ddim_timesteps(self, steps: int) -> torch.Tensor:
        if not 1 <= steps <= self.timesteps:
            raise ValueError("DDIM steps must be in [1, training timesteps]")
        return torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prev_t: torch.Tensor,
        pred_epsilon: torch.Tensor,
        *,
        eta: float = 0.0,
    ) -> torch.Tensor:
        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = torch.ones_like(alpha_t)
        valid_prev = prev_t >= 0
        if valid_prev.any():
            alpha_prev[valid_prev] = extract(
                self.alphas_cumprod, prev_t[valid_prev], x_t[valid_prev].shape
            )
        pred_x0 = ((x_t - torch.sqrt(1 - alpha_t) * pred_epsilon) / torch.sqrt(alpha_t)).clamp(
            -1, 1
        )
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        direction = torch.sqrt((1 - alpha_prev - sigma**2).clamp_min(0)) * pred_epsilon
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_prev) * pred_x0 + direction + sigma * noise
