from __future__ import annotations

import math

import torch
from torch import nn

from .conditioning import ConditionBatch


def _groups(channels: int) -> int:
    for group in (32, 16, 8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels * 2))
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(flat).chunk(3, dim=1)
        scale = c**-0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        return x + self.proj(out).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 64,
        base_channels: int = 96,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 3, 4),
        attention_resolutions: list[int] | tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        emb_dim: int = 384,
        num_animals: int = 11,
        num_objects: int = 11,
        num_pairs: int = 101,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.metadata = {
            "image_size": image_size,
            "base_channels": base_channels,
            "channel_mults": list(channel_mults),
            "attention_resolutions": list(attention_resolutions),
            "dropout": dropout,
            "emb_dim": emb_dim,
        }
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.animal_emb = nn.Embedding(num_animals, emb_dim)
        self.object_emb = nn.Embedding(num_objects, emb_dim)
        self.pair_emb = nn.Embedding(num_pairs, emb_dim)
        self.input_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = base_channels
        resolution = image_size
        skip_channels: list[int] = []
        for index, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            block = ResBlock(channels, out_channels, emb_dim, dropout)
            attention = (
                AttentionBlock(out_channels)
                if resolution in attention_resolutions
                else nn.Identity()
            )
            self.down_blocks.append(nn.ModuleList([block, attention]))
            skip_channels.append(out_channels)
            channels = out_channels
            if index != len(channel_mults) - 1:
                self.downsamples.append(Downsample(channels))
                resolution //= 2

        self.mid1 = ResBlock(channels, channels, emb_dim, dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid2 = ResBlock(channels, channels, emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            skip = skip_channels.pop()
            block = ResBlock(channels + skip, out_channels, emb_dim, dropout)
            attention = (
                AttentionBlock(out_channels)
                if resolution in attention_resolutions
                else nn.Identity()
            )
            self.up_blocks.append(nn.ModuleList([block, attention]))
            channels = out_channels
            if index != len(channel_mults) - 1:
                self.upsamples.append(Upsample(channels))
                resolution *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def _embedding(self, timesteps: torch.Tensor, conditions: ConditionBatch) -> torch.Tensor:
        return (
            self.time_mlp(timesteps)
            + self.animal_emb(conditions.animal)
            + self.object_emb(conditions.object)
            + self.pair_emb(conditions.pair)
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, conditions: ConditionBatch
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                f"expected x shape [B, 3, {self.image_size}, {self.image_size}], got {tuple(x.shape)}"
            )
        if timesteps.shape != (x.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        if conditions.animal.shape != (x.shape[0],):
            raise ValueError("condition tensors must have shape [B]")
        emb = self._embedding(timesteps, conditions)
        h = self.input_conv(x)
        skips = []
        for index, (block, attention) in enumerate(self.down_blocks):
            h = attention(block(h, emb))
            skips.append(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for index, (block, attention) in enumerate(self.up_blocks):
            h = torch.cat([h, skips.pop()], dim=1)
            h = attention(block(h, emb))
            if index < len(self.upsamples):
                h = self.upsamples[index](h)
        return self.out(h)


def build_model(config, mapper) -> ConditionalUNet:
    return ConditionalUNet(
        image_size=config.image_size,
        base_channels=config.model.base_channels,
        channel_mults=config.model.channel_mults,
        attention_resolutions=config.model.attention_resolutions,
        dropout=config.model.dropout,
        emb_dim=config.model.emb_dim,
        num_animals=mapper.num_animals_with_null,
        num_objects=mapper.num_objects_with_null,
        num_pairs=mapper.num_pairs_with_null,
    )
