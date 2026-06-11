from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .conditions import CONDITION_KEYS


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


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
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class DownLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        blocks: int,
        dropout: float,
        use_attention: bool,
        add_downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleList()
        ch = in_channels
        for _ in range(blocks):
            self.blocks.append(ResBlock(ch, out_channels, emb_dim, dropout))
            self.attn.append(SelfAttention(out_channels) if use_attention else nn.Identity())
            ch = out_channels
        self.downsample = Downsample(ch) if add_downsample else None
        self.out_channels = ch

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for block, attn in zip(self.blocks, self.attn):
            x = attn(block(x, emb))
            skips.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skips


class UpLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: list[int],
        out_channels: int,
        emb_dim: int,
        dropout: float,
        use_attention: bool,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleList()
        ch = in_channels
        for skip_ch in skip_channels:
            self.blocks.append(ResBlock(ch + skip_ch, out_channels, emb_dim, dropout))
            self.attn.append(SelfAttention(out_channels) if use_attention else nn.Identity())
            ch = out_channels
        self.upsample = Upsample(ch) if add_upsample else None
        self.out_channels = ch

    def forward(self, x: torch.Tensor, emb: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for block, attn in zip(self.blocks, self.attn):
            skip = skips.pop()
            if skip.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"Skip shape {skip.shape} does not match current shape {x.shape}")
            x = torch.cat([x, skip], dim=1)
            x = attn(block(x, emb))
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        num_animals: int,
        num_objects: int,
        num_pairs: int,
        image_channels: int = 3,
        base_channels: int = 96,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 2, 4),
        blocks_per_level: int = 2,
        dropout: float = 0.1,
        attention_resolutions: list[int] | tuple[int, ...] = (16,),
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ConditionalUNet currently supports image_size=64")
        self.num_animals = int(num_animals)
        self.num_objects = int(num_objects)
        self.num_pairs = int(num_pairs)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.architecture = {
            "num_animals": self.num_animals,
            "num_objects": self.num_objects,
            "num_pairs": self.num_pairs,
            "image_channels": self.image_channels,
            "base_channels": int(base_channels),
            "channel_mults": list(channel_mults),
            "blocks_per_level": int(blocks_per_level),
            "dropout": float(dropout),
            "attention_resolutions": list(attention_resolutions),
            "image_size": self.image_size,
        }

        emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.animal_emb = nn.Embedding(self.num_animals + 1, emb_dim)
        self.object_emb = nn.Embedding(self.num_objects + 1, emb_dim)
        self.pair_emb = nn.Embedding(self.num_pairs + 1, emb_dim)
        self.stem = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        resolutions: list[int] = []
        skip_channels: list[int] = []
        ch = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * int(mult)
            use_attention = resolution in set(attention_resolutions)
            add_downsample = level != len(channel_mults) - 1
            self.downs.append(
                DownLevel(ch, out_ch, emb_dim, blocks_per_level, dropout, use_attention, add_downsample)
            )
            skip_channels.extend([out_ch] * blocks_per_level)
            resolutions.append(resolution)
            ch = out_ch
            if add_downsample:
                resolution //= 2

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * int(mult)
            use_attention = resolutions[level] in set(attention_resolutions)
            level_skips = [skip_channels.pop() for _ in range(blocks_per_level)]
            add_upsample = level != 0
            self.ups.append(UpLevel(ch, level_skips, out_ch, emb_dim, dropout, use_attention, add_upsample))
            ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, image_channels, 3, padding=1),
        )

    def _validate_inputs(self, x: torch.Tensor, t: torch.Tensor, conditions: dict[str, torch.Tensor]) -> None:
        if x.ndim != 4 or x.shape[1] != self.image_channels:
            raise ValueError(f"x must have shape [B, {self.image_channels}, H, W]")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [B]")
        for key in CONDITION_KEYS:
            if key not in conditions:
                raise ValueError(f"Missing condition key {key!r}")
            value = conditions[key]
            if value.ndim != 1 or value.shape[0] != x.shape[0]:
                raise ValueError(f"conditions[{key!r}] must have shape [B]")
        limits = {
            "animal_id": self.num_animals,
            "object_id": self.num_objects,
            "pair_id": self.num_pairs,
        }
        for key, max_id in limits.items():
            value = conditions[key]
            if value.min().item() < 0 or value.max().item() > max_id:
                raise ValueError(f"conditions[{key!r}] contains ids outside [0, {max_id}]")

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditions: dict[str, torch.Tensor]) -> torch.Tensor:
        self._validate_inputs(x, t, conditions)
        emb = self.time_mlp(sinusoidal_embedding(t, self.architecture["base_channels"]))
        emb = (
            emb
            + self.animal_emb(conditions["animal_id"])
            + self.object_emb(conditions["object_id"])
            + self.pair_emb(conditions["pair_id"])
        )
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for level in self.downs:
            h, level_skips = level(h, emb)
            skips.extend(level_skips)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for level in self.ups:
            h = level(h, emb, skips)
        if skips:
            raise ValueError("Internal UNet skip stack was not fully consumed")
        return self.out(h)


def build_unet_from_config(config: dict[str, Any], mappings: dict[str, Any]) -> ConditionalUNet:
    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    return ConditionalUNet(
        num_animals=int(mappings["num_animals"]),
        num_objects=int(mappings["num_objects"]),
        num_pairs=int(mappings["num_pairs"]),
        image_channels=int(model_cfg.get("image_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 96)),
        channel_mults=list(model_cfg.get("channel_mults", [1, 2, 2, 4])),
        blocks_per_level=int(model_cfg.get("blocks_per_level", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        attention_resolutions=list(model_cfg.get("attention_resolutions", [16])),
        image_size=int(data_cfg.get("image_size", 64)),
    )
