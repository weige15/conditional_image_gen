"""From-scratch conditional UNet-style noise predictor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .conditions import ConditionIDs, ConditionMappings


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    image_size: int
    in_channels: int
    out_channels: int
    base_channels: int
    channel_multipliers: tuple[int, ...]
    num_res_blocks: int
    attention_resolutions: tuple[int, ...]
    embedding_dim: int
    dropout: float
    num_animals: int
    num_objects: int
    num_pairs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "channel_multipliers": list(self.channel_multipliers),
            "num_res_blocks": self.num_res_blocks,
            "attention_resolutions": list(self.attention_resolutions),
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "num_animals": self.num_animals,
            "num_objects": self.num_objects,
            "num_pairs": self.num_pairs,
        }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) * torch.arange(half, device=device).float()
        exponent = exponent / max(half - 1, 1)
        args = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.embedding = nn.Linear(embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.embedding(embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        attention = torch.bmm(q.transpose(1, 2), k).mul(c**-0.5).softmax(dim=-1)
        out = torch.bmm(v, attention.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    """Small UNet that predicts epsilon for `64x64` RGB diffusion."""

    def __init__(
        self,
        *,
        num_animals: int,
        num_objects: int,
        num_pairs: int,
        image_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        name: str = "compact_unet",
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ConditionalUNet only supports image_size=64")
        for label, value in {
            "num_animals": num_animals,
            "num_objects": num_objects,
            "num_pairs": num_pairs,
            "base_channels": base_channels,
            "embedding_dim": embedding_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if in_channels != 3 or out_channels != 3:
            raise ValueError("assignment model expects 3 input and output channels")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.metadata = ModelMetadata(
            name=name,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(),
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_animals=num_animals,
            num_objects=num_objects,
            num_pairs=num_pairs,
        )

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.animal_embedding = nn.Embedding(num_animals, embedding_dim)
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.pair_embedding = nn.Embedding(num_pairs, embedding_dim)
        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        c = base_channels
        self.input = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.down_block1 = ResidualBlock(c, c, embedding_dim, dropout)
        self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.down_block2 = ResidualBlock(c * 2, c * 2, embedding_dim, dropout)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(c * 4, c * 4, embedding_dim, dropout),
            ResidualBlock(c * 4, c * 4, embedding_dim, dropout),
        )
        self.upsample1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResidualBlock(c * 4, c * 2, embedding_dim, dropout)
        self.upsample2 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResidualBlock(c * 2, c, embedding_dim, dropout)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(c), c),
            nn.SiLU(),
            nn.Conv2d(c, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: Mapping[str, object] | ConditionIDs,
    ) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1:] != (3, 64, 64):
            raise ValueError("x_t must have shape [B, 3, 64, 64]")
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape [B]")

        animal_id = _condition_tensor(conditions, "animal_id", x_t.shape[0], x_t.device, self.metadata.num_animals)
        object_id = _condition_tensor(conditions, "object_id", x_t.shape[0], x_t.device, self.metadata.num_objects)
        pair_id = _condition_tensor(conditions, "pair_id", x_t.shape[0], x_t.device, self.metadata.num_pairs)

        embedding = self.time_embedding(timesteps.to(x_t.device))
        embedding = embedding + self.animal_embedding(animal_id)
        embedding = embedding + self.object_embedding(object_id)
        embedding = embedding + self.pair_embedding(pair_id)
        embedding = self.condition_projection(embedding)

        h0 = self.input(x_t)
        h0 = self.down_block1(h0, embedding)
        h1 = self.downsample1(h0)
        h1 = self.down_block2(h1, embedding)
        h2 = self.downsample2(h1)
        for block in self.middle:
            h2 = block(h2, embedding)
        h = self.upsample1(h2)
        h = self.up_block1(torch.cat([h, h1], dim=1), embedding)
        h = self.upsample2(h)
        h = self.up_block2(torch.cat([h, h0], dim=1), embedding)
        return self.output(h)


class AttentionUNet(nn.Module):
    """Deeper UNet with low-resolution self-attention for final training runs."""

    def __init__(
        self,
        *,
        num_animals: int,
        num_objects: int,
        num_pairs: int,
        image_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        embedding_dim: int = 256,
        dropout: float = 0.0,
        name: str = "attention_unet",
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("AttentionUNet only supports image_size=64")
        if in_channels != 3 or out_channels != 3:
            raise ValueError("assignment model expects 3 input and output channels")
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        multipliers = _int_tuple(channel_multipliers, "channel_multipliers")
        if len(multipliers) != 3:
            raise ValueError("channel_multipliers must contain exactly 3 values")
        resolutions = _int_tuple(attention_resolutions, "attention_resolutions", allow_empty=True)
        unsupported = sorted(set(resolutions) - {16, 32, 64})
        if unsupported:
            raise ValueError(f"unsupported attention resolution(s): {unsupported}")
        for label, value in {
            "num_animals": num_animals,
            "num_objects": num_objects,
            "num_pairs": num_pairs,
            "base_channels": base_channels,
            "embedding_dim": embedding_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{label} must be positive")

        c1, c2, c3 = [base_channels * multiplier for multiplier in multipliers]
        self.metadata = ModelMetadata(
            name=name,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=resolutions,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_animals=num_animals,
            num_objects=num_objects,
            num_pairs=num_pairs,
        )

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.animal_embedding = nn.Embedding(num_animals, embedding_dim)
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.pair_embedding = nn.Embedding(num_pairs, embedding_dim)
        self.condition_projection = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))

        self.input = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.down_blocks1 = nn.ModuleList(
            [ResidualBlock(c1, c1, embedding_dim, dropout) for _ in range(num_res_blocks)]
        )
        self.attn64 = _attention(c1, 64, resolutions)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.down_blocks2 = nn.ModuleList(
            [ResidualBlock(c2, c2, embedding_dim, dropout) for _ in range(num_res_blocks)]
        )
        self.attn32 = _attention(c2, 32, resolutions)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.middle1 = ResidualBlock(c3, c3, embedding_dim, dropout)
        self.middle_attn1 = _attention(c3, 16, resolutions)
        self.middle2 = ResidualBlock(c3, c3, embedding_dim, dropout)
        self.middle_attn2 = _attention(c3, 16, resolutions)

        self.upsample1 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up_blocks1 = nn.ModuleList([ResidualBlock(c2 * 2, c2, embedding_dim, dropout)])
        self.up_blocks1.extend(
            ResidualBlock(c2, c2, embedding_dim, dropout) for _ in range(num_res_blocks - 1)
        )
        self.up_attn32 = _attention(c2, 32, resolutions)
        self.upsample2 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up_blocks2 = nn.ModuleList([ResidualBlock(c1 * 2, c1, embedding_dim, dropout)])
        self.up_blocks2.extend(
            ResidualBlock(c1, c1, embedding_dim, dropout) for _ in range(num_res_blocks - 1)
        )
        self.up_attn64 = _attention(c1, 64, resolutions)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: Mapping[str, object] | ConditionIDs,
    ) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1:] != (3, 64, 64):
            raise ValueError("x_t must have shape [B, 3, 64, 64]")
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        embedding = _combined_embedding(self, x_t, timesteps, conditions)

        h0 = self.input(x_t)
        for block in self.down_blocks1:
            h0 = block(h0, embedding)
        h0 = self.attn64(h0)

        h1 = self.downsample1(h0)
        for block in self.down_blocks2:
            h1 = block(h1, embedding)
        h1 = self.attn32(h1)

        h2 = self.downsample2(h1)
        h2 = self.middle_attn1(self.middle1(h2, embedding))
        h2 = self.middle_attn2(self.middle2(h2, embedding))

        h = self.upsample1(h2)
        for index, block in enumerate(self.up_blocks1):
            h = block(torch.cat([h, h1], dim=1) if index == 0 else h, embedding)
        h = self.up_attn32(h)

        h = self.upsample2(h)
        for index, block in enumerate(self.up_blocks2):
            h = block(torch.cat([h, h0], dim=1) if index == 0 else h, embedding)
        h = self.up_attn64(h)
        return self.output(h)


def build_model_from_config(config: Mapping[str, object], mappings: ConditionMappings) -> nn.Module:
    model_config = config.get("model") if "model" in config else config
    if not isinstance(model_config, Mapping):
        raise ValueError("model config must be a mapping")
    name = str(model_config.get("name", "compact_unet"))
    common = {
        "num_animals": mappings.num_animals,
        "num_objects": mappings.num_objects,
        "num_pairs": mappings.num_pairs,
        "image_size": int(model_config.get("image_size", 64)),
        "in_channels": int(model_config.get("in_channels", 3)),
        "out_channels": int(model_config.get("out_channels", 3)),
        "base_channels": int(model_config.get("base_channels", 64)),
        "embedding_dim": int(model_config.get("embedding_dim", 256)),
        "dropout": float(model_config.get("dropout", 0.0 if name == "attention_unet" else 0.1)),
        "name": name,
    }
    if name == "compact_unet":
        return ConditionalUNet(**common)
    if name == "attention_unet":
        return AttentionUNet(
            **common,
            channel_multipliers=model_config.get("channel_multipliers", (1, 2, 4)),
            num_res_blocks=int(model_config.get("num_res_blocks", 2)),
            attention_resolutions=model_config.get("attention_resolutions", (16,)),
        )
    raise ValueError(f"unsupported model name: {name}")


def _combined_embedding(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    conditions: Mapping[str, object] | ConditionIDs,
) -> torch.Tensor:
    metadata = model.metadata
    animal_id = _condition_tensor(conditions, "animal_id", x_t.shape[0], x_t.device, metadata.num_animals)
    object_id = _condition_tensor(conditions, "object_id", x_t.shape[0], x_t.device, metadata.num_objects)
    pair_id = _condition_tensor(conditions, "pair_id", x_t.shape[0], x_t.device, metadata.num_pairs)
    embedding = model.time_embedding(timesteps.to(x_t.device))
    embedding = embedding + model.animal_embedding(animal_id)
    embedding = embedding + model.object_embedding(object_id)
    embedding = embedding + model.pair_embedding(pair_id)
    return model.condition_projection(embedding)


def _attention(channels: int, resolution: int, attention_resolutions: Sequence[int]) -> nn.Module:
    return SelfAttention2d(channels) if resolution in attention_resolutions else nn.Identity()


def _int_tuple(values: Sequence[int], name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of positive integers")
    result = tuple(int(value) for value in values)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _condition_tensor(
    conditions: Mapping[str, object] | ConditionIDs,
    key: str,
    batch_size: int,
    device: torch.device,
    limit: int,
) -> torch.Tensor:
    if isinstance(conditions, ConditionIDs):
        value = getattr(conditions, key)
    else:
        value = conditions.get(key)
    if value is None:
        raise ValueError(f"missing condition key: {key}")
    tensor = torch.as_tensor(value, dtype=torch.long, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f"{key} must have shape [B]")
    if torch.any(tensor < 0) or torch.any(tensor >= limit):
        raise ValueError(f"{key} contains out-of-range IDs")
    return tensor


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1
