from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in state.items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for key in ["decay", "updates", "shadow"]:
            if key not in state:
                raise ValueError(f"EMA state missing key {key!r}")
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.shadow = {name: value.clone() for name, value in state["shadow"].items()}  # type: ignore[union-attr]

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))

    def clone_model_with_ema(self, model: nn.Module) -> nn.Module:
        cloned = copy.deepcopy(model)
        self.copy_to(cloned)
        return cloned
