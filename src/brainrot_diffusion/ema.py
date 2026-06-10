from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if "model" not in state:
            raise ValueError("EMA state missing model weights")
        self.decay = float(state.get("decay", self.decay))
        self.model.load_state_dict(state["model"])
