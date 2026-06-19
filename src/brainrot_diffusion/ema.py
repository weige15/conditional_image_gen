"""Exponential moving average state for PyTorch modules."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999, enabled: bool = True) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.enabled = enabled
        self.shadow: dict[str, torch.Tensor] | None = None
        if enabled:
            self.shadow = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
                if torch.is_floating_point(value)
            }

    def update(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        assert self.shadow is not None
        current = model.state_dict()
        for key, shadow_value in self.shadow.items():
            if key not in current:
                raise ValueError(f"EMA key missing from model: {key}")
            value = current[key]
            if value.shape != shadow_value.shape:
                raise ValueError(f"EMA shape mismatch for {key}")
            shadow_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object] | None:
        if not self.enabled:
            return None
        assert self.shadow is not None
        return {
            "decay": self.decay,
            "shadow": {key: value.detach().clone() for key, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object], model: torch.nn.Module | None = None) -> None:
        if not self.enabled:
            raise ValueError("cannot load EMA state when EMA is disabled")
        if "decay" not in state or "shadow" not in state:
            raise ValueError("EMA state must contain decay and shadow")
        shadow = state["shadow"]
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA shadow must be a mapping")
        loaded = {str(key): value.detach().clone() for key, value in shadow.items() if isinstance(value, torch.Tensor)}
        if len(loaded) != len(shadow):
            raise ValueError("EMA shadow values must be tensors")
        if model is not None:
            self._validate_against_model(loaded, model)
        self.decay = float(state["decay"])
        if not 0 <= self.decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.shadow = loaded

    def copy_to(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            raise ValueError("cannot copy disabled EMA to model")
        assert self.shadow is not None
        self._validate_against_model(self.shadow, model)
        state = model.state_dict()
        state.update({key: value.detach().clone().to(state[key].device) for key, value in self.shadow.items()})
        model.load_state_dict(state)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(backup)

    def _validate_against_model(self, shadow: Mapping[str, torch.Tensor], model: torch.nn.Module) -> None:
        current = model.state_dict()
        for key, value in shadow.items():
            if key not in current:
                raise ValueError(f"EMA key missing from model: {key}")
            if current[key].shape != value.shape:
                raise ValueError(f"EMA shape mismatch for {key}")

