from __future__ import annotations

import pytest
import torch

from brainrot_diffusion.ema import EMA


def _linear() -> torch.nn.Linear:
    layer = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    return layer


def test_one_step_ema_math() -> None:
    model = _linear()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)

    ema.update(model)

    assert ema.shadow is not None
    assert torch.equal(ema.shadow["weight"], torch.tensor([[2.0]]))


def test_disabled_ema_is_noop_and_not_serialized() -> None:
    model = _linear()
    ema = EMA(model, enabled=False)

    ema.update(model)

    assert ema.state_dict() is None
    with pytest.raises(ValueError, match="disabled"):
        ema.copy_to(model)


def test_state_round_trip_and_copy_to_model() -> None:
    model = _linear()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(model)

    restored = EMA(_linear(), decay=0.9)
    restored.load_state_dict(ema.state_dict(), _linear())
    target = _linear()
    restored.copy_to(target)

    assert torch.equal(target.weight, torch.tensor([[3.0]]))


def test_shape_mismatch_fails() -> None:
    model = _linear()
    ema = EMA(model)

    with pytest.raises(ValueError, match="shape mismatch"):
        ema.load_state_dict({"decay": 0.5, "shadow": {"weight": torch.zeros(2, 2)}}, model)

