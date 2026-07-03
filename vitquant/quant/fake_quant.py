import torch
from torch import nn

from vitquant.quant.observers import build_observer, qrange
from vitquant.quant.qconfig import TensorQConfig


def fake_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                  qmin: int, qmax: int, ch_axis: int | None = None) -> torch.Tensor:
    """Quantize -> dequantize in fp32, with straight-through estimator for gradients."""
    if ch_axis is not None:
        shape = [1] * x.dim()
        shape[ch_axis] = -1
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    dq = (q - zero_point) * scale
    return x + (dq - x).detach()  # STE: forward=dq, backward=identity


class FakeQuantize(nn.Module):
    """Modes: observing (collect stats, pass through), quantizing (apply fake-quant),
    neither (identity). freeze() computes qparams from the observer."""

    def __init__(self, cfg: TensorQConfig):
        super().__init__()
        self.cfg = cfg
        self.observer = build_observer(cfg)
        self.observing = False
        self.quantizing = False
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("zero_point", torch.empty(0))

    def freeze(self) -> None:
        self.scale, self.zero_point = self.observer.compute_qparams()
        self.observing = False
        self.quantizing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self.observer(x)
        if self.quantizing:
            qmin, qmax = qrange(self.cfg.bits, self.cfg.symmetric)
            ch_axis = self.cfg.ch_axis if self.cfg.per_channel else None
            return fake_quantize(x, self.scale.to(x.device),
                                 self.zero_point.to(x.device), qmin, qmax, ch_axis)
        return x


def set_observing(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.observing = enabled


def set_quantizing(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.quantizing = enabled


def freeze_qparams(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.freeze()
