import torch
from torch import nn

from vitquant.quant.qconfig import TensorQConfig


class CalibrationError(RuntimeError):
    """Raised when qparams are requested before any statistics were collected."""


def qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer range. Symmetric uses restricted range so zero_point can be 0."""
    if symmetric:
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


class ObserverBase(nn.Module):
    def __init__(self, cfg: TensorQConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("min_val", torch.empty(0))
        self.register_buffer("max_val", torch.empty(0))

    @property
    def has_stats(self) -> bool:
        return self.min_val.numel() > 0

    def _reduce(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel: min/max over all dims except ch_axis. Per-tensor: global."""
        x = x.detach().float()
        if self.cfg.per_channel:
            flat = x.transpose(0, self.cfg.ch_axis).flatten(1)
            return flat.min(dim=1).values, flat.max(dim=1).values
        return x.min().reshape(1), x.max().reshape(1)

    def compute_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_stats:
            raise CalibrationError(
                f"{type(self).__name__} has no statistics; run calibration first.")
        qmin, qmax = qrange(self.cfg.bits, self.cfg.symmetric)
        # Range must include 0 so that real zero is exactly representable.
        min_val = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        if self.cfg.symmetric:
            scale = torch.maximum(max_val.abs(), min_val.abs()) / qmax
            scale = torch.clamp(scale, min=1e-12)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-12)
            zero_point = torch.clamp(torch.round(qmin - min_val / scale),
                                     float(qmin), float(qmax))
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self._reduce(x)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = torch.minimum(self.min_val, lo)
            self.max_val = torch.maximum(self.max_val, hi)
        return x


class MovingAvgMinMaxObserver(ObserverBase):
    MOMENTUM = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self._reduce(x)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = self.min_val + self.MOMENTUM * (lo - self.min_val)
            self.max_val = self.max_val + self.MOMENTUM * (hi - self.max_val)
        return x


class PercentileObserver(ObserverBase):
    MAX_SAMPLES = 1_000_000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.per_channel:
            raise NotImplementedError("PercentileObserver supports per-tensor only")
        flat = x.detach().float().flatten()
        if flat.numel() > self.MAX_SAMPLES:  # torch.quantile has input size limits
            idx = torch.randint(flat.numel(), (self.MAX_SAMPLES,), device=flat.device)
            flat = flat[idx]
        lo = torch.quantile(flat, 1.0 - self.cfg.percentile).reshape(1)
        hi = torch.quantile(flat, self.cfg.percentile).reshape(1)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = torch.minimum(self.min_val, lo)
            self.max_val = torch.maximum(self.max_val, hi)
        return x


_OBSERVERS = {
    "minmax": MinMaxObserver,
    "moving_avg": MovingAvgMinMaxObserver,
    "percentile": PercentileObserver,
}


def build_observer(cfg: TensorQConfig) -> ObserverBase:
    try:
        return _OBSERVERS[cfg.observer](cfg)
    except KeyError:
        raise ValueError(f"Unknown observer '{cfg.observer}'. Choose from {list(_OBSERVERS)}")
