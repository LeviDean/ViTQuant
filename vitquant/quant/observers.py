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


def qparams_from_range(min_val: torch.Tensor, max_val: torch.Tensor,
                       cfg: TensorQConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """scale/zero_point from a clip range, the single formula shared by every
    observer (and by MSEObserver's candidate search, so a searched range yields
    exactly the qparams that compute_qparams will later derive from it)."""
    qmin, qmax = qrange(cfg.bits, cfg.symmetric)
    # Range must include 0 so that real zero is exactly representable.
    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))
    if cfg.symmetric:
        scale = torch.maximum(max_val.abs(), min_val.abs()) / qmax
        scale = torch.clamp(scale, min=1e-12)
        zero_point = torch.zeros_like(scale)
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-12)
        zero_point = torch.clamp(torch.round(qmin - min_val / scale),
                                 float(qmin), float(qmax))
    return scale, zero_point


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
        # Stats are always kept in fp32 regardless of input dtype: calibration in
        # this framework runs on the fp32 pretrained model, and qparams derived
        # from fp32 stats stay precise even if inputs were lower precision.
        x = x.detach().float()
        if self.cfg.per_channel:
            flat = x.transpose(0, self.cfg.ch_axis).flatten(1)
            return flat.min(dim=1).values, flat.max(dim=1).values
        return x.min().reshape(1), x.max().reshape(1)

    def compute_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_stats:
            raise CalibrationError(
                f"{type(self).__name__} has no statistics; run calibration first.")
        return qparams_from_range(self.min_val, self.max_val, self.cfg)


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


class MSEObserver(ObserverBase):
    """MSE-optimal clipping: instead of trusting the raw min/max (which a single
    outlier can inflate, wasting resolution on everyone else), grid-search a
    shrink factor alpha over candidate clip ranges [alpha*min, alpha*max] and
    keep the one whose quantize-dequantize MSE against the observed tensor is
    smallest — the standard calibration used by NPU/TensorRT-style toolchains.
    Per-channel configs search alpha independently per channel (vectorized).
    Across multiple observations (activations) the chosen ranges are merged with
    an exponential moving average, like MovingAvgMinMaxObserver; weights are
    observed once so the search result is used directly."""

    STEPS = 40        # alpha candidates, log-spaced in [MIN_ALPHA, 1.0]
    MIN_ALPHA = 0.01  # search down to 1% of the observed range — a linear grid
                      # can't reach useful clips when an outlier is 100x the bulk
    MOMENTUM = 0.1

    def _search(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach().float()
        lo_full, hi_full = self._reduce(x)  # (C,) per-channel or (1,) per-tensor
        # rows: (C, N) with one row per channel, or (1, numel) per-tensor —
        # the same reduction axes _reduce uses, so errors align with lo/hi.
        if self.cfg.per_channel:
            rows = x.transpose(0, self.cfg.ch_axis).flatten(1)
        else:
            rows = x.reshape(1, -1)
        qmin, qmax = qrange(self.cfg.bits, self.cfg.symmetric)

        best_err = torch.full_like(lo_full, float("inf"))
        best_lo, best_hi = lo_full.clone(), hi_full.clone()
        for i in range(self.STEPS):
            alpha = self.MIN_ALPHA ** (i / (self.STEPS - 1))  # 1.0 → MIN_ALPHA, log-spaced
            lo, hi = alpha * lo_full, alpha * hi_full
            scale, zp = qparams_from_range(lo, hi, self.cfg)
            scale, zp = scale.unsqueeze(1), zp.unsqueeze(1)
            dq = (torch.clamp(torch.round(rows / scale) + zp, qmin, qmax) - zp) * scale
            err = ((dq - rows) ** 2).mean(dim=1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_lo = torch.where(better, lo, best_lo)
            best_hi = torch.where(better, hi, best_hi)
        return best_lo, best_hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self._search(x)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = self.min_val + self.MOMENTUM * (lo - self.min_val)
            self.max_val = self.max_val + self.MOMENTUM * (hi - self.max_val)
        return x


_OBSERVERS = {
    "minmax": MinMaxObserver,
    "moving_avg": MovingAvgMinMaxObserver,
    "percentile": PercentileObserver,
    "mse": MSEObserver,
}


def build_observer(cfg: TensorQConfig) -> ObserverBase:
    try:
        return _OBSERVERS[cfg.observer](cfg)
    except KeyError:
        raise ValueError(f"Unknown observer '{cfg.observer}'. Choose from {list(_OBSERVERS)}")
