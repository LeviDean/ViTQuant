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
    neither (identity). freeze() computes qparams from the observer but does NOT
    enable quantizing — call set_quantizing() to activate. Separating "derive
    qparams" from "apply quantization" lets block_sensitivity toggle quantizing
    on/off after calibration, and lets weights be calibrated independently of
    activations (see calibrate_weights)."""

    def __init__(self, cfg: TensorQConfig):
        super().__init__()
        self.cfg = cfg
        self.observer = build_observer(cfg)
        self.observing = False
        self.quantizing = False
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("zero_point", torch.empty(0))

    @property
    def is_frozen(self) -> bool:
        """True once qparams have been derived (freeze() has run)."""
        return self.scale.numel() > 0

    def freeze(self) -> None:
        self.scale, self.zero_point = self.observer.compute_qparams()
        self.observing = False

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
    """Toggle observing on quantizers that are not yet frozen. Frozen quantizers
    (e.g. weights calibrated by calibrate_weights) are skipped so a subsequent
    activation-calibration pass doesn't needlessly re-observe them."""
    for m in model.modules():
        if isinstance(m, FakeQuantize) and not m.is_frozen:
            m.observing = enabled


def set_quantizing(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.quantizing = enabled


def freeze_qparams(model: nn.Module) -> None:
    """Derive qparams for every not-yet-frozen quantizer from its observer.
    Already-frozen quantizers (weights) are left as-is."""
    for m in model.modules():
        if isinstance(m, FakeQuantize) and not m.is_frozen:
            m.freeze()


def calibrate_weights(model: nn.Module) -> None:
    """Weight quantization is data-independent: a weight quantizer's qparams
    depend only on the weight tensor, not on any calibration data. Observe each
    weight once and freeze it here, so the calibration data pass only has to
    collect activation statistics. Leaves weight quantizers frozen but not yet
    quantizing (set_quantizing enables application). Numerically identical to
    observing the weight on every batch — for MinMax it's idempotent, for
    moving-average/percentile the stat converges to the same single-tensor
    value — just without the redundant repeated observation."""
    for m in model.modules():
        weight_fq = getattr(m, "weight_fq", None)
        weight = getattr(m, "weight", None)
        if isinstance(weight_fq, FakeQuantize) and isinstance(weight, torch.Tensor):
            weight_fq.observer(weight.detach())
            weight_fq.freeze()
