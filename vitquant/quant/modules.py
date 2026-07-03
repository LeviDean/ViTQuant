import torch
from torch import nn
from torch.nn import functional as F

from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.qconfig import QConfig


class QuantLinear(nn.Linear):
    """nn.Linear with fake-quant on input activation and weight.
    Construct via from_float(); shares parameter storage with the source module."""

    @classmethod
    def from_float(cls, mod: nn.Linear, qconfig: QConfig) -> "QuantLinear":
        new = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new.weight = mod.weight
        new.bias = mod.bias
        new.input_fq = FakeQuantize(qconfig.activation)
        new.weight_fq = FakeQuantize(qconfig.weight)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.input_fq(x), self.weight_fq(self.weight), self.bias)


class QuantConv2d(nn.Conv2d):
    """nn.Conv2d with fake-quant on input and weight (used for ViT patch embed)."""

    @classmethod
    def from_float(cls, mod: nn.Conv2d, qconfig: QConfig) -> "QuantConv2d":
        new = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                  stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                  groups=mod.groups, bias=mod.bias is not None)
        new.weight = mod.weight
        new.bias = mod.bias
        new.input_fq = FakeQuantize(qconfig.activation)
        new.weight_fq = FakeQuantize(qconfig.weight)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(self.input_fq(x), self.weight_fq(self.weight), self.bias)


class QuantMatMul(nn.Module):
    """Fake-quantized a @ b for attention score/value matmuls.
    Both inputs are activations -> per-tensor activation config."""

    def __init__(self, qconfig: QConfig):
        super().__init__()
        self.a_fq = FakeQuantize(qconfig.activation)
        self.b_fq = FakeQuantize(qconfig.activation)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.a_fq(a) @ self.b_fq(b)
