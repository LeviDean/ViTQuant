from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from timm.layers.attention import maybe_add_mask, resolve_self_attn_mask

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
                  groups=mod.groups, bias=mod.bias is not None,
                  padding_mode=mod.padding_mode)
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


class QuantAttention(nn.Module):
    """timm ViT Attention rewritten with explicit matmuls so the score (q@k^T)
    and context (attn@v) matmuls can be fake-quantized. qkv/proj become QuantLinear."""

    @classmethod
    def from_float(cls, attn: nn.Module, qconfig: QConfig) -> "QuantAttention":
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.num_heads = attn.num_heads
        new.head_dim = attn.head_dim
        new.attn_dim = attn.attn_dim
        new.scale = attn.scale
        new.qkv = QuantLinear.from_float(attn.qkv, qconfig)
        new.q_norm = attn.q_norm
        new.k_norm = attn.k_norm
        new.attn_drop = attn.attn_drop
        new.norm = attn.norm
        new.proj = QuantLinear.from_float(attn.proj, qconfig)
        new.proj_drop = attn.proj_drop
        new.qk_matmul = QuantMatMul(qconfig)
        new.av_matmul = QuantMatMul(qconfig)
        return new

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
               is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
        attn = attn.softmax(dim=-1)  # softmax stays fp32 per spec
        attn = self.attn_drop(attn)
        x = self.av_matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
