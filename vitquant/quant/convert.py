from torch import nn
from timm.models.vision_transformer import Attention

from vitquant.quant.modules import QuantAttention, QuantConv2d, QuantLinear
from vitquant.quant.qconfig import QConfig

DEFAULT_SKIP = ("head",)  # classifier head stays fp32


def convert_vit(model: nn.Module, qconfig: QConfig,
                skip: tuple[str, ...] = DEFAULT_SKIP) -> nn.Module:
    """In-place replacement: Attention -> QuantAttention, Linear -> QuantLinear,
    Conv2d -> QuantConv2d. LayerNorm/Softmax/GELU are untouched (stay fp32)."""
    _convert(model, qconfig, skip, prefix="")
    return model


def _convert(module: nn.Module, qconfig: QConfig, skip: tuple[str, ...],
             prefix: str) -> None:
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if any(full == s or full.startswith(s + ".") for s in skip):
            continue
        if isinstance(child, Attention):
            setattr(module, name, QuantAttention.from_float(child, qconfig))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear.from_float(child, qconfig))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d.from_float(child, qconfig))
        else:
            _convert(child, qconfig, skip, full)
