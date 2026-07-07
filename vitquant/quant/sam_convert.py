from torch import nn
from transformers.models.sam.modeling_sam import SamVisionAttention

from vitquant.quant.modules import QuantConv2d, QuantLinear
from vitquant.quant.qconfig import QConfig
from vitquant.quant.sam_modules import QuantSamAttention


def convert_sam_vision_encoder(model: nn.Module, qconfig: QConfig) -> nn.Module:
    """In-place replacement scoped to model.vision_encoder only:
    Attention -> QuantSamAttention, Linear -> QuantLinear, Conv2d -> QuantConv2d.
    prompt_encoder and mask_decoder are never touched (stay fp32)."""
    _convert(model.vision_encoder, qconfig)
    return model


def _convert(module: nn.Module, qconfig: QConfig) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, SamVisionAttention):
            setattr(module, name, QuantSamAttention.from_float(child, qconfig))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear.from_float(child, qconfig))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d.from_float(child, qconfig))
        else:
            _convert(child, qconfig)
