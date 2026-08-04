from torch import nn
from transformers.models.sam.modeling_sam import SamVisionAttention
from transformers.models.sam3.modeling_sam3 import Sam3ViTRoPEAttention

from vitquant.quant.modules import QuantConv2d, QuantLinear
from vitquant.quant.qconfig import QConfig
from vitquant.quant.sam_modules import QuantSam3ViTAttention, QuantSamAttention

DEFAULT_SCOPE = ("vision_encoder",)


def convert_sam_modules(model: nn.Module, qconfig: QConfig,
                        scope=DEFAULT_SCOPE) -> nn.Module:
    """In-place quantization conversion of the top-level SAM/SAM3 submodules
    named in `scope`: Linear -> QuantLinear, Conv2d -> QuantConv2d, and the
    vision attention classes -> their decomposed-matmul Quant rewrites.
    `scope="all"` converts every top-level child that has parameters.

    Scope semantics: weights and linear-input activations are quantized in
    every scoped module (all SAM3 attention variants — CLIPAttention,
    Sam3Attention — build on plain nn.Linear q/k/v/o, so the generic Linear
    replacement covers them). The attention score matmuls (q@k^T, attn@v) are
    additionally quantized only in the vision encoder, whose attention classes
    have explicit rewrites. Embedding, LayerNorm/GroupNorm and the mask
    upscaler's ConvTranspose2d stay fp32 everywhere (standard PTQ practice).

    Valid scope names are the model's top-level children, e.g. Sam3Model:
    vision_encoder, text_encoder, text_projection, geometry_encoder,
    detr_encoder, detr_decoder, mask_decoder, dot_product_scoring;
    Sam3TrackerModel/SamModel: vision_encoder, prompt_encoder, mask_decoder.
    Downstream stages (SmoothQuant, calibration, AdaRound, sensitivity
    grouping, persistence) discover quantized modules by named_modules(), so
    they follow whatever scope was converted with no further wiring."""
    for name in resolve_scope(model, scope):
        child = model.get_submodule(name)
        # a scoped module can itself be a bare Linear (e.g. Sam3Model's
        # text_projection) — _convert only rewrites children, so replace it
        # on the parent directly
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantLinear.from_float(child, qconfig))
        elif isinstance(child, nn.Conv2d):
            setattr(model, name, QuantConv2d.from_float(child, qconfig))
        else:
            _convert(child, qconfig)
    return model


def resolve_scope(model: nn.Module, scope=DEFAULT_SCOPE) -> list[str]:
    """Validate a quant scope against the model and return it as an explicit
    list ("all" -> every parametrized top-level child). Run scripts resolve
    before saving, so a persisted artifact's meta always names concrete
    modules rather than "all"."""
    children = dict(model.named_children())
    if scope == "all":
        return [n for n, c in children.items()
                if any(True for _ in c.parameters())]
    scope = list(scope)
    unknown = [n for n in scope if n not in children]
    if unknown:
        raise ValueError(
            f"unknown quant scope module(s) {unknown}; this model's top-level "
            f"modules are {sorted(children)}")
    return scope


def convert_sam_vision_encoder(model: nn.Module, qconfig: QConfig) -> nn.Module:
    """Backward-compatible wrapper: convert only model.vision_encoder (the
    default scope; prompt/text/decoder paths stay fp32)."""
    return convert_sam_modules(model, qconfig, DEFAULT_SCOPE)


def _convert(module: nn.Module, qconfig: QConfig) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, SamVisionAttention):
            # isinstance (not exact-type) deliberately also matches the
            # SamVisionSdpaAttention subclass, which is what from_pretrained
            # actually constructs by default (config._attn_implementation ==
            # "sdpa"). QuantSamAttention always reimplements the eager,
            # decomposed-matmul math regardless of which variant it replaces,
            # since that's what exposes q@k^T/attn@v for fake-quant hooks —
            # verified numerically identical (exact match) to the source
            # SdpaAttention's output before calibration.
            setattr(module, name, QuantSamAttention.from_float(child, qconfig))
        elif isinstance(child, Sam3ViTRoPEAttention):
            # SAM3 Perception-Encoder ViT attention (RoPE, split q/k/v/o).
            # Same decomposed-matmul rewrite; the attention_interface dispatch
            # (sdpa by default) is replaced by explicit eager math.
            setattr(module, name, QuantSam3ViTAttention.from_float(child, qconfig))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear.from_float(child, qconfig))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d.from_float(child, qconfig))
        else:
            _convert(child, qconfig)
