"""Block-grouping of module/quantizer names, shared by the sensitivity /
mixed-precision sweeps (vitquant.eval.evaluate) and by layer-sequential PTQ
algorithms (vitquant.quant.adaround) so "what counts as a block" can never
drift between measurement and optimization."""


def group_key(name: str) -> str:
    """Map a quantizer/module name to its per-block group, for both naming
    schemes: timm ViT ("blocks.N.*") and HF SAM vision encoder
    ("vision_encoder.layers.N.*"). A transformer block groups by the path
    through its index; everything else (patch embed, SAM neck) groups by its
    owning component (drop the leaf module + the fq buffer name)."""
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] in ("blocks", "layers") and parts[i + 1].isdigit():
            return ".".join(parts[: i + 2])
    return ".".join(parts[:-2]) if len(parts) > 2 else parts[0]
