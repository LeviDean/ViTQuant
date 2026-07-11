"""SmoothQuant (Xiao et al., 2022): activation outliers make activations the
hard part of W8A8-style PTQ — a few channels are 10-100x larger than the rest,
and a per-tensor activation scale wastes almost all of its resolution on them.
Weights, in contrast, are flat and easy. SmoothQuant migrates the difficulty:
for each Linear, pick a per-input-channel factor s and rewrite

    y = x @ W^T  ==  (x / s) @ (s * W)^T

which is exactly equal in fp32 but gives the quantizers a tamed input (x/s)
and a mildly-rescaled weight (s*W) that per-channel weight quantization
absorbs. s_j = amax|x_j|^alpha / amax|W_:,j|^(1-alpha); alpha=0.5 balances the
migration (the paper's default).

Must run AFTER convert and BEFORE calibration: the smoothing factors are
derived from per-channel activation statistics collected in a dedicated fp32
pre-pass (quantizers transparent), and calibration must then observe the
smoothed input / scaled weight. Orthogonal to the MSE observer and AdaRound —
they compose. Only QuantLinear is smoothed; conv inputs (images / patch grids)
don't show the outlier-channel pathology."""
from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn

from vitquant.quant.calibrate import FeedFn, _feed_tensor_batch
from vitquant.quant.modules import QuantLinear


@torch.no_grad()
def _collect_channel_amax(model: nn.Module, modules: list[QuantLinear],
                          batches: list[Any], device: torch.device,
                          feed: FeedFn) -> dict[nn.Module, torch.Tensor]:
    """One fp32 pass over the calibration data recording, for every listed
    QuantLinear, the absolute max of each input channel (last dim)."""
    amax: dict[nn.Module, torch.Tensor] = {}

    def hook(mod, args):
        x = args[0].detach()
        cur = x.abs().reshape(-1, x.shape[-1]).amax(dim=0)
        prev = amax.get(mod)
        amax[mod] = cur if prev is None else torch.maximum(prev, cur)

    handles = [m.register_forward_pre_hook(hook) for m in modules]
    try:
        for batch in batches:
            feed(model, batch, device)
    finally:
        for h in handles:
            h.remove()
    return amax


def smooth_quant(model: nn.Module, batches: Iterable[Any], device: torch.device,
                 feed: FeedFn = _feed_tensor_batch, alpha: float = 0.5,
                 log: Optional[Callable[[str], None]] = None) -> nn.Module:
    """Compute and install SmoothQuant factors on every QuantLinear. Call on a
    converted-but-not-yet-calibrated model (quantizers transparent, so the
    statistics pass sees pure fp32)."""
    model = model.eval().to(device)
    linears = [(name, m) for name, m in model.named_modules()
               if isinstance(m, QuantLinear)]
    assert linears, "smooth_quant: no QuantLinear modules (run convert first)"
    for name, m in linears:
        assert not m.weight_fq.is_frozen, (
            f"smooth_quant must run before calibration (found frozen weight "
            f"quantizer on {name})")

    amax = _collect_channel_amax(model, [m for _, m in linears],
                                 list(batches), device, feed)
    for name, m in linears:
        act_amax = amax[m].clamp(min=1e-5)
        w_amax = m.weight.detach().abs().amax(dim=0).clamp(min=1e-5)
        s = act_amax.pow(alpha) / w_amax.pow(1.0 - alpha)
        # A channel that never fired gets a tiny s, exploding s*W for no
        # benefit — keep such channels neutral.
        s = torch.where(amax[m] > 1e-5, s.clamp(min=1e-5), torch.ones_like(s))
        m.smooth_scale = s
        if log is not None:
            log(f"{name}: act amax {float(amax[m].max()):.2f} -> "
                f"{float((amax[m] / s).max()):.2f} "
                f"(s range {float(s.min()):.3f}..{float(s.max()):.3f})")
    return model
