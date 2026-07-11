"""AdaRound (Nagel et al., "Up or Down? Adaptive Rounding for Post-Training
Quantization", ICML 2020): instead of rounding each quantized weight to the
nearest grid point, learn per-weight whether to round up or down by minimizing
the layer's output reconstruction error on calibration data. No labels and no
end-to-end training — each layer is a small independent optimization (a few
hundred Adam steps on a continuous rounding variable, annealed to binary).

Runs AFTER standard calibration (run_calibration): quantization scales stay
exactly as calibrated, only the rounding decision changes, so activation
statistics remain valid. The learned choice is stored as a {0,1} round_offset
buffer on each weight FakeQuantize — the quantizing on/off toggle keeps its
semantics, so block_sensitivity / mixed_precision sweeps work unchanged on an
adarounded model.

Layers are processed in block groups (vitquant.quant.groups) in forward
order: one calibration pass per group captures the inputs of every layer in
that group with all upstream quantizers active (already-adarounded earlier
blocks included), matching the paper's sequential setup. The reconstruction
target is the layer's fp32 output on those same inputs, so the learned
rounding also compensates the input-activation quantization error
(asymmetric reconstruction, as in AIMET/BRECQ)."""
from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from vitquant.quant.calibrate import FeedFn, _feed_tensor_batch
from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.groups import group_key
from vitquant.quant.observers import qrange

# Rectified-sigmoid stretch (paper's values): h(V) = clamp(sigmoid(V)*(Z-G)+G, 0, 1)
ZETA, GAMMA = 1.1, -0.1
BETA_HIGH, BETA_LOW = 20.0, 2.0  # regularizer annealing range
WARMUP = 0.2                     # fraction of iters with reconstruction loss only


def _rectified_sigmoid(v: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


def _init_v(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Initialize V so that h(V) equals each weight's fractional remainder —
    the soft-quantized weight starts exactly at the fp32 weight's grid position
    (finite for any remainder in [0, 1] thanks to the stretched sigmoid)."""
    rest = w / scale - torch.floor(w / scale)
    return torch.log((rest - GAMMA) / (ZETA - rest))


def _effective_weight(module: nn.Module) -> torch.Tensor:
    """The weight the quantizer sees (SmoothQuant-scaled when applicable)."""
    if hasattr(module, "effective_weight"):
        return module.effective_weight()
    return module.weight


def _quant_input(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The input the matmul sees on the quantized path: smoothed (when
    applicable) then fake-quantized by the frozen activation quantizer."""
    if hasattr(module, "smooth_input"):
        x = module.smooth_input(x)
    return module.input_fq(x)


def _ops_for(module: nn.Module):
    """(fp32_target_op, quant_pred_op) for the module type. Both take the
    (possibly fake-quantized) input; the pred op additionally takes the
    soft-quantized weight. Bias is included in both, so it cancels out of the
    reconstruction error."""
    if isinstance(module, nn.Linear):
        return (lambda x: F.linear(x, module.weight, module.bias),
                lambda x, wq: F.linear(x, wq, module.bias))
    if isinstance(module, nn.Conv2d):
        conv = lambda x, w: F.conv2d(x, w, module.bias, module.stride,
                                     module.padding, module.dilation, module.groups)
        return (lambda x: conv(x, module.weight), conv)
    raise TypeError(f"AdaRound: unsupported module type {type(module).__name__}")


def _subsample_rows(x: torch.Tensor, in_features: int, max_tokens: int,
                    gen: torch.Generator) -> torch.Tensor:
    """For Linear inputs: flatten leading dims to rows of in_features and keep
    a random subset — the per-row reconstruction error is i.i.d. across tokens,
    so a subsample is an unbiased, much cheaper estimate (SAM has 4096 tokens
    per image; optimizing on all of them every step is wasted work)."""
    rows = x.reshape(-1, in_features)
    if rows.shape[0] <= max_tokens:
        return rows
    idx = torch.randint(rows.shape[0], (max_tokens,), generator=gen)
    return rows[idx.to(rows.device)]


def _optimize_module(module: nn.Module, inputs: list[torch.Tensor],
                     device: torch.device, iters: int, lr: float,
                     reg_weight: float, max_tokens: int,
                     gen: torch.Generator) -> tuple[float, float, float]:
    """Learn the rounding offsets for one layer; stores the binarized result on
    module.weight_fq. Returns (nearest_mse, adaround_mse, flipped_fraction):
    the layer's reconstruction MSE under round-to-nearest vs under the learned
    rounding (both with hard/binary decisions, evaluated on the first captured
    batch), and the fraction of weights whose rounding direction changed."""
    fq: FakeQuantize = module.weight_fq
    qmin, qmax = qrange(fq.cfg.bits, fq.cfg.symmetric)
    w = _effective_weight(module).detach()
    scale, zp = fq.scale, fq.zero_point
    if fq.cfg.per_channel:
        shape = [1] * w.dim()
        shape[fq.cfg.ch_axis] = -1
        scale, zp = scale.reshape(shape), zp.reshape(shape)
    w_floor = torch.floor(w / scale)

    v = nn.Parameter(_init_v(w, scale))
    opt = torch.optim.Adam([v], lr=lr)
    target_op, pred_op = _ops_for(module)
    is_linear = isinstance(module, nn.Linear)

    def hard_recon(offset: torch.Tensor) -> float:
        """Reconstruction MSE with a hard {0,1} offset, on a fixed eval slice."""
        with torch.no_grad():
            x = inputs[0].to(device)
            if is_linear:
                x = x.reshape(-1, module.in_features)[:max_tokens]
            wq = (torch.clamp(w_floor + offset + zp, qmin, qmax) - zp) * scale
            return float(F.mse_loss(pred_op(_quant_input(module, x), wq), target_op(x)))

    nearest = (torch.round(w / scale) - w_floor).clamp(0, 1)  # {0,1} nearest choice
    nearest_mse = hard_recon(nearest)

    for it in range(iters):
        x = inputs[int(torch.randint(len(inputs), (1,), generator=gen))].to(device)
        if is_linear:
            x = _subsample_rows(x, module.in_features, max_tokens, gen)
        with torch.no_grad():
            y = target_op(x)               # fp32 weights on the raw input
            xq = _quant_input(module, x)   # smoothed + frozen activation quantizer
        h = _rectified_sigmoid(v)
        wq = (torch.clamp(w_floor + h + zp, qmin, qmax) - zp) * scale
        loss = F.mse_loss(pred_op(xq, wq), y)
        if it >= WARMUP * iters:
            p = (it - WARMUP * iters) / ((1 - WARMUP) * iters)  # 0 -> 1
            beta = BETA_HIGH + (BETA_LOW - BETA_HIGH) * p
            loss = loss + reg_weight * (1 - (2 * h - 1).abs().pow(beta)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        offset = (_rectified_sigmoid(v) >= 0.5).to(w.dtype)
    ada_mse = hard_recon(offset)
    if ada_mse > nearest_mse:
        # The learned rounding must never ship worse than round-to-nearest
        # (can happen if the regularizer dominates a very easy layer).
        offset, ada_mse = nearest, nearest_mse
    fq.round_offset = offset
    flipped = float((offset != nearest).float().mean())
    return nearest_mse, ada_mse, flipped


@torch.no_grad()
def _capture_inputs(model: nn.Module, modules: list[nn.Module],
                    batches: list[Any], device: torch.device,
                    feed: FeedFn) -> dict[nn.Module, list[torch.Tensor]]:
    """One pass over the calibration data, recording every listed module's
    input (kept on the compute device; per-group capture keeps the footprint
    to one block's activations at a time)."""
    store: dict[nn.Module, list[torch.Tensor]] = {m: [] for m in modules}
    hooks = [m.register_forward_pre_hook(
        lambda mod, args: store[mod].append(args[0].detach()))
        for m in modules]
    try:
        for batch in batches:
            feed(model, batch, device)
    finally:
        for h in hooks:
            h.remove()
    return store


def adaround(model: nn.Module, batches: Iterable[Any], device: torch.device,
             feed: FeedFn = _feed_tensor_batch, iters: int = 1000,
             lr: float = 1e-2, reg_weight: float = 0.01, max_tokens: int = 2048,
             seed: int = 0,
             log: Optional[Callable[[str], None]] = None) -> nn.Module:
    """Refine every weight quantizer's rounding on calibration data. Call on a
    calibrated model (run_calibration done, quantizing enabled); scales/zero
    points are left untouched. Deterministic for a fixed seed."""
    model = model.eval().to(device)
    batches = list(batches)

    groups: dict[str, list[tuple[str, nn.Module]]] = {}
    for name, m in model.named_modules():
        if (isinstance(getattr(m, "weight_fq", None), FakeQuantize)
                and isinstance(getattr(m, "weight", None), torch.Tensor)):
            assert m.weight_fq.is_frozen, f"adaround before calibration: {name}"
            groups.setdefault(group_key(f"{name}.weight_fq"), []).append((name, m))

    gen = torch.Generator().manual_seed(seed)
    for gi, (gkey, mods) in enumerate(groups.items()):
        store = _capture_inputs(model, [m for _, m in mods], batches, device, feed)
        for name, m in mods:
            near, ada, flipped = _optimize_module(m, store[m], device, iters, lr,
                                                  reg_weight, max_tokens, gen)
            if log is not None:
                log(f"[{gi + 1}/{len(groups)}] {name}: recon MSE nearest "
                    f"{near:.3e} -> adaround {ada:.3e} ({flipped:.1%} flipped)")
        del store
    return model
