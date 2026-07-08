from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.eval.metrics import AccuracyMeter
from vitquant.quant.fake_quant import FakeQuantize, set_quantizing

ProgressFn = Callable[[int, int], None]  # progress(batch_index, total_batches)


def _num_batches(loader: DataLoader, max_batches: Optional[int]) -> int:
    total = len(loader)
    return total if max_batches is None else min(max_batches, total)


@torch.no_grad()
def evaluate_torch(model: nn.Module, loader: DataLoader, class_indices: list[int],
                   device: torch.device, max_batches: Optional[int] = None,
                   progress: Optional[ProgressFn] = None) -> dict:
    """Top-1/top-5 on Imagenette: slice the 1000-class logits down to the 10
    Imagenette classes, then argmax against the 0-9 dataset labels."""
    model = model.eval().to(device)
    meter = AccuracyMeter()
    total = _num_batches(loader, max_batches)
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(x.to(device))[:, class_indices]
        meter.update(logits.cpu(), y)
        if progress is not None:
            progress(i, total)
    return {"top1": meter.top1, "top5": meter.top5}


def _group_key(name: str) -> str:
    """Map a module/quantizer name to its block group: "blocks.N" for anything
    under a transformer block, else the top-level component (e.g. patch_embed)."""
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]


def _fq_groups(model: nn.Module) -> dict[str, list[FakeQuantize]]:
    """Group FakeQuantize modules by top-level component; blocks are split
    per-block ("blocks.0", "blocks.1", ...)."""
    groups: dict[str, list[FakeQuantize]] = {}
    for name, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            groups.setdefault(_group_key(name), []).append(m)
    return groups


def _weight_params_per_group(model: nn.Module) -> dict[str, int]:
    """Number of quantizable weight parameters per block group. Used to weight
    the average-bit-width of a mixed-precision plan by real parameter count."""
    counts: dict[str, int] = {}
    for name, m in model.named_modules():
        wfq = getattr(m, "weight_fq", None)
        weight = getattr(m, "weight", None)
        if isinstance(wfq, FakeQuantize) and isinstance(weight, torch.Tensor):
            counts[_group_key(name)] = counts.get(_group_key(name), 0) + weight.numel()
    return counts


def mixed_precision_sweep(
    model: nn.Module, sensitivity: dict[str, float], loader: DataLoader,
    class_indices: list[int], device: torch.device, weight_bits: int,
    max_batches: Optional[int] = None, ks: Optional[list[int]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Build mixed-precision plans from a per-block sensitivity ranking and
    measure each one, producing an accuracy-vs-compression trade-off curve.

    Plan family: "protect the K most sensitive blocks". A protected block stays
    fp32 (its quantizers off); every other block is quantized at the model's
    configured `weight_bits`. K=0 is the uniform low-bit baseline; K=all is full
    fp32. Requires a calibrated model (all quantizers frozen) and the ranking
    from block_sensitivity (already sorted most-sensitive first).

    Each row's top-1 is measured by real evaluation, not predicted by summing
    per-block drops: quantization error across blocks is not additive, so the
    ranking only decides *what to protect first*, never the resulting accuracy.

    Returns one dict per K: {k, protected, top1, avg_weight_bits, compression}.
    `avg_weight_bits`/`compression` are over the quantizable weights only
    (protected -> 32 bits, quantized -> weight_bits; activations aren't stored).
    Restores the model to fully-quantizing on exit."""
    groups = _fq_groups(model)
    params = _weight_params_per_group(model)
    total = sum(params.values())
    ranked = list(sensitivity.keys())  # most-sensitive first
    if ks is None:
        ks = sorted({0, 1, 2, 3, 4, len(ranked)} & set(range(len(ranked) + 1)))
    rows: list[dict] = []
    try:
        for k in ks:
            protected = set(ranked[:k])
            for g, fqs in groups.items():
                on = g not in protected
                for m in fqs:
                    m.quantizing = on
            if log is not None:
                shown = ", ".join(sorted(protected)) or "(none)"
                log(f"K={k}: protect {shown}")
            top1 = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
            prot_params = sum(p for g, p in params.items() if g in protected)
            quant_params = total - prot_params
            avg_bits = (quant_params * weight_bits + prot_params * 32) / total
            rows.append({
                "k": k,
                "protected": sorted(protected),
                "top1": top1,
                "avg_weight_bits": avg_bits,
                "compression": 32 / avg_bits,
            })
    finally:
        set_quantizing(model, True)  # never leave the model half-quantized
    return rows


def block_sensitivity(model: nn.Module, loader: DataLoader, class_indices: list[int],
                      device: torch.device, max_batches: Optional[int] = None,
                      log: Optional[Callable[[str], None]] = None) -> dict:
    """Quantize one block at a time (rest fp32); report top-1 drop vs full fp32.
    Returns {group_name: top1_drop} sorted most-sensitive first.
    Requires a calibrated model; leaves it fully quantizing afterwards.
    If log is given, it's called with a status line before each group's pass —
    this sweep runs len(groups)+1 full evaluation passes and can take a while."""
    groups = _fq_groups(model)
    try:
        set_quantizing(model, False)
        if log is not None:
            log("baseline (all fp32)")
        base = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
        drops = {}
        for i, (key, fqs) in enumerate(groups.items(), 1):
            if log is not None:
                log(f"group {i}/{len(groups)}: {key}")
            for m in fqs:
                m.quantizing = True
            acc = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
            drops[key] = base - acc
            for m in fqs:
                m.quantizing = False
    finally:
        set_quantizing(model, True)  # never leave the model half-quantized
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))
