from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.eval.metrics import AccuracyMeter
from vitquant.quant.fake_quant import FakeQuantize, set_quantizing
from vitquant.quant.groups import group_key as _group_key

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
    the average-bit-width of a mixed-precision plan by real parameter count.
    Keys off the weight_fq's name (not the owning module's) so the group keys
    match _fq_groups exactly — _group_key expects an fq-suffixed name, and using
    the bare module name would mis-group deeply-nested SAM modules."""
    counts: dict[str, int] = {}
    for name, m in model.named_modules():
        wfq = getattr(m, "weight_fq", None)
        weight = getattr(m, "weight", None)
        if isinstance(wfq, FakeQuantize) and isinstance(weight, torch.Tensor):
            key = _group_key(f"{name}.weight_fq")
            counts[key] = counts.get(key, 0) + weight.numel()
    return counts


def mixed_precision_scored(
    model: nn.Module, sensitivity: dict[str, float], weight_bits: int,
    measure: Callable[[], float], score_key: str = "score",
    ks: Optional[list[int]] = None, log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Task-agnostic mixed-precision sweep core. Build "protect the K most
    sensitive blocks" plans from a sensitivity ranking and measure each, giving
    a score-vs-compression trade-off curve. A protected block stays fp32 (its
    quantizers off); every other block is quantized at `weight_bits`. K=0 is the
    uniform low-bit baseline; K=all is full fp32.

    `measure()` returns a scalar score (top-1, IoU) for the model in its current
    quantizing state — the sweep only toggles which blocks are protected, so
    classification and SAM share this one implementation. Each score is MEASURED,
    not predicted by summing per-block drops (cross-block quant error is not
    additive), so the ranking only decides *what to protect first*.

    Returns one dict per K: {k, protected, <score_key>, avg_weight_bits,
    compression}. avg_weight_bits/compression are over quantizable weights only
    (protected -> 32 bits, quantized -> weight_bits; activations aren't stored).
    Requires a calibrated model; restores full quantization on exit."""
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
            score = measure()
            prot_params = sum(p for g, p in params.items() if g in protected)
            quant_params = total - prot_params
            avg_bits = (quant_params * weight_bits + prot_params * 32) / total
            rows.append({
                "k": k,
                "protected": sorted(protected),
                score_key: score,
                "avg_weight_bits": avg_bits,
                "compression": 32 / avg_bits,
            })
    finally:
        set_quantizing(model, True)  # never leave the model half-quantized
    return rows


def mixed_precision_sweep(
    model: nn.Module, sensitivity: dict[str, float], loader: DataLoader,
    class_indices: list[int], device: torch.device, weight_bits: int,
    max_batches: Optional[int] = None, ks: Optional[list[int]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Classification mixed-precision sweep (top-1). Thin wrapper over
    mixed_precision_scored; rows carry a "top1" score field."""
    def measure() -> float:
        return evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
    return mixed_precision_scored(model, sensitivity, weight_bits, measure,
                                  score_key="top1", ks=ks, log=log)


def block_sensitivity_scored(model: nn.Module, measure: Callable[[], float],
                             log: Optional[Callable[[str], None]] = None) -> dict:
    """Task-agnostic per-block sensitivity core. Quantize one block at a time
    (rest fp32) and report each block's score drop vs the all-fp32 baseline.

    `measure()` returns a scalar score (higher is better) for the model in its
    current quantizing state — top-1 accuracy for classification, self-
    consistency IoU for SAM. The sweep only toggles which block is quantized;
    the metric is entirely `measure`'s concern, so classification and SAM share
    this one implementation.

    Returns {group_name: score_drop} sorted most-sensitive first. Requires a
    calibrated model; restores full quantization on exit. If log is given it's
    called before each group's pass — the sweep runs len(groups)+1 measurement
    passes and can take a while."""
    groups = _fq_groups(model)
    try:
        set_quantizing(model, False)
        if log is not None:
            log("baseline (all fp32)")
        base = measure()
        drops = {}
        for i, (key, fqs) in enumerate(groups.items(), 1):
            if log is not None:
                log(f"group {i}/{len(groups)}: {key}")
            for m in fqs:
                m.quantizing = True
            drops[key] = base - measure()
            for m in fqs:
                m.quantizing = False
    finally:
        set_quantizing(model, True)  # never leave the model half-quantized
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))


def block_sensitivity(model: nn.Module, loader: DataLoader, class_indices: list[int],
                      device: torch.device, max_batches: Optional[int] = None,
                      log: Optional[Callable[[str], None]] = None) -> dict:
    """Classification per-block sensitivity: top-1 drop when only that block is
    quantized (rest fp32), sorted most-sensitive first. Thin wrapper over
    block_sensitivity_scored with a top-1 measure."""
    def measure() -> float:
        return evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
    return block_sensitivity_scored(model, measure, log)
