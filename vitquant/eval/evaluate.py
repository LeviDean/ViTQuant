from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.eval.metrics import AccuracyMeter
from vitquant.quant.fake_quant import FakeQuantize, set_quantizing
from vitquant.utils.ort_session import create_cpu_session

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


def evaluate_onnx(onnx_path: str | Path, loader: DataLoader,
                  class_indices: list[int], max_batches: Optional[int] = None,
                  progress: Optional[ProgressFn] = None,
                  graph_optimization_level: Optional[str] = None) -> dict:
    sess = create_cpu_session(onnx_path, graph_optimization_level)
    meter = AccuracyMeter()
    total = _num_batches(loader, max_batches)
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = sess.run(None, {"input": x.numpy()})[0]
        meter.update(torch.from_numpy(out)[:, class_indices], y)
        if progress is not None:
            progress(i, total)
    return {"top1": meter.top1, "top5": meter.top5}


def _fq_groups(model: nn.Module) -> dict[str, list[FakeQuantize]]:
    """Group FakeQuantize modules by top-level component; blocks are split
    per-block ("blocks.0", "blocks.1", ...)."""
    groups: dict[str, list[FakeQuantize]] = {}
    for name, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            parts = name.split(".")
            key = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
            groups.setdefault(key, []).append(m)
    return groups


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
