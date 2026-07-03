from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.eval.metrics import AccuracyMeter
from vitquant.quant.fake_quant import FakeQuantize, set_quantizing


@torch.no_grad()
def evaluate_torch(model: nn.Module, loader: DataLoader, class_indices: list[int],
                   device: torch.device, max_batches: Optional[int] = None) -> dict:
    """Top-1/top-5 on Imagenette: slice the 1000-class logits down to the 10
    Imagenette classes, then argmax against the 0-9 dataset labels."""
    model = model.eval().to(device)
    meter = AccuracyMeter()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(x.to(device))[:, class_indices]
        meter.update(logits.cpu(), y)
    return {"top1": meter.top1, "top5": meter.top5}


def evaluate_onnx(onnx_path: str | Path, loader: DataLoader,
                  class_indices: list[int], max_batches: Optional[int] = None) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meter = AccuracyMeter()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = sess.run(None, {"input": x.numpy()})[0]
        meter.update(torch.from_numpy(out)[:, class_indices], y)
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
                      device: torch.device, max_batches: Optional[int] = None) -> dict:
    """Quantize one block at a time (rest fp32); report top-1 drop vs full fp32.
    Returns {group_name: top1_drop} sorted most-sensitive first.
    Requires a calibrated model; leaves it fully quantizing afterwards."""
    groups = _fq_groups(model)
    try:
        set_quantizing(model, False)
        base = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
        drops = {}
        for key, fqs in groups.items():
            for m in fqs:
                m.quantizing = True
            acc = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
            drops[key] = base - acc
            for m in fqs:
                m.quantizing = False
    finally:
        set_quantizing(model, True)  # never leave the model half-quantized
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))
