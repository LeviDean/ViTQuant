#!/usr/bin/env python
"""Evaluate fp32 PyTorch baseline, or an ONNX model with --onnx."""
import argparse

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K, build_val_loader
from vitquant.eval.evaluate import evaluate_onnx, evaluate_torch
from vitquant.models.loader import load_model
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--onnx", default=None, help="evaluate this ONNX file instead")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"],
                           d["download"])

    if args.onnx:
        ort_opt = cfg.get("onnx", {}).get("graph_optimization_level")
        res = evaluate_onnx(args.onnx, val, IMAGENETTE_TO_IMAGENET1K,
                            cfg["eval"]["max_batches"],
                            graph_optimization_level=ort_opt)
        print(f"[evaluate] onnx {args.onnx}: top1={res['top1']:.4f} top5={res['top5']:.4f}")
    else:
        device = resolve_device(cfg["device"])
        res = evaluate_torch(model, val, IMAGENETTE_TO_IMAGENET1K, device,
                             cfg["eval"]["max_batches"])
        print(f"[evaluate] fp32 torch: top1={res['top1']:.4f} top5={res['top5']:.4f}")


if __name__ == "__main__":
    main()
