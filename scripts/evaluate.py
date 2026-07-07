#!/usr/bin/env python
"""Evaluate the fp32 PyTorch baseline top-1/top-5 on Imagenette."""
import argparse

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K, build_val_loader
from vitquant.eval.evaluate import evaluate_torch
from vitquant.models.loader import load_model
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"],
                           d["download"])

    device = resolve_device(cfg["device"])
    res = evaluate_torch(model, val, IMAGENETTE_TO_IMAGENET1K, device,
                         cfg["eval"]["max_batches"])
    print(f"[evaluate] fp32 torch: top1={res['top1']:.4f} top5={res['top5']:.4f}")


if __name__ == "__main__":
    main()
