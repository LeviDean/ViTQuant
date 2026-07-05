#!/usr/bin/env python
"""Run one simulated-quantization experiment: convert -> calibrate -> evaluate."""
import argparse
import json
from pathlib import Path

from vitquant.data.imagenette import (IMAGENETTE_TO_IMAGENET1K, build_calib_loader,
                                      build_val_loader)
from vitquant.eval.evaluate import evaluate_torch
from vitquant.models.loader import load_model
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"],
                           d["download"])
    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])

    print(f"[quantize] device={device}  model={cfg['model']['name']}")
    fp32 = evaluate_torch(model, val, IMAGENETTE_TO_IMAGENET1K, device,
                          cfg["eval"]["max_batches"])
    print(f"[quantize] fp32 top1={fp32['top1']:.4f} top5={fp32['top5']:.4f}")

    qmodel = convert_vit(model, qconfig_from_dict(cfg["quant"]))
    calibrate(qmodel, calib, device)
    int8 = evaluate_torch(qmodel, val, IMAGENETTE_TO_IMAGENET1K, device,
                          cfg["eval"]["max_batches"])
    print(f"[quantize] int8(sim) top1={int8['top1']:.4f} top5={int8['top5']:.4f}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    result = {"fp32": fp32, "int8_simulated": int8,
              "top1_drop": fp32["top1"] - int8["top1"], "qconfig": cfg["quant"]}
    (out / "quantize_result.json").write_text(json.dumps(result, indent=2))
    print(f"[quantize] wrote {out / 'quantize_result.json'}")


if __name__ == "__main__":
    main()
