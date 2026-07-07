#!/usr/bin/env python
"""Quantize a SAM vision encoder and report fp32-vs-quantized mask
self-consistency (IoU) — no ground truth needed. Only the vision encoder
(ViT backbone) is quantized; prompt_encoder/mask_decoder stay fp32."""
import argparse
import json
import sys
from pathlib import Path

from vitquant.data.sam_samples import build_sam_inputs
from vitquant.eval.sam_evaluate import evaluate_sam_consistency
from vitquant.models.sam_loader import load_sam_model
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.quant.sam_calibrate import calibrate_sam
from vitquant.quant.sam_convert import convert_sam_vision_encoder
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    print(f"Loading checkpoint {cfg['model']['checkpoint']} ...")
    fp32_model, processor = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])

    print(f"Building {d['calib_samples']} calibration samples ...")
    calib_samples = build_sam_inputs(d["root"], processor, d["calib_samples"],
                                     seed=0, download=d["download"])
    print(f"Building {d['eval_samples']} evaluation samples ...")
    eval_samples = build_sam_inputs(d["root"], processor, d["eval_samples"],
                                    seed=1, download=d["download"])

    print("Converting + calibrating quantized vision encoder ...")
    quant_model, _ = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    convert_sam_vision_encoder(quant_model, qconfig_from_dict(cfg["quant"]))
    calibrate_sam(quant_model, calib_samples, device)

    print("Evaluating self-consistency (fp32 vs quantized masks) ...")
    result = evaluate_sam_consistency(fp32_model, quant_model, eval_samples, device)

    print(f"\nmean IoU: {result['mean_iou']:.4f}")
    print(f"min IoU:  {result['min_iou']:.4f}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "sam_consistency.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out / 'sam_consistency.json'}")


if __name__ == "__main__":
    main()
