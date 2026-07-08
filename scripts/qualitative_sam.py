#!/usr/bin/env python
"""Standalone SAM qualitative comparison: overlay fp32 vs simulated-quantized
SAM mask contours over real sample images (point prompt marked), sorted
worst-IoU-first. run_sam.py also produces this automatically; use this
script to regenerate it on its own or with a different --num-samples."""
import argparse

from vitquant.data.sam_samples import build_sam_inputs
from vitquant.eval.qualitative import save_sam_qualitative
from vitquant.models.sam_loader import load_sam_model
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.quant.sam_calibrate import calibrate_sam
from vitquant.quant.sam_convert import convert_sam_vision_encoder
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    print(f"Loading checkpoint {cfg['model']['checkpoint']} ...")
    fp32_model, processor = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])

    print("Converting + calibrating quantized vision encoder ...")
    calib_samples = build_sam_inputs(d["root"], processor, d["calib_samples"], seed=0, download=d["download"])
    quant_model, _ = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    convert_sam_vision_encoder(quant_model, qconfig_from_dict(cfg["quant"]))
    calibrate_sam(quant_model, calib_samples, device)

    print(f"Building {args.num_samples} qualitative samples ...")
    grid = save_sam_qualitative(
        fp32_model, quant_model, processor, d["root"], args.num_samples, device,
        cfg["output_dir"], cfg["model"]["name"], download=d["download"])
    print(f"Wrote {grid} (+ qualitative_sam.json alongside)")


if __name__ == "__main__":
    main()
