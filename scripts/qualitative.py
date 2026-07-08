#!/usr/bin/env python
"""Standalone qualitative comparison: run fp32 vs simulated-quantized models on
real sample images and save a contact-sheet grid annotated with both models'
predictions (flipped cases first, titled red). run_classification.py also
produces this automatically; use this script to regenerate the visualization on
its own or with a different --num-samples."""
import argparse

from vitquant.eval.qualitative import save_classification_qualitative
from vitquant.models.loader import load_model
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.data.imagenette import build_calib_loader
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    fp32_model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])
    qmodel, _ = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    qmodel = convert_vit(qmodel, qconfig_from_dict(cfg["quant"]))
    calibrate(qmodel, calib, device)

    grid = save_classification_qualitative(
        fp32_model, qmodel, data_cfg, d["root"], args.num_samples, device,
        cfg["output_dir"], cfg["model"]["name"], download=d["download"])
    print(f"Wrote {grid} (+ qualitative.json / qualitative.md alongside)")


if __name__ == "__main__":
    main()
