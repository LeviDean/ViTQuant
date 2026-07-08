#!/usr/bin/env python
"""SAM research pipeline (one command): simulated-quantize the vision encoder
(ViT backbone) and report fp32-vs-quantized mask self-consistency (IoU) — no
ground truth needed — plus a per-image qualitative mask-overlay grid and a
markdown report. Only the vision encoder is quantized; prompt_encoder/mask_decoder
stay fp32. Simulated (fake-quant) quantization measures the accuracy impact of a
scheme; it is device-independent and does not run a real int8 kernel.

The classification counterpart is scripts/run_classification.py; reporting,
progress, and calibration helpers are shared under vitquant/."""
import argparse
import sys
from pathlib import Path

from vitquant.data.sam_samples import build_sam_inputs
from vitquant.eval.qualitative import save_sam_qualitative
from vitquant.eval.report import sam_report, write_outputs
from vitquant.eval.sam_evaluate import (block_sensitivity_sam, evaluate_sam_consistency,
                                        mixed_precision_sweep_sam)
from vitquant.models.sam_loader import load_sam_model
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.quant.sam_calibrate import calibrate_sam
from vitquant.quant.sam_convert import convert_sam_vision_encoder
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device
from vitquant.utils.progress import calib_progress

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="skip the per-block (IoU) sensitivity sweep")
    ap.add_argument("--skip-mixed-precision", action="store_true",
                    help="skip the mixed-precision (top-K block protection) sweep")
    ap.add_argument("--mixed-precision-ks", type=str, default=None,
                    help="comma-separated K values for the sweep (default: 0,1,2,3,4 + full-fp32)")
    ap.add_argument("--no-qualitative", action="store_true",
                    help="skip the per-image mask visualization grid")
    ap.add_argument("--qualitative-samples", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]
    base_qc = qconfig_from_dict(cfg["quant"])

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
    convert_sam_vision_encoder(quant_model, base_qc)
    calibrate_sam(quant_model, calib_samples, device,
                  progress=calib_progress("calib", len(calib_samples)))

    print("Evaluating simulated self-consistency (fp32 vs fake-quant masks) ...")
    sim_result = evaluate_sam_consistency(fp32_model, quant_model, eval_samples, device)
    print(f"\nsimulated mean IoU: {sim_result['mean_iou']:.4f}")
    print(f"simulated min IoU:  {sim_result['min_iou']:.4f}")

    out = Path(cfg["output_dir"])
    results = {
        "model": cfg["model"]["name"],
        "device": str(device),
        "weight_bits": base_qc.weight.bits,
        "activation_bits": base_qc.activation.bits,
        "iou_simulated": sim_result,
    }

    # Per-block sensitivity (IoU): quantize one vision-encoder block at a time,
    # measure how much the masks change vs full fp32. Restores full quant on exit.
    if not args.skip_sensitivity:
        print("Per-block (IoU) sensitivity sweep ...")
        results["sensitivity"] = block_sensitivity_sam(
            quant_model, fp32_model, eval_samples, device,
            log=lambda msg: print(f"    [sensitivity] {msg}"))

    # Mixed-precision sweep (needs the sensitivity ranking; reuses quant_model,
    # which mixed_precision_sweep_sam restores to fully-quantizing on exit)
    if not args.skip_mixed_precision and "sensitivity" in results:
        ks = ([int(x) for x in args.mixed_precision_ks.split(",")]
              if args.mixed_precision_ks else None)
        print("Mixed-precision (top-K block protection) sweep ...")
        results["mixed_precision"] = mixed_precision_sweep_sam(
            quant_model, fp32_model, eval_samples, device, results["sensitivity"],
            base_qc.weight.bits, ks=ks, log=lambda msg: print(f"    [mixed-prec] {msg}"))

    # Qualitative visualization examples (mask contour overlays)
    if not args.no_qualitative:
        print(f"\nGenerating qualitative mask visualization "
             f"({args.qualitative_samples} samples) ...")
        grid = save_sam_qualitative(
            fp32_model, quant_model, processor, d["root"], args.qualitative_samples,
            device, out, cfg["model"]["name"], download=d["download"])
        results["qualitative_grid"] = str(grid)

    note = f", and {out / 'qualitative_sam_grid.png'}" if not args.no_qualitative else ""
    write_outputs(out, results, sam_report(results), note)


if __name__ == "__main__":
    main()
