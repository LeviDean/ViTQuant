#!/usr/bin/env python
"""SAM3 concept-segmentation (text prompt) research pipeline: simulated-quantize
the shared PE vision encoder and report fp32-vs-quantized instance-set
self-consistency — each image is prompted with its own Imagenette class name,
the two models' instance sets are greedy-matched by mask IoU, and
consistency = sum(matched IoU) / max(n_fp32, n_quant), so missed and
hallucinated instances both count against the score. Text encoder, DETR
decoder and mask head stay fp32.

The point-prompt counterpart is scripts/run_sam.py (family: sam | sam3);
reporting, progress, calibration and the advanced-PTQ switches are shared."""
import argparse
import sys
from pathlib import Path

from vitquant.data.sam3_concept_samples import build_sam3_concept_samples
from vitquant.eval.qualitative import save_sam3_concept_qualitative
from vitquant.eval.report import sam3_concept_report, write_outputs
from vitquant.eval.sam3_concept_evaluate import (block_sensitivity_sam3_concept,
                                                 evaluate_sam3_concept_consistency,
                                                 mixed_precision_sweep_sam3_concept)
from vitquant.models.sam_loader import load_sam3_concept_model
from vitquant.quant.persist import save_quantized
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.quant.sam_calibrate import adaround_sam, calibrate_sam, smooth_quant_sam
from vitquant.quant.sam_convert import convert_sam_vision_encoder
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device
from vitquant.utils.progress import calib_progress

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="skip the per-block (consistency) sensitivity sweep")
    ap.add_argument("--skip-mixed-precision", action="store_true",
                    help="skip the mixed-precision (top-K block protection) sweep")
    ap.add_argument("--mixed-precision-ks", type=str, default=None,
                    help="comma-separated K values for the sweep (default: 0,1,2,3,4 + full-fp32)")
    ap.add_argument("--no-qualitative", action="store_true",
                    help="skip the per-image instance-overlay grid")
    ap.add_argument("--qualitative-samples", type=int, default=8)
    ap.add_argument("--save-quantized", action="store_true",
                    help="save the calibrated quantization state into output_dir for "
                         "later label-free inference via scripts/infer_sam.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]
    base_qc = qconfig_from_dict(cfg["quant"])

    print(f"Loading checkpoint {cfg['model']['checkpoint']} (sam3 concept) ...")
    fp32_model, processor = load_sam3_concept_model(cfg["model"]["name"],
                                                    cfg["model"]["checkpoint"])

    print(f"Building {d['calib_samples']} calibration samples ...")
    calib_rich = build_sam3_concept_samples(d["root"], processor, d["calib_samples"],
                                            seed=0, download=d["download"])
    calib_samples = [s["inputs"] for s in calib_rich]
    print(f"Building {d['eval_samples']} evaluation samples (text prompts) ...")
    eval_samples = build_sam3_concept_samples(d["root"], processor, d["eval_samples"],
                                              seed=1, download=d["download"])

    print("Converting + calibrating quantized vision encoder ...")
    quant_model, _ = load_sam3_concept_model(cfg["model"]["name"],
                                             cfg["model"]["checkpoint"])
    convert_sam_vision_encoder(quant_model, base_qc)
    sq = cfg.get("smoothquant") or {}
    if sq.get("enabled"):
        sq_alpha = float(sq.get("alpha", 0.5))
        print(f"SmoothQuant outlier migration (alpha={sq_alpha}) ...")
        smooth_quant_sam(quant_model, calib_samples, device, alpha=sq_alpha)
    calibrate_sam(quant_model, calib_samples, device,
                  progress=calib_progress("calib", len(calib_samples)))
    ar = cfg.get("adaround") or {}
    if ar.get("enabled"):
        ar_iters = int(ar.get("iters", 1000))
        print(f"AdaRound weight-rounding refinement ({ar_iters} iters/layer) ...")
        adaround_sam(quant_model, calib_samples, device, iters=ar_iters,
                     lr=float(ar.get("lr", 1e-2)),
                     reg_weight=float(ar.get("reg_weight", 0.01)),
                     max_tokens=int(ar.get("max_tokens", 2048)),
                     log=lambda msg: print(f"    [adaround] {msg}"))

    if args.save_quantized:
        meta = {"model": cfg["model"]["name"], "family": "sam3_concept",
                "checkpoint": cfg["model"]["checkpoint"], "quant": cfg["quant"]}
        if sq.get("enabled"):
            meta["smoothquant"] = {"alpha": float(sq.get("alpha", 0.5))}
        if ar.get("enabled"):
            meta["adaround"] = {"iters": int(ar.get("iters", 1000))}
        path = save_quantized(quant_model, cfg["output_dir"], meta)
        print(f"Saved quantized state to {path}")

    print("Evaluating instance-set self-consistency (fp32 vs fake-quant) ...")
    sim_result = evaluate_sam3_concept_consistency(
        fp32_model, quant_model, processor, eval_samples, device)
    print(f"\nmean consistency: {sim_result['mean_consistency']:.4f}")
    print(f"detection F1:     {sim_result['mean_f1']:.4f}")
    print(f"matched IoU:      {sim_result['mean_matched_iou']:.4f}")
    print(f"avg instances:    fp32 {sim_result['avg_instances_fp32']:.1f} / "
         f"quant {sim_result['avg_instances_quant']:.1f}")

    out = Path(cfg["output_dir"])
    results = {
        "model": cfg["model"]["name"],
        "family": "sam3_concept",
        "device": str(device),
        "weight_bits": base_qc.weight.bits,
        "activation_bits": base_qc.activation.bits,
        "concept_consistency": sim_result,
    }
    if ar.get("enabled"):
        results["adaround"] = {"iters": int(ar.get("iters", 1000))}
    if sq.get("enabled"):
        results["smoothquant"] = {"alpha": float(sq.get("alpha", 0.5))}

    if not args.skip_sensitivity:
        print("Per-block (consistency) sensitivity sweep ...")
        results["sensitivity"] = block_sensitivity_sam3_concept(
            quant_model, fp32_model, processor, eval_samples, device,
            log=lambda msg: print(f"    [sensitivity] {msg}"))

    if not args.skip_mixed_precision and "sensitivity" in results:
        ks = ([int(x) for x in args.mixed_precision_ks.split(",")]
              if args.mixed_precision_ks else None)
        print("Mixed-precision (top-K block protection) sweep ...")
        results["mixed_precision"] = mixed_precision_sweep_sam3_concept(
            quant_model, fp32_model, processor, eval_samples, device,
            results["sensitivity"], base_qc.weight.bits, ks=ks,
            log=lambda msg: print(f"    [mixed-prec] {msg}"))

    if not args.no_qualitative:
        print(f"\nGenerating qualitative instance visualization "
             f"({args.qualitative_samples} samples) ...")
        grid_png = save_sam3_concept_qualitative(
            fp32_model, quant_model, processor, d["root"], args.qualitative_samples,
            device, out, cfg["model"]["name"], download=d["download"])
        results["qualitative_grid"] = str(grid_png)

    note = (f", and {out / 'qualitative_sam3_concept_grid.png'}"
            if not args.no_qualitative else "")
    write_outputs(out, results, sam3_concept_report(results), note)


if __name__ == "__main__":
    main()
