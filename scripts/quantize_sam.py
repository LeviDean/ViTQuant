#!/usr/bin/env python
"""Quantize a SAM vision encoder (research layer) and report fp32-vs-quantized
mask self-consistency (IoU) — no ground truth needed. Only the vision encoder
(ViT backbone) is quantized; prompt_encoder/mask_decoder stay fp32. Simulated
(fake-quant) quantization measures the accuracy impact of a scheme; it is
device-independent and does not run a real int8 kernel."""
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


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_report(r: dict) -> str:
    sim = r["iou_simulated"]
    wbits, abits = r.get("weight_bits", 8), r.get("activation_bits", 8)
    scheme = f"W{wbits}A{abits}"
    ratio = 32 / wbits  # theoretical weight compression: int8 = 1/4 fp32

    parts = [f"# SAM Quantization Report: {r['model']} ({scheme})",
             f"\nDevice: `{r['device']}`  ·  simulated (fake-quant) self-consistency "
             f"— device-independent, no real int8 kernel run.\n",
             "Only `vision_encoder` is quantized; `prompt_encoder` and "
             "`mask_decoder` stay fp32.\n",
             "## Self-Consistency IoU (fp32 vs simulated-quant masks)\n",
             md_table(["Variant", "Mean IoU", "Min IoU"], [
                 [f"{scheme} simulated (custom kernel, fake-quant)",
                  f"{sim['mean_iou']:.4f}", f"{sim['min_iou']:.4f}"],
             ])]
    parts.append("\nHigh IoU means quantization barely changed the predicted "
                 "masks vs the fp32 model. These are self-consistency scores "
                 "(fp32 vs quantized on the same inputs), not ground-truth "
                 "mIoU against a labeled benchmark.")

    parts.append(f"\n## Theoretical Weight Compression\n")
    parts.append(md_table(["Scheme", "Weight bits", "Compression vs FP32"], [
        [scheme, str(wbits), f"{ratio:.1f}x"]]))

    parts.append("\n## Scope\n")
    parts.append("Only the vision encoder (ViT backbone) is quantized. "
                 "`prompt_encoder` and `mask_decoder` remain fp32 PyTorch.")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
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
    calibrate_sam(quant_model, calib_samples, device)

    print("Evaluating simulated self-consistency (fp32 vs fake-quant masks) ...")
    sim_result = evaluate_sam_consistency(fp32_model, quant_model, eval_samples, device)
    print(f"\nsimulated mean IoU: {sim_result['mean_iou']:.4f}")
    print(f"simulated min IoU:  {sim_result['min_iou']:.4f}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "model": cfg["model"]["name"],
        "device": str(device),
        "weight_bits": base_qc.weight.bits,
        "activation_bits": base_qc.activation.bits,
        "iou_simulated": sim_result,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    report = build_report(results)
    (out / "report.md").write_text(report)
    print(f"\nWrote {out / 'results.json'} and {out / 'report.md'}\n")
    print(report)


if __name__ == "__main__":
    main()
