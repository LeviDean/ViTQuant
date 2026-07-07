#!/usr/bin/env python
"""Quantize a SAM vision encoder and report fp32-vs-quantized mask
self-consistency (IoU) — no ground truth needed. Only the vision encoder
(ViT backbone) is quantized; prompt_encoder/mask_decoder stay fp32.

Two layers, mirroring the classification pipeline (scripts/run_all.py):
  - Research layer: custom fake-quant kernel, simulated INT8 IoU.
  - Delivery layer: real fp32 ONNX export of the vision encoder + ORT
    static INT8 (QDQ), real on-disk size, real CPU latency, and real
    (non-simulated) INT8 IoU via the ONNX graph."""
import argparse
import json
import sys
from pathlib import Path

from vitquant.data.sam_samples import build_sam_inputs
from vitquant.deploy.benchmark import benchmark_onnx, model_size_mb
from vitquant.deploy.quantize_ort import quantize_onnx
from vitquant.deploy.sam_export_onnx import export_sam_vision_encoder_onnx
from vitquant.eval.sam_evaluate import evaluate_sam_consistency, evaluate_sam_consistency_onnx
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
    sim, real = r["iou_simulated"], r["iou_real"]
    sz, lat = r["size_mb"], r["latency_ms"]
    wbits, abits = r.get("weight_bits", 8), r.get("activation_bits", 8)
    scheme = f"W{wbits}A{abits}"
    onnx_format = r.get("onnx_quant_format", "qdq").upper()

    parts = [f"# SAM Quantization Report: {r['model']} ({scheme})",
             f"\nDevice: `{r['device']}`\n",
             "Only `vision_encoder` is quantized; `prompt_encoder` and "
             "`mask_decoder` stay fp32 in both the simulated and real "
             "delivery-layer pipelines below.\n",
             "## Self-Consistency IoU (fp32 vs quantized masks)\n",
             md_table(["Variant", "Mean IoU", "Min IoU"], [
                 [f"{scheme} simulated (custom kernel, fake-quant)",
                  f"{sim['mean_iou']:.4f}", f"{sim['min_iou']:.4f}"],
                 [f"{scheme} real (ORT {onnx_format})",
                  f"{real['mean_iou']:.4f}", f"{real['min_iou']:.4f}"],
             ])]
    parts.append("\nHigh IoU means quantization barely changed the predicted "
                 "masks vs the fp32 model. These are self-consistency scores "
                 "(fp32 vs quantized on the same inputs), not ground-truth "
                 "mIoU against a labeled benchmark.")
    gap = abs(sim["mean_iou"] - real["mean_iou"])
    parts.append(f"\nSimulated vs real mean-IoU gap: {gap:.4f} — a small gap "
                 f"cross-validates that the research layer's fake-quant "
                 f"simulation is representative of the real ORT delivery "
                 f"layer.")

    parts.append(f"\n## Real Model Size (ONNX vision encoder on disk, {scheme})\n")
    parts.append(md_table(["Variant", "Size (MB)", "Compression"], [
        ["FP32", f"{sz['fp32']:.2f}", "1.0x"],
        [scheme, f"{sz['int8']:.2f}", f"{sz['fp32'] / sz['int8']:.2f}x"]]))

    parts.append(f"\n## Real Latency (ORT CPU EP, batch=1, median, {scheme})\n")
    parts.append(md_table(["Variant", "Latency (ms)", "Speedup"], [
        ["FP32", f"{lat['fp32']:.2f}", "1.0x"],
        [scheme, f"{lat['int8']:.2f}", f"{lat['fp32'] / lat['int8']:.2f}x"]]))
    opt_level = r.get("onnx_graph_optimization_level")
    if onnx_format == "QDQ" and opt_level in ("basic", "disable"):
        parts.append(
            f"\n> **Note**: `onnx.graph_optimization_level: {opt_level}` is set "
            f"(likely a workaround for an ORT crash on this CPU — see README). "
            f"At this level ORT does not fuse QDQ into a real int8 kernel: the "
            f"graph runs dequantize→fp32-compute→requantize for every "
            f"quantized op, so the {scheme} latency above reflects fp32 compute "
            f"plus quantize/dequantize overhead, not real int8 hardware speed. "
            f"Treat only the IoU and size/compression numbers above as "
            f"valid delivery-layer results on this machine.")

    parts.append("\n## Scope\n")
    parts.append("Only the vision encoder (ViT backbone) is quantized and "
                 "exported to ONNX. `prompt_encoder` and `mask_decoder` "
                 "remain fp32 PyTorch in every variant above.")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]
    base_qc = qconfig_from_dict(cfg["quant"])
    onnx_cfg = cfg.get("onnx", {})
    ort_opt = onnx_cfg.get("graph_optimization_level")
    ort_quant_format = onnx_cfg.get("quant_format", "qdq")
    bench_cfg = cfg.get("benchmark", {})
    bench_runs = bench_cfg.get("runs", 20)
    bench_warmup = bench_cfg.get("warmup", 5)

    print(f"Loading checkpoint {cfg['model']['checkpoint']} ...")
    fp32_model, processor = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    img_size = fp32_model.config.vision_config.image_size

    print(f"Building {d['calib_samples']} calibration samples ...")
    calib_samples = build_sam_inputs(d["root"], processor, d["calib_samples"],
                                     seed=0, download=d["download"])
    print(f"Building {d['eval_samples']} evaluation samples ...")
    eval_samples = build_sam_inputs(d["root"], processor, d["eval_samples"],
                                    seed=1, download=d["download"])

    print("Converting + calibrating quantized vision encoder (research layer) ...")
    quant_model, _ = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    convert_sam_vision_encoder(quant_model, base_qc)
    calibrate_sam(quant_model, calib_samples, device)

    print("Evaluating simulated self-consistency (fp32 vs fake-quant masks) ...")
    sim_result = evaluate_sam_consistency(fp32_model, quant_model, eval_samples, device)
    print(f"\nsimulated mean IoU: {sim_result['mean_iou']:.4f}")
    print(f"simulated min IoU:  {sim_result['min_iou']:.4f}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Keep the original file for back-compat with anything reading it directly.
    (out / "sam_consistency.json").write_text(json.dumps(sim_result, indent=2))
    print(f"\nWrote {out / 'sam_consistency.json'}")

    # ---- Delivery layer: real ONNX export + ORT INT8 ----
    print(f"\nExporting fp32 vision encoder to ONNX (img_size={img_size}) ...")
    fp32_onnx = export_sam_vision_encoder_onnx(
        fp32_model, out / "sam_vision_encoder.fp32.onnx", img_size=img_size)
    print(f"Wrote {fp32_onnx}")

    print("Quantizing vision encoder ONNX to real INT8 (ORT static quantization) ...")
    calib_loader = [(s["pixel_values"], None) for s in calib_samples]
    int8_onnx = quantize_onnx(fp32_onnx, out / "sam_vision_encoder.int8.onnx",
                              calib_loader, weight_bits=base_qc.weight.bits,
                              quant_format=ort_quant_format)
    print(f"Wrote {int8_onnx}")

    print("Evaluating real self-consistency (fp32 vs ORT INT8 masks) ...")
    real_result = evaluate_sam_consistency_onnx(
        fp32_model, int8_onnx, eval_samples, device,
        graph_optimization_level=ort_opt)
    print(f"\nreal mean IoU: {real_result['mean_iou']:.4f}")
    print(f"real min IoU:  {real_result['min_iou']:.4f}")

    print("Measuring real on-disk size ...")
    size_mb = {"fp32": model_size_mb(fp32_onnx), "int8": model_size_mb(int8_onnx)}
    print(f"fp32: {size_mb['fp32']:.2f} MB, int8: {size_mb['int8']:.2f} MB, "
         f"compression: {size_mb['fp32'] / size_mb['int8']:.2f}x")

    print(f"Benchmarking real CPU latency (runs={bench_runs}, warmup={bench_warmup}, "
         f"img_size={img_size}) ... SAM's full 1024x1024 input makes this much "
         f"heavier than the classification pipeline's 224x224 — can take a few "
         f"minutes on CPU.")
    latency_ms = {
        "fp32": benchmark_onnx(fp32_onnx, bench_runs, bench_warmup,
                               img_size=img_size, graph_optimization_level=ort_opt),
        "int8": benchmark_onnx(int8_onnx, bench_runs, bench_warmup,
                               img_size=img_size, graph_optimization_level=ort_opt),
    }
    print(f"fp32: {latency_ms['fp32']:.2f} ms, int8: {latency_ms['int8']:.2f} ms, "
         f"speedup: {latency_ms['fp32'] / latency_ms['int8']:.2f}x")

    results = {
        "model": cfg["model"]["name"],
        "device": str(device),
        "weight_bits": base_qc.weight.bits,
        "activation_bits": base_qc.activation.bits,
        "onnx_quant_format": ort_quant_format,
        "onnx_graph_optimization_level": ort_opt,
        "iou_simulated": sim_result,
        "iou_real": real_result,
        "size_mb": size_mb,
        "latency_ms": latency_ms,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    report = build_report(results)
    (out / "report.md").write_text(report)
    print(f"\nWrote {out / 'results.json'} and {out / 'report.md'}\n")
    print(report)


if __name__ == "__main__":
    main()
