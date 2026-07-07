#!/usr/bin/env python
"""One-command full evaluation: fp32 baseline, simulated INT8, block sensitivity,
ablation matrix, ORT real INT8 (accuracy + size + latency), markdown report."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from vitquant.data.imagenette import (IMAGENETTE_TO_IMAGENET1K, build_calib_loader,
                                      build_val_loader)
from vitquant.deploy.benchmark import benchmark_onnx, model_size_mb
from vitquant.deploy.export_onnx import export_fp32_onnx
from vitquant.deploy.quantize_ort import quantize_onnx
from vitquant.eval.evaluate import block_sensitivity, evaluate_onnx, evaluate_torch
from vitquant.models.loader import load_model
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import QConfig, qconfig_from_dict
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device

# stdout is fully buffered (not line-buffered) unless attached to a real tty —
# common under nohup/log redirection/some containers — so progress prints can
# sit invisible until the process exits. Force line buffering unconditionally.
sys.stdout.reconfigure(line_buffering=True)

CLS = IMAGENETTE_TO_IMAGENET1K
ACC_GAP_WARN = 0.01  # spec: flag if |simulated - ORT real| top-1 gap > 1%


def ablation_qconfigs(base: QConfig) -> dict[str, QConfig]:
    return {
        "default (w:per-ch sym / a:per-t asym ema)": base,
        "weight per-tensor": replace(base, weight=replace(base.weight, per_channel=False)),
        "activation symmetric": replace(base, activation=replace(base.activation, symmetric=True)),
        "activation minmax obs": replace(base, activation=replace(base.activation, observer="minmax")),
        "activation percentile obs": replace(base, activation=replace(base.activation, observer="percentile")),
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _progress(label: str, total: int) -> callable:
    """Print a status line every ~20% of batches, so long eval passes don't
    look hung on a server terminal. Matches evaluate_torch/onnx's progress(i, n)."""
    step = max(1, total // 5)

    def cb(i: int, n: int) -> None:
        if (i + 1) % step == 0 or i + 1 == n:
            print(f"    {label}: batch {i + 1}/{n}")
    return cb


def _calib_progress(label: str, total: int) -> callable:
    """Same as _progress but matches calibrate()'s progress(i)-only signature."""
    inner = _progress(label, total)
    return lambda i: inner(i, total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d, name = cfg["data"], cfg["model"]["name"]
    ckpt = cfg["model"]["checkpoint"]
    max_b = cfg["eval"]["max_batches"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    base_qc = qconfig_from_dict(cfg["quant"])
    results: dict = {"model": name, "device": str(device),
                     "weight_bits": base_qc.weight.bits,
                     "activation_bits": base_qc.activation.bits}

    print(f"Loading checkpoint {ckpt} ...")
    model, data_cfg = load_model(name, ckpt)
    print(f"Preparing data under {d['root']} "
         f"(downloading Imagenette if missing — can take a few minutes) ...")
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"], d["download"])
    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])
    n_val = len(val) if max_b is None else min(max_b, len(val))
    n_calib = len(calib)  # calibrate() has no max_batches limit in this pipeline

    # 1. fp32 torch baseline
    print(f"[1/6] fp32 baseline on {device} ({n_val} batches)")
    results["fp32_torch"] = evaluate_torch(model, val, CLS, device, max_b,
                                           progress=_progress("fp32", n_val))

    # 2. simulated INT8 (research layer)
    print(f"[2/6] simulated INT8 (custom kernel): calibrating ({n_calib} batches)")
    qmodel = convert_vit(model, base_qc)
    calibrate(qmodel, calib, device, progress=_calib_progress("calib", n_calib))
    print(f"[2/6] simulated INT8: evaluating ({n_val} batches)")
    results["int8_simulated"] = evaluate_torch(qmodel, val, CLS, device, max_b,
                                               progress=_progress("int8 sim", n_val))

    # 3. block sensitivity (len(groups)+1 full eval passes — can take a while)
    if not args.skip_sensitivity:
        print("[3/6] per-block sensitivity sweep")
        results["sensitivity"] = block_sensitivity(
            qmodel, val, CLS, device, max_b,
            log=lambda msg: print(f"    [sensitivity] {msg}"))
    else:
        print("[3/6] skipped")

    # 4. ablation matrix (each variant needs a fresh model: weights were shared in-place)
    if not args.skip_ablation:
        variants = ablation_qconfigs(base_qc)
        print(f"[4/6] ablation matrix ({len(variants)} variants)")
        results["ablation"] = {}
        for i, (label, qc) in enumerate(variants.items(), 1):
            print(f"    variant {i}/{len(variants)}: {label}")
            m, _ = load_model(name, ckpt)
            qm = convert_vit(m, qc)
            calibrate(qm, calib, device)
            results["ablation"][label] = evaluate_torch(
                qm, val, CLS, device, max_b, progress=_progress(f"variant {i}", n_val))
            print(f"    {label}: top1={results['ablation'][label]['top1']:.4f}")
    else:
        print("[4/6] skipped")

    # 5. delivery layer: ONNX export + ORT real INT8
    print("[5/6] ONNX export + ORT static INT8 (export/quantize give no progress "
         "signal — ORT internal, may take a few minutes on larger models)")
    fp32_model, _ = load_model(name, ckpt)  # fresh fp32 weights for export
    fp32_onnx = export_fp32_onnx(fp32_model, out / f"{name}.fp32.onnx")
    int8_onnx = quantize_onnx(fp32_onnx, out / f"{name}.int8.onnx", calib,
                              weight_bits=base_qc.weight.bits)
    results["fp32_onnx"] = evaluate_onnx(fp32_onnx, val, CLS, max_b,
                                         progress=_progress("fp32 onnx", n_val))
    results["int8_onnx"] = evaluate_onnx(int8_onnx, val, CLS, max_b,
                                         progress=_progress("int8 onnx", n_val))
    results["size_mb"] = {"fp32": model_size_mb(fp32_onnx), "int8": model_size_mb(int8_onnx)}

    # 6. latency benchmark (CPU EP — the representative int8 speedup metric)
    print("[6/6] latency benchmark (ORT CPU EP)")
    b = cfg["benchmark"]
    results["latency_ms"] = {
        "fp32": benchmark_onnx(fp32_onnx, b["runs"], b["warmup"]),
        "int8": benchmark_onnx(int8_onnx, b["runs"], b["warmup"]),
    }

    (out / "results.json").write_text(json.dumps(results, indent=2))
    report = build_report(results, args)
    (out / "report.md").write_text(report)
    print(f"\nWrote {out / 'results.json'} and {out / 'report.md'}\n")
    print(report)


def build_report(r: dict, args) -> str:
    fp32_t, int8_s = r["fp32_torch"], r["int8_simulated"]
    fp32_o, int8_o = r["fp32_onnx"], r["int8_onnx"]
    sz, lat = r["size_mb"], r["latency_ms"]
    wbits, abits = r.get("weight_bits", 8), r.get("activation_bits", 8)
    scheme = f"W{wbits}A{abits}"

    acc_rows = [
        ["FP32 (PyTorch)", pct(fp32_t["top1"]), pct(fp32_t["top5"]), "-"],
        ["FP32 (ONNX)", pct(fp32_o["top1"]), pct(fp32_o["top5"]),
         pct(fp32_t["top1"] - fp32_o["top1"])],
        [f"{scheme} simulated (custom kernel)", pct(int8_s["top1"]), pct(int8_s["top5"]),
         pct(fp32_t["top1"] - int8_s["top1"])],
        [f"{scheme} real (ORT QDQ)", pct(int8_o["top1"]), pct(int8_o["top5"]),
         pct(fp32_t["top1"] - int8_o["top1"])],
    ]
    parts = [f"# Quantization Report: {r['model']} ({scheme})",
             f"\nDevice (torch eval): `{r['device']}`\n",
             "## Accuracy\n",
             md_table(["Variant", "Top-1", "Top-5", "Top-1 drop vs FP32"], acc_rows)]

    gap = abs(int8_s["top1"] - int8_o["top1"])
    if gap > ACC_GAP_WARN:
        parts.append(f"\n> **WARNING**: simulated vs ORT-real top-1 gap {pct(gap)} "
                     f"exceeds {pct(ACC_GAP_WARN)} — investigate qscheme mismatch.")
    else:
        parts.append(f"\nSimulated vs ORT-real top-1 gap: {pct(gap)} (cross-validation OK)")

    parts.append(f"\n## Real Model Size (ONNX on disk, {scheme})\n")
    parts.append(md_table(["Variant", "Size (MB)", "Compression"], [
        ["FP32", f"{sz['fp32']:.1f}", "1.0x"],
        [scheme, f"{sz['int8']:.1f}", f"{sz['fp32'] / sz['int8']:.2f}x"]]))

    parts.append(f"\n## Real Latency (ORT CPU EP, batch=1, median, {scheme})\n")
    parts.append(md_table(["Variant", "Latency (ms)", "Speedup"], [
        ["FP32", f"{lat['fp32']:.2f}", "1.0x"],
        [scheme, f"{lat['int8']:.2f}", f"{lat['fp32'] / lat['int8']:.2f}x"]]))
    if wbits < 8:
        parts.append(f"\n> Note: ORT's CPU EP has no native INT{wbits} matmul kernel — "
                     f"the {scheme} latency above reflects the QDQ-graph overhead of "
                     f"dequantizing on the fly, not a real hardware speedup. Treat the "
                     f"size/compression numbers as the meaningful {scheme} result.")

    if "sensitivity" in r:
        parts.append("\n## Per-Block Sensitivity (top-1 drop when only that block is quantized)\n")
        parts.append(md_table(["Block", "Top-1 drop"],
                              [[k, pct(v)] for k, v in r["sensitivity"].items()]))

    if "ablation" in r:
        parts.append("\n## Ablation (simulated INT8)\n")
        parts.append(md_table(["Config", "Top-1", "Top-5"],
                              [[k, pct(v["top1"]), pct(v["top5"])]
                               for k, v in r["ablation"].items()]))
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
