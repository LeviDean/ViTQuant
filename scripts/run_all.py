#!/usr/bin/env python
"""One-command research-layer evaluation: fp32 baseline, simulated INT8
(fake-quant) accuracy, per-block sensitivity, and a quantization-scheme
ablation matrix, plus a markdown report. Simulated quantization measures the
accuracy impact of a scheme (device-independent); it does not run a real int8
kernel. The theoretical compression ratio is reported from the weight
bit-width (int8 weights are 1/4 of fp32)."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from vitquant.data.imagenette import (IMAGENETTE_TO_IMAGENET1K, build_calib_loader,
                                      build_val_loader)
from vitquant.eval.evaluate import block_sensitivity, evaluate_torch
from vitquant.eval.qualitative import save_classification_qualitative
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
    look hung on a server terminal. Matches evaluate_torch's progress(i, n)."""
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
    ap.add_argument("--no-qualitative", action="store_true",
                    help="skip the per-image visualization grid")
    ap.add_argument("--qualitative-samples", type=int, default=30)
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
    print(f"[1/5] fp32 baseline on {device} ({n_val} batches)")
    results["fp32_torch"] = evaluate_torch(model, val, CLS, device, max_b,
                                           progress=_progress("fp32", n_val))

    # 2. simulated INT8 (research layer)
    print(f"[2/5] simulated INT8 (custom kernel): calibrating ({n_calib} batches)")
    qmodel = convert_vit(model, base_qc)
    calibrate(qmodel, calib, device, progress=_calib_progress("calib", n_calib))
    print(f"[2/5] simulated INT8: evaluating ({n_val} batches)")
    results["int8_simulated"] = evaluate_torch(qmodel, val, CLS, device, max_b,
                                               progress=_progress("int8 sim", n_val))

    # 3. block sensitivity (len(groups)+1 full eval passes — can take a while)
    if not args.skip_sensitivity:
        print("[3/5] per-block sensitivity sweep")
        results["sensitivity"] = block_sensitivity(
            qmodel, val, CLS, device, max_b,
            log=lambda msg: print(f"    [sensitivity] {msg}"))
    else:
        print("[3/5] skipped")

    # 4. ablation matrix (each variant needs a fresh model: weights were shared in-place)
    if not args.skip_ablation:
        variants = ablation_qconfigs(base_qc)
        print(f"[4/5] ablation matrix ({len(variants)} variants)")
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
        print("[4/5] skipped")

    # 5. qualitative visualization examples (fresh fp32 model — convert_vit
    # mutated the original in place at step 2; qmodel is the calibrated one)
    if not args.no_qualitative:
        print(f"[5/5] qualitative visualization ({args.qualitative_samples} samples)")
        fp32_fresh, _ = load_model(name, ckpt)
        grid = save_classification_qualitative(
            fp32_fresh, qmodel, data_cfg, d["root"], args.qualitative_samples,
            device, out, name, download=d["download"])
        results["qualitative_grid"] = str(grid)
    else:
        print("[5/5] skipped")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    report = build_report(results)
    (out / "report.md").write_text(report)
    print(f"\nWrote {out / 'results.json'}, {out / 'report.md'}"
         + (f", and {out / 'qualitative_grid.png'}" if not args.no_qualitative else "") + "\n")
    print(report)


def build_report(r: dict) -> str:
    fp32_t, int8_s = r["fp32_torch"], r["int8_simulated"]
    wbits, abits = r.get("weight_bits", 8), r.get("activation_bits", 8)
    scheme = f"W{wbits}A{abits}"

    acc_rows = [
        ["FP32 (PyTorch)", pct(fp32_t["top1"]), pct(fp32_t["top5"]), "-"],
        [f"{scheme} simulated (custom kernel)", pct(int8_s["top1"]), pct(int8_s["top5"]),
         pct(fp32_t["top1"] - int8_s["top1"])],
    ]
    parts = [f"# Quantization Report: {r['model']} ({scheme})",
             f"\nDevice: `{r['device']}`  ·  simulated (fake-quant) accuracy — "
             f"device-independent, no real int8 kernel run.\n",
             "## Accuracy\n",
             md_table(["Variant", "Top-1", "Top-5", "Top-1 drop vs FP32"], acc_rows)]

    # Theoretical weight compression (arithmetic from bit-width): int8 weights
    # are 8/32 = 1/4 of fp32; W4 is 4/32 = 1/8. Activations aren't stored.
    ratio = 32 / wbits
    parts.append(f"\n## Theoretical Weight Compression\n")
    parts.append(md_table(["Scheme", "Weight bits", "Compression vs FP32"], [
        [scheme, str(wbits), f"{ratio:.1f}x"]]))

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
