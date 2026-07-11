"""Shared reporting for the classification and SAM pipelines: markdown-table
rendering, the theoretical weight-compression section, the two report bodies,
and writing the results.json + report.md pair. Centralizing this keeps the
pipeline scripts (run_classification.py / run_sam.py) as thin orchestration —
they build a results dict and call the matching report function here.

Theoretical weight compression is the arithmetic ratio from bit-width alone
(int8 weights are 8/32 = 1/4 of fp32; W4 is 4/32 = 1/8). Activations aren't
stored, so only weight bits factor in. It's device-independent and holds
regardless of the eventual NPU target."""
import json
from pathlib import Path


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def scheme_str(weight_bits: int, activation_bits: int) -> str:
    return f"W{weight_bits}A{activation_bits}"


def theoretical_compression_section(weight_bits: int) -> str:
    """The '## Theoretical Weight Compression' heading + table for a scheme."""
    ratio = 32 / weight_bits
    return ("## Theoretical Weight Compression\n\n"
            + md_table(["Weight bits", "Compression vs FP32"],
                       [[str(weight_bits), f"{ratio:.1f}x"]]))


def write_outputs(out_dir, results: dict, report: str, extra_note: str = "") -> Path:
    """Write results.json + report.md, print a summary line and the report.
    Returns the output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "report.md").write_text(report)
    print(f"\nWrote {out / 'results.json'}, {out / 'report.md'}{extra_note}\n")
    print(report)
    return out


def adaround_note(r: dict) -> str:
    """Marker lines for the advanced-PTQ passes a run used (AdaRound and/or
    SmoothQuant), so two reports are never comparable without saying so."""
    note = ""
    if r.get("smoothquant"):
        note += (f"\nActivation smoothing: **SmoothQuant** "
                 f"(alpha={r['smoothquant']['alpha']}, per-input-channel outlier "
                 "migration into weights before calibration).\n")
    if r.get("adaround"):
        note += (f"\nWeight rounding: **AdaRound** "
                 f"({r['adaround']['iters']} iters/layer, learned on the calibration "
                 "set) on top of the calibrated scales.\n")
    return note


def classification_report(r: dict) -> str:
    """Markdown report for the classification ViT pipeline: accuracy, theoretical
    compression, and (when present) per-block sensitivity, mixed-precision
    trade-off, and scheme ablation."""
    fp32_t, int8_s = r["fp32_torch"], r["int8_simulated"]
    wbits, abits = r.get("weight_bits", 8), r.get("activation_bits", 8)
    scheme = scheme_str(wbits, abits)

    acc_rows = [
        ["FP32 (PyTorch)", pct(fp32_t["top1"]), pct(fp32_t["top5"]), "-"],
        [f"{scheme} simulated (custom kernel)", pct(int8_s["top1"]), pct(int8_s["top5"]),
         pct(fp32_t["top1"] - int8_s["top1"])],
    ]
    parts = [f"# Quantization Report: {r['model']} ({scheme})",
             f"\nDevice: `{r['device']}`  ·  simulated (fake-quant) accuracy — "
             f"device-independent, no real int8 kernel run.\n"
             + adaround_note(r),
             "## Accuracy\n",
             md_table(["Variant", "Top-1", "Top-5", "Top-1 drop vs FP32"], acc_rows),
             "\n" + theoretical_compression_section(wbits)]

    if "sensitivity" in r:
        parts.append("\n## Per-Block Sensitivity (top-1 drop when only that block is quantized)\n")
        parts.append(md_table(["Block", "Top-1 drop"],
                              [[k, pct(v)] for k, v in r["sensitivity"].items()]))

    if r.get("mixed_precision"):
        parts.append(f"\n## Mixed-Precision Trade-off ({scheme} base, protected blocks kept FP32)\n")
        parts.append("Protect the K most-sensitive blocks (kept FP32); quantize the rest at "
                     f"{scheme}. Top-1 is measured, not predicted from summed per-block drops. "
                     "Compression is over quantizable weights only (protected = 32-bit).\n")
        mp_rows = []
        for row in r["mixed_precision"]:
            prot = ", ".join(row["protected"]) or "(none — uniform)"
            mp_rows.append([str(row["k"]), prot, pct(row["top1"]),
                            f"{row['avg_weight_bits']:.2f}", f"{row['compression']:.2f}x"])
        parts.append(md_table(
            ["K protected", "Blocks kept FP32", "Top-1", "Avg weight bits", "Compression vs FP32"],
            mp_rows))

    if "ablation" in r:
        parts.append("\n## Ablation (simulated INT8)\n")
        parts.append(md_table(["Config", "Top-1", "Top-5"],
                              [[k, pct(v["top1"]), pct(v["top5"])]
                               for k, v in r["ablation"].items()]))
    return "\n".join(parts) + "\n"


def sam_report(r: dict) -> str:
    """Markdown report for the SAM pipeline: self-consistency IoU (fp32 vs
    simulated-quant masks) + theoretical weight compression. Only the vision
    encoder is quantized."""
    sim = r["iou_simulated"]
    wbits, abits = r.get("weight_bits", 8), r.get("activation_bits", 8)
    scheme = scheme_str(wbits, abits)
    grid = r.get("prompt_grid", 1)
    prompts = (f"a {grid}x{grid} grid of point prompts per image (covering the whole image)"
              if grid > 1 else "a single center-point prompt per image")

    parts = [f"# SAM Quantization Report: {r['model']} ({scheme})",
             f"\nDevice: `{r['device']}`  ·  simulated (fake-quant) self-consistency "
             f"— device-independent, no real int8 kernel run.\n"
             + adaround_note(r),
             "Only `vision_encoder` is quantized; `prompt_encoder` and "
             "`mask_decoder` stay fp32. "
             f"Evaluation uses {prompts}; IoU aggregates over every "
             "(image, point, mask-hypothesis) triple.\n",
             "## Self-Consistency IoU (fp32 vs simulated-quant masks)\n",
             md_table(["Variant", "Mean IoU", "Min IoU"], [
                 [f"{scheme} simulated (custom kernel, fake-quant)",
                  f"{sim['mean_iou']:.4f}", f"{sim['min_iou']:.4f}"],
             ]),
             "\nHigh IoU means quantization barely changed the predicted masks vs "
             "the fp32 model. These are self-consistency scores (fp32 vs quantized "
             "on the same inputs), not ground-truth mIoU against a labeled benchmark.",
             "\n" + theoretical_compression_section(wbits)]

    if "sensitivity" in r:
        parts.append("\n## Per-Block Sensitivity (self-consistency IoU drop when only "
                     "that block is quantized)\n")
        parts.append("Bigger drop = quantizing that vision-encoder block changes the "
                     "masks more (drop = 1.0 − mean IoU vs full fp32).\n")
        parts.append(md_table(["Block", "IoU drop"],
                              [[k, f"{v:.4f}"] for k, v in r["sensitivity"].items()]))

    if r.get("mixed_precision"):
        parts.append(f"\n## Mixed-Precision Trade-off ({scheme} base, protected blocks kept FP32)\n")
        parts.append("Protect the K most-sensitive blocks (kept FP32); quantize the rest at "
                     f"{scheme}. Mean IoU is measured, not predicted from summed per-block "
                     "drops. Compression is over quantizable weights only (protected = 32-bit).\n")
        mp_rows = []
        for row in r["mixed_precision"]:
            prot = ", ".join(row["protected"]) or "(none — uniform)"
            mp_rows.append([str(row["k"]), prot, f"{row['iou']:.4f}",
                            f"{row['avg_weight_bits']:.2f}", f"{row['compression']:.2f}x"])
        parts.append(md_table(
            ["K protected", "Blocks kept FP32", "Mean IoU", "Avg weight bits", "Compression vs FP32"],
            mp_rows))

    parts += ["\n## Scope\n",
              "Only the vision encoder (ViT backbone) is quantized. `prompt_encoder` "
              "and `mask_decoder` remain fp32 PyTorch."]
    return "\n".join(parts) + "\n"
