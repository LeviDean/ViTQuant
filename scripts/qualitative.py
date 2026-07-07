#!/usr/bin/env python
"""Qualitative comparison: run fp32 vs quantized models on real sample images
and inspect where their top-1 predictions actually differ, with confidence
scores. Aggregate accuracy is already covered by scripts/run_all.py — this is
for eyeballing specific cases quantization flips. Saves a contact-sheet image
of the actual sample photos annotated with both models' predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: servers have no display
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vitquant.data.imagenette import (IMAGENETTE_CLASS_NAMES, IMAGENETTE_TO_IMAGENET1K,
                                      build_calib_loader, build_sample_loader)
from vitquant.models.loader import load_model
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device

CLS = IMAGENETTE_TO_IMAGENET1K


def _predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> tuple[int, float]:
    """Top-1 (class index into IMAGENETTE_CLASS_NAMES, confidence) over the 10 Imagenette classes."""
    with torch.no_grad():
        logits = model(x.to(device))[:, CLS].cpu()
    probs = F.softmax(logits, dim=1)[0]
    top1 = int(probs.argmax())
    return top1, float(probs[top1])


def _denormalize(x: torch.Tensor, mean: tuple, std: tuple):
    """x: (1, 3, H, W) normalized tensor -> (H, W, 3) float array in [0, 1] for imshow."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = x[0].cpu() * std_t + mean_t
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _predict_onnx(sess, x: torch.Tensor) -> tuple[int, float]:
    out = sess.run(None, {"input": x.numpy()})[0]
    probs = F.softmax(torch.from_numpy(out)[:, CLS], dim=1)[0]
    top1 = int(probs.argmax())
    return top1, float(probs[top1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--onnx", default=None,
                    help="also compare against this real ORT-quantized ONNX model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    fp32_model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    fp32_model = fp32_model.eval().to(device)

    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])
    qmodel, _ = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    qmodel = convert_vit(qmodel, qconfig_from_dict(cfg["quant"]))
    calibrate(qmodel, calib, device)
    qmodel = qmodel.eval()

    ort_sess = None
    if args.onnx:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    samples = build_sample_loader(d["root"], data_cfg, args.num_samples)
    mean, std = data_cfg["mean"], data_cfg["std"]

    rows, images = [], []
    for i, (x, y) in enumerate(samples):
        true_idx = int(y)
        fp32_top1, fp32_conf = _predict(fp32_model, x, device)
        int8_top1, int8_conf = _predict(qmodel, x, device)
        row = {
            "index": i,
            "true": IMAGENETTE_CLASS_NAMES[true_idx],
            "fp32_pred": IMAGENETTE_CLASS_NAMES[fp32_top1],
            "fp32_conf": fp32_conf,
            "fp32_correct": fp32_top1 == true_idx,
            "int8_sim_pred": IMAGENETTE_CLASS_NAMES[int8_top1],
            "int8_sim_conf": int8_conf,
            "int8_sim_correct": int8_top1 == true_idx,
            "sim_prediction_changed": fp32_top1 != int8_top1,
        }
        if ort_sess is not None:
            ort_top1, ort_conf = _predict_onnx(ort_sess, x)
            row.update({
                "int8_real_pred": IMAGENETTE_CLASS_NAMES[ort_top1],
                "int8_real_conf": ort_conf,
                "int8_real_correct": ort_top1 == true_idx,
                "real_prediction_changed": fp32_top1 != ort_top1,
            })
        rows.append(row)
        images.append(_denormalize(x, mean, std))
        flag = " <-- prediction changed" if row["sim_prediction_changed"] else ""
        print(f"[{i:3d}] true={row['true']:<18} "
             f"fp32={row['fp32_pred']:<18}({row['fp32_conf']:.2f}) "
             f"int8={row['int8_sim_pred']:<18}({row['int8_sim_conf']:.2f}){flag}")

    n = len(rows)
    n_changed = sum(r["sim_prediction_changed"] for r in rows)
    n_fp32_correct = sum(r["fp32_correct"] for r in rows)
    n_int8_correct = sum(r["int8_sim_correct"] for r in rows)
    print(f"\n{n} samples: fp32 correct {n_fp32_correct}/{n}, "
         f"int8(sim) correct {n_int8_correct}/{n}, "
         f"predictions changed by quantization: {n_changed}/{n}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "qualitative.json").write_text(json.dumps(rows, indent=2))
    report = _build_markdown(rows, cfg["model"]["name"])
    (out / "qualitative.md").write_text(report)
    grid_path = out / "qualitative_grid.png"
    _save_grid(rows, images, cfg["model"]["name"], grid_path)
    print(f"\nWrote {out / 'qualitative.json'}, {out / 'qualitative.md'}, and {grid_path}")


def _build_markdown(rows: list[dict], model_name: str) -> str:
    has_real = bool(rows) and "int8_real_pred" in rows[0]
    headers = ["#", "True", "FP32 pred (conf)", "INT8-sim pred (conf)"]
    if has_real:
        headers.append("INT8-real pred (conf)")
    headers.append("Changed?")

    changed = [r for r in rows if r["sim_prediction_changed"]]
    unchanged = [r for r in rows if not r["sim_prediction_changed"]]

    lines = [f"# Qualitative Sample: {model_name}", "",
             f"{len(rows)} real Imagenette validation images, fp32 vs simulated INT8 "
             "(research-layer fake-quant). Rows where quantization flipped the top-1 "
             "prediction are listed first.", "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in changed + unchanged:
        cells = [str(r["index"]), r["true"],
                 f"{r['fp32_pred']} ({r['fp32_conf']:.2f})",
                 f"{r['int8_sim_pred']} ({r['int8_sim_conf']:.2f})"]
        if has_real:
            cells.append(f"{r['int8_real_pred']} ({r['int8_real_conf']:.2f})")
        cells.append("YES" if r["sim_prediction_changed"] else "")
        lines.append("| " + " | ".join(cells) + " |")

    n = len(rows)
    n_fp32_correct = sum(r["fp32_correct"] for r in rows)
    n_int8_correct = sum(r["int8_sim_correct"] for r in rows)
    lines += ["", f"**Summary**: fp32 correct {n_fp32_correct}/{n}, "
             f"int8(sim) correct {n_int8_correct}/{n}, "
             f"predictions flipped by quantization: {len(changed)}/{n}."]
    return "\n".join(lines) + "\n"


def _save_grid(rows: list[dict], images: list, model_name: str, out_path: Path,
              cols: int = 5) -> None:
    """Contact-sheet visualization: the actual sample photos, each annotated
    with true label and both models' predictions. Flipped predictions (the
    interesting cases) are sorted first and titled in red."""
    order = sorted(range(len(rows)), key=lambda i: not rows[i]["sim_prediction_changed"])
    n = len(rows)
    n_rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.6, n_rows * 3.6))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, i in zip(axes, order):
        r = rows[i]
        ax.imshow(images[i])
        ax.axis("off")
        changed = r["sim_prediction_changed"]
        title = (f"true: {r['true']}\n"
                f"fp32: {r['fp32_pred']} ({r['fp32_conf']:.2f})\n"
                f"int8: {r['int8_sim_pred']} ({r['int8_sim_conf']:.2f})")
        ax.set_title(title, fontsize=7.5, color="red" if changed else "black", pad=6)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Qualitative sample: {model_name} "
                f"(red title = quantization flipped the prediction)", fontsize=11)
    fig.subplots_adjust(hspace=0.8, wspace=0.15, top=0.92)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
