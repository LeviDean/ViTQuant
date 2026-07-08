"""Qualitative visualization for quantization comparison — shared by the
standalone scripts (scripts/qualitative*.py) and the main evaluation scripts
(run_classification.py / run_sam.py), so a comparison run always produces both the
metric report AND visual examples from one code path."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: servers have no display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vitquant.data.imagenette import (IMAGENETTE_CLASS_NAMES, IMAGENETTE_TO_IMAGENET1K,
                                      build_sample_loader)
from vitquant.data.sam_samples import build_sam_qualitative_samples
from vitquant.eval.sam_evaluate import mask_iou

CLS = IMAGENETTE_TO_IMAGENET1K
SAM_BAD_IOU_THRESHOLD = 0.9  # SAM IoU below this: title in red as "worth a look"


# ----------------------------- classification -----------------------------

def _predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> tuple[int, float]:
    """Top-1 (index into IMAGENETTE_CLASS_NAMES, confidence) over the 10 classes."""
    with torch.no_grad():
        logits = model(x.to(device))[:, CLS].cpu()
    probs = F.softmax(logits, dim=1)[0]
    top1 = int(probs.argmax())
    return top1, float(probs[top1])


def _denormalize(x: torch.Tensor, mean: tuple, std: tuple):
    """x: (1, 3, H, W) normalized -> (H, W, 3) float array in [0, 1] for imshow."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = x[0].cpu() * std_t + mean_t
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _cls_markdown(rows: list[dict], model_name: str) -> str:
    headers = ["#", "True", "FP32 pred (conf)", "INT8-sim pred (conf)", "Changed?"]
    changed = [r for r in rows if r["sim_prediction_changed"]]
    unchanged = [r for r in rows if not r["sim_prediction_changed"]]
    lines = [f"# Qualitative Sample: {model_name}", "",
             f"{len(rows)} real Imagenette validation images, fp32 vs simulated INT8 "
             "(research-layer fake-quant). Rows where quantization flipped the top-1 "
             "prediction are listed first.", "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in changed + unchanged:
        lines.append("| " + " | ".join([
            str(r["index"]), r["true"],
            f"{r['fp32_pred']} ({r['fp32_conf']:.2f})",
            f"{r['int8_sim_pred']} ({r['int8_sim_conf']:.2f})",
            "YES" if r["sim_prediction_changed"] else ""]) + " |")
    n = len(rows)
    lines += ["", f"**Summary**: fp32 correct {sum(r['fp32_correct'] for r in rows)}/{n}, "
             f"int8(sim) correct {sum(r['int8_sim_correct'] for r in rows)}/{n}, "
             f"predictions flipped by quantization: {len(changed)}/{n}."]
    return "\n".join(lines) + "\n"


def _cls_grid(rows: list[dict], images: list, model_name: str, out_path: Path,
              cols: int = 5) -> None:
    order = sorted(range(len(rows)), key=lambda i: not rows[i]["sim_prediction_changed"])
    n = len(rows)
    n_rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.6, n_rows * 3.6), squeeze=False)
    axes = axes.flatten()
    for ax, i in zip(axes, order):
        r = rows[i]
        ax.imshow(images[i])
        ax.axis("off")
        title = (f"true: {r['true']}\n"
                f"fp32: {r['fp32_pred']} ({r['fp32_conf']:.2f})\n"
                f"int8: {r['int8_sim_pred']} ({r['int8_sim_conf']:.2f})")
        ax.set_title(title, fontsize=7.5,
                    color="red" if r["sim_prediction_changed"] else "black", pad=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Qualitative sample: {model_name} "
                f"(red title = quantization flipped the prediction)", fontsize=11)
    fig.subplots_adjust(hspace=0.8, wspace=0.15, top=0.92)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_classification_qualitative(fp32_model, qmodel, data_cfg: dict, root,
                                    num_samples: int, device: torch.device,
                                    out_dir: Path, model_name: str,
                                    download: bool = True) -> Path:
    """Run fp32 and simulated-quant models on real sample images, write
    qualitative.json/.md and a contact-sheet qualitative_grid.png (flipped
    predictions sorted first, titled red). Returns the grid image path."""
    fp32_model = fp32_model.eval().to(device)
    qmodel = qmodel.eval().to(device)
    samples = build_sample_loader(root, data_cfg, num_samples)
    mean, std = data_cfg["mean"], data_cfg["std"]

    rows, images = [], []
    for i, (x, y) in enumerate(samples):
        true_idx = int(y)
        fp32_top1, fp32_conf = _predict(fp32_model, x, device)
        int8_top1, int8_conf = _predict(qmodel, x, device)
        rows.append({
            "index": i, "true": IMAGENETTE_CLASS_NAMES[true_idx],
            "fp32_pred": IMAGENETTE_CLASS_NAMES[fp32_top1], "fp32_conf": fp32_conf,
            "fp32_correct": fp32_top1 == true_idx,
            "int8_sim_pred": IMAGENETTE_CLASS_NAMES[int8_top1], "int8_sim_conf": int8_conf,
            "int8_sim_correct": int8_top1 == true_idx,
            "sim_prediction_changed": fp32_top1 != int8_top1,
        })
        images.append(_denormalize(x, mean, std))

    n = len(rows)
    n_changed = sum(r["sim_prediction_changed"] for r in rows)
    print(f"    qualitative: {n} samples, "
         f"fp32 correct {sum(r['fp32_correct'] for r in rows)}/{n}, "
         f"int8(sim) correct {sum(r['int8_sim_correct'] for r in rows)}/{n}, "
         f"predictions changed by quantization: {n_changed}/{n}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualitative.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "qualitative.md").write_text(_cls_markdown(rows, model_name))
    grid_path = out_dir / "qualitative_grid.png"
    _cls_grid(rows, images, model_name, grid_path)
    return grid_path


# --------------------------------- SAM ------------------------------------

def _sam_grid(results: list[dict], model_name: str, out_path: Path, cols: int = 4) -> None:
    n = len(results)
    cols = min(cols, n) if n > 0 else 1
    n_rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.2, n_rows * 3.6), squeeze=False)
    axes = axes.flatten()
    for ax, r in zip(axes, results):
        ax.imshow(r["image"])
        # fp32 mask: semi-transparent filled overlay (the "before" reference)
        fp32 = r["fp32_mask"].cpu().numpy().astype(bool)
        overlay = np.zeros((*fp32.shape, 4), dtype=float)
        overlay[fp32] = (0.20, 0.95, 0.20, 0.35)  # translucent lime fill
        ax.imshow(overlay)
        # quantized mask: thin boundary line (the "after", easy to compare against fill)
        ax.contour(r["quant_mask"].cpu().numpy().astype(float), levels=[0.5],
                  colors="magenta", linewidths=1.0)
        px, py = r["point"]
        ax.scatter([px], [py], marker="*", s=200, c="yellow", edgecolors="black", linewidths=1)
        ax.axis("off")
        idx_note = "" if r["index_agrees"] else f" [quant picks #{r['quant_best_idx']}]"
        ax.set_title(f"IoU={r['iou']:.3f} (mask #{r['mask_idx']}){idx_note}", fontsize=9,
                    color="red" if r["iou"] < SAM_BAD_IOU_THRESHOLD else "black")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"SAM qualitative: {model_name}\n"
                f"lime translucent fill = fp32 mask, magenta line = quantized boundary, "
                f"star = point prompt, sorted worst-IoU-first", fontsize=10, y=0.99)
    # Fixed inch-based top margin so the two-line suptitle never collides with
    # row-0 subplot titles regardless of grid size.
    fig.subplots_adjust(hspace=0.5, wspace=0.15, top=1 - 0.85 / (n_rows * 3.6))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_sam_qualitative(fp32_model, quant_model, processor, root, num_samples: int,
                         device: torch.device, out_dir: Path, model_name: str,
                         seed: int = 2, download: bool = True) -> Path:
    """For real sample images + a point prompt, overlay fp32 vs quantized SAM
    masks (the hypothesis fp32's own iou_scores ranks best, used for both so the
    comparison is apples-to-apples), sorted worst-IoU-first. Writes
    qualitative_sam.json and qualitative_sam_grid.png. Returns the grid path.
    seed defaults to 2 (different from calibration's seed=0) so visualization
    uses images not seen during calibration."""
    fp32_model = fp32_model.eval().to(device)
    quant_model = quant_model.eval().to(device)
    samples = build_sam_qualitative_samples(root, processor, num_samples,
                                            seed=seed, download=download)

    results = []
    for s in samples:
        inputs = {k: v.to(device) for k, v in s["inputs"].items()}
        with torch.no_grad():
            fp32_out = fp32_model(**inputs)
            quant_out = quant_model(**inputs)
        fp32_best_idx = int(fp32_out.iou_scores[0, 0].argmax())
        quant_best_idx = int(quant_out.iou_scores[0, 0].argmax())
        fp32_full = processor.image_processor.post_process_masks(
            fp32_out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        quant_full = processor.image_processor.post_process_masks(
            quant_out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        fp32_mask = fp32_full[0, fp32_best_idx]
        quant_mask = quant_full[0, fp32_best_idx]
        results.append({
            "image": s["image"], "point": s["point"], "mask_idx": fp32_best_idx,
            "quant_best_idx": quant_best_idx, "index_agrees": quant_best_idx == fp32_best_idx,
            "iou": mask_iou(fp32_mask, quant_mask),
            "fp32_mask": fp32_mask, "quant_mask": quant_mask,
        })
    results.sort(key=lambda r: r["iou"])  # worst first

    ious = [r["iou"] for r in results]
    print(f"    qualitative: {len(results)} samples, "
         f"mask IoU mean {sum(ious) / len(ious):.4f}, min {min(ious):.4f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "qualitative_sam_grid.png"
    _sam_grid(results, model_name, grid_path)
    summary = [{"point": r["point"], "mask_idx": r["mask_idx"], "iou": r["iou"],
               "quant_best_idx": r["quant_best_idx"], "index_agrees": r["index_agrees"]}
              for r in results]
    (out_dir / "qualitative_sam.json").write_text(json.dumps(summary, indent=2))
    return grid_path
