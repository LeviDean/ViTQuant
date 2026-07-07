#!/usr/bin/env python
"""Qualitative comparison: for real sample images, visualize the fp32 vs
quantized SAM vision encoder's predicted mask (the one SAM's own iou_scores
ranks best), overlaid on the actual image with the point prompt marked.
Samples are sorted worst-agreement-first (lowest mask IoU) so the most
interesting cases are easy to spot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: servers have no display
import matplotlib.pyplot as plt
import torch

from vitquant.data.sam_samples import build_sam_inputs, build_sam_qualitative_samples
from vitquant.eval.sam_evaluate import mask_iou
from vitquant.models.sam_loader import load_sam_model
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.quant.sam_calibrate import calibrate_sam
from vitquant.quant.sam_convert import convert_sam_vision_encoder
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device

BAD_IOU_THRESHOLD = 0.9  # below this, title in red as "worth a second look"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    print(f"Loading checkpoint {cfg['model']['checkpoint']} ...")
    fp32_model, processor = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    fp32_model = fp32_model.eval().to(device)

    print("Converting + calibrating quantized vision encoder ...")
    calib_samples = build_sam_inputs(d["root"], processor, d["calib_samples"], seed=0, download=d["download"])
    quant_model, _ = load_sam_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    convert_sam_vision_encoder(quant_model, qconfig_from_dict(cfg["quant"]))
    calibrate_sam(quant_model, calib_samples, device)
    quant_model = quant_model.eval().to(device)

    print(f"Building {args.num_samples} qualitative samples ...")
    # seed differs from calib_samples' seed=0 above so visualization uses images
    # not seen during calibration
    samples = build_sam_qualitative_samples(d["root"], processor, args.num_samples, seed=2, download=d["download"])

    results = []
    for s in samples:
        inputs = {k: v.to(device) for k, v in s["inputs"].items()}
        with torch.no_grad():
            fp32_out = fp32_model(**inputs)
            quant_out = quant_model(**inputs)

        fp32_best_idx = int(fp32_out.iou_scores[0, 0].argmax())
        # quantization could in principle change which mask index the model
        # itself would pick; the contour/IoU below always uses fp32's index
        # (the "ground truth" choice) for both models, so that disagreement
        # is invisible there — surface it separately here instead of hiding it.
        quant_best_idx = int(quant_out.iou_scores[0, 0].argmax())
        index_agrees = quant_best_idx == fp32_best_idx

        fp32_full = processor.image_processor.post_process_masks(
            fp32_out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        quant_full = processor.image_processor.post_process_masks(
            quant_out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        # use fp32's chosen index for BOTH, so the comparison is apples-to-apples
        # (the mask hypothesis SAM would actually pick in real usage)
        fp32_mask = fp32_full[0, fp32_best_idx]
        quant_mask = quant_full[0, fp32_best_idx]
        iou = mask_iou(fp32_mask, quant_mask)

        results.append({
            "image": s["image"], "point": s["point"], "mask_idx": fp32_best_idx,
            "quant_best_idx": quant_best_idx, "index_agrees": index_agrees,
            "iou": iou, "fp32_mask": fp32_mask, "quant_mask": quant_mask,
        })
        flag = "" if index_agrees else f"  <-- quantization would pick mask #{quant_best_idx} instead"
        print(f"  IoU={iou:.4f}  point={s['point']}  mask_idx={fp32_best_idx}{flag}")

    results.sort(key=lambda r: r["iou"])  # worst first

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / "qualitative_sam_grid.png"
    _save_grid(results, cfg["model"]["name"], grid_path)

    summary = [{"point": r["point"], "mask_idx": r["mask_idx"], "iou": r["iou"],
               "quant_best_idx": r["quant_best_idx"], "index_agrees": r["index_agrees"]}
              for r in results]
    (out / "qualitative_sam.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {grid_path} and {out / 'qualitative_sam.json'}")


def _save_grid(results: list[dict], model_name: str, out_path: Path, cols: int = 4) -> None:
    n = len(results)
    cols = min(cols, n) if n > 0 else 1
    n_rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.2, n_rows * 3.6), squeeze=False)
    axes = axes.flatten()

    for ax, r in zip(axes, results):
        ax.imshow(r["image"])
        # fp32 contour: lime; quantized contour: magenta (both distinguishable from
        # the red used for a low-IoU title, and from each other)
        ax.contour(r["fp32_mask"].cpu().numpy().astype(float), levels=[0.5], colors="lime", linewidths=2)
        ax.contour(r["quant_mask"].cpu().numpy().astype(float), levels=[0.5], colors="magenta", linewidths=2, linestyles="dashed")
        px, py = r["point"]
        ax.scatter([px], [py], marker="*", s=200, c="yellow", edgecolors="black", linewidths=1)
        ax.axis("off")
        bad = r["iou"] < BAD_IOU_THRESHOLD
        idx_note = "" if r["index_agrees"] else f" [quant picks #{r['quant_best_idx']}]"
        ax.set_title(f"IoU={r['iou']:.3f} (mask #{r['mask_idx']}){idx_note}", fontsize=9,
                    color="red" if bad else "black")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"SAM qualitative: {model_name}\n"
                f"lime = fp32 mask, magenta dashed = quantized mask, "
                f"star = point prompt, sorted worst-IoU-first", fontsize=10, y=0.99)
    # Reserve a fixed amount of vertical space for the two-line suptitle
    # regardless of n_rows, so it never collides with row-0 subplot titles
    # (a fixed fraction like top=0.88 shrinks that reserved space as rows
    # grow, and blows up as rows shrink to 1 — an absolute inch-based
    # margin stays legible either way).
    top_margin_in = 0.85
    fig_height_in = n_rows * 3.6
    fig.subplots_adjust(hspace=0.5, wspace=0.15, top=1 - top_margin_in / fig_height_in)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
