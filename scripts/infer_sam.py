#!/usr/bin/env python
"""Label-free inference with a SAVED quantized SAM/SAM3 model on arbitrary
images (e.g. an unlabeled test set). Loads the quantization artifact written by
run_sam.py / run_sam3_concept.py --save-quantized — no calibration data, no
AdaRound rerun, bit-identical to the model that was evaluated.

Point families (sam / sam3): each image gets an n×n grid of point prompts;
per point the model's own best mask hypothesis is kept. Concept family
(sam3_concept): every image is searched for --text, all instances returned.

Outputs per image: <name>_masks.npz (boolean masks + metadata) and
<name>_overlay.png (masks drawn over the image)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vitquant.data.sam_samples import grid_points
from vitquant.eval.qualitative import _post_process_masks
from vitquant.quant.persist import load_quantized_sam
from vitquant.utils.device import resolve_device

sys.stdout.reconfigure(line_buffering=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


@torch.no_grad()
def _infer_points(model, processor, img, grid, with_labels, device):
    """n×n grid of point prompts -> per point the model's best mask.
    Returns (masks (P,H,W) bool ndarray, points list)."""
    points = grid_points(img.width, img.height, grid)
    kwargs = {}
    if with_labels:
        kwargs["input_labels"] = [[[1] for _ in points]]
    inputs = processor(images=img, input_points=[[[list(p)] for p in points]],
                       return_tensors="pt", **kwargs)
    inputs = {k: (v.float() if v.dtype == torch.float64 else v)
             for k, v in inputs.items()}
    out = model(**{k: v.to(device) for k, v in inputs.items()})
    full = _post_process_masks(processor, out.pred_masks, inputs)[0]  # (P, M, H, W)
    best = out.iou_scores[0].argmax(dim=-1)  # (P,)
    masks = torch.stack([full[p, int(best[p])] for p in range(len(points))])
    return masks.cpu().numpy().astype(bool), points


@torch.no_grad()
def _infer_concept(model, processor, img, text, device):
    """Text prompt -> all matching instances. Returns (masks (N,H,W), scores)."""
    inputs = processor(images=img, text=text, return_tensors="pt")
    inputs = {k: (v.float() if v.dtype == torch.float64 else v)
             for k, v in inputs.items()}
    out = model(**{k: v.to(device) for k, v in inputs.items()})
    res = processor.post_process_instance_segmentation(
        out, threshold=0.3, mask_threshold=0.5,
        target_sizes=[(img.height, img.width)])[0]
    return (res["masks"].bool().cpu().numpy(),
            res["scores"].cpu().tolist())


def _save_overlay(img, masks, out_png, title, points=None):
    fig, ax = plt.subplots(figsize=(7, 7 * img.height / max(img.width, 1)))
    ax.imshow(img)
    cmap = plt.get_cmap("tab20")
    overlay = np.zeros((img.height, img.width, 4))
    for i, m in enumerate(masks):
        color = cmap(i % 20)
        overlay[m] = (*color[:3], 0.45)
    ax.imshow(overlay)
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, marker="o", s=14, c="white", edgecolors="black", linewidths=0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True,
                    help="directory containing quantized_state.pt + quant_meta.json "
                         "(the output_dir of a --save-quantized run)")
    ap.add_argument("--images", required=True, help="image file or directory")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <artifact>/inference)")
    ap.add_argument("--prompt-grid", type=int, default=None,
                    help="point families: n×n prompt grid (default: the grid the "
                         "artifact was evaluated with, else 4)")
    ap.add_argument("--text", default=None,
                    help="sam3_concept artifacts: the concept phrase to segment "
                         "(applied to every image)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    # A sam3 (point-prompt tracker) artifact can serve text-prompted concept
    # inference too: tracker and Sam3Model share the quantized vision encoder,
    # and its calibration doesn't depend on prompts. Passing --text on a sam3
    # artifact loads the concept model with the same quantization state.
    meta_family = None
    if args.text:
        meta_family = "sam3_concept"
    model, processor, meta = load_quantized_sam(args.artifact, device=device,
                                                family=meta_family)
    family = meta["family"]
    if family == "sam3_concept" and not args.text:
        ap.error("--text is required for concept (text-prompt) inference")

    images = _list_images(Path(args.images))
    if not images:
        ap.error(f"no images found under {args.images}")
    out_dir = Path(args.out) if args.out else Path(args.artifact) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = args.prompt_grid or meta.get("prompt_grid", 4)
    scheme = f"W{meta['quant']['weight']['bits']}A{meta['quant']['activation']['bits']}"
    print(f"{family} {scheme}: {len(images)} image(s) -> {out_dir}")

    for path in images:
        img = Image.open(path).convert("RGB")
        if family == "sam3_concept":
            masks, scores = _infer_concept(model, processor, img, args.text, device)
            points = None
            note = f"\"{args.text}\": {len(masks)} instance(s)"
            extra = {"scores": np.array(scores), "text": args.text}
        else:
            masks, points = _infer_points(model, processor, img, grid,
                                          with_labels=(family == "sam3"), device=device)
            note = f"{grid}x{grid} point grid"
            extra = {"points": np.array(points)}
        np.savez_compressed(out_dir / f"{path.stem}_masks.npz", masks=masks, **extra)
        _save_overlay(img, masks, out_dir / f"{path.stem}_overlay.png",
                      f"{path.name} · {scheme} quantized · {note}", points)
        print(f"  {path.name}: {len(masks)} mask(s)")

    print(f"Done. Masks (.npz) and overlays (.png) in {out_dir}")


if __name__ == "__main__":
    main()
