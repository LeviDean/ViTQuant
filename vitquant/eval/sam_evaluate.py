from typing import Callable, Optional

import torch
from torch import nn

from vitquant.eval.evaluate import block_sensitivity_scored, mixed_precision_scored

MASK_THRESHOLD = 0.0  # matches SAM's own binarization convention


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU between two boolean masks of the same shape. Both-empty -> 1.0
    (perfect agreement), not an undefined 0/0."""
    intersection = (a & b).sum().item()
    union = (a | b).sum().item()
    return 1.0 if union == 0 else intersection / union


@torch.no_grad()
def evaluate_sam_consistency(fp32_model: nn.Module, quant_model: nn.Module,
                             samples: list[dict], device: torch.device) -> dict:
    """Self-consistency check (no ground truth needed): for the same
    image+prompts, compare fp32 vs quantized-vision-encoder predicted masks
    via IoU, per (point prompt, mask hypothesis) pair. Samples may carry a
    single point or an n×n grid of points (see build_sam_inputs `grid`) — a
    grid covers objects across the whole image instead of only the center one,
    while the quantized vision encoder still runs once per image. High IoU
    means quantization didn't change what the model segments. IoU is computed
    on SAM's native low-res mask logits (not upsampled to original image
    resolution) — a same-basis comparison between the two models on one run,
    not a substitute for full-resolution mIoU against ground truth.
    mean_iou/min_iou aggregate equally across every (sample, point, mask
    hypothesis) triple, not weighted per image. Returns {"per_sample_iou":
    [[iou_per_point_and_mask], ...], "mean_iou": float, "min_iou": float}."""
    assert samples, "evaluate_sam_consistency: samples is empty"
    fp32_model = fp32_model.eval().to(device)
    quant_model = quant_model.eval().to(device)

    per_sample = []
    all_ious = []
    for inputs in samples:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        fp32_masks = fp32_model(**inputs).pred_masks > MASK_THRESHOLD
        quant_masks = quant_model(**inputs).pred_masks > MASK_THRESHOLD
        assert fp32_masks.shape[0] == 1, (
            "evaluate_sam_consistency assumes batch_size=1 per sample "
            "(matches build_sam_inputs)")
        num_points, num_masks = fp32_masks.shape[1], fp32_masks.shape[2]
        ious = [mask_iou(fp32_masks[0, p, m], quant_masks[0, p, m])
                for p in range(num_points) for m in range(num_masks)]
        per_sample.append(ious)
        all_ious.extend(ious)

    return {
        "per_sample_iou": per_sample,
        "mean_iou": sum(all_ious) / len(all_ious),
        "min_iou": min(all_ious),
    }


def block_sensitivity_sam(quant_model: nn.Module, fp32_model: nn.Module,
                          samples: list[dict], device: torch.device,
                          log: Optional[Callable[[str], None]] = None) -> dict:
    """Per-block sensitivity for the SAM vision encoder, measured by
    self-consistency IoU instead of top-1 (SAM has no classification accuracy).
    Quantize one vision-encoder block at a time (rest fp32) and report the mean
    self-consistency IoU drop vs the all-fp32 baseline — a bigger drop means that
    block's quantization changes the predicted masks more, i.e. it's more
    sensitive. Blocks sorted most-sensitive first.

    With every quantizer off, the "quantized" model reproduces fp32's masks
    exactly, so the baseline IoU is 1.0; each block's drop is 1.0 minus its IoU.
    Shares the sweep implementation with the classification pipeline via
    block_sensitivity_scored — only the measure differs. `quant_model` must be
    calibrated; it's restored to fully-quantizing on exit."""
    def measure() -> float:
        return evaluate_sam_consistency(fp32_model, quant_model, samples, device)["mean_iou"]
    return block_sensitivity_scored(quant_model, measure, log)


def mixed_precision_sweep_sam(quant_model: nn.Module, fp32_model: nn.Module,
                              samples: list[dict], device: torch.device,
                              sensitivity: dict, weight_bits: int,
                              ks: Optional[list[int]] = None,
                              log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Mixed-precision sweep for the SAM vision encoder, scored by self-
    consistency IoU. Protect the K most sensitive blocks (kept fp32) and quantize
    the rest at `weight_bits`, sweeping K to give an IoU-vs-compression curve.
    With every block protected (K=all) the quantized model reproduces fp32's
    masks, so the IoU is 1.0. Shares the sweep core with the classification
    pipeline via mixed_precision_scored — only the measure differs; rows carry an
    "iou" score field. `quant_model` must be calibrated; restored to
    fully-quantizing on exit."""
    def measure() -> float:
        return evaluate_sam_consistency(fp32_model, quant_model, samples, device)["mean_iou"]
    return mixed_precision_scored(quant_model, sensitivity, weight_bits, measure,
                                  score_key="iou", ks=ks, log=log)
