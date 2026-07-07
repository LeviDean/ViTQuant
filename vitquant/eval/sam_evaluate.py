import torch
from torch import nn

MASK_THRESHOLD = 0.0  # matches SAM's own binarization convention


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU between two boolean masks of the same shape. Both-empty -> 1.0
    (perfect agreement), not an undefined 0/0."""
    intersection = (a & b).sum().item()
    union = (a | b).sum().item()
    return 1.0 if union == 0 else intersection / union


@torch.no_grad()
def evaluate_sam_consistency(fp32_model: nn.Module, quant_model: nn.Module,
                             samples: list[dict], device: torch.device) -> dict:
    """Self-consistency check (no ground truth needed): for the same
    image+prompt, compare fp32 vs quantized-vision-encoder predicted masks
    via IoU, per mask hypothesis. High IoU means quantization didn't change
    what the model segments. Returns {"per_sample_iou": [[iou_per_mask], ...],
    "mean_iou": float, "min_iou": float}."""
    fp32_model = fp32_model.eval().to(device)
    quant_model = quant_model.eval().to(device)

    per_sample = []
    all_ious = []
    for inputs in samples:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        fp32_masks = fp32_model(**inputs).pred_masks > MASK_THRESHOLD
        quant_masks = quant_model(**inputs).pred_masks > MASK_THRESHOLD
        num_masks = fp32_masks.shape[2]
        ious = [_mask_iou(fp32_masks[0, 0, m], quant_masks[0, 0, m]) for m in range(num_masks)]
        per_sample.append(ious)
        all_ious.extend(ious)

    return {
        "per_sample_iou": per_sample,
        "mean_iou": sum(all_ious) / len(all_ious),
        "min_iou": min(all_ious),
    }
