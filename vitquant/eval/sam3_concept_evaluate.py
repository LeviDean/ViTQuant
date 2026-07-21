"""Instance-set self-consistency for SAM3 concept segmentation (text prompts).

Unlike point prompts, a text prompt returns a VARIABLE-SIZED set of instances,
so fp32 and quantized outputs have no natural 1:1 pairing. We match instances
greedily by mask IoU (highest-IoU pairs first, no reuse — the same scheme COCO
evaluation uses), then report:

- consistency  = sum(matched IoU) / max(n_fp32, n_quant)   per image, averaged.
  One number folding mask-quality AND detection agreement together: every
  missed or hallucinated instance contributes 0. Equal sets with perfect masks
  give 1.0; both-empty gives 1.0 (perfect agreement). This is the scalar the
  sensitivity / mixed-precision sweeps optimize against.
- detection F1 = 2*matched / (n_fp32 + n_quant): count agreement alone.
- matched IoU  = mean IoU over matched pairs alone: mask quality where the two
  models agree an instance exists.

Same caveat as the point-prompt protocol: this measures how much quantization
changed the predictions, not ground-truth accuracy."""
from typing import Callable, Optional

import torch
from torch import nn

from vitquant.eval.evaluate import block_sensitivity_scored, mixed_precision_scored

SCORE_THRESHOLD = 0.3  # instance-confidence cutoff, HF default for SAM3
MASK_THRESHOLD = 0.5


@torch.no_grad()
def _instances(model: nn.Module, processor, sample: dict,
               device: torch.device) -> torch.Tensor:
    """Run one image+text sample, return binary instance masks
    (num_instances, H, W) at original resolution (possibly empty)."""
    inputs = {k: v.to(device) for k, v in sample["inputs"].items()}
    out = model(**inputs)
    res = processor.post_process_instance_segmentation(
        out, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD,
        target_sizes=[sample["target_size"]])[0]
    return res["masks"].bool().cpu()


def pairwise_mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(Na, H, W) x (Nb, H, W) boolean masks -> (Na, Nb) IoU matrix."""
    fa = a.flatten(1).float()
    fb = b.flatten(1).float()
    inter = fa @ fb.T
    union = fa.sum(1, keepdim=True) + fb.sum(1, keepdim=True).T - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def greedy_match(iou: torch.Tensor) -> list[tuple[int, int, float]]:
    """Greedy bipartite matching on an IoU matrix: repeatedly take the highest
    remaining pair with IoU > 0, never reusing a row or column. Returns
    (i, j, iou) triples. Greedy (not Hungarian) is deliberate: it needs no
    scipy dependency and for IoU-based matching of small instance sets it is
    the standard choice (COCO eval does the same)."""
    if iou.numel() == 0:
        return []
    pairs = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    nb = iou.shape[1]
    for k in iou.flatten().argsort(descending=True).tolist():
        i, j = divmod(k, nb)
        if i in used_a or j in used_b:
            continue
        if iou[i, j] <= 0:
            break
        pairs.append((i, j, float(iou[i, j])))
        used_a.add(i)
        used_b.add(j)
    return pairs


@torch.no_grad()
def evaluate_sam3_concept_consistency(fp32_model: nn.Module, quant_model: nn.Module,
                                      processor, samples: list[dict],
                                      device: torch.device) -> dict:
    """Instance-set self-consistency between fp32 and quantized Sam3Model over
    image+text samples. Returns per-sample rows plus mean_consistency /
    mean_f1 / mean_matched_iou / min_consistency and average instance counts."""
    assert samples, "evaluate_sam3_concept_consistency: samples is empty"
    fp32_model = fp32_model.eval().to(device)
    quant_model = quant_model.eval().to(device)

    rows = []
    for s in samples:
        fm = _instances(fp32_model, processor, s, device)
        qm = _instances(quant_model, processor, s, device)
        pairs = greedy_match(pairwise_mask_iou(fm, qm))
        n_f, n_q, n_m = len(fm), len(qm), len(pairs)
        sum_iou = sum(p[2] for p in pairs)
        denom = max(n_f, n_q)
        rows.append({
            "text": s["text"],
            "n_fp32": n_f, "n_quant": n_q, "n_matched": n_m,
            "consistency": 1.0 if denom == 0 else sum_iou / denom,
            "f1": 1.0 if n_f + n_q == 0 else 2 * n_m / (n_f + n_q),
            "matched_iou": (sum_iou / n_m) if n_m else (1.0 if denom == 0 else 0.0),
        })

    n = len(rows)
    return {
        "per_sample": rows,
        "mean_consistency": sum(r["consistency"] for r in rows) / n,
        "min_consistency": min(r["consistency"] for r in rows),
        "mean_f1": sum(r["f1"] for r in rows) / n,
        "mean_matched_iou": sum(r["matched_iou"] for r in rows) / n,
        "avg_instances_fp32": sum(r["n_fp32"] for r in rows) / n,
        "avg_instances_quant": sum(r["n_quant"] for r in rows) / n,
    }


def block_sensitivity_sam3_concept(quant_model: nn.Module, fp32_model: nn.Module,
                                   processor, samples: list[dict],
                                   device: torch.device,
                                   log: Optional[Callable[[str], None]] = None) -> dict:
    """Per-block sensitivity with the concept-consistency score as the measure —
    same scan core as every other pipeline, only the ruler differs."""
    def measure() -> float:
        return evaluate_sam3_concept_consistency(
            fp32_model, quant_model, processor, samples, device)["mean_consistency"]
    return block_sensitivity_scored(quant_model, measure, log)


def mixed_precision_sweep_sam3_concept(quant_model: nn.Module, fp32_model: nn.Module,
                                       processor, samples: list[dict],
                                       device: torch.device,
                                       sensitivity: dict, weight_bits: int,
                                       ks: Optional[list[int]] = None,
                                       log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Mixed-precision sweep scored by concept consistency; rows carry a
    "consistency" score field."""
    def measure() -> float:
        return evaluate_sam3_concept_consistency(
            fp32_model, quant_model, processor, samples, device)["mean_consistency"]
    return mixed_precision_scored(quant_model, sensitivity, weight_bits, measure,
                                  score_key="consistency", ks=ks, log=log)
