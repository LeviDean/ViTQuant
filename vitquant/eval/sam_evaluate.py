from typing import Optional

import torch
from torch import nn

from vitquant.utils.ort_session import create_cpu_session

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
    image+prompt, compare fp32 vs quantized-vision-encoder predicted masks
    via IoU, per mask hypothesis. High IoU means quantization didn't change
    what the model segments. IoU is computed on SAM's native low-res mask
    logits (not upsampled to original image resolution) — a same-basis
    comparison between the two models on one run, not a substitute for
    full-resolution mIoU against ground truth. mean_iou/min_iou aggregate
    equally across every (sample, mask hypothesis) pair, not weighted per
    image. Returns {"per_sample_iou": [[iou_per_mask], ...], "mean_iou":
    float, "min_iou": float}."""
    assert samples, "evaluate_sam_consistency: samples is empty"
    fp32_model = fp32_model.eval().to(device)
    quant_model = quant_model.eval().to(device)

    per_sample = []
    all_ious = []
    for inputs in samples:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        fp32_masks = fp32_model(**inputs).pred_masks > MASK_THRESHOLD
        quant_masks = quant_model(**inputs).pred_masks > MASK_THRESHOLD
        assert fp32_masks.shape[0] == 1 and fp32_masks.shape[1] == 1, (
            "evaluate_sam_consistency assumes batch_size=1 and a single "
            "point-prompt-group per sample (matches build_sam_inputs)")
        num_masks = fp32_masks.shape[2]
        ious = [mask_iou(fp32_masks[0, 0, m], quant_masks[0, 0, m]) for m in range(num_masks)]
        per_sample.append(ious)
        all_ious.extend(ious)

    return {
        "per_sample_iou": per_sample,
        "mean_iou": sum(all_ious) / len(all_ious),
        "min_iou": min(all_ious),
    }


@torch.no_grad()
def evaluate_sam_consistency_onnx(fp32_model: nn.Module, onnx_vision_encoder_path,
                                  samples: list[dict], device: torch.device,
                                  graph_optimization_level: Optional[str] = None) -> dict:
    """Real-delivery self-consistency check: run each sample's pixel_values
    through the ORT-quantized vision encoder ONNX graph (real INT8, not
    simulated), then feed the resulting image_embeddings into the same fp32
    PyTorch prompt_encoder+mask_decoder used for the fp32 reference — so the
    only thing that differs between the two pipelines being compared is
    whether the vision encoder ran as fp32 PyTorch or real ORT INT8. Compares
    against the pure-fp32 (pixel_values all the way through) pipeline via
    mask_iou, same aggregation as evaluate_sam_consistency. This is the
    "real quantization" counterpart to evaluate_sam_consistency's simulated
    fake-quant comparison — the two should be close if the research layer's
    simulated quantization is representative of the real delivery layer.
    graph_optimization_level is forwarded to create_cpu_session; set it to
    "basic" on CPUs where ORT's quantized-op fusion SIGILL-crashes (see
    vitquant/utils/ort_session.py)."""
    assert samples, "evaluate_sam_consistency_onnx: samples is empty"
    fp32_model = fp32_model.eval().to(device)
    # ORT always runs the vision encoder on CPU regardless of `device`; goes
    # through create_cpu_session (not raw InferenceSession) so the graph-opt
    # SIGILL workaround is available on affected hardware.
    sess = create_cpu_session(onnx_vision_encoder_path, graph_optimization_level)

    per_sample = []
    all_ious = []
    for inputs in samples:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        fp32_masks = fp32_model(**inputs).pred_masks > MASK_THRESHOLD

        embeds_np = sess.run(None, {"input": inputs["pixel_values"].cpu().numpy()})[0]
        # ORT output is CPU numpy float32; this is the CPU->device handoff
        embeds = torch.from_numpy(embeds_np).to(device)
        bridged_inputs = {k: v for k, v in inputs.items() if k != "pixel_values"}
        real_out = fp32_model(image_embeddings=embeds, **bridged_inputs)
        real_masks = real_out.pred_masks > MASK_THRESHOLD

        assert fp32_masks.shape[0] == 1 and fp32_masks.shape[1] == 1, (
            "evaluate_sam_consistency_onnx assumes batch_size=1 and a single "
            "point-prompt-group per sample (matches build_sam_inputs)")
        num_masks = fp32_masks.shape[2]
        ious = [mask_iou(fp32_masks[0, 0, m], real_masks[0, 0, m]) for m in range(num_masks)]
        per_sample.append(ious)
        all_ious.extend(ious)

    return {
        "per_sample_iou": per_sample,
        "mean_iou": sum(all_ious) / len(all_ious),
        "min_iou": min(all_ious),
    }
