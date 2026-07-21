"""Image + text-phrase samples for SAM3 concept segmentation. Each Imagenette
image is paired with its own class name as the text prompt (e.g. "gas pump"),
so the model is asked to find every instance of a concept that is actually
present — the quantization question becomes "does the quantized encoder still
support the same concept-level detections?"."""
from pathlib import Path

import torch

from vitquant.data.imagenette import IMAGENETTE_CLASS_NAMES
from vitquant.data.sam_samples import _sample_indices


def build_sam3_concept_samples(root: str | Path, processor, num_samples: int = 8,
                               seed: int = 0, download: bool = True,
                               keep_images: bool = False) -> list[dict]:
    """A seeded-random subset of Imagenette images, each run through the
    Sam3Processor with its class name as the text prompt. Returns dicts with
    keys: "inputs" (model-ready dict for Sam3Model), "text" (the phrase),
    "target_size" ((h, w) of the original image, for instance post-processing),
    and optionally "image" (PIL, for visualization). Calibration should feed
    [s["inputs"] for s in samples] — the quantized vision encoder's activations
    don't depend on the text, but keeping the real per-image phrase makes the
    calibration forward identical to the evaluation forward."""
    ds, idx = _sample_indices(root, download, num_samples, seed)

    samples = []
    for i in idx:
        img, label = ds[i]
        text = IMAGENETTE_CLASS_NAMES[int(label)]
        inputs = processor(images=img, text=text, return_tensors="pt")
        inputs = {k: (v.float() if v.dtype == torch.float64 else v)
                 for k, v in inputs.items()}
        s = {"inputs": inputs, "text": text, "target_size": (img.height, img.width)}
        if keep_images:
            s["image"] = img
        samples.append(s)
    return samples
