from pathlib import Path
from typing import Optional

import torch
from torchvision.datasets import Imagenette


def _sample_indices(root: str | Path, download: bool, num_samples: int,
                    seed: int) -> tuple[Imagenette, list[int]]:
    """Build the Imagenette dataset and pick a seeded-random subset of
    indices, shared by build_sam_inputs and build_sam_qualitative_samples so
    the two selection strategies can't silently drift apart."""
    root = Path(root)
    need_download = download and not (root / "imagenette2-160").exists()
    ds = Imagenette(str(root), split="train", size="160px", download=need_download)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:num_samples].tolist()
    return ds, idx


def _processor_inputs(processor, img, point: Optional[tuple[int, int]]) -> tuple[dict, tuple[int, int]]:
    """Run the processor on a single PIL image with a point prompt (default:
    image center), returning the model-ready inputs dict and the point
    actually used. SamProcessor emits input_points as float64, which MPS
    can't run ops on (int64 sizes are fine there, only float64 is
    unsupported); float32 has ample precision for pixel coordinates."""
    w, h = img.size
    p = point if point is not None else (w // 2, h // 2)
    inputs = processor(images=img, input_points=[[list(p)]], return_tensors="pt")
    inputs = {k: (v.float() if v.dtype == torch.float64 else v)
             for k, v in inputs.items()}
    return inputs, p


def build_sam_inputs(root: str | Path, processor, num_samples: int = 8,
                     seed: int = 0, download: bool = True,
                     point: Optional[tuple[int, int]] = None) -> list[dict]:
    """A seeded-random subset of real Imagenette images (raw PIL, no
    classification transform — SAM doesn't care about ImageNet classes), each
    run through the given SAM processor with a single point prompt (default:
    image center) to produce ready-to-feed `model(**inputs)` dicts. Reuses
    whatever Imagenette copy the classification pipeline already has."""
    ds, idx = _sample_indices(root, download, num_samples, seed)

    samples = []
    for i in idx:
        img, _ = ds[i]  # raw PIL Image; classification label is irrelevant here
        inputs, _ = _processor_inputs(processor, img, point)
        samples.append(inputs)
    return samples


def build_sam_qualitative_samples(root: str | Path, processor, num_samples: int = 8,
                                  seed: int = 0, download: bool = True,
                                  point: Optional[tuple[int, int]] = None) -> list[dict]:
    """Like build_sam_inputs, but also retains the raw PIL image and the point
    prompt used, for visualization. Each returned dict has keys: "inputs"
    (model-ready dict, same construction as build_sam_inputs), "image" (PIL
    Image), "point" (the (x, y) prompt coordinate actually used)."""
    ds, idx = _sample_indices(root, download, num_samples, seed)

    samples = []
    for i in idx:
        img, _ = ds[i]  # raw PIL Image; classification label is irrelevant here
        inputs, p = _processor_inputs(processor, img, point)
        samples.append({"inputs": inputs, "image": img, "point": p})
    return samples
