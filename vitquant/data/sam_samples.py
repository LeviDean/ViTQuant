from pathlib import Path
from typing import Optional

import torch
from torchvision.datasets import Imagenette


def build_sam_inputs(root: str | Path, processor, num_samples: int = 8,
                     seed: int = 0, download: bool = True,
                     point: Optional[tuple[int, int]] = None) -> list[dict]:
    """A seeded-random subset of real Imagenette images (raw PIL, no
    classification transform — SAM doesn't care about ImageNet classes), each
    run through the given SAM processor with a single point prompt (default:
    image center) to produce ready-to-feed `model(**inputs)` dicts. Reuses
    whatever Imagenette copy the classification pipeline already has."""
    root = Path(root)
    need_download = download and not (root / "imagenette2-160").exists()
    ds = Imagenette(str(root), split="train", size="160px", download=need_download)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:num_samples].tolist()

    samples = []
    for i in idx:
        img, _ = ds[i]  # raw PIL Image; classification label is irrelevant here
        w, h = img.size
        p = point if point is not None else (w // 2, h // 2)
        inputs = processor(images=img, input_points=[[list(p)]], return_tensors="pt")
        samples.append(inputs)
    return samples
