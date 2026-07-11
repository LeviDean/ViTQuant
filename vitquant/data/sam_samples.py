from pathlib import Path

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


def grid_points(width: int, height: int, n: int) -> list[tuple[int, int]]:
    """An n×n grid of prompt points at cell centers, covering the whole image
    (no points on the border). n=1 degenerates to the single image-center
    point, so the grid is a strict generalization of the old center-point
    prompt. Row-major order (left→right, top→bottom)."""
    return [(int((i + 0.5) * width / n), int((j + 0.5) * height / n))
            for j in range(n) for i in range(n)]


def _processor_inputs(processor, img, grid: int) -> tuple[dict, list[tuple[int, int]]]:
    """Run the processor on a single PIL image with an n×n grid of point
    prompts (grid=1: just the image center), returning the model-ready inputs
    dict and the points actually used. Each grid point is its own point-prompt
    group (one point per group), so SAM predicts an independent mask set per
    point — the vision encoder still runs only once per image, only the light
    prompt-encoder/mask-decoder repeat per point. Shared by build_sam_inputs
    and build_sam_qualitative_samples so the processor-call and dtype-fix
    logic can't drift apart either.
    SamProcessor emits input_points as float64, which MPS can't run ops on
    (int64 sizes are fine there, only float64 is unsupported); float32 has
    ample precision for pixel coordinates."""
    w, h = img.size
    points = grid_points(w, h, grid)
    # shape per image: (point_batch=n*n, points_per_group=1, 2)
    inputs = processor(images=img, input_points=[[[list(p)] for p in points]],
                       return_tensors="pt")
    inputs = {k: (v.float() if v.dtype == torch.float64 else v)
             for k, v in inputs.items()}
    return inputs, points


def build_sam_inputs(root: str | Path, processor, num_samples: int = 8,
                     seed: int = 0, download: bool = True,
                     grid: int = 1) -> list[dict]:
    """A seeded-random subset of real Imagenette images (raw PIL, no
    classification transform — SAM doesn't care about ImageNet classes), each
    run through the given SAM processor with an n×n grid of point prompts
    (grid=1: single image-center point, the old behavior) to produce
    ready-to-feed `model(**inputs)` dicts. Reuses whatever Imagenette copy the
    classification pipeline already has."""
    ds, idx = _sample_indices(root, download, num_samples, seed)

    samples = []
    for i in idx:
        img, _ = ds[i]  # raw PIL Image; classification label is irrelevant here
        inputs, _ = _processor_inputs(processor, img, grid)
        samples.append(inputs)
    return samples


def build_sam_qualitative_samples(root: str | Path, processor, num_samples: int = 8,
                                  seed: int = 0, download: bool = True,
                                  grid: int = 1) -> list[dict]:
    """Like build_sam_inputs, but also retains the raw PIL image and the point
    prompts used, for visualization. Each returned dict has keys: "inputs"
    (model-ready dict, same construction as build_sam_inputs), "image" (PIL
    Image), "points" (list of (x, y) prompt coordinates actually used)."""
    ds, idx = _sample_indices(root, download, num_samples, seed)

    samples = []
    for i in idx:
        img, _ = ds[i]  # raw PIL Image; classification label is irrelevant here
        inputs, points = _processor_inputs(processor, img, grid)
        samples.append({"inputs": inputs, "image": img, "points": points})
    return samples
