from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Imagenette

# Imagenette classes in wordnet-id sorted order (tench, English springer, cassette
# player, chain saw, church, French horn, garbage truck, gas pump, golf ball,
# parachute) -> their ImageNet-1k class indices.
IMAGENETTE_TO_IMAGENET1K = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

IMAGENETTE_CLASS_NAMES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]


def _dataset(root: str | Path, split: str, data_cfg: dict, download: bool) -> Imagenette:
    root = Path(root)
    need_download = download and not (root / "imagenette2-160").exists()
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return Imagenette(str(root), split=split, size="160px",
                      download=need_download, transform=transform)


def build_val_loader(root: str | Path, data_cfg: dict, batch_size: int = 64,
                     num_workers: int = 4, download: bool = True) -> DataLoader:
    ds = _dataset(root, "val", data_cfg, download)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def build_calib_loader(root: str | Path, data_cfg: dict, calib_images: int = 256,
                       batch_size: int = 32, num_workers: int = 4,
                       download: bool = True) -> DataLoader:
    """Fixed random subset of the train split (seeded for reproducibility)."""
    ds = _dataset(root, "train", data_cfg, download)
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=gen)[:calib_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def build_sample_loader(root: str | Path, data_cfg: dict, num_samples: int = 30,
                        seed: int = 0, download: bool = True) -> DataLoader:
    """Fixed random subset of the val split, batch_size=1, for per-image
    qualitative inspection. The val split is class-sorted and build_val_loader
    doesn't shuffle, so a plain slice would only cover one or two classes —
    this shuffles (seeded, reproducible) to get a spread across classes."""
    ds = _dataset(root, "val", data_cfg, download)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:num_samples].tolist()
    return DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=0)
