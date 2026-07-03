import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.qconfig import QConfig


def _loader(n=4, bs=2):
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.zeros(n, dtype=torch.long)), batch_size=bs)


def test_calibrate_freezes_all_fake_quants():
    model = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                        QConfig())
    calibrate(model, _loader(), device=torch.device("cpu"))
    fqs = [m for m in model.modules() if isinstance(m, FakeQuantize)]
    assert all(m.quantizing and not m.observing for m in fqs)
    assert all(m.scale.numel() > 0 for m in fqs)


def test_calibrate_respects_num_batches():
    model = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                        QConfig())
    seen = []
    loader = _loader(n=8, bs=2)
    calibrate(model, loader, device=torch.device("cpu"), num_batches=1,
              progress=seen.append)
    assert seen == [0]
