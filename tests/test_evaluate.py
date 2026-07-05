import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K
from vitquant.eval.evaluate import block_sensitivity, evaluate_torch
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import QConfig

CPU = torch.device("cpu")


def _loader(n=4, bs=2):
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.randint(0, 10, (n,))), batch_size=bs)


def test_evaluate_torch_returns_fractions():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    res = evaluate_torch(model, _loader(), IMAGENETTE_TO_IMAGENET1K, CPU)
    assert set(res) == {"top1", "top5"}
    assert 0.0 <= res["top1"] <= res["top5"] <= 1.0


def test_evaluate_perfect_model():
    class Oracle(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 1000)
            logits[:, IMAGENETTE_TO_IMAGENET1K[3]] = 1.0
            return logits

    x = torch.randn(4, 3, 224, 224)
    loader = DataLoader(TensorDataset(x, torch.full((4,), 3)), batch_size=2)
    assert evaluate_torch(Oracle(), loader, IMAGENETTE_TO_IMAGENET1K, CPU)["top1"] == 1.0


def test_block_sensitivity_groups_and_restores():
    model = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                        QConfig())
    calibrate(model, _loader(), device=CPU)
    result = block_sensitivity(model, _loader(), IMAGENETTE_TO_IMAGENET1K, CPU)
    assert "patch_embed" in result and "blocks.0" in result and len(result) >= 13
    # after the sweep every FakeQuantize must be back in quantizing mode
    from vitquant.quant.fake_quant import FakeQuantize
    assert all(m.quantizing for m in model.modules() if isinstance(m, FakeQuantize))
