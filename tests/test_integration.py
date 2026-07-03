import pytest
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K
from vitquant.deploy.benchmark import model_size_mb
from vitquant.deploy.export_onnx import export_fp32_onnx
from vitquant.deploy.quantize_ort import quantize_onnx
from vitquant.eval.evaluate import evaluate_onnx, evaluate_torch
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import QConfig

CPU = torch.device("cpu")


def _loader(n=8, bs=4):
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.randint(0, 10, (n,))), batch_size=bs)


def test_research_layer_end_to_end():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    qmodel = convert_vit(model, QConfig())
    calibrate(qmodel, _loader(), device=CPU)
    res = evaluate_torch(qmodel, _loader(), IMAGENETTE_TO_IMAGENET1K, CPU)
    assert 0.0 <= res["top1"] <= 1.0


@pytest.mark.slow
def test_delivery_layer_end_to_end(tmp_path):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    fp32 = export_fp32_onnx(model, tmp_path / "m.onnx")
    int8 = quantize_onnx(fp32, tmp_path / "m.int8.onnx", _loader(), num_batches=1)
    res = evaluate_onnx(int8, _loader(), IMAGENETTE_TO_IMAGENET1K)
    assert 0.0 <= res["top1"] <= 1.0
    assert model_size_mb(int8) < 0.5 * model_size_mb(fp32)
