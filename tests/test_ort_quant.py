import pytest
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.deploy.benchmark import benchmark_onnx, model_size_mb
from vitquant.deploy.export_onnx import export_fp32_onnx
from vitquant.deploy.quantize_ort import TorchCalibrationReader, quantize_onnx


def _loader(n=4, bs=2):
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.zeros(n, dtype=torch.long)), batch_size=bs)


def test_calibration_reader_yields_then_none():
    reader = TorchCalibrationReader(_loader(n=4, bs=2), num_batches=2)
    batches = [reader.get_next(), reader.get_next()]
    assert all(b is not None and "input" in b for b in batches)
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


@pytest.mark.slow
def test_quantize_shrinks_model(tmp_path):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    fp32 = export_fp32_onnx(model, tmp_path / "m.onnx")
    int8 = quantize_onnx(fp32, tmp_path / "m.int8.onnx", _loader(), num_batches=1)
    assert int8.exists()
    assert model_size_mb(int8) < 0.5 * model_size_mb(fp32)  # ~4x expected
    lat = benchmark_onnx(int8, runs=3, warmup=1)
    assert lat > 0
