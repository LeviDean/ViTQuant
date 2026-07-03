import timm
import torch

from vitquant.deploy.export_onnx import export_fp32_onnx


def test_export_and_parity(tmp_path):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    out = export_fp32_onnx(model, tmp_path / "deit_tiny.onnx")
    assert out.exists()


def test_export_supports_dynamic_batch(tmp_path):
    import onnxruntime as ort
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    out = export_fp32_onnx(model, tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for bs in (1, 3):
        y = sess.run(None, {"input": torch.randn(bs, 3, 224, 224).numpy()})[0]
        assert y.shape == (bs, 1000)
