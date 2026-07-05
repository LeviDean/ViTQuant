from pathlib import Path

import numpy as np
import onnx
import torch


def export_fp32_onnx(model: torch.nn.Module, out_path: str | Path,
                     img_size: int = 224, opset: int = 17) -> Path:
    """Export the *original fp32* model (not the fake-quant one) to ONNX,
    validate with onnx.checker, and verify numerical parity against PyTorch."""
    import onnxruntime as ort

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # legacy exporter: stable for timm ViT + ORT quantization tooling
    )
    onnx.checker.check_model(str(out_path))

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    if not np.allclose(ref, got, atol=1e-4):
        raise RuntimeError(f"ONNX parity check failed: max diff {max_diff:.2e} > 1e-4")
    return out_path
