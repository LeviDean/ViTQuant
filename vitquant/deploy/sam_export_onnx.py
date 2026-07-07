from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch


class _VisionEncoderWrapper(torch.nn.Module):
    """Wraps SamVisionEncoder so forward() returns a plain tensor (the image
    embeddings) instead of a SamVisionEncoderOutput dataclass, which
    torch.onnx.export cannot trace cleanly."""

    def __init__(self, vision_encoder: torch.nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(pixel_values)[0]


def export_sam_vision_encoder_onnx(model, out_path: str | Path,
                                   img_size: Optional[int] = None,
                                   opset: int = 17) -> Path:
    """Export model.vision_encoder (the ONLY part this framework quantizes)
    to ONNX as a standalone pixel_values -> image_embeddings graph — the
    prompt_encoder/mask_decoder stay in PyTorch. Validates with onnx.checker
    and a numerical parity gate (atol 1e-4) against the same wrapper run in
    PyTorch, same pattern as vitquant/deploy/export_onnx.py::export_fp32_onnx.
    img_size defaults to model.config.vision_config.image_size if not given."""
    import onnxruntime as ort

    if img_size is None:
        img_size = model.config.vision_config.image_size

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _VisionEncoderWrapper(model.vision_encoder).eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        wrapper, (dummy,), str(out_path),
        input_names=["input"], output_names=["image_embeddings"],
        dynamic_axes={"input": {0: "batch"}, "image_embeddings": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # legacy exporter: stable for ORT quantization tooling
    )
    onnx.checker.check_model(str(out_path))

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    if not np.allclose(ref, got, atol=1e-4):
        raise RuntimeError(f"ONNX parity check failed: max diff {max_diff:.2e} > 1e-4")
    return out_path
