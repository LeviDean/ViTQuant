from pathlib import Path
from typing import Optional

from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)
from torch.utils.data import DataLoader


class TorchCalibrationReader(CalibrationDataReader):
    """Feeds batches from a torch DataLoader to ORT static quantization."""

    def __init__(self, loader: DataLoader, num_batches: Optional[int] = None,
                 input_name: str = "input"):
        self._batches = []
        for i, (x, _) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            self._batches.append({input_name: x.numpy()})
        self._it = iter(self._batches)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._batches)


def quantize_onnx(fp32_path: str | Path, int8_path: str | Path,
                  calib_loader: DataLoader,
                  num_batches: Optional[int] = None) -> Path:
    """ORT static INT8 quantization: QDQ format, per-channel weights (matches the
    research-layer default: weight per-channel symmetric int8)."""
    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    pre_path = fp32_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre_path))  # shape inference + optimization
    quantize_static(
        str(pre_path), str(int8_path),
        TorchCalibrationReader(calib_loader, num_batches),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,  # U8S8: recommended for x86-64 CPU EP
    )
    pre_path.unlink(missing_ok=True)
    return int8_path
