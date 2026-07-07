from pathlib import Path
from typing import Optional

from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)
from torch.utils.data import DataLoader


class TorchCalibrationReader(CalibrationDataReader):
    """Feeds batches from a torch DataLoader to ORT static quantization.
    All batches are materialized in memory up front (fine for ~256 calib images)."""

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


_WEIGHT_TYPE_BY_BITS = {8: QuantType.QInt8, 4: QuantType.QInt4}


def quantize_onnx(fp32_path: str | Path, int8_path: str | Path,
                  calib_loader: DataLoader,
                  num_batches: Optional[int] = None,
                  weight_bits: int = 8) -> Path:
    """ORT static quantization: QDQ format, per-channel weights (matches the
    research-layer default scheme), activation fixed at U8 (recommended for x86-64
    CPU EP). weight_bits selects the weight grid: 8 (default, real INT8 speedup on
    CPU) or 4 (real size/accuracy numbers via ORT's contrib Q/DQ ops, but ORT's CPU
    EP has no int4 matmul kernel, so this does NOT yield a real latency speedup)."""
    try:
        weight_type = _WEIGHT_TYPE_BY_BITS[weight_bits]
    except KeyError:
        raise ValueError(f"Unsupported weight_bits={weight_bits}; "
                         f"choose from {sorted(_WEIGHT_TYPE_BY_BITS)}")

    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    pre_path = fp32_path.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(fp32_path), str(pre_path))  # shape inference + optimization
        quantize_static(
            str(pre_path), str(int8_path),
            TorchCalibrationReader(calib_loader, num_batches),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=weight_type,
            activation_type=QuantType.QUInt8,  # U8S8/U8S4: recommended for x86-64 CPU EP
        )
    finally:
        pre_path.unlink(missing_ok=True)
    return int8_path
