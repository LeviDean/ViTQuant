import platform
from pathlib import Path
from typing import Optional

from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)
from torch.utils.data import DataLoader


def _check_int4_cpu_support() -> None:
    """ORT's int4 QuantizeLinear/DequantizeLinear contrib kernel (com.microsoft
    domain) requires AVX-512 on x86-64; without it the process is killed with
    SIGILL (Illegal instruction) deep in native code, with no Python traceback.
    Fail loudly and early instead. Only x86-64/Linux is checked: Apple Silicon
    (arm64) uses a different kernel path and is unaffected; other platforms
    can't be checked here so we don't block them."""
    if platform.machine() not in ("x86_64", "AMD64"):
        return
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return
    flags = cpuinfo.read_text()
    if "avx512f" not in flags:
        raise RuntimeError(
            "weight_bits=4 needs ORT's int4 contrib kernel, which requires AVX-512 "
            "on x86-64 CPUs. This CPU lacks it (checked /proc/cpuinfo for 'avx512f') "
            "and would crash the process with SIGILL instead of raising a Python "
            "error. Use weight_bits=8 for the real ORT delivery layer on this "
            "machine — the research layer's simulated W4A8 accuracy numbers "
            "(vitquant.quant, no ORT involved) remain valid regardless.")


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
_QUANT_FORMATS = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}


def quantize_onnx(fp32_path: str | Path, int8_path: str | Path,
                  calib_loader: DataLoader,
                  num_batches: Optional[int] = None,
                  weight_bits: int = 8,
                  quant_format: str = "qdq") -> Path:
    """ORT static quantization, per-channel weights (matches the research-layer
    default scheme), activation fixed at U8 (recommended for x86-64 CPU EP).
    weight_bits selects the weight grid: 8 (default, real INT8 speedup on CPU)
    or 4 (real size/accuracy numbers via ORT's contrib Q/DQ ops, but ORT's CPU
    EP has no int4 matmul kernel, so this does NOT yield a real latency
    speedup). quant_format: "qdq" (default) inserts Quantize/DequantizeLinear
    nodes and relies on ORT's graph optimizer to fuse them into a fast int8
    kernel at runtime (see vitquant/utils/ort_session.py — that fusion is what
    crashes on some CPUs); "qoperator" bakes the quantized op (e.g.
    QLinearMatMul) directly into the graph at quantization time, so the fast
    kernel is used regardless of the runtime graph optimization level."""
    try:
        weight_type = _WEIGHT_TYPE_BY_BITS[weight_bits]
    except KeyError:
        raise ValueError(f"Unsupported weight_bits={weight_bits}; "
                         f"choose from {sorted(_WEIGHT_TYPE_BY_BITS)}")
    try:
        fmt = _QUANT_FORMATS[quant_format]
    except KeyError:
        raise ValueError(f"Unsupported quant_format={quant_format!r}; "
                         f"choose from {sorted(_QUANT_FORMATS)}")
    if weight_bits == 4:
        _check_int4_cpu_support()

    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    pre_path = fp32_path.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(fp32_path), str(pre_path))  # shape inference + optimization
        quantize_static(
            str(pre_path), str(int8_path),
            TorchCalibrationReader(calib_loader, num_batches),
            quant_format=fmt,
            per_channel=True,
            weight_type=weight_type,
            activation_type=QuantType.QUInt8,  # U8S8/U8S4: recommended for x86-64 CPU EP
        )
    finally:
        pre_path.unlink(missing_ok=True)
    return int8_path
