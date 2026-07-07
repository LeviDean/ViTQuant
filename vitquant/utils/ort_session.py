from pathlib import Path
from typing import Optional

_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def create_cpu_session(onnx_path: str | Path, graph_optimization_level: Optional[str] = None):
    """CPU-only ORT InferenceSession. graph_optimization_level caps ORT's graph
    fusion passes: confirmed on an AMD EPYC cloud VM (nested virtualization,
    inconsistent CPUID reporting) that ORT_ENABLE_EXTENDED/ORT_ENABLE_ALL fuse
    quantized ops into a kernel that crashes the whole process with SIGILL
    (Illegal instruction) — not a catchable Python exception, so this can't be
    guarded with try/except. If you hit that, set this to "basic" (verified
    safe on that hardware) via your config's `onnx.graph_optimization_level`.
    Leave unset to use ORT's own default (ORT_ENABLE_ALL), which is faster on
    healthy hardware."""
    import onnxruntime as ort

    if graph_optimization_level is None:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    try:
        level_name = _LEVELS[graph_optimization_level]
    except KeyError:
        raise ValueError(f"Unknown graph_optimization_level={graph_optimization_level!r}; "
                         f"choose from {sorted(_LEVELS)}")
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level_name)
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
