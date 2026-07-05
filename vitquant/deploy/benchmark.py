import statistics
import time
from pathlib import Path

import numpy as np


def model_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6


def benchmark_onnx(path: str | Path, runs: int = 50, warmup: int = 10,
                   img_size: int = 224) -> float:
    """Median single-image CPU latency in milliseconds."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    feed = {"input": x}
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)
