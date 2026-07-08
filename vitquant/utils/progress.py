"""Progress-callback factories for long calibration / evaluation passes. On a
server terminal (nohup, log redirect) a multi-minute pass with no output looks
hung; these print a status line every ~20% of batches. Shared by both the
classification and SAM pipelines."""
from typing import Callable


def batch_progress(label: str, total: int) -> Callable[[int, int], None]:
    """Matches evaluate_torch's progress(i, n) signature."""
    step = max(1, total // 5)

    def cb(i: int, n: int) -> None:
        if (i + 1) % step == 0 or i + 1 == n:
            print(f"    {label}: batch {i + 1}/{n}")
    return cb


def calib_progress(label: str, total: int) -> Callable[[int], None]:
    """Matches calibrate()'s progress(i)-only signature."""
    inner = batch_progress(label, total)
    return lambda i: inner(i, total)
