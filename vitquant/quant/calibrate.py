from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn

from vitquant.quant.fake_quant import (calibrate_weights, freeze_qparams,
                                       set_observing, set_quantizing)

# A feed function runs one calibration batch through the model. It exists so the
# observing→freeze→quantize skeleton is written once and shared: classification
# batches are (x, label) tuples, SAM batches are dicts of named tensors.
FeedFn = Callable[[nn.Module, Any, torch.device], None]


def _feed_tensor_batch(model: nn.Module, batch: Any, device: torch.device) -> None:
    """Classification: batch is (inputs, labels); only inputs are needed."""
    x, _ = batch
    model(x.to(device))


def run_calibration(model: nn.Module, batches: Iterable[Any], device: torch.device,
                    feed: FeedFn = _feed_tensor_batch, num_batches: Optional[int] = None,
                    progress: Optional[Callable[[int], None]] = None) -> nn.Module:
    """Shared calibration skeleton. Calibrate weights (data-independent) then
    activations (from the data), freeze qparams, and switch every FakeQuantize
    into quantizing mode.

    Weight qparams depend only on the weight tensors, so they're derived up front
    without touching the data. The data pass then collects only activation
    statistics (with fp32 weights, matching standard PTQ), so activation ranges
    are numerically identical to observing everything at once — just without
    redundantly re-observing the static weights on every batch. `feed` adapts
    this to a task's batch shape."""
    model = model.eval().to(device)
    calibrate_weights(model)  # data-independent: freeze weight quantizers first
    set_observing(model, True)  # only the (unfrozen) activation quantizers observe
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if num_batches is not None and i >= num_batches:
                break
            feed(model, batch, device)
            if progress is not None:
                progress(i)
    set_observing(model, False)
    freeze_qparams(model)  # activations; raises CalibrationError if none saw data
    set_quantizing(model, True)
    return model


def calibrate(model: nn.Module, loader, device: torch.device,
              num_batches: Optional[int] = None,
              progress: Optional[Callable[[int], None]] = None) -> nn.Module:
    """Calibrate a classification model over a DataLoader of (inputs, labels)."""
    return run_calibration(model, loader, device, _feed_tensor_batch,
                           num_batches, progress)
