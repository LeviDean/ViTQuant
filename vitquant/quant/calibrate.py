from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.quant.fake_quant import (calibrate_weights, freeze_qparams,
                                       set_observing, set_quantizing)


def calibrate(model: nn.Module, loader: DataLoader, device: torch.device,
              num_batches: Optional[int] = None,
              progress: Optional[Callable[[int], None]] = None) -> nn.Module:
    """Calibrate weights (data-independent) then activations (from the data),
    freeze qparams, and switch every FakeQuantize into quantizing mode.

    Weight qparams depend only on the weight tensors, so they're derived up front
    without touching the data. The calibration data pass then collects only
    activation statistics (with fp32 weights, matching standard PTQ), so
    activation ranges are numerically identical to observing everything at once —
    just without redundantly re-observing the static weights on every batch."""
    model = model.eval().to(device)
    calibrate_weights(model)  # data-independent: freeze weight quantizers first
    set_observing(model, True)  # only the (unfrozen) activation quantizers observe
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            model(x.to(device))
            if progress is not None:
                progress(i)
    set_observing(model, False)
    freeze_qparams(model)  # activations; raises CalibrationError if none saw data
    set_quantizing(model, True)
    return model
