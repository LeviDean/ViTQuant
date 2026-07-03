from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.quant.fake_quant import freeze_qparams, set_observing, set_quantizing


def calibrate(model: nn.Module, loader: DataLoader, device: torch.device,
              num_batches: Optional[int] = None,
              progress: Optional[Callable[[int], None]] = None) -> nn.Module:
    """Run calibration data through the model to collect activation statistics,
    then freeze qparams and switch every FakeQuantize into quantizing mode."""
    model = model.eval().to(device)
    set_observing(model, True)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            model(x.to(device))
            if progress is not None:
                progress(i)
    set_observing(model, False)
    freeze_qparams(model)  # raises CalibrationError if an observer saw no data
    set_quantizing(model, True)
    return model
