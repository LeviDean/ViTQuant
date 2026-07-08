from typing import Callable, Optional

import torch
from torch import nn

from vitquant.quant.calibrate import run_calibration


def _feed_sam_inputs(model: nn.Module, inputs: dict, device: torch.device) -> None:
    """SAM: a batch is a dict of named tensors (image + point prompt), all moved
    to device and passed as keyword args."""
    model(**{k: v.to(device) for k, v in inputs.items()})


def calibrate_sam(model: nn.Module, samples: list[dict], device: torch.device,
                  progress: Optional[Callable[[int], None]] = None) -> nn.Module:
    """Calibrate a converted SAM model over image+prompt samples. Only
    vision_encoder has FakeQuantize modules after convert_sam_vision_encoder —
    prompt_encoder/mask_decoder are untouched fp32 and just run normally. Shares
    the observing→freeze→quantize skeleton with the classification pipeline via
    run_calibration; the only difference is the dict-of-named-tensors feed. Each
    dict in `samples` (from build_sam_inputs with return_tensors="pt") must have
    tensor values only, since every value is moved to `device`. Weight quantizers
    are calibrated up front (data-independent); the sample pass only collects
    activation statistics."""
    return run_calibration(model, samples, device, _feed_sam_inputs, progress=progress)
