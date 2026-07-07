from typing import Callable, Optional

import torch
from torch import nn

from vitquant.quant.fake_quant import (calibrate_weights, freeze_qparams,
                                       set_observing, set_quantizing)


def calibrate_sam(model: nn.Module, samples: list[dict], device: torch.device,
                  progress: Optional[Callable[[int], None]] = None) -> nn.Module:
    """Run calibration image+prompt pairs through the model to collect
    activation statistics (only vision_encoder has FakeQuantize modules after
    convert_sam_vision_encoder — prompt_encoder/mask_decoder are untouched
    fp32 and just run normally), then freeze qparams and switch every
    FakeQuantize into quantizing mode. Mirrors vitquant.quant.calibrate but
    adapted for SAM's dict-of-named-tensors calling convention. Each dict in
    `samples` (as produced by build_sam_inputs with return_tensors="pt") must
    have tensor values only, since every value is moved to `device`.

    Weight quantizers are calibrated up front (data-independent); the sample
    pass only collects activation statistics."""
    model = model.eval().to(device)
    calibrate_weights(model)  # data-independent: freeze weight quantizers first
    set_observing(model, True)  # only the (unfrozen) activation quantizers observe
    with torch.no_grad():
        for i, inputs in enumerate(samples):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
            if progress is not None:
                progress(i)
    set_observing(model, False)
    freeze_qparams(model)
    set_quantizing(model, True)
    return model
