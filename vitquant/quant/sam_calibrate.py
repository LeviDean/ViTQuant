import torch
from torch import nn

from vitquant.quant.fake_quant import freeze_qparams, set_observing, set_quantizing


def calibrate_sam(model: nn.Module, samples: list[dict], device: torch.device) -> nn.Module:
    """Run calibration image+prompt pairs through the model to collect
    activation statistics (only vision_encoder has FakeQuantize modules after
    convert_sam_vision_encoder — prompt_encoder/mask_decoder are untouched
    fp32 and just run normally), then freeze qparams and switch every
    FakeQuantize into quantizing mode. Mirrors vitquant.quant.calibrate but
    adapted for SAM's dict-of-named-tensors calling convention."""
    model = model.eval().to(device)
    set_observing(model, True)
    with torch.no_grad():
        for inputs in samples:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
    set_observing(model, False)
    freeze_qparams(model)
    set_quantizing(model, True)
    return model
