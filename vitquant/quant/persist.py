"""Save/load the quantization state of a calibrated model, so an expensive
calibration + AdaRound run can be reused for inference (e.g. on an unlabeled
test set) without redoing it.

No algorithm in this framework mutates the original fp32 weights — MSE clip
writes scale/zero_point, AdaRound writes round_offset, SmoothQuant writes
smooth_scale, all registered buffers applied on the fly in forward. So the
artifact only needs those buffers (a few MB), not a copy of the checkpoint:

    save:  quantized_state.pt (quantizer buffers) + quant_meta.json (how to
           rebuild: model name/family/checkpoint path + quant config + switches)
    load:  load the fp32 checkpoint -> convert with the same qconfig ->
           pour the buffers back -> set_quantizing(True). No calibration data,
           no labels, bit-identical to the model that was saved."""
import json
from pathlib import Path

import torch
from torch import nn

from vitquant.quant.fake_quant import set_quantizing

STATE_FILE = "quantized_state.pt"
META_FILE = "quant_meta.json"

# Everything quantization-related lives under a FakeQuantize submodule
# (weight_fq/input_fq/a_fq/b_fq: scale, zero_point, round_offset, observer
# stats) or is a QuantLinear's SmoothQuant scale.
_QUANT_KEY_MARKERS = ("_fq.",)
_QUANT_KEY_SUFFIXES = (".smooth_scale",)


def quant_state_dict(model: nn.Module) -> dict:
    """The subset of state_dict holding quantization state only."""
    return {k: v for k, v in model.state_dict().items()
            if any(m in k for m in _QUANT_KEY_MARKERS)
            or k.endswith(_QUANT_KEY_SUFFIXES)}


def save_quantized(model: nn.Module, out_dir, meta: dict) -> Path:
    """Write quantized_state.pt + quant_meta.json into out_dir. `meta` must
    carry whatever load needs to rebuild the structure — for the SAM families:
    model, family, checkpoint, quant (the config dict); algorithm switches are
    informational. AdaRound's round_offset is per-weight but binary {0,1}, so
    it's stored as uint8 (4x smaller; load casts it back to float)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {}
    for k, v in quant_state_dict(model).items():
        v = v.cpu()
        if k.endswith(".round_offset") and v.numel():
            v = v.to(torch.uint8)
        state[k] = v
    torch.save(state, out / STATE_FILE)
    (out / META_FILE).write_text(json.dumps(meta, indent=2))
    return out / STATE_FILE


def read_meta(artifact_dir) -> dict:
    return json.loads((Path(artifact_dir) / META_FILE).read_text())


def load_quant_state(model: nn.Module, artifact_dir) -> int:
    """Pour a saved quantization state into a freshly CONVERTED model (same
    qconfig as at save time) and switch it to quantizing. Buffers are assigned
    (not copied) because convert registers them as empty placeholders whose
    shapes only materialize at calibration. Returns the number of buffers
    applied; raises if a key doesn't correspond to a registered buffer, which
    catches structure/qconfig mismatches early."""
    state = torch.load(Path(artifact_dir) / STATE_FILE,
                       map_location="cpu", weights_only=True)
    for key, value in state.items():
        mod_path, buf_name = key.rsplit(".", 1)
        mod = model.get_submodule(mod_path)
        if buf_name not in mod._buffers:
            raise KeyError(
                f"{key}: not a registered buffer — the model structure does not "
                "match the artifact (was it converted with the same quant config?)")
        if buf_name == "round_offset" and value.dtype == torch.uint8:
            value = value.float()  # stored compact; forward does float arithmetic
        mod._buffers[buf_name] = value
    set_quantizing(model, True)
    return len(state)


def load_quantized_sam(artifact_dir, device=None):
    """One-call loader for the SAM families: rebuild fp32 model + processor
    from the meta, convert, load the quantization state, move to device.
    Returns (model, processor, meta)."""
    from vitquant.models.sam_loader import (load_sam3_concept_model,
                                            load_sam3_model, load_sam_model)
    from vitquant.quant.qconfig import qconfig_from_dict
    from vitquant.quant.sam_convert import convert_sam_vision_encoder

    meta = read_meta(artifact_dir)
    loaders = {"sam": load_sam_model, "sam3": load_sam3_model,
               "sam3_concept": load_sam3_concept_model}
    model, processor = loaders[meta["family"]](meta["model"], meta["checkpoint"])
    convert_sam_vision_encoder(model, qconfig_from_dict(meta["quant"]))
    n = load_quant_state(model, artifact_dir)
    if device is not None:
        model = model.to(device)
    print(f"    loaded quantized model: {meta['family']} "
         f"W{meta['quant']['weight']['bits']}A{meta['quant']['activation']['bits']}, "
         f"{n} quantizer buffers")
    return model.eval(), processor, meta
