import timm
import torch
import pytest

from vitquant.models.loader import load_model


def test_missing_checkpoint_raises_with_instructions(tmp_path):
    missing = tmp_path / "nope.pth"
    with pytest.raises(FileNotFoundError, match="pretrained=True"):
        load_model("deit_tiny_patch16_224", missing)


def test_loads_local_checkpoint(tmp_path):
    src = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    ckpt = tmp_path / "deit_tiny.pth"
    torch.save(src.state_dict(), ckpt)
    model, data_cfg = load_model("deit_tiny_patch16_224", ckpt)
    assert not model.training
    assert data_cfg["input_size"] == (3, 224, 224)
    assert torch.equal(model.head.weight, src.head.weight)


def test_unwraps_model_key(tmp_path):
    src = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    ckpt = tmp_path / "deit_tiny_wrapped.pth"
    torch.save({"model": src.state_dict()}, ckpt)  # facebookresearch/deit format
    model, _ = load_model("deit_tiny_patch16_224", ckpt)
    assert torch.equal(model.head.weight, src.head.weight)
