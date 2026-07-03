import pytest
import torch

from vitquant.quant.qconfig import TensorQConfig
from vitquant.quant.observers import CalibrationError
from vitquant.quant.fake_quant import (FakeQuantize, fake_quantize, freeze_qparams,
                                       set_observing, set_quantizing)


def test_fake_quantize_per_channel_conv_shaped():
    x = torch.randn(4, 3, 2, 2)  # Conv2d weight layout [out, in, kh, kw]
    scale = x.abs().amax(dim=(1, 2, 3)) / 127
    out = fake_quantize(x, scale, torch.zeros(4), -127, 127, ch_axis=0)
    assert (out - x).abs().max() <= scale.max().item() / 2 + 1e-6


def test_freeze_without_stats_raises():
    fq = FakeQuantize(TensorQConfig())
    with pytest.raises(CalibrationError):
        fq.freeze()


def test_fake_quantize_roundtrip_error_bounded():
    x = torch.randn(64)
    scale = torch.tensor([x.abs().max().item() / 127])
    out = fake_quantize(x, scale, torch.zeros(1), -127, 127)
    assert (out - x).abs().max() <= scale.item() / 2 + 1e-6


def test_fake_quantize_exact_grid_points():
    scale = torch.tensor([0.5])
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0])  # already on the grid
    out = fake_quantize(x, scale, torch.zeros(1), -127, 127)
    assert torch.allclose(out, x)


def test_fake_quantize_per_channel():
    x = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    scale = torch.tensor([1.0 / 127, 10.0 / 127])
    out = fake_quantize(x, scale, torch.zeros(2), -127, 127, ch_axis=0)
    assert torch.allclose(out, x, atol=0.05)


def test_ste_gradient_passthrough():
    x = torch.randn(8, requires_grad=True)
    out = fake_quantize(x, torch.tensor([0.1]), torch.zeros(1), -127, 127)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones(8))  # STE: gradient passes through


def test_module_identity_before_freeze():
    fq = FakeQuantize(TensorQConfig())
    x = torch.randn(4)
    assert torch.equal(fq(x), x)


def test_module_observe_freeze_quantize():
    fq = FakeQuantize(TensorQConfig(symmetric=True))
    fq.observing = True
    x = torch.randn(100)
    fq(x)
    fq.freeze()
    assert fq.quantizing and not fq.observing
    out = fq(x)
    assert not torch.equal(out, x)
    assert (out - x).abs().max() < x.abs().max() / 64  # int8 error is small


def test_model_wide_helpers():
    model = torch.nn.Sequential(FakeQuantize(TensorQConfig()))
    set_observing(model, True)
    model(torch.randn(2, 10))
    set_observing(model, False)
    freeze_qparams(model)
    assert model[0].quantizing
    set_quantizing(model, False)
    assert not model[0].quantizing
