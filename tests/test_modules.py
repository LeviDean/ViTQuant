import torch
from torch import nn

from vitquant.quant.qconfig import QConfig
from vitquant.quant.fake_quant import freeze_qparams, set_observing
from vitquant.quant.modules import QuantConv2d, QuantLinear, QuantMatMul


def _calibrate(mod, *inputs):
    set_observing(mod, True)
    with torch.no_grad():
        mod(*inputs)
    set_observing(mod, False)
    freeze_qparams(mod)


def test_quant_linear_fp32_equivalent_before_freeze():
    lin = nn.Linear(16, 8)
    qlin = QuantLinear.from_float(lin, QConfig())
    x = torch.randn(4, 16)
    assert torch.allclose(qlin(x), lin(x), atol=1e-6)


def test_quant_linear_close_after_quantize():
    lin = nn.Linear(64, 32)
    qlin = QuantLinear.from_float(lin, QConfig())
    x = torch.randn(8, 64)
    _calibrate(qlin, x)
    ref, out = lin(x), qlin(x)
    assert not torch.equal(out, ref)
    assert (out - ref).abs().mean() < 0.1 * ref.abs().mean()


def test_quant_linear_shares_weight_storage():
    lin = nn.Linear(4, 4)
    qlin = QuantLinear.from_float(lin, QConfig())
    assert qlin.weight is lin.weight and qlin.bias is lin.bias


def test_quant_conv2d():
    conv = nn.Conv2d(3, 8, kernel_size=4, stride=4)
    qconv = QuantConv2d.from_float(conv, QConfig())
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(qconv(x), conv(x), atol=1e-6)
    _calibrate(qconv, x)
    assert (qconv(x) - conv(x)).abs().mean() < 0.1 * conv(x).abs().mean()


def test_quant_matmul():
    mm = QuantMatMul(QConfig())
    a, b = torch.randn(2, 4, 8), torch.randn(2, 8, 4)
    assert torch.allclose(mm(a, b), a @ b, atol=1e-6)
    _calibrate(mm, a, b)
    assert ((mm(a, b) - a @ b).abs().mean()) < 0.1 * (a @ b).abs().mean()
