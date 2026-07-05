import pytest
import torch

from vitquant.quant.qconfig import TensorQConfig
from vitquant.quant.observers import (CalibrationError, MinMaxObserver,
                                      MovingAvgMinMaxObserver, PercentileObserver,
                                      build_observer, qrange)


def test_qrange():
    assert qrange(8, symmetric=True) == (-127, 127)
    assert qrange(8, symmetric=False) == (-128, 127)
    assert qrange(4, symmetric=True) == (-7, 7)


def test_minmax_per_tensor_symmetric():
    obs = MinMaxObserver(TensorQConfig(symmetric=True, per_channel=False))
    obs(torch.tensor([-2.0, 0.5, 1.0]))
    obs(torch.tensor([-1.0, 4.0]))  # running min/max: [-2, 4]
    scale, zp = obs.compute_qparams()
    assert torch.allclose(scale, torch.tensor([4.0 / 127]))
    assert torch.equal(zp, torch.zeros(1))


def test_minmax_asymmetric_zero_point():
    obs = MinMaxObserver(TensorQConfig(symmetric=False))
    obs(torch.tensor([0.0, 10.0]))
    scale, zp = obs.compute_qparams()
    assert torch.allclose(scale, torch.tensor([10.0 / 255]))
    assert zp.item() == -128  # min 0 maps to qmin


def test_minmax_per_channel():
    obs = MinMaxObserver(TensorQConfig(symmetric=True, per_channel=True, ch_axis=0))
    obs(torch.tensor([[-1.0, 1.0], [-8.0, 2.0]]))
    scale, _ = obs.compute_qparams()
    assert scale.shape == (2,)
    assert torch.allclose(scale, torch.tensor([1.0 / 127, 8.0 / 127]))


def test_moving_avg_updates_smoothly():
    obs = MovingAvgMinMaxObserver(TensorQConfig(symmetric=True))
    obs(torch.tensor([-1.0, 1.0]))
    obs(torch.tensor([-11.0, 11.0]))  # EMA: 1 + 0.1*(11-1) = 2.0
    assert torch.allclose(obs.max_val, torch.tensor([2.0]))


def test_percentile_clips_outliers():
    obs = PercentileObserver(TensorQConfig(symmetric=True, percentile=0.99))
    x = torch.cat([torch.linspace(-1, 1, 1000), torch.tensor([100.0])])
    obs(x)
    assert obs.max_val.item() < 100.0


def test_percentile_rejects_per_channel():
    obs = PercentileObserver(TensorQConfig(per_channel=True))
    with pytest.raises(NotImplementedError):
        obs(torch.randn(4, 4))


def test_compute_before_observe_raises():
    obs = MinMaxObserver(TensorQConfig())
    with pytest.raises(CalibrationError):
        obs.compute_qparams()


def test_build_observer():
    assert isinstance(build_observer(TensorQConfig(observer="percentile")), PercentileObserver)
