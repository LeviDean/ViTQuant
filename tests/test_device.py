import torch

from vitquant.utils.device import resolve_device


def test_explicit_spec_wins():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_returns_available_device():
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")
