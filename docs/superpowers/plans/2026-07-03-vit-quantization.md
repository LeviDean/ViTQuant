# ViT Quantization Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-layer ViT INT8 quantization framework — a research layer (custom fake-quant kernel for accuracy analysis) plus a delivery layer (ONNX Runtime real INT8) — with a one-command evaluation report.

**Architecture:** Research layer replaces `nn.Linear`/`nn.Conv2d`/timm `Attention` with quant-aware wrappers holding pluggable Observers + FakeQuantize (STE). Delivery layer exports fp32 ONNX and applies ORT static QDQ quantization. `scripts/run_all.py` produces accuracy/size/latency/sensitivity/ablation report.

**Tech Stack:** Python ≥3.10, PyTorch, timm, torchvision (Imagenette), onnx, onnxruntime, pyyaml, pytest.

**Key constraints (from spec):**
- No network downloads for model weights: `timm.create_model(pretrained=False)` + local checkpoint. Clear error with download instructions when missing.
- Device auto-detect `cuda > mps > cpu`, overridable via config. No hardcoded devices/paths.
- All unit tests run with random weights and random tensors — no downloads needed on the dev machine.

**Spec:** `docs/superpowers/specs/2026-07-03-vit-quantization-design.md`

---

## File Structure

```
pyproject.toml
configs/{deit_tiny,vit_base}.yaml
vitquant/
├── __init__.py
├── utils/{__init__,device,config}.py
├── quant/{__init__,qconfig,observers,fake_quant,modules,convert,calibrate}.py
├── models/{__init__,loader}.py
├── data/{__init__,imagenette}.py
├── eval/{__init__,metrics,evaluate}.py
└── deploy/{__init__,export_onnx,quantize_ort,benchmark}.py
scripts/{quantize,evaluate,run_all}.py
tests/test_{device,qconfig,observers,fake_quant,modules,convert,calibrate,metrics,loader,data,evaluate,export,ort_quant,integration}.py
README.md
```

Naming contract (used consistently across all tasks):
`TensorQConfig`, `QConfig`, `qconfig_from_dict`, `qrange`, `CalibrationError`, `MinMaxObserver`, `MovingAvgMinMaxObserver`, `PercentileObserver`, `build_observer`, `fake_quantize`, `FakeQuantize`, `set_observing`, `set_quantizing`, `freeze_qparams`, `QuantLinear`, `QuantConv2d`, `QuantMatMul`, `QuantAttention`, `convert_vit`, `calibrate`, `load_model`, `IMAGENETTE_TO_IMAGENET1K`, `build_val_loader`, `build_calib_loader`, `topk_correct`, `AccuracyMeter`, `evaluate_torch`, `evaluate_onnx`, `block_sensitivity`, `export_fp32_onnx`, `quantize_onnx`, `TorchCalibrationReader`, `benchmark_onnx`, `model_size_mb`, `resolve_device`, `load_config`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `vitquant/__init__.py` and all package `__init__.py` files, `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "vitquant"
version = "0.1.0"
description = "ViT INT8 quantization framework: simulated quant research kernel + ONNX Runtime delivery"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "torchvision",
    "timm",
    "onnx",
    "onnxruntime",
    "pyyaml",
    "numpy",
]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["vitquant*"]

[tool.pytest.ini_options]
markers = ["slow: long-running integration tests (deselect with '-m \"not slow\"')"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
*.egg-info/
outputs/
weights/
data/
*.onnx
.DS_Store
```

- [ ] **Step 3: Create package skeleton**

```bash
mkdir -p vitquant/{utils,quant,models,data,eval,deploy} tests configs scripts
touch vitquant/__init__.py vitquant/utils/__init__.py vitquant/quant/__init__.py \
      vitquant/models/__init__.py vitquant/data/__init__.py vitquant/eval/__init__.py \
      vitquant/deploy/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create venv and install**

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Expected: installs torch/timm/onnx/onnxruntime without errors. (Use `.venv/bin/pytest` / `.venv/bin/python` in all subsequent steps.)

- [ ] **Step 5: Sanity check imports**

```bash
.venv/bin/python -c "import torch, timm, onnx, onnxruntime, torchvision; import vitquant; print('ok')"
```

Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore vitquant tests configs scripts
git commit -m "chore: project scaffolding with package skeleton"
```

---

### Task 2: Device resolution

**Files:**
- Create: `vitquant/utils/device.py`
- Test: `tests/test_device.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device.py
import torch

from vitquant.utils.device import resolve_device


def test_explicit_spec_wins():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_returns_available_device():
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vitquant.utils.device'`

- [ ] **Step 3: Write implementation**

```python
# vitquant/utils/device.py
import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec. "auto" picks cuda > mps > cpu."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_device.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/utils/device.py tests/test_device.py
git commit -m "feat: device auto-detection (cuda > mps > cpu)"
```

---

### Task 3: QConfig

**Files:**
- Create: `vitquant/quant/qconfig.py`
- Test: `tests/test_qconfig.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qconfig.py
from vitquant.quant.qconfig import QConfig, TensorQConfig, qconfig_from_dict


def test_defaults_match_spec():
    qc = QConfig()
    assert qc.weight.symmetric and qc.weight.per_channel and qc.weight.bits == 8
    assert not qc.activation.symmetric and not qc.activation.per_channel


def test_from_dict():
    qc = qconfig_from_dict({
        "weight": {"bits": 8, "symmetric": True, "per_channel": True, "observer": "minmax"},
        "activation": {"bits": 8, "symmetric": False, "per_channel": False, "observer": "moving_avg"},
    })
    assert qc.weight.per_channel is True
    assert qc.activation.observer == "moving_avg"


def test_frozen():
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        QConfig().weight.bits = 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_qconfig.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/quant/qconfig.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TensorQConfig:
    """Quantization config for one tensor kind (weight or activation)."""
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    ch_axis: int = 0
    observer: str = "minmax"  # minmax | moving_avg | percentile
    percentile: float = 0.999  # only used by PercentileObserver


@dataclass(frozen=True)
class QConfig:
    """Default per spec: weight per-channel symmetric, activation per-tensor asymmetric."""
    weight: TensorQConfig = field(
        default_factory=lambda: TensorQConfig(symmetric=True, per_channel=True))
    activation: TensorQConfig = field(
        default_factory=lambda: TensorQConfig(symmetric=False, per_channel=False,
                                              observer="moving_avg"))


def qconfig_from_dict(d: dict) -> QConfig:
    return QConfig(weight=TensorQConfig(**d["weight"]),
                   activation=TensorQConfig(**d["activation"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_qconfig.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/quant/qconfig.py tests/test_qconfig.py
git commit -m "feat: pluggable QConfig (bits/symmetry/granularity/observer)"
```

---

### Task 4: Observers

**Files:**
- Create: `vitquant/quant/observers.py`
- Test: `tests/test_observers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observers.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_observers.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/quant/observers.py
import torch
from torch import nn

from vitquant.quant.qconfig import TensorQConfig


class CalibrationError(RuntimeError):
    """Raised when qparams are requested before any statistics were collected."""


def qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer range. Symmetric uses restricted range so zero_point can be 0."""
    if symmetric:
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


class ObserverBase(nn.Module):
    def __init__(self, cfg: TensorQConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("min_val", torch.empty(0))
        self.register_buffer("max_val", torch.empty(0))

    @property
    def has_stats(self) -> bool:
        return self.min_val.numel() > 0

    def _reduce(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel: min/max over all dims except ch_axis. Per-tensor: global."""
        x = x.detach().float()
        if self.cfg.per_channel:
            flat = x.transpose(0, self.cfg.ch_axis).flatten(1)
            return flat.min(dim=1).values, flat.max(dim=1).values
        return x.min().reshape(1), x.max().reshape(1)

    def compute_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_stats:
            raise CalibrationError(
                f"{type(self).__name__} has no statistics; run calibration first.")
        qmin, qmax = qrange(self.cfg.bits, self.cfg.symmetric)
        # Range must include 0 so that real zero is exactly representable.
        min_val = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        if self.cfg.symmetric:
            scale = torch.maximum(max_val.abs(), min_val.abs()) / qmax
            scale = torch.clamp(scale, min=1e-12)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-12)
            zero_point = torch.clamp(torch.round(qmin - min_val / scale),
                                     float(qmin), float(qmax))
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self._reduce(x)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = torch.minimum(self.min_val, lo)
            self.max_val = torch.maximum(self.max_val, hi)
        return x


class MovingAvgMinMaxObserver(ObserverBase):
    MOMENTUM = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self._reduce(x)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = self.min_val + self.MOMENTUM * (lo - self.min_val)
            self.max_val = self.max_val + self.MOMENTUM * (hi - self.max_val)
        return x


class PercentileObserver(ObserverBase):
    MAX_SAMPLES = 1_000_000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.per_channel:
            raise NotImplementedError("PercentileObserver supports per-tensor only")
        flat = x.detach().float().flatten()
        if flat.numel() > self.MAX_SAMPLES:  # torch.quantile has input size limits
            idx = torch.randint(flat.numel(), (self.MAX_SAMPLES,), device=flat.device)
            flat = flat[idx]
        lo = torch.quantile(flat, 1.0 - self.cfg.percentile).reshape(1)
        hi = torch.quantile(flat, self.cfg.percentile).reshape(1)
        if not self.has_stats:
            self.min_val, self.max_val = lo, hi
        else:
            self.min_val = torch.minimum(self.min_val, lo)
            self.max_val = torch.maximum(self.max_val, hi)
        return x


_OBSERVERS = {
    "minmax": MinMaxObserver,
    "moving_avg": MovingAvgMinMaxObserver,
    "percentile": PercentileObserver,
}


def build_observer(cfg: TensorQConfig) -> ObserverBase:
    try:
        return _OBSERVERS[cfg.observer](cfg)
    except KeyError:
        raise ValueError(f"Unknown observer '{cfg.observer}'. Choose from {list(_OBSERVERS)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_observers.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/quant/observers.py tests/test_observers.py
git commit -m "feat: MinMax/MovingAvg/Percentile observers with per-channel support"
```

---

### Task 5: FakeQuantize

**Files:**
- Create: `vitquant/quant/fake_quant.py`
- Test: `tests/test_fake_quant.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fake_quant.py
import torch

from vitquant.quant.qconfig import TensorQConfig
from vitquant.quant.fake_quant import (FakeQuantize, fake_quantize, freeze_qparams,
                                       set_observing, set_quantizing)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fake_quant.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/quant/fake_quant.py
import torch
from torch import nn

from vitquant.quant.observers import build_observer, qrange
from vitquant.quant.qconfig import TensorQConfig


def fake_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                  qmin: int, qmax: int, ch_axis: int | None = None) -> torch.Tensor:
    """Quantize -> dequantize in fp32, with straight-through estimator for gradients."""
    if ch_axis is not None:
        shape = [1] * x.dim()
        shape[ch_axis] = -1
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    dq = (q - zero_point) * scale
    return x + (dq - x).detach()  # STE: forward=dq, backward=identity


class FakeQuantize(nn.Module):
    """Modes: observing (collect stats, pass through), quantizing (apply fake-quant),
    neither (identity). freeze() computes qparams from the observer."""

    def __init__(self, cfg: TensorQConfig):
        super().__init__()
        self.cfg = cfg
        self.observer = build_observer(cfg)
        self.observing = False
        self.quantizing = False
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("zero_point", torch.empty(0))

    def freeze(self) -> None:
        self.scale, self.zero_point = self.observer.compute_qparams()
        self.observing = False
        self.quantizing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self.observer(x)
        if self.quantizing:
            qmin, qmax = qrange(self.cfg.bits, self.cfg.symmetric)
            ch_axis = self.cfg.ch_axis if self.cfg.per_channel else None
            return fake_quantize(x, self.scale.to(x.device),
                                 self.zero_point.to(x.device), qmin, qmax, ch_axis)
        return x


def set_observing(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.observing = enabled


def set_quantizing(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.quantizing = enabled


def freeze_qparams(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.freeze()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fake_quant.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/quant/fake_quant.py tests/test_fake_quant.py
git commit -m "feat: FakeQuantize with STE and model-wide mode helpers"
```

---

### Task 6: Quantized modules (Linear / Conv2d / MatMul)

**Files:**
- Create: `vitquant/quant/modules.py` (QuantAttention added in Task 7)
- Test: `tests/test_modules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modules.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_modules.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/quant/modules.py
import torch
from torch import nn
from torch.nn import functional as F

from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.qconfig import QConfig


class QuantLinear(nn.Linear):
    """nn.Linear with fake-quant on input activation and weight.
    Construct via from_float(); shares parameter storage with the source module."""

    @classmethod
    def from_float(cls, mod: nn.Linear, qconfig: QConfig) -> "QuantLinear":
        new = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new.weight = mod.weight
        new.bias = mod.bias
        new.input_fq = FakeQuantize(qconfig.activation)
        new.weight_fq = FakeQuantize(qconfig.weight)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.input_fq(x), self.weight_fq(self.weight), self.bias)


class QuantConv2d(nn.Conv2d):
    """nn.Conv2d with fake-quant on input and weight (used for ViT patch embed)."""

    @classmethod
    def from_float(cls, mod: nn.Conv2d, qconfig: QConfig) -> "QuantConv2d":
        new = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                  stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                  groups=mod.groups, bias=mod.bias is not None)
        new.weight = mod.weight
        new.bias = mod.bias
        new.input_fq = FakeQuantize(qconfig.activation)
        new.weight_fq = FakeQuantize(qconfig.weight)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(self.input_fq(x), self.weight_fq(self.weight), self.bias)


class QuantMatMul(nn.Module):
    """Fake-quantized a @ b for attention score/value matmuls.
    Both inputs are activations -> per-tensor activation config."""

    def __init__(self, qconfig: QConfig):
        super().__init__()
        self.a_fq = FakeQuantize(qconfig.activation)
        self.b_fq = FakeQuantize(qconfig.activation)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.a_fq(a) @ self.b_fq(b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_modules.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/quant/modules.py tests/test_modules.py
git commit -m "feat: QuantLinear/QuantConv2d/QuantMatMul modules"
```

---

### Task 7: QuantAttention + model conversion

**Files:**
- Modify: `vitquant/quant/modules.py` (append QuantAttention)
- Create: `vitquant/quant/convert.py`
- Test: `tests/test_convert.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_convert.py
import timm
import torch
from timm.models.vision_transformer import Attention

from vitquant.quant.convert import convert_vit
from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.modules import QuantAttention, QuantConv2d, QuantLinear
from vitquant.quant.qconfig import QConfig


def _tiny_vit():
    torch.manual_seed(0)
    return timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()


def test_quant_attention_fp32_equivalent():
    torch.manual_seed(0)
    attn = Attention(dim=64, num_heads=4).eval()
    qattn = QuantAttention.from_float(attn, QConfig()).eval()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        assert torch.allclose(qattn(x), attn(x), atol=1e-5)


def test_convert_replaces_modules():
    model = convert_vit(_tiny_vit(), QConfig())
    types = [type(m) for m in model.modules()]
    assert QuantAttention in types
    assert QuantConv2d in types  # patch embed
    assert QuantLinear in types  # mlp
    assert not any(isinstance(m, Attention) for m in model.modules())


def test_convert_skips_head():
    model = convert_vit(_tiny_vit(), QConfig())
    assert type(model.head) is torch.nn.Linear  # classifier stays fp32


def test_converted_model_fp32_equivalent():
    ref = _tiny_vit()
    torch.manual_seed(0)
    qmodel = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                         QConfig()).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(qmodel(x), ref(x), atol=1e-4)


def test_converted_model_has_fake_quant():
    qmodel = convert_vit(_tiny_vit(), QConfig())
    n = sum(1 for m in qmodel.modules() if isinstance(m, FakeQuantize))
    assert n > 50  # 12 blocks x (qkv+proj+2 mlp) x 2 + matmuls + patch embed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_convert.py -v`
Expected: FAIL with `ImportError` (QuantAttention / convert_vit missing)

- [ ] **Step 3: Append QuantAttention to `vitquant/quant/modules.py`**

```python
# append to vitquant/quant/modules.py

class QuantAttention(nn.Module):
    """timm ViT Attention rewritten with explicit matmuls so the score (q@k^T)
    and context (attn@v) matmuls can be fake-quantized. qkv/proj become QuantLinear."""

    @classmethod
    def from_float(cls, attn: nn.Module, qconfig: QConfig) -> "QuantAttention":
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.num_heads = attn.num_heads
        new.head_dim = attn.head_dim
        new.scale = attn.scale
        new.qkv = QuantLinear.from_float(attn.qkv, qconfig)
        new.q_norm = attn.q_norm
        new.k_norm = attn.k_norm
        new.attn_drop = attn.attn_drop
        new.proj = QuantLinear.from_float(attn.proj, qconfig)
        new.proj_drop = attn.proj_drop
        new.qk_matmul = QuantMatMul(qconfig)
        new.av_matmul = QuantMatMul(qconfig)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)  # softmax stays fp32 per spec
        attn = self.attn_drop(attn)
        x = self.av_matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

- [ ] **Step 4: Write `vitquant/quant/convert.py`**

```python
# vitquant/quant/convert.py
from torch import nn
from timm.models.vision_transformer import Attention

from vitquant.quant.modules import QuantAttention, QuantConv2d, QuantLinear
from vitquant.quant.qconfig import QConfig

DEFAULT_SKIP = ("head",)  # classifier head stays fp32


def convert_vit(model: nn.Module, qconfig: QConfig,
                skip: tuple[str, ...] = DEFAULT_SKIP) -> nn.Module:
    """In-place replacement: Attention -> QuantAttention, Linear -> QuantLinear,
    Conv2d -> QuantConv2d. LayerNorm/Softmax/GELU are untouched (stay fp32)."""
    _convert(model, qconfig, skip, prefix="")
    return model


def _convert(module: nn.Module, qconfig: QConfig, skip: tuple[str, ...],
             prefix: str) -> None:
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if any(full == s or full.startswith(s + ".") for s in skip):
            continue
        if isinstance(child, Attention):
            setattr(module, name, QuantAttention.from_float(child, qconfig))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear.from_float(child, qconfig))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d.from_float(child, qconfig))
        else:
            _convert(child, qconfig, skip, full)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_convert.py -v`
Expected: 5 PASS. If `Attention(dim=64, num_heads=4)` signature errors on the installed timm version, check `timm.models.vision_transformer.Attention.__init__` and adjust the test constructor args only (not the implementation).

- [ ] **Step 6: Commit**

```bash
git add vitquant/quant/modules.py vitquant/quant/convert.py tests/test_convert.py
git commit -m "feat: QuantAttention and fp32-equivalent ViT conversion"
```

---

### Task 8: Calibration

**Files:**
- Create: `vitquant/quant/calibrate.py`
- Test: `tests/test_calibrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.qconfig import QConfig


def _loader(n=4, bs=2):
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.zeros(n, dtype=torch.long)), batch_size=bs)


def test_calibrate_freezes_all_fake_quants():
    model = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                        QConfig())
    calibrate(model, _loader(), device=torch.device("cpu"))
    fqs = [m for m in model.modules() if isinstance(m, FakeQuantize)]
    assert all(m.quantizing and not m.observing for m in fqs)
    assert all(m.scale.numel() > 0 for m in fqs)


def test_calibrate_respects_num_batches():
    model = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                        QConfig())
    seen = []
    loader = _loader(n=8, bs=2)
    calibrate(model, loader, device=torch.device("cpu"), num_batches=1,
              progress=seen.append)
    assert seen == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/quant/calibrate.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_calibrate.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/quant/calibrate.py tests/test_calibrate.py
git commit -m "feat: calibration driver (observe -> freeze -> quantize)"
```

---

### Task 9: Metrics

**Files:**
- Create: `vitquant/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import torch

from vitquant.eval.metrics import AccuracyMeter, topk_correct


def test_topk_correct():
    logits = torch.tensor([[0.1, 0.9, 0.0],   # pred 1, target 1: top1 hit
                           [0.8, 0.1, 0.1],   # pred 0, target 2: top1 miss, top2 miss
                           [0.4, 0.5, 0.1]])  # pred 1, target 0: top1 miss, top2 hit
    counts = topk_correct(logits, torch.tensor([1, 2, 0]), ks=(1, 2))
    assert counts == {1: 1, 2: 2}


def test_accuracy_meter_accumulates():
    m = AccuracyMeter(ks=(1, 5))
    logits = torch.zeros(4, 10)
    logits[torch.arange(4), torch.tensor([0, 1, 2, 3])] = 1.0
    m.update(logits, torch.tensor([0, 1, 2, 9]))  # 3/4 top1
    m.update(logits, torch.tensor([0, 1, 2, 3]))  # 4/4 top1
    assert abs(m.top1 - 7 / 8) < 1e-9
    assert m.total == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/eval/metrics.py
import torch


def topk_correct(logits: torch.Tensor, targets: torch.Tensor,
                 ks: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    """Number of samples whose target is within the top-k predictions, per k."""
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.unsqueeze(1))
    return {k: int(correct[:, :min(k, maxk)].any(dim=1).sum()) for k in ks}


class AccuracyMeter:
    def __init__(self, ks: tuple[int, ...] = (1, 5)):
        self.ks = ks
        self.correct = {k: 0 for k in ks}
        self.total = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        counts = topk_correct(logits, targets, self.ks)
        for k in self.ks:
            self.correct[k] += counts[k]
        self.total += targets.numel()

    @property
    def top1(self) -> float:
        return self.correct[1] / max(self.total, 1)

    @property
    def top5(self) -> float:
        return self.correct[5] / max(self.total, 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/eval/metrics.py tests/test_metrics.py
git commit -m "feat: top-k accuracy metrics"
```

---

### Task 10: Model loader (offline) + Imagenette data

**Files:**
- Create: `vitquant/models/loader.py`, `vitquant/data/imagenette.py`
- Test: `tests/test_loader.py`, `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loader.py
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
```

```python
# tests/test_data.py
from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K


def test_class_mapping():
    # Imagenette wordnet-id sorted order -> ImageNet-1k indices
    assert IMAGENETTE_TO_IMAGENET1K == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    assert len(set(IMAGENETTE_TO_IMAGENET1K)) == 10
    assert IMAGENETTE_TO_IMAGENET1K == sorted(IMAGENETTE_TO_IMAGENET1K)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_loader.py tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the loader**

```python
# vitquant/models/loader.py
from pathlib import Path

import timm
import torch

DOWNLOAD_HINT = """Checkpoint not found: {path}

This framework never downloads weights. On a machine with network access run:

    python -c "import timm, torch; m = timm.create_model('{name}', pretrained=True); \\
torch.save(m.state_dict(), '{filename}')"

then copy '{filename}' to {path}."""


def load_model(name: str, checkpoint: str | Path) -> tuple[torch.nn.Module, dict]:
    """Build a timm architecture offline and load weights from a local checkpoint.
    Returns (model.eval(), timm data config for building transforms)."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            DOWNLOAD_HINT.format(path=path, name=name, filename=path.name))
    model = timm.create_model(name, pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:  # facebookresearch/deit format
        state = state["model"]
    model.load_state_dict(state)
    data_cfg = timm.data.resolve_model_data_config(model)
    return model.eval(), data_cfg
```

- [ ] **Step 4: Write the data module**

```python
# vitquant/data/imagenette.py
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Imagenette

# Imagenette classes in wordnet-id sorted order (tench, English springer, cassette
# player, chain saw, church, French horn, garbage truck, gas pump, golf ball,
# parachute) -> their ImageNet-1k class indices.
IMAGENETTE_TO_IMAGENET1K = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def _dataset(root: str | Path, split: str, data_cfg: dict, download: bool) -> Imagenette:
    root = Path(root)
    need_download = download and not (root / "imagenette2-160").exists()
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return Imagenette(str(root), split=split, size="160px",
                      download=need_download, transform=transform)


def build_val_loader(root: str | Path, data_cfg: dict, batch_size: int = 64,
                     num_workers: int = 4, download: bool = True) -> DataLoader:
    ds = _dataset(root, "val", data_cfg, download)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def build_calib_loader(root: str | Path, data_cfg: dict, calib_images: int = 256,
                       batch_size: int = 32, num_workers: int = 4,
                       download: bool = True) -> DataLoader:
    """Fixed random subset of the train split (seeded for reproducibility)."""
    ds = _dataset(root, "train", data_cfg, download)
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=gen)[:calib_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_loader.py tests/test_data.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add vitquant/models/loader.py vitquant/data/imagenette.py tests/test_loader.py tests/test_data.py
git commit -m "feat: offline model loader and Imagenette data pipeline"
```

---

### Task 11: Evaluation loops + block sensitivity

**Files:**
- Create: `vitquant/eval/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K
from vitquant.eval.evaluate import block_sensitivity, evaluate_torch
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import QConfig

CPU = torch.device("cpu")


def _loader(n=4, bs=2):
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.randint(0, 10, (n,))), batch_size=bs)


def test_evaluate_torch_returns_fractions():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    res = evaluate_torch(model, _loader(), IMAGENETTE_TO_IMAGENET1K, CPU)
    assert set(res) == {"top1", "top5"}
    assert 0.0 <= res["top1"] <= res["top5"] <= 1.0


def test_evaluate_perfect_model():
    class Oracle(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 1000)
            logits[:, IMAGENETTE_TO_IMAGENET1K[3]] = 1.0
            return logits

    x = torch.randn(4, 3, 224, 224)
    loader = DataLoader(TensorDataset(x, torch.full((4,), 3)), batch_size=2)
    assert evaluate_torch(Oracle(), loader, IMAGENETTE_TO_IMAGENET1K, CPU)["top1"] == 1.0


def test_block_sensitivity_groups_and_restores():
    model = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                        QConfig())
    calibrate(model, _loader(), device=CPU)
    result = block_sensitivity(model, _loader(), IMAGENETTE_TO_IMAGENET1K, CPU)
    assert "patch_embed" in result and "blocks.0" in result and len(result) >= 13
    # after the sweep every FakeQuantize must be back in quantizing mode
    from vitquant.quant.fake_quant import FakeQuantize
    assert all(m.quantizing for m in model.modules() if isinstance(m, FakeQuantize))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/eval/evaluate.py
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitquant.eval.metrics import AccuracyMeter
from vitquant.quant.fake_quant import FakeQuantize, set_quantizing


@torch.no_grad()
def evaluate_torch(model: nn.Module, loader: DataLoader, class_indices: list[int],
                   device: torch.device, max_batches: Optional[int] = None) -> dict:
    """Top-1/top-5 on Imagenette: slice the 1000-class logits down to the 10
    Imagenette classes, then argmax against the 0-9 dataset labels."""
    model = model.eval().to(device)
    meter = AccuracyMeter()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(x.to(device))[:, class_indices]
        meter.update(logits.cpu(), y)
    return {"top1": meter.top1, "top5": meter.top5}


def evaluate_onnx(onnx_path: str | Path, loader: DataLoader,
                  class_indices: list[int], max_batches: Optional[int] = None) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meter = AccuracyMeter()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = sess.run(None, {"input": x.numpy()})[0]
        meter.update(torch.from_numpy(out)[:, class_indices], y)
    return {"top1": meter.top1, "top5": meter.top5}


def _fq_groups(model: nn.Module) -> dict[str, list[FakeQuantize]]:
    """Group FakeQuantize modules by top-level component; blocks are split
    per-block ("blocks.0", "blocks.1", ...)."""
    groups: dict[str, list[FakeQuantize]] = {}
    for name, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            parts = name.split(".")
            key = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
            groups.setdefault(key, []).append(m)
    return groups


def block_sensitivity(model: nn.Module, loader: DataLoader, class_indices: list[int],
                      device: torch.device, max_batches: Optional[int] = None) -> dict:
    """Quantize one block at a time (rest fp32); report top-1 drop vs full fp32.
    Returns {group_name: top1_drop} sorted most-sensitive first.
    Requires a calibrated model; leaves it fully quantizing afterwards."""
    groups = _fq_groups(model)
    set_quantizing(model, False)
    base = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
    drops = {}
    for key, fqs in groups.items():
        for m in fqs:
            m.quantizing = True
        acc = evaluate_torch(model, loader, class_indices, device, max_batches)["top1"]
        drops[key] = base - acc
        for m in fqs:
            m.quantizing = False
    set_quantizing(model, True)
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: 3 PASS (sensitivity test takes ~1-2 min on CPU: 15 groups x 2 batches of deit_tiny)

- [ ] **Step 5: Commit**

```bash
git add vitquant/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: torch/onnx evaluation and per-block sensitivity analysis"
```

---

### Task 12: ONNX export with parity check

**Files:**
- Create: `vitquant/deploy/export_onnx.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
import timm
import torch

from vitquant.deploy.export_onnx import export_fp32_onnx


def test_export_and_parity(tmp_path):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    out = export_fp32_onnx(model, tmp_path / "deit_tiny.onnx")
    assert out.exists()


def test_export_supports_dynamic_batch(tmp_path):
    import onnxruntime as ort
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    out = export_fp32_onnx(model, tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for bs in (1, 3):
        y = sess.run(None, {"input": torch.randn(bs, 3, 224, 224).numpy()})[0]
        assert y.shape == (bs, 1000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# vitquant/deploy/export_onnx.py
from pathlib import Path

import numpy as np
import onnx
import torch


def export_fp32_onnx(model: torch.nn.Module, out_path: str | Path,
                     img_size: int = 224, opset: int = 17) -> Path:
    """Export the *original fp32* model (not the fake-quant one) to ONNX,
    validate with onnx.checker, and verify numerical parity against PyTorch."""
    import onnxruntime as ort

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # legacy exporter: stable for timm ViT + ORT quantization tooling
    )
    onnx.checker.check_model(str(out_path))

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    if not np.allclose(ref, got, atol=1e-4):
        raise RuntimeError(f"ONNX parity check failed: max diff {max_diff:.2e} > 1e-4")
    return out_path
```

Note: if the installed torch version rejects the `dynamo` kwarg, drop that single kwarg — everything else stays.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add vitquant/deploy/export_onnx.py tests/test_export.py
git commit -m "feat: fp32 ONNX export with checker and numerical parity gate"
```

---

### Task 13: ORT INT8 quantization + benchmark

**Files:**
- Create: `vitquant/deploy/quantize_ort.py`, `vitquant/deploy/benchmark.py`
- Test: `tests/test_ort_quant.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ort_quant.py
import pytest
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.deploy.benchmark import benchmark_onnx, model_size_mb
from vitquant.deploy.export_onnx import export_fp32_onnx
from vitquant.deploy.quantize_ort import TorchCalibrationReader, quantize_onnx


def _loader(n=4, bs=2):
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.zeros(n, dtype=torch.long)), batch_size=bs)


def test_calibration_reader_yields_then_none():
    reader = TorchCalibrationReader(_loader(n=4, bs=2), num_batches=2)
    batches = [reader.get_next(), reader.get_next()]
    assert all(b is not None and "input" in b for b in batches)
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


@pytest.mark.slow
def test_quantize_shrinks_model(tmp_path):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    fp32 = export_fp32_onnx(model, tmp_path / "m.onnx")
    int8 = quantize_onnx(fp32, tmp_path / "m.int8.onnx", _loader(), num_batches=1)
    assert int8.exists()
    assert model_size_mb(int8) < 0.5 * model_size_mb(fp32)  # ~4x expected
    lat = benchmark_onnx(int8, runs=3, warmup=1)
    assert lat > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ort_quant.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the quantizer**

```python
# vitquant/deploy/quantize_ort.py
from pathlib import Path
from typing import Optional

from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader


class TorchCalibrationReader(CalibrationDataReader):
    """Feeds batches from a torch DataLoader to ORT static quantization."""

    def __init__(self, loader: DataLoader, num_batches: Optional[int] = None,
                 input_name: str = "input"):
        self._batches = []
        for i, (x, _) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            self._batches.append({input_name: x.numpy()})
        self._it = iter(self._batches)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._batches)


def quantize_onnx(fp32_path: str | Path, int8_path: str | Path,
                  calib_loader: DataLoader,
                  num_batches: Optional[int] = None) -> Path:
    """ORT static INT8 quantization: QDQ format, per-channel weights (matches the
    research-layer default: weight per-channel symmetric int8)."""
    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    pre_path = fp32_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre_path))  # shape inference + optimization
    quantize_static(
        str(pre_path), str(int8_path),
        TorchCalibrationReader(calib_loader, num_batches),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,  # U8S8: recommended for x86-64 CPU EP
    )
    pre_path.unlink(missing_ok=True)
    return int8_path
```

- [ ] **Step 4: Write the benchmark**

```python
# vitquant/deploy/benchmark.py
import statistics
import time
from pathlib import Path

import numpy as np


def model_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6


def benchmark_onnx(path: str | Path, runs: int = 50, warmup: int = 10,
                   img_size: int = 224) -> float:
    """Median single-image CPU latency in milliseconds."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    feed = {"input": x}
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ort_quant.py -v`
Expected: 2 PASS (slow test takes ~1-2 min: preprocess + quantize deit_tiny)

- [ ] **Step 6: Commit**

```bash
git add vitquant/deploy/quantize_ort.py vitquant/deploy/benchmark.py tests/test_ort_quant.py
git commit -m "feat: ORT static INT8 quantization and latency/size benchmark"
```

---

### Task 14: Config loading + YAML configs

**Files:**
- Create: `vitquant/utils/config.py`, `configs/deit_tiny.yaml`, `configs/vit_base.yaml`
- Test: extend `tests/test_qconfig.py`

- [ ] **Step 1: Write the failing test (append to tests/test_qconfig.py)**

```python
# append to tests/test_qconfig.py
from pathlib import Path

from vitquant.utils.config import load_config


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name: deit_tiny_patch16_224\ndevice: auto\n")
    cfg = load_config(p)
    assert cfg["model"]["name"] == "deit_tiny_patch16_224"


def test_shipped_configs_parse():
    for name in ("deit_tiny", "vit_base"):
        cfg = load_config(Path("configs") / f"{name}.yaml")
        for key in ("model", "data", "quant", "eval", "benchmark", "device", "output_dir"):
            assert key in cfg, f"{name}.yaml missing '{key}'"
        from vitquant.quant.qconfig import qconfig_from_dict
        qconfig_from_dict(cfg["quant"])  # must build without error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_qconfig.py -v`
Expected: new tests FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write config loader**

```python
# vitquant/utils/config.py
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Write `configs/deit_tiny.yaml`**

```yaml
# Development config: small model for fast iteration on the dev machine.
model:
  name: deit_tiny_patch16_224
  checkpoint: weights/deit_tiny_patch16_224.pth

data:
  root: data
  download: true          # set false on offline servers with data pre-copied
  batch_size: 64
  num_workers: 4
  calib_images: 256
  calib_batch_size: 32

quant:
  weight: {bits: 8, symmetric: true, per_channel: true, observer: minmax}
  activation: {bits: 8, symmetric: false, per_channel: false, observer: moving_avg}

eval:
  max_batches: null       # null = full val set; set small int for smoke runs

benchmark:
  runs: 50
  warmup: 10

device: auto              # auto | cpu | cuda | mps
output_dir: outputs/deit_tiny
```

- [ ] **Step 5: Write `configs/vit_base.yaml`**

```yaml
# Server config: full evaluation target.
model:
  name: vit_base_patch16_224
  checkpoint: weights/vit_base_patch16_224.pth

data:
  root: data
  download: true
  batch_size: 64
  num_workers: 8
  calib_images: 256
  calib_batch_size: 32

quant:
  weight: {bits: 8, symmetric: true, per_channel: true, observer: minmax}
  activation: {bits: 8, symmetric: false, per_channel: false, observer: moving_avg}

eval:
  max_batches: null

benchmark:
  runs: 50
  warmup: 10

device: auto
output_dir: outputs/vit_base
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_qconfig.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add vitquant/utils/config.py configs tests/test_qconfig.py
git commit -m "feat: YAML config loading and shipped model configs"
```

---

### Task 15: Scripts (quantize / evaluate / run_all with report)

**Files:**
- Create: `scripts/quantize.py`, `scripts/evaluate.py`, `scripts/run_all.py`

These are thin CLI entry points over tested library code, so no TDD here; each is verified by `--help` and by the integration test in Task 16.

- [ ] **Step 1: Write `scripts/quantize.py` (single simulated-quant experiment)**

```python
#!/usr/bin/env python
"""Run one simulated-quantization experiment: convert -> calibrate -> evaluate."""
import argparse
import json
from pathlib import Path

from vitquant.data.imagenette import (IMAGENETTE_TO_IMAGENET1K, build_calib_loader,
                                      build_val_loader)
from vitquant.eval.evaluate import evaluate_torch
from vitquant.models.loader import load_model
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import qconfig_from_dict
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d = cfg["data"]

    model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"],
                           d["download"])
    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])

    print(f"[quantize] device={device}  model={cfg['model']['name']}")
    fp32 = evaluate_torch(model, val, IMAGENETTE_TO_IMAGENET1K, device,
                          cfg["eval"]["max_batches"])
    print(f"[quantize] fp32 top1={fp32['top1']:.4f} top5={fp32['top5']:.4f}")

    qmodel = convert_vit(model, qconfig_from_dict(cfg["quant"]))
    calibrate(qmodel, calib, device)
    int8 = evaluate_torch(qmodel, val, IMAGENETTE_TO_IMAGENET1K, device,
                          cfg["eval"]["max_batches"])
    print(f"[quantize] int8(sim) top1={int8['top1']:.4f} top5={int8['top5']:.4f}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    result = {"fp32": fp32, "int8_simulated": int8,
              "top1_drop": fp32["top1"] - int8["top1"], "qconfig": cfg["quant"]}
    (out / "quantize_result.json").write_text(json.dumps(result, indent=2))
    print(f"[quantize] wrote {out / 'quantize_result.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/evaluate.py` (baseline / ONNX accuracy)**

```python
#!/usr/bin/env python
"""Evaluate fp32 PyTorch baseline, or an ONNX model with --onnx."""
import argparse

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K, build_val_loader
from vitquant.eval.evaluate import evaluate_onnx, evaluate_torch
from vitquant.models.loader import load_model
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--onnx", default=None, help="evaluate this ONNX file instead")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    model, data_cfg = load_model(cfg["model"]["name"], cfg["model"]["checkpoint"])
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"],
                           d["download"])

    if args.onnx:
        res = evaluate_onnx(args.onnx, val, IMAGENETTE_TO_IMAGENET1K,
                            cfg["eval"]["max_batches"])
        print(f"[evaluate] onnx {args.onnx}: top1={res['top1']:.4f} top5={res['top5']:.4f}")
    else:
        device = resolve_device(cfg["device"])
        res = evaluate_torch(model, val, IMAGENETTE_TO_IMAGENET1K, device,
                             cfg["eval"]["max_batches"])
        print(f"[evaluate] fp32 torch: top1={res['top1']:.4f} top5={res['top5']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `scripts/run_all.py` (full matrix + markdown report)**

```python
#!/usr/bin/env python
"""One-command full evaluation: fp32 baseline, simulated INT8, block sensitivity,
ablation matrix, ORT real INT8 (accuracy + size + latency), markdown report."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from vitquant.data.imagenette import (IMAGENETTE_TO_IMAGENET1K, build_calib_loader,
                                      build_val_loader)
from vitquant.deploy.benchmark import benchmark_onnx, model_size_mb
from vitquant.deploy.export_onnx import export_fp32_onnx
from vitquant.deploy.quantize_ort import quantize_onnx
from vitquant.eval.evaluate import block_sensitivity, evaluate_onnx, evaluate_torch
from vitquant.models.loader import load_model
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import QConfig, qconfig_from_dict
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device

CLS = IMAGENETTE_TO_IMAGENET1K
ACC_GAP_WARN = 0.01  # spec: flag if |simulated - ORT real| top-1 gap > 1%


def ablation_qconfigs(base: QConfig) -> dict[str, QConfig]:
    return {
        "default (w:per-ch sym / a:per-t asym ema)": base,
        "weight per-tensor": replace(base, weight=replace(base.weight, per_channel=False)),
        "activation symmetric": replace(base, activation=replace(base.activation, symmetric=True)),
        "activation minmax obs": replace(base, activation=replace(base.activation, observer="minmax")),
        "activation percentile obs": replace(base, activation=replace(base.activation, observer="percentile")),
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d, name = cfg["data"], cfg["model"]["name"]
    ckpt = cfg["model"]["checkpoint"]
    max_b = cfg["eval"]["max_batches"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": name, "device": str(device)}

    model, data_cfg = load_model(name, ckpt)
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"], d["download"])
    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])
    base_qc = qconfig_from_dict(cfg["quant"])

    # 1. fp32 torch baseline
    print(f"[1/6] fp32 baseline on {device}")
    results["fp32_torch"] = evaluate_torch(model, val, CLS, device, max_b)

    # 2. simulated INT8 (research layer)
    print("[2/6] simulated INT8 (custom kernel)")
    qmodel = convert_vit(model, base_qc)
    calibrate(qmodel, calib, device)
    results["int8_simulated"] = evaluate_torch(qmodel, val, CLS, device, max_b)

    # 3. block sensitivity
    if not args.skip_sensitivity:
        print("[3/6] per-block sensitivity sweep")
        results["sensitivity"] = block_sensitivity(qmodel, val, CLS, device, max_b)
    else:
        print("[3/6] skipped")

    # 4. ablation matrix (each variant needs a fresh model: weights were shared in-place)
    if not args.skip_ablation:
        print("[4/6] ablation matrix")
        results["ablation"] = {}
        for label, qc in ablation_qconfigs(base_qc).items():
            m, _ = load_model(name, ckpt)
            qm = convert_vit(m, qc)
            calibrate(qm, calib, device)
            results["ablation"][label] = evaluate_torch(qm, val, CLS, device, max_b)
            print(f"    {label}: top1={results['ablation'][label]['top1']:.4f}")
    else:
        print("[4/6] skipped")

    # 5. delivery layer: ONNX export + ORT real INT8
    print("[5/6] ONNX export + ORT static INT8")
    fp32_model, _ = load_model(name, ckpt)  # fresh fp32 weights for export
    fp32_onnx = export_fp32_onnx(fp32_model, out / f"{name}.fp32.onnx")
    int8_onnx = quantize_onnx(fp32_onnx, out / f"{name}.int8.onnx", calib)
    results["fp32_onnx"] = evaluate_onnx(fp32_onnx, val, CLS, max_b)
    results["int8_onnx"] = evaluate_onnx(int8_onnx, val, CLS, max_b)
    results["size_mb"] = {"fp32": model_size_mb(fp32_onnx), "int8": model_size_mb(int8_onnx)}

    # 6. latency benchmark (CPU EP — the representative int8 speedup metric)
    print("[6/6] latency benchmark (ORT CPU EP)")
    b = cfg["benchmark"]
    results["latency_ms"] = {
        "fp32": benchmark_onnx(fp32_onnx, b["runs"], b["warmup"]),
        "int8": benchmark_onnx(int8_onnx, b["runs"], b["warmup"]),
    }

    (out / "results.json").write_text(json.dumps(results, indent=2))
    report = build_report(results, args)
    (out / "report.md").write_text(report)
    print(f"\nWrote {out / 'results.json'} and {out / 'report.md'}\n")
    print(report)


def build_report(r: dict, args) -> str:
    fp32_t, int8_s = r["fp32_torch"], r["int8_simulated"]
    fp32_o, int8_o = r["fp32_onnx"], r["int8_onnx"]
    sz, lat = r["size_mb"], r["latency_ms"]

    acc_rows = [
        ["FP32 (PyTorch)", pct(fp32_t["top1"]), pct(fp32_t["top5"]), "-"],
        ["FP32 (ONNX)", pct(fp32_o["top1"]), pct(fp32_o["top5"]),
         pct(fp32_t["top1"] - fp32_o["top1"])],
        ["INT8 simulated (custom kernel)", pct(int8_s["top1"]), pct(int8_s["top5"]),
         pct(fp32_t["top1"] - int8_s["top1"])],
        ["INT8 real (ORT QDQ)", pct(int8_o["top1"]), pct(int8_o["top5"]),
         pct(fp32_t["top1"] - int8_o["top1"])],
    ]
    parts = [f"# Quantization Report: {r['model']}",
             f"\nDevice (torch eval): `{r['device']}`\n",
             "## Accuracy\n",
             md_table(["Variant", "Top-1", "Top-5", "Top-1 drop vs FP32"], acc_rows)]

    gap = abs(int8_s["top1"] - int8_o["top1"])
    if gap > ACC_GAP_WARN:
        parts.append(f"\n> **WARNING**: simulated vs ORT-real top-1 gap {pct(gap)} "
                     f"exceeds {pct(ACC_GAP_WARN)} — investigate qscheme mismatch.")
    else:
        parts.append(f"\nSimulated vs ORT-real top-1 gap: {pct(gap)} (cross-validation OK)")

    parts.append("\n## Real Model Size (ONNX on disk)\n")
    parts.append(md_table(["Variant", "Size (MB)", "Compression"], [
        ["FP32", f"{sz['fp32']:.1f}", "1.0x"],
        ["INT8", f"{sz['int8']:.1f}", f"{sz['fp32'] / sz['int8']:.2f}x"]]))

    parts.append("\n## Real Latency (ORT CPU EP, batch=1, median)\n")
    parts.append(md_table(["Variant", "Latency (ms)", "Speedup"], [
        ["FP32", f"{lat['fp32']:.2f}", "1.0x"],
        ["INT8", f"{lat['int8']:.2f}", f"{lat['fp32'] / lat['int8']:.2f}x"]]))

    if "sensitivity" in r:
        parts.append("\n## Per-Block Sensitivity (top-1 drop when only that block is quantized)\n")
        parts.append(md_table(["Block", "Top-1 drop"],
                              [[k, pct(v)] for k, v in r["sensitivity"].items()]))

    if "ablation" in r:
        parts.append("\n## Ablation (simulated INT8)\n")
        parts.append(md_table(["Config", "Top-1", "Top-5"],
                              [[k, pct(v["top1"]), pct(v["top5"])]
                               for k, v in r["ablation"].items()]))
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify CLIs parse**

```bash
.venv/bin/python scripts/quantize.py --help
.venv/bin/python scripts/evaluate.py --help
.venv/bin/python scripts/run_all.py --help
```

Expected: each prints usage without error.

- [ ] **Step 5: Commit**

```bash
git add scripts
git commit -m "feat: quantize/evaluate/run_all CLI with markdown report"
```

---

### Task 16: Integration test + README

**Files:**
- Create: `tests/test_integration.py`, `README.md`

- [ ] **Step 1: Write the integration test (random weights, no downloads)**

```python
# tests/test_integration.py
import pytest
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from vitquant.data.imagenette import IMAGENETTE_TO_IMAGENET1K
from vitquant.deploy.benchmark import model_size_mb
from vitquant.deploy.export_onnx import export_fp32_onnx
from vitquant.deploy.quantize_ort import quantize_onnx
from vitquant.eval.evaluate import evaluate_onnx, evaluate_torch
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.qconfig import QConfig

CPU = torch.device("cpu")


def _loader(n=8, bs=4):
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, 3, 224, 224),
                                    torch.randint(0, 10, (n,))), batch_size=bs)


def test_research_layer_end_to_end():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    qmodel = convert_vit(model, QConfig())
    calibrate(qmodel, _loader(), device=CPU)
    res = evaluate_torch(qmodel, _loader(), IMAGENETTE_TO_IMAGENET1K, CPU)
    assert 0.0 <= res["top1"] <= 1.0


@pytest.mark.slow
def test_delivery_layer_end_to_end(tmp_path):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    fp32 = export_fp32_onnx(model, tmp_path / "m.onnx")
    int8 = quantize_onnx(fp32, tmp_path / "m.int8.onnx", _loader(), num_batches=1)
    res = evaluate_onnx(int8, _loader(), IMAGENETTE_TO_IMAGENET1K)
    assert 0.0 <= res["top1"] <= 1.0
    assert model_size_mb(int8) < 0.5 * model_size_mb(fp32)
```

- [ ] **Step 2: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: all PASS (use `-m "not slow"` for the fast subset)

- [ ] **Step 3: Write `README.md`**

````markdown
# ViTQuant

ViT INT8 quantization framework with two layers:

- **Research layer** (`vitquant/quant/`): custom fake-quant kernel (pluggable
  Observers, per-tensor/per-channel, symmetric/asymmetric) for accuracy analysis,
  per-block sensitivity, and ablations. Simulated quantization — measures accuracy,
  not speed.
- **Delivery layer** (`vitquant/deploy/`): fp32 ONNX export + ONNX Runtime static
  INT8 (QDQ). Real on-disk compression (~4x) and real CPU int8 latency.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Weights (manual, offline)

This framework never downloads model weights. On a machine with network access:

```bash
python -c "import timm, torch; m = timm.create_model('deit_tiny_patch16_224', pretrained=True); torch.save(m.state_dict(), 'deit_tiny_patch16_224.pth')"
python -c "import timm, torch; m = timm.create_model('vit_base_patch16_224', pretrained=True); torch.save(m.state_dict(), 'vit_base_patch16_224.pth')"
```

Copy the files into `weights/`. Imagenette (~100MB) downloads automatically on
first run; set `data.download: false` and pre-copy `data/imagenette2-160/` for
offline servers.

## Usage

```bash
# smoke run on dev machine (small model, subset of val)
.venv/bin/python scripts/run_all.py --config configs/deit_tiny.yaml

# full evaluation on the server
python scripts/run_all.py --config configs/vit_base.yaml
```

Outputs land in `outputs/<model>/`: `report.md` (accuracy / size / latency /
sensitivity / ablation tables) and `results.json`.

Single experiments: `scripts/quantize.py --config ...` (simulated INT8 only),
`scripts/evaluate.py --config ... [--onnx path]`.

## Tests

```bash
.venv/bin/pytest -m "not slow"   # fast unit tests (~2 min)
.venv/bin/pytest                 # includes ORT quantization integration (~5 min)
```

All tests use random weights and random tensors — no downloads required.
````

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "test: end-to-end integration tests; docs: README"
```

---

### Task 17: Smoke run on dev machine (manual gate)

No new code. Validates the full pipeline with real weights/data before server handoff.

- [ ] **Step 1: Obtain deit_tiny weights** (requires network once — user said they will download models manually; ask the user to place `weights/deit_tiny_patch16_224.pth`, or run the README snippet if network is available and permitted)

- [ ] **Step 2: Smoke run with tiny eval subset**

Temporarily set `eval.max_batches: 5` in `configs/deit_tiny.yaml`, then:

```bash
.venv/bin/python scripts/run_all.py --config configs/deit_tiny.yaml --skip-sensitivity --skip-ablation
```

Expected: completes all 6 stages; `outputs/deit_tiny/report.md` shows fp32 top-1 well above 10% (random = 10%), int8 close to fp32, int8 onnx ~4x smaller.

- [ ] **Step 3: Restore `eval.max_batches: null`, commit any fixes**

```bash
git add -A && git commit -m "fix: smoke-run fixes" # only if changes were needed
```

---

## Verification checklist (spec → tasks)

| Spec requirement | Task |
|---|---|
| Observers (MinMax/MovingAvg/Percentile, per-tensor/per-channel) | 4 |
| FakeQuantize + STE | 5 |
| QuantLinear / QuantConv2d / QuantMatMul | 6 |
| QuantAttention, conversion, LayerNorm/Softmax/GELU stay fp32, head skipped | 7 |
| Calibration + CalibrationError on missing stats | 4, 8 |
| Offline model loading with clear error | 10 |
| Imagenette + ImageNet-1k class mapping | 10 |
| top-1/top-5 evaluation | 9, 11 |
| Per-block sensitivity | 11 |
| ONNX export + checker + parity gate | 12 |
| ORT static QDQ INT8, per-channel weights | 13 |
| Real latency + size benchmark | 13 |
| Device auto-detect, config-driven paths | 2, 14 |
| Ablation matrix (granularity/symmetry/observer) | 15 |
| Sim-vs-real gap >1% warning in report | 15 |
| One-command report (markdown + JSON) | 15 |
| Unit + integration tests without downloads | all, 16 |
