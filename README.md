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
.venv/bin/pip install -e .
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

> macOS + python.org Python: if the Imagenette download fails with
> `SSL: CERTIFICATE_VERIFY_FAILED`, prefix the command with
> `SSL_CERT_FILE=$(.venv/bin/python -m certifi)`.

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

## Quantization schemes (W8A8, W4A8, ...)

Weight and activation bit-width are independent, set per config under `quant`:

```yaml
quant:
  weight: {bits: 4, symmetric: true, per_channel: true, observer: minmax}
  activation: {bits: 8, symmetric: false, per_channel: false, observer: moving_avg}
```

- **Research layer**: any bit-width works out of the box (fake-quant kernel is
  generic over `bits`) — just edit the config and rerun.
- **Delivery layer**: `weight_bits` of 8 or 4 are supported for real ORT export
  (`configs/deit_tiny_w4a8.yaml` is a ready-made W4A8 example). Activation stays
  8-bit (`QUInt8`, ORT's recommended x86-64 CPU EP type). W4 weights give a real,
  larger on-disk compression (~6.5x vs ~3.6x for W8), but ORT's CPU EP has no
  native int4 matmul kernel, so **W4A8 latency does not reflect a real hardware
  speedup** — treat the accuracy and size numbers as the meaningful W4A8 result,
  and the INT8 latency number as the real deployment speedup reference.

  > **x86-64 CPUs without AVX-512**: ORT's int4 contrib kernel (`com.microsoft`
  > domain) needs AVX-512 and otherwise crashes the process with `Illegal
  > instruction (core dumped)` — an OS-level SIGILL, not a catchable Python
  > error. `quantize_onnx(weight_bits=4)` now checks `/proc/cpuinfo` for
  > `avx512f` on x86-64 Linux first and raises a clear `RuntimeError` instead.
  > Apple Silicon (arm64) is unaffected and uses a different kernel path. If
  > your server lacks AVX-512, use `weight_bits=8` for the real delivery
  > layer — the research layer's simulated W4A8 accuracy numbers stay valid
  > either way, since they never touch ORT.

## ORT graph-optimization crashes on some CPUs (cloud/virtualized x86-64)

Confirmed on an AMD EPYC cloud VM (nested virtualization, no AVX-512): ORT's
`sess.run()` on a **plain W8A8** quantized graph crashed with the same
`Illegal instruction (core dumped)` SIGILL — not int4-specific after all.
Bisected by trying each `onnxruntime.GraphOptimizationLevel` in turn:
`ORT_DISABLE_ALL` and `ORT_ENABLE_BASIC` ran fine; `ORT_ENABLE_EXTENDED`
(ORT's default is `ORT_ENABLE_ALL`, which includes everything `EXTENDED`
does) crashed — ORT fuses the QuantizeLinear/DequantizeLinear pattern into a
specialized kernel at that level, and that fused kernel is what crashes on
this hardware.

Every ORT session in the delivery layer (`evaluate_onnx`, `benchmark_onnx`)
now goes through `vitquant/utils/ort_session.py::create_cpu_session`, which
accepts an optional `graph_optimization_level`. Default is `None` (ORT's own
default — fastest on healthy hardware, unchanged behavior). If you hit this
crash, add to your config:

```yaml
onnx:
  graph_optimization_level: basic   # disable | basic | extended | all
```

`scripts/run_all.py` and `scripts/evaluate.py --onnx` both read this key
automatically. This trades a bit of ORT's fusion-based speed for stability;
size/accuracy numbers are unaffected (optimization level only changes how
the graph executes, not what it computes).

**`graph_optimization_level: basic` alone gives back correctness but not
speed**: with fusion disabled, the QDQ graph runs as
dequantize→fp32-compute→requantize for every quantized op, i.e. full fp32
compute *plus* quantize/dequantize overhead — slower than plain fp32, not
faster. To get a real kernel without depending on the (crashing) fusion pass,
quantize with `QuantFormat.QOperator` instead of the default `QDQ`:
`quantize_static` bakes the quantized op (e.g. `QLinearMatMul`) into the
graph directly, so the fast kernel is already present regardless of the
runtime optimization level.

```yaml
onnx:
  graph_optimization_level: basic   # keep the confirmed-safe level
  quant_format: qoperator           # qdq (default) | qoperator
```

Try `quant_format: qoperator` together with `graph_optimization_level: basic`
first (both individually confirmed not to crash) — if this gets real speedup
back, no need to touch `graph_optimization_level` further. Only experiment
with raising it back toward `all` if you want to test whether QOperator format
itself avoids the crash independent of the optimization level, since that
hasn't been confirmed and could still crash.
