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
