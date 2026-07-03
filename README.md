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
.venv/bin/pytest -m "not slow"   # fast unit tests
.venv/bin/pytest                 # includes ORT quantization integration
```

All tests use random weights and random tensors — no downloads required.
