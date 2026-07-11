# ViTQuant

A simulated-quantization research framework for accuracy analysis and **NPU
pre-research** — ViT image classification (timm) and SAM segmentation
(HuggingFace `transformers`, vision-encoder-only). It is not a CPU-deployment
tool: there is no ONNX export, no runtime, and no latency benchmark. The
framework's only job is to answer "how much accuracy does a given
quantization scheme cost?" before you commit to real hardware.

- **Research layer** (`vitquant/quant/`): custom fake-quant kernel (pluggable
  Observers, per-tensor/per-channel, symmetric/asymmetric, any bit-width) for
  accuracy analysis, per-block sensitivity, and quantization-scheme ablations.
  Simulated quantization — numerically quantizes and dequantizes back to fp32
  and runs fp32 compute, so it measures accuracy precisely but produces no
  real speedup and no real on-disk compression.

> **文档**：上手使用看 [USER_GUIDE.md](USER_GUIDE.md)（使用手册，中文）；实验目标与设置看
> [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)；实现细节看 [PROJECT_REPORT.md](PROJECT_REPORT.md)。

## Why simulation only (NPU pre-research)

The actual deployment target for this project is an edge NPU, not a CPU. Real
INT8 latency and real on-disk compression are meaningless numbers to chase
right now: they depend on a specific NPU's runtime, kernels, and toolchain,
none of which exist yet. What *is* transferable to any future target device:

- **Accuracy retention** under a given quantization scheme (bit-width,
  granularity, symmetry) vs the fp32 baseline.
- **Per-block sensitivity** — which layers/blocks lose the most accuracy when
  quantized, useful for deciding what to leave at higher precision on a real
  target.
- **Scheme ablation** — comparing observer types, symmetric vs asymmetric,
  per-tensor vs per-channel, independent of any runtime.
- **Theoretical compression** — the arithmetic ratio from bit-width alone
  (e.g. int8 weights are 1/4 the size of fp32), which holds regardless of
  target hardware.

Real latency requires the actual target NPU, its SDK, and its kernels — none
of that can be simulated meaningfully on a dev machine, so it's out of scope
here. The framework's `QConfig` is pluggable specifically so that once an NPU
is chosen, you can match its exact required scheme (per-tensor vs
per-channel, symmetric vs asymmetric, bit-width) and get an accuracy read
before touching the vendor toolchain. The W4A8 config
(`configs/deit_tiny_w4a8.yaml`) exists for this reason: many edge NPUs prefer
4-bit weights, so it's worth knowing the accuracy cost of dropping to 4-bit
ahead of time.

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
.venv/bin/python scripts/run_classification.py --config configs/deit_tiny.yaml

# full evaluation on the server
python scripts/run_classification.py --config configs/vit_base.yaml
```

`run_classification.py` runs the fp32 baseline, simulated INT8 accuracy, per-block
sensitivity, a mixed-precision (top-K block protection) trade-off sweep, a
quantization-scheme ablation matrix, the theoretical weight compression ratio
(arithmetic, from bit-width), AND a per-image visualization grid — one command
gives you the metric report and the visual examples together. Flags:
`--skip-sensitivity`, `--skip-mixed-precision`, `--mixed-precision-ks 0,1,2,3`,
`--skip-ablation`, `--no-qualitative`, `--qualitative-samples N` (default 30).
Outputs land in `outputs/<model>/`: `report.md` (accuracy /
theoretical-compression / sensitivity / mixed-precision / ablation tables),
`results.json`, and `qualitative_grid.png` (the actual sample photos with
fp32-vs-quantized predictions annotated, flipped cases sorted first and titled
red) plus `qualitative.json`/`.md`.

The mixed-precision sweep turns the sensitivity ranking into deployable plans:
it keeps the K most-sensitive blocks at FP32 and quantizes the rest at the
config's bit-width, measuring each K to produce an accuracy-vs-compression
curve. Each row's top-1 is measured, not predicted from summing per-block drops
(quantization error across blocks is not additive). It needs the sensitivity
ranking, so it's skipped when `--skip-sensitivity` is set, and it's only
meaningful at low bit-width (e.g. `configs/deit_tiny_w4a8.yaml`) — at W8A8 every
block is insensitive and the curve is flat.

Single experiments: `scripts/quantize.py --config ...` (one simulated INT8
experiment: convert -> calibrate -> evaluate, writes `quantize_result.json`),
`scripts/evaluate.py --config ...` (fp32 baseline only).

Standalone visualization (already produced by `run_classification.py`; use this to
regenerate it on its own or with a different sample count):
`scripts/qualitative.py --config ... [--num-samples N]`.

## Quantization schemes (W8A8, W4A8, ...)

Weight and activation bit-width are independent, set per config under `quant`:

```yaml
quant:
  weight: {bits: 4, symmetric: true, per_channel: true, observer: minmax}
  activation: {bits: 8, symmetric: false, per_channel: false, observer: moving_avg}
```

Any bit-width works out of the box — the fake-quant kernel is generic over
`bits` — just edit the config and rerun. `configs/deit_tiny_w4a8.yaml` is a
ready-made W4A8 example: it measures the accuracy cost of 4-bit weights
(relevant for edge NPUs that prefer 4-bit weights), purely as a simulated
accuracy number. There is no real W4A8 runtime path in this project.

## SAM: vision-encoder-only quantization (self-consistency IoU)

Alongside the classification ViT pipeline above, this framework also quantizes
the **image encoder (ViT backbone)** of Segment Anything (SAM), via
HuggingFace `transformers.SamModel`. Only `model.vision_encoder` is converted
to simulated INT8 — `prompt_encoder` and `mask_decoder` stay fp32. Since SAM
has no single "top-1 accuracy" metric, evaluation instead measures
**self-consistency**: for the same image + point prompts, how similar are the
fp32 model's predicted masks to the simulated-quantized model's predicted
masks (IoU, per point and mask hypothesis)? By default each eval image gets an
n×n **grid of point prompts** (`data.prompt_grid`, default 4×4; 1 = single
center point), so the IoU measures quantization's effect on segmenting objects
across the whole image, not just the center one — the quantized vision encoder
still runs once per image, only the light fp32 prompt-encoder/mask-decoder
repeat per point. High IoU means quantization didn't change what the model
segments. This is **not** a ground-truth accuracy benchmark (e.g. mIoU against
COCO) — that's a deliberate scope decision, not an oversight.

### Weights (manual, offline)

Same "never downloads" policy as the classification models, but a SAM/HF
checkpoint is a local **directory** (`config.json` + weight files produced by
`save_pretrained`), not a single `.pth` file:

```bash
python -c "from transformers import SamModel, SamProcessor; \
SamModel.from_pretrained('facebook/sam-vit-base').save_pretrained('weights/sam-vit-base'); \
SamProcessor.from_pretrained('facebook/sam-vit-base').save_pretrained('weights/sam-vit-base')"
```

Copy (or generate directly on the target machine) the resulting
`weights/sam-vit-base/` directory before running offline.

### Usage

```bash
python scripts/run_sam.py --config configs/sam_vit_b.yaml
```

One command writes the metric report AND the visual examples to
`outputs/sam_vit_b/`:

- `results.json` — model/device info, weight/activation bit-width, simulated
  self-consistency IoU (`mean_iou`, `min_iou`, `per_sample_iou`), and per-block
  sensitivity.
- `report.md` — a human-readable report (IoU table + theoretical weight
  compression table + per-block sensitivity table + mixed-precision trade-off
  table), printed to stdout at the end too.
- `qualitative_sam_grid.png` (+ `qualitative_sam.json`) — fp32 masks
  (translucent fills) vs quantized masks (thin boundary lines) over the actual
  images (prompt points marked; in grid mode each point gets its own color),
  sorted worst-IoU-first, low-IoU cases titled red.

The per-block sensitivity sweep quantizes one vision-encoder block at a time
(`vision_encoder.layers.N` / `patch_embed` / `neck`) and reports how much the
masks change, measured by self-consistency IoU drop (SAM has no top-1). The
mixed-precision sweep then protects the K most-sensitive blocks at FP32 and
quantizes the rest, giving an IoU-vs-compression curve. Both share their sweep
core with the classification pipeline — only the metric (IoU vs top-1) differs.

Flags: `--skip-sensitivity`, `--skip-mixed-precision`,
`--mixed-precision-ks 0,1,2,3`, `--no-qualitative`, `--qualitative-samples N`
(default 8). The standalone `scripts/qualitative_sam.py --config ...
[--num-samples N]` regenerates just the grid.

### Current scope boundaries

These are deliberate staged decisions for this phase, not bugs:

- **Only the vision encoder is quantized.** `mask_decoder` (and
  `prompt_encoder`) are not touched and remain fp32.
- **Evaluation is self-consistency only, not ground-truth mIoU** (e.g. against
  COCO). The IoU numbers here measure agreement between fp32 and
  simulated-quantized masks, not segmentation quality against a labeled
  benchmark.
- **No real deployment path.** There is no ONNX export or runtime for SAM in
  this project; all numbers are simulated/theoretical.
</content>
