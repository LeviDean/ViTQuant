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

Qualitative check: `scripts/qualitative.py --config ... [--num-samples N] [--onnx path]`
runs fp32 vs quantized on real sample images and prints/saves per-image
predictions and confidence, flagging cases where quantization actually flips
the top-1 prediction — complements the aggregate accuracy tables with
concrete examples. Writes `qualitative.json`/`.md` (tables) and
`qualitative_grid.png` (the actual sample photos with predictions annotated,
flipped cases sorted first and titled in red) to the config's `output_dir`.

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

## SAM: vision-encoder-only quantization (self-consistency IoU)

Alongside the classification ViT pipeline above, this framework also quantizes
the **image encoder (ViT backbone)** of Segment Anything (SAM), via
HuggingFace `transformers.SamModel`. Only `model.vision_encoder` is converted
to INT8 — `prompt_encoder` and `mask_decoder` stay fp32. Since SAM has no
single "top-1 accuracy" metric, evaluation instead measures **self-consistency**:
for the same image + point prompt, how similar are the fp32 model's predicted
masks to the quantized model's predicted masks (IoU, per mask hypothesis)?
High IoU means quantization didn't change what the model segments. This is
**not** a ground-truth accuracy benchmark (e.g. mIoU against COCO) — that's a
deliberate current-phase scope decision, not an oversight.

Like the classification pipeline, SAM now has both layers:

- **Research layer**: custom fake-quant kernel on the vision encoder,
  simulated INT8 self-consistency IoU.
- **Delivery layer** (`vitquant/deploy/`): real fp32 ONNX export of the
  vision encoder + ONNX Runtime static INT8 (QDQ), real on-disk compression,
  real CPU latency, and real (non-simulated) self-consistency IoU computed by
  running the actual ORT INT8 graph and feeding its image embeddings into the
  fp32 prompt_encoder/mask_decoder.

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
python scripts/quantize_sam.py --config configs/sam_vit_b.yaml
```

This now runs both layers in one pass and writes to `outputs/sam_vit_b/`:

- `sam_consistency.json` — simulated (research-layer) IoU only, kept for
  back-compat (`mean_iou`, `min_iou`, `per_sample_iou`).
- `sam_vision_encoder.fp32.onnx` / `sam_vision_encoder.int8.onnx` — the real
  ONNX export and its ORT-quantized INT8 counterpart.
- `results.json` — everything: simulated IoU, real ORT IoU, real on-disk
  size, real CPU latency.
- `report.md` — a human-readable report (mirroring `run_all.py`'s report
  style) with IoU / size / latency tables, printed to stdout at the end too.

### Current scope boundaries

These are deliberate staged decisions for this phase, not bugs — follow-ups
for a future phase:

- **Only the vision encoder is quantized.** `mask_decoder` (and
  `prompt_encoder`) are not touched and remain fp32 in both the simulated and
  real delivery-layer pipelines.
- **Evaluation is self-consistency only, not ground-truth mIoU** (e.g. against
  COCO). The IoU numbers here (both simulated and real) measure agreement
  between fp32 and quantized masks, not segmentation quality against a
  labeled benchmark.

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

**`graph_optimization_level: basic` gives back correctness but not speed**:
with fusion disabled, the QDQ graph runs as dequantize→fp32-compute→requantize
for every quantized op — full fp32 compute *plus* quantize/dequantize
overhead, slower than plain fp32. We tried working around this with
`QuantFormat.QOperator` (`onnx.quant_format: qoperator`), which bakes the
quantized op directly into the graph at quantize time instead of relying on
runtime fusion — **also confirmed to crash with the same SIGILL**, at every
optimization level including `ORT_DISABLE_ALL`, and on two onnxruntime
versions (1.27.0 and 1.24.1). This rules out a fusion-pass or
version-specific bug: **on this class of hardware, ORT's int8 quantized
kernel itself cannot execute at all**, regardless of how the graph reaches
it. There is no further code-level workaround.

Practical takeaway for affected hardware: use `quant_format: qdq` (default)
with `graph_optimization_level: basic` — the only combination that doesn't
crash. The report will show a latency number for this, but it does **not**
reflect real int8 speed (it's the fp32-fallback path); `scripts/run_all.py`
detects this configuration and adds a warning note to the report so it isn't
mistaken for a real speedup. The accuracy and size/compression numbers remain
valid — only the latency comparison is unusable on this hardware. Getting a
genuine ORT CPU int8 speedup number requires different hardware (e.g. a
non-virtualized/physical x86-64 machine, or a different cloud instance).
