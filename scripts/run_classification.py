#!/usr/bin/env python
"""Classification ViT research pipeline (one command): fp32 baseline, simulated
INT8 (fake-quant) accuracy, per-block sensitivity, a mixed-precision (top-K block
protection) trade-off sweep, a quantization-scheme ablation matrix, and a
per-image qualitative grid — plus a markdown report. Simulated quantization
measures the accuracy impact of a scheme (device-independent); it does not run a
real int8 kernel. The theoretical compression ratio is reported from the weight
bit-width (int8 weights are 1/4 of fp32).

The SAM counterpart is scripts/run_sam.py; reporting/progress/calibration
helpers are shared under vitquant/."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

from vitquant.data.imagenette import (IMAGENETTE_TO_IMAGENET1K, build_calib_loader,
                                      build_val_loader)
from vitquant.eval.evaluate import (block_sensitivity, evaluate_torch,
                                    mixed_precision_sweep)
from vitquant.eval.qualitative import save_classification_qualitative
from vitquant.eval.report import classification_report, write_outputs
from vitquant.models.loader import load_model
from vitquant.quant.adaround import adaround
from vitquant.quant.calibrate import calibrate
from vitquant.quant.convert import convert_vit
from vitquant.quant.smoothquant import smooth_quant
from vitquant.quant.qconfig import QConfig, qconfig_from_dict
from vitquant.utils.config import load_config
from vitquant.utils.device import resolve_device
from vitquant.utils.progress import batch_progress, calib_progress

# stdout is fully buffered (not line-buffered) unless attached to a real tty —
# common under nohup/log redirection/some containers — so progress prints can
# sit invisible until the process exits. Force line buffering unconditionally.
sys.stdout.reconfigure(line_buffering=True)

CLS = IMAGENETTE_TO_IMAGENET1K


def ablation_qconfigs(base: QConfig) -> dict[str, QConfig]:
    return {
        "default (w:per-ch sym / a:per-t asym ema)": base,
        "weight per-tensor": replace(base, weight=replace(base.weight, per_channel=False)),
        "activation symmetric": replace(base, activation=replace(base.activation, symmetric=True)),
        "activation minmax obs": replace(base, activation=replace(base.activation, observer="minmax")),
        "activation percentile obs": replace(base, activation=replace(base.activation, observer="percentile")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-mixed-precision", action="store_true",
                    help="skip the mixed-precision (top-K block protection) sweep")
    ap.add_argument("--mixed-precision-ks", type=str, default=None,
                    help="comma-separated K values for the sweep (default: 0,1,2,3,4 + full-fp32)")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--no-qualitative", action="store_true",
                    help="skip the per-image visualization grid")
    ap.add_argument("--qualitative-samples", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    d, name = cfg["data"], cfg["model"]["name"]
    ckpt = cfg["model"]["checkpoint"]
    max_b = cfg["eval"]["max_batches"]
    out = Path(cfg["output_dir"])
    base_qc = qconfig_from_dict(cfg["quant"])
    results: dict = {"model": name, "device": str(device),
                     "weight_bits": base_qc.weight.bits,
                     "activation_bits": base_qc.activation.bits}

    print(f"Loading checkpoint {ckpt} ...")
    model, data_cfg = load_model(name, ckpt)
    print(f"Preparing data under {d['root']} "
         f"(downloading Imagenette if missing — can take a few minutes) ...")
    val = build_val_loader(d["root"], data_cfg, d["batch_size"], d["num_workers"], d["download"])
    calib = build_calib_loader(d["root"], data_cfg, d["calib_images"],
                               d["calib_batch_size"], d["num_workers"], d["download"])
    n_val = len(val) if max_b is None else min(max_b, len(val))
    n_calib = len(calib)  # calibrate() has no max_batches limit in this pipeline

    # 1. fp32 torch baseline
    print(f"[1/6] fp32 baseline on {device} ({n_val} batches)")
    results["fp32_torch"] = evaluate_torch(model, val, CLS, device, max_b,
                                           progress=batch_progress("fp32", n_val))

    # 2. simulated INT8 (research layer)
    print(f"[2/6] simulated INT8 (custom kernel): calibrating ({n_calib} batches)")
    qmodel = convert_vit(model, base_qc)
    sq = cfg.get("smoothquant") or {}
    if sq.get("enabled"):
        sq_alpha = float(sq.get("alpha", 0.5))
        print(f"[2/6] SmoothQuant outlier migration (alpha={sq_alpha}) ...")
        smooth_quant(qmodel, calib, device, alpha=sq_alpha)
        results["smoothquant"] = {"alpha": sq_alpha}
    calibrate(qmodel, calib, device, progress=calib_progress("calib", n_calib))
    ar = cfg.get("adaround") or {}
    if ar.get("enabled"):
        ar_iters = int(ar.get("iters", 1000))
        print(f"[2/6] AdaRound weight-rounding refinement ({ar_iters} iters/layer)")
        adaround(qmodel, calib, device, iters=ar_iters,
                 lr=float(ar.get("lr", 1e-2)),
                 reg_weight=float(ar.get("reg_weight", 0.01)),
                 max_tokens=int(ar.get("max_tokens", 2048)),
                 log=lambda msg: print(f"    [adaround] {msg}"))
        results["adaround"] = {"iters": ar_iters}
    print(f"[2/6] simulated INT8: evaluating ({n_val} batches)")
    results["int8_simulated"] = evaluate_torch(qmodel, val, CLS, device, max_b,
                                               progress=batch_progress("int8 sim", n_val))

    # 3. block sensitivity (len(groups)+1 full eval passes — can take a while)
    if not args.skip_sensitivity:
        print("[3/6] per-block sensitivity sweep")
        results["sensitivity"] = block_sensitivity(
            qmodel, val, CLS, device, max_b,
            log=lambda msg: print(f"    [sensitivity] {msg}"))
    else:
        print("[3/6] skipped")

    # 4. mixed-precision sweep (needs the sensitivity ranking; reuses qmodel,
    # which mixed_precision_sweep restores to fully-quantizing on exit)
    if not args.skip_mixed_precision and "sensitivity" in results:
        ks = ([int(x) for x in args.mixed_precision_ks.split(",")]
              if args.mixed_precision_ks else None)
        print("[4/6] mixed-precision (top-K block protection) sweep")
        results["mixed_precision"] = mixed_precision_sweep(
            qmodel, results["sensitivity"], val, CLS, device, base_qc.weight.bits,
            max_b, ks=ks, log=lambda msg: print(f"    [mixed-prec] {msg}"))
    elif args.skip_mixed_precision:
        print("[4/6] skipped")
    else:
        print("[4/6] skipped (needs sensitivity ranking; --skip-sensitivity was set)")

    # 5. ablation matrix (each variant needs a fresh model: weights were shared in-place)
    if not args.skip_ablation:
        variants = ablation_qconfigs(base_qc)
        print(f"[5/6] ablation matrix ({len(variants)} variants)")
        results["ablation"] = {}
        for i, (label, qc) in enumerate(variants.items(), 1):
            print(f"    variant {i}/{len(variants)}: {label}")
            m, _ = load_model(name, ckpt)
            qm = convert_vit(m, qc)
            calibrate(qm, calib, device)
            results["ablation"][label] = evaluate_torch(
                qm, val, CLS, device, max_b, progress=batch_progress(f"variant {i}", n_val))
            print(f"    {label}: top1={results['ablation'][label]['top1']:.4f}")
    else:
        print("[5/6] skipped")

    # 6. qualitative visualization examples (fresh fp32 model — convert_vit
    # mutated the original in place at step 2; qmodel is the calibrated one)
    if not args.no_qualitative:
        print(f"[6/6] qualitative visualization ({args.qualitative_samples} samples)")
        fp32_fresh, _ = load_model(name, ckpt)
        grid = save_classification_qualitative(
            fp32_fresh, qmodel, data_cfg, d["root"], args.qualitative_samples,
            device, out, name, download=d["download"])
        results["qualitative_grid"] = str(grid)
    else:
        print("[6/6] skipped")

    note = f", and {out / 'qualitative_grid.png'}" if not args.no_qualitative else ""
    write_outputs(out, results, classification_report(results), note)


if __name__ == "__main__":
    main()
