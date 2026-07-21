from pathlib import Path

from transformers import (Sam3Model, Sam3Processor, Sam3TrackerModel,
                          Sam3TrackerProcessor, SamModel, SamProcessor)

DOWNLOAD_HINT = """SAM checkpoint directory not found: {path}

This framework never downloads model weights. On a machine with network access run:

    python -c "from transformers import SamModel, SamProcessor; \\
SamModel.from_pretrained('{name}').save_pretrained('{filename}'); \\
SamProcessor.from_pretrained('{name}').save_pretrained('{filename}')"

then copy the '{filename}' directory to {path}."""

SAM3_DOWNLOAD_HINT = """SAM3 checkpoint directory not found: {path}

This framework never downloads model weights. facebook/sam3 is a GATED repo:
first request access on https://huggingface.co/{name} (accept the license),
then on a machine with network access and `hf auth login` done, run:

    python -c "from transformers import Sam3TrackerModel, Sam3TrackerProcessor; \\
Sam3TrackerModel.from_pretrained('{name}').save_pretrained('{filename}'); \\
Sam3TrackerProcessor.from_pretrained('{name}').save_pretrained('{filename}')"

then copy the '{filename}' directory to {path}."""


def _load_offline(model_cls, processor_cls, name: str, checkpoint: str | Path,
                  hint: str):
    """Shared offline-loading skeleton: local HF-format checkpoint directory
    (config.json + weights, produced by save_pretrained), never touching the
    network — local_files_only=True on both from_pretrained calls."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(hint.format(path=path, name=name, filename=path.name))
    try:
        model = model_cls.from_pretrained(str(path), local_files_only=True)
        processor = processor_cls.from_pretrained(str(path), local_files_only=True)
    except OSError as e:
        # directory exists but is incomplete/corrupt (e.g. missing weights file)
        raise FileNotFoundError(
            hint.format(path=path, name=name, filename=path.name)) from e
    return model.eval(), processor


def load_sam_model(name: str, checkpoint: str | Path) -> tuple[SamModel, SamProcessor]:
    """SAM 1 (facebook/sam-vit-*): SamModel + SamProcessor, offline."""
    return _load_offline(SamModel, SamProcessor, name, checkpoint, DOWNLOAD_HINT)


def load_sam3_model(name: str, checkpoint: str | Path) -> tuple[Sam3TrackerModel, Sam3TrackerProcessor]:
    """SAM 3 (facebook/sam3): the point-promptable tracker head —
    Sam3TrackerModel + Sam3TrackerProcessor, offline. The tracker is the
    SAM1/SAM2-style interactive-segmentation entry point (image + point
    prompts -> masks), matching this framework's point-prompt self-consistency
    protocol. For the text-prompted concept path use load_sam3_concept_model."""
    return _load_offline(Sam3TrackerModel, Sam3TrackerProcessor, name, checkpoint,
                         SAM3_DOWNLOAD_HINT)


def load_sam3_concept_model(name: str, checkpoint: str | Path) -> tuple[Sam3Model, Sam3Processor]:
    """SAM 3 concept segmentation (text prompts): Sam3Model + Sam3Processor,
    offline, from the same weights/sam3 checkpoint directory as the tracker —
    the checkpoint holds the full model; each class loads its own subset
    (Sam3Model additionally uses the text encoder + DETR decoder, which stay
    fp32; only the shared PE vision encoder is quantized)."""
    return _load_offline(Sam3Model, Sam3Processor, name, checkpoint,
                         SAM3_DOWNLOAD_HINT)
