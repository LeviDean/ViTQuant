from pathlib import Path

from transformers import SamModel, SamProcessor

DOWNLOAD_HINT = """SAM checkpoint directory not found: {path}

This framework never downloads model weights. On a machine with network access run:

    python -c "from transformers import SamModel, SamProcessor; \\
SamModel.from_pretrained('{name}').save_pretrained('{filename}'); \\
SamProcessor.from_pretrained('{name}').save_pretrained('{filename}')"

then copy the '{filename}' directory to {path}."""


def load_sam_model(name: str, checkpoint: str | Path) -> tuple[SamModel, SamProcessor]:
    """Load a SAM model + processor offline from a local HF-format checkpoint
    directory (config.json + weights, produced by save_pretrained). Never
    touches the network — pass local_files_only=True to from_pretrained."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            DOWNLOAD_HINT.format(path=path, name=name, filename=path.name))
    try:
        model = SamModel.from_pretrained(str(path), local_files_only=True)
        processor = SamProcessor.from_pretrained(str(path), local_files_only=True)
    except OSError as e:
        # directory exists but is incomplete/corrupt (e.g. missing weights file)
        raise FileNotFoundError(
            DOWNLOAD_HINT.format(path=path, name=name, filename=path.name)) from e
    return model.eval(), processor
