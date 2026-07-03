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
