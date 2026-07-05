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
