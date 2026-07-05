from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config as a plain dict (see configs/*.yaml)."""
    with open(path) as f:
        return yaml.safe_load(f)
