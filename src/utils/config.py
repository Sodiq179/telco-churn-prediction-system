from __future__ import annotations

from pathlib import Path
import yaml


def load_yaml_config(config_path: str | Path) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config