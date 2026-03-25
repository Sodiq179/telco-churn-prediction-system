from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json


def build_model_version() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


def get_versioned_model_path(base_dir: str = "artifacts/model") -> Path:
    version = build_model_version()
    return Path(base_dir) / f"model_pipeline_{version}.joblib"


def save_model_metadata(model_path: str, metrics: dict, output_path: str = "artifacts/model/model_metadata.json") -> None:
    metadata = {
        "model_path": model_path,
        "metrics": metrics,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)