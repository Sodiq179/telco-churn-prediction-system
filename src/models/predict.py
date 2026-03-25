from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = "artifacts/model/model_pipeline.joblib"


def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found at: {path}")

    return joblib.load(path)


def predict(data: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    model_pipeline = load_model(model_path)

    predictions = model_pipeline.predict(data)
    probabilities = model_pipeline.predict_proba(data)[:, 1]

    results = data.copy()
    results["prediction"] = predictions
    results["prediction_label"] = results["prediction"].map({0: "No", 1: "Yes"})
    results["prediction_probability"] = probabilities

    return results