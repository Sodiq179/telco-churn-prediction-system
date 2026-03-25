from __future__ import annotations

import pandas as pd

from src.models.predict import predict


def run_inference(input_data: dict) -> dict:
    input_df = pd.DataFrame([input_data])
    results = predict(input_df)

    return {
        "prediction": results["prediction_label"].iloc[0],
        "prediction_probability": float(results["prediction_probability"].iloc[0]),
    }