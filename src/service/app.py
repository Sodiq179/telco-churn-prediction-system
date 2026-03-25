from __future__ import annotations

import pandas as pd
from fastapi import FastAPI

from src.models.predict import predict
from src.service.schemas import ChurnPredictionRequest


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict:
    return {"message": "Telco Churn Prediction API is running."}


@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest) -> dict:
    input_df = pd.DataFrame([request.model_dump()])
    result = predict(input_df)

    return {
        "prediction": result["prediction_label"].iloc[0],
        "prediction_probability": float(result["prediction_probability"].iloc[0]),
    }