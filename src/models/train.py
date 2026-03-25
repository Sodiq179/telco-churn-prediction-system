from __future__ import annotations

from pathlib import Path
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.ingest import ingest_data
from src.data.preprocess import build_preprocessor, prepare_features_and_target


RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
MODEL_OUTPUT_PATH = "artifacts/model/model_pipeline.joblib"
METRICS_OUTPUT_PATH = "artifacts/metrics/metrics.json"


def train_model(
    raw_data_path: str = RAW_DATA_PATH,
    model_output_path: str = MODEL_OUTPUT_PATH,
    metrics_output_path: str = METRICS_OUTPUT_PATH,
) -> dict:
    df, ingestion_report = ingest_data(raw_data_path)
    print("Ingestion report:", ingestion_report)

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Metrics:", metrics)

    model_path = Path(model_output_path)
    metrics_path = Path(metrics_output_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model pipeline to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_model()