from __future__ import annotations

from pathlib import Path
import joblib

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


def train_model(preprocessor, X_train, y_train, model_params: dict) -> Pipeline:
    """
    Train XGBoost model with given preprocessing and hyperparameters.
    """

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **model_params
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    return model_pipeline


def save_model(model_pipeline: Pipeline, output_path: str) -> None:
    """
    Save trained pipeline to disk.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, path)

    print(f"Saved model pipeline to: {path}")