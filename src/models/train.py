from __future__ import annotations

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline

from src.models.model_factory import build_estimator
from src.utils.config import load_yaml_config


MODEL_CONFIG = load_yaml_config("configs/model.yaml")


def get_active_model_config() -> tuple[str, dict]:
    """
    Get active model and its parameter from model.yaml.
    """
    active_model = MODEL_CONFIG["active_model"]

    model_cfg = MODEL_CONFIG["models"][active_model]

    return active_model, model_cfg


def train_model(preprocessor, X_train, y_train, model_params: dict | None = None) -> Pipeline:
    """
    Train models.
    """

    active_model, model_cfg = get_active_model_config()

    base_params = model_cfg["params"].copy()
    if model_params:
        base_params.update(model_params)

    estimator = build_estimator(
        estimator_class_name=model_cfg["estimator_class"],
        params=base_params,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
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