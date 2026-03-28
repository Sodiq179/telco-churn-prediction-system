from __future__ import annotations

from src.models.train import train_model, save_model
from sklearn.model_selection import train_test_split

from src.data.ingest import ingest_data
from src.data.validate import validate_dataframe
from src.features.build_features import add_engineered_features
from src.data.preprocess import prepare_features_and_target, build_preprocessor
from src.models.tune import run_tuning
from src.models.evaluate import evaluate_classification_model, save_metrics
from src.models.registry import get_versioned_model_path, save_model_metadata
from src.utils.config import load_yaml_config

APP_CONFIG = load_yaml_config("configs/config.yaml")
DATA_CONFIG = load_yaml_config("configs/data.yaml")

RAW_DATA_PATH = DATA_CONFIG["raw_data_path"]
TEST_SIZE = DATA_CONFIG["test_size"]
RANDOM_STATE = DATA_CONFIG["random_state"]

METRICS_OUTPUT_PATH = f"{APP_CONFIG['metrics_dir']}/metrics.json"
METADATA_OUTPUT_PATH = f"{APP_CONFIG['model_dir']}/model_metadata.json"


def main():
    # Load + validate
    df, _ = ingest_data(RAW_DATA_PATH)
    validate_dataframe(df)

    # Feature engineering
    df = add_engineered_features(df)

    # Prepare features
    X, y = prepare_features_and_target(df)

    # Train / Val / Test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
    )

    preprocessor = build_preprocessor()

    # 🔥 Optuna tuning
    best_params = run_tuning(X_train, y_train, X_val, y_val, preprocessor)

    # Train final model on full train set
    model_pipeline = train_model(
        preprocessor=preprocessor,
        X_train=X_train_full,
        y_train=y_train_full,
        model_params=best_params,
    )

    # Evaluate on test
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_classification_model(y_test, y_pred, y_proba)

    # Save outputs
    model_path = get_versioned_model_path(APP_CONFIG["model_dir"])
    save_model(model_pipeline, str(model_path))
    save_metrics(metrics, METRICS_OUTPUT_PATH)
    save_model_metadata(
        model_path=str(model_path),
        metrics=metrics,
        output_path=METADATA_OUTPUT_PATH,
    )

    print("Training with tuning completed.")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()