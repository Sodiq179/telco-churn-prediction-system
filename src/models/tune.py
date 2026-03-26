from __future__ import annotations

import optuna
import mlflow
import mlflow.sklearn

from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils.config import load_yaml_config


MODEL_CONFIG = load_yaml_config("configs/model.yaml")


def suggest_params(trial: optuna.Trial) -> dict:
    space = MODEL_CONFIG["tuning"]["search_space"]

    params = {}
    for name, cfg in space.items():
        if cfg["type"] == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
        elif cfg["type"] == "float":
            if cfg.get("log", False):
                params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], log=True)
            else:
                params[name] = trial.suggest_float(name, cfg["low"], cfg["high"])
    return params


def objective(trial, X_train, y_train, X_val, y_val, preprocessor):
    params = suggest_params(trial)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **params
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        recall = recall_score(y_val, y_pred)

        mlflow.log_metric("recall", recall)

    return recall


def run_tuning(X_train, y_train, X_val, y_val, preprocessor):
    study = optuna.create_study(
        direction=MODEL_CONFIG["tuning"]["direction"]
    )

    with mlflow.start_run(run_name="optuna_tuning"):
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, preprocessor),
            n_trials=MODEL_CONFIG["tuning"]["n_trials"],
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_recall", study.best_value)

    return study.best_params