from __future__ import annotations

import optuna
import mlflow

from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils.config import load_yaml_config


MODEL_CONFIG = load_yaml_config("configs/model.yaml")
APP_CONFIG = load_yaml_config("configs/config.yaml")

mlflow.set_tracking_uri(APP_CONFIG["mlruns_dir"])
mlflow.set_experiment("telco_churn_xgboost_tuning")

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

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        recall = recall_score(y_val, y_pred)

        mlflow.log_metric("val_recall", recall)

    return recall


def run_tuning(X_train, y_train, X_val, y_val, preprocessor):
    study = optuna.create_study(
        direction=MODEL_CONFIG["tuning"]["direction"]
    )

    with mlflow.start_run(run_name="optuna_tuning"):
        mlflow.log_param("n_trials", MODEL_CONFIG["tuning"]["n_trials"])
        mlflow.log_param("optimization_metric", MODEL_CONFIG["tuning"]["optimization_metric"])

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, preprocessor),
            n_trials=MODEL_CONFIG["tuning"]["n_trials"],
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_recall", study.best_value)

    return study.best_params