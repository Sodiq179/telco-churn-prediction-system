from __future__ import annotations

import mlflow
import optuna

from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

from src.models.model_factory import build_estimator
from src.utils.config import load_yaml_config


MODEL_CONFIG = load_yaml_config("configs/model.yaml")
APP_CONFIG = load_yaml_config("configs/config.yaml")


def get_active_model_config() -> tuple[str, dict]:

    active_model = MODEL_CONFIG["active_model"]
    model_cfg = MODEL_CONFIG["models"][active_model]

    return active_model, model_cfg


def setup_mlflow_for_tuning(active_model: str) -> None:
    mlflow.set_tracking_uri(APP_CONFIG["mlruns_dir"])
    experiment_prefix = MODEL_CONFIG["tracking"]["experiment_prefix"]
    mlflow.set_experiment(f"{experiment_prefix}_{active_model}_tuning")


def suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}

    for name, cfg in search_space.items():
        param_type = cfg["type"]

        if param_type == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
        elif param_type == "float":
            params[name] = trial.suggest_float(
                name,
                cfg["low"],
                cfg["high"],
                log=cfg.get("log", False),
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
        else:
            raise ValueError(f"Unsupported search space type: {param_type}")

    return params

def objective(trial, X_train, y_train, X_val, y_val, preprocessor, model_cfg):
    params = suggest_params(trial, model_cfg["search_space"])

    final_params = model_cfg["params"].copy()
    final_params.update(params)

    estimator = build_estimator(
        estimator_class_name=model_cfg["estimator_class"],
        params=final_params,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    threshold = MODEL_CONFIG["tuning"].get("threshold", 0.5)

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_param("active_model", MODEL_CONFIG["active_model"])
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        recall = recall_score(y_val, y_pred)

        mlflow.log_metric("val_recall", recall)
        mlflow.log_param("threshold", threshold)

    return recall

def run_tuning(X_train, y_train, X_val, y_val, preprocessor):
    active_model, model_cfg = get_active_model_config()
    setup_mlflow_for_tuning(active_model)

    study = optuna.create_study(
        direction=MODEL_CONFIG["tuning"]["direction"]
    )

    with mlflow.start_run(run_name=f"{active_model}_optuna_tuning"):
        mlflow.log_param("active_model", active_model)
        mlflow.log_param("n_trials", MODEL_CONFIG["tuning"]["n_trials"])
        mlflow.log_param("optimization_metric", MODEL_CONFIG["tuning"]["optimization_metric"])

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, preprocessor, model_cfg),
            n_trials=MODEL_CONFIG["tuning"]["n_trials"],
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_recall", study.best_value)

    return study.best_params