from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


MODEL_REGISTRY = {
    "XGBClassifier": XGBClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LGBMClassifier": LGBMClassifier,
}


def build_estimator(estimator_class_name: str, params: dict):
    estimator_class = MODEL_REGISTRY.get(estimator_class_name)

    if estimator_class is None:
        raise ValueError(f"Unsupported or unavailable estimator class: {estimator_class_name}")

    return estimator_class(**params)