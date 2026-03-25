from __future__ import annotations

from pathlib import Path
import json

from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, precision_score, recall_score


def evaluate_classification_model(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


def save_metrics(metrics: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)