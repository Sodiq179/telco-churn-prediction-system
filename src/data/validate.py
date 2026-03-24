from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_unique_customer_id(df: pd.DataFrame) -> None:
    duplicate_count = int(df["customerID"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate customerID values.")


def validate_target_values(df: pd.DataFrame) -> None:
    valid_targets = {"Yes", "No"}
    actual_targets = set(df["Churn"].dropna().unique())
    invalid = actual_targets - valid_targets
    if invalid:
        raise ValueError(f"Invalid target values found: {sorted(invalid)}")


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    validate_required_columns(df)
    validate_unique_customer_id(df)
    validate_target_values(df)