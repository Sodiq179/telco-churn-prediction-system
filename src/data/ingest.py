import os
import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")


def validate_basic_structure(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["customerID"].duplicated().any():
        dup_count = df["customerID"].duplicated().sum()
        raise ValueError(f"Found {dup_count} duplicate customerID values.")

    valid_targets = {"Yes", "No"}
    actual_targets = set(df["Churn"].dropna().unique())
    invalid_targets = actual_targets - valid_targets
    if invalid_targets:
        raise ValueError(f"Invalid Churn values found: {invalid_targets}")


def inspect_raw_issues(df: pd.DataFrame) -> dict:
    total_charges_blank = df["TotalCharges"].astype(str).str.strip().eq("").sum()

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values_per_column": df.isna().sum().to_dict(),
        "blank_total_charges": int(total_charges_blank),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "churn_distribution": df["Churn"].value_counts(dropna=False).to_dict(),
    }
    return report


def ingest_data(file_path: str) -> tuple[pd.DataFrame, dict]:
    df = load_raw_data(file_path)
    validate_columns(df)
    validate_basic_structure(df)
    report = inspect_raw_issues(df)
    return df, report


if __name__ == "__main__":
    raw_path = "data/raw/Telco-Customer-Churn.csv"
    df, report = ingest_data(raw_path)

    print("Raw data loaded successfully.")
    print(f"Shape: {df.shape}")
    print("Ingestion report:")
    for key, value in report.items():
        print(f"{key}: {value}")