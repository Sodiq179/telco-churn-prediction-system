from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


BINARY_CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

ONEHOT_CATEGORICAL_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERICAL_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    )

    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_raw_dataframe(df)

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Unexpected values found in target column 'Churn'.")

    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    multiclass_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_COLUMNS),
            ("bin", binary_pipeline, BINARY_CATEGORICAL_COLUMNS),
            ("cat", multiclass_pipeline, ONEHOT_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessor