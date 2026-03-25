from __future__ import annotations

from src.models.train import train_model
from src.data.ingest import ingest_data

def main() -> None:
    print("Training pipeline started...")
    df, report = ingest_data("data/raw/Telco-Customer-Churn.csv")
    print(report)

    metrics = train_model()
    print("\nTraining pipeline completed.")
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()