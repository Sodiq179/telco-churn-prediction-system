from src.data.ingest import ingest_data

def main():
    print("Training pipeline started...")
    df, report = ingest_data("data/raw/Telco-Customer-Churn.csv")
    print(report)

if __name__ == "__main__":
    main()