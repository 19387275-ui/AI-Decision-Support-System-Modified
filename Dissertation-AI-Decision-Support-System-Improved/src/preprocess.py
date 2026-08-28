"""Clean, encode and scale the loan dataset for the CBR case base."""

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DATASET_PATH, PROCESSED_DIR, SUMMARIES_DIR, NUMERICAL_COLUMNS
)


def load_and_clean_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()
    for col in ["education", "self_employed", "loan_status"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.drop(columns=["loan_id"])
    df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
    df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    if df[["education", "self_employed", "loan_status"]].isnull().any().any():
        raise ValueError("Categorical/target encoding produced missing values.")

    negative_count = int((df["residential_assets_value"] < 0).sum())
    df["residential_assets_value"] = df["residential_assets_value"].clip(lower=0)
    df.attrs["negative_residential_count"] = negative_count
    return df


def preprocess_dataset() -> pd.DataFrame:
    df = load_and_clean_dataset()
    scaler = MinMaxScaler()
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, PROCESSED_DIR / "minmax_scaler.pkl")
    processed_path = PROCESSED_DIR / "processed_loan_dataset.csv"
    df.to_csv(processed_path, index=False)

    with open(SUMMARIES_DIR / "preprocessing_report.txt", "w", encoding="utf-8") as report:
        report.write("PREPROCESSING REPORT\n\n")
        report.write(f"Original Dataset Shape: {load_and_clean_dataset().shape}\n")
        report.write(f"Processed Dataset Shape: {df.shape}\n\n")
        report.write("Encoding Validation\n")
        report.write(f"Education Null Values: {df['education'].isnull().sum()}\n")
        report.write(f"Self Employed Null Values: {df['self_employed'].isnull().sum()}\n")
        report.write(f"Loan Status Null Values: {df['loan_status'].isnull().sum()}\n")
        report.write(f"Negative Residential Asset Records Found: {df.attrs.get('negative_residential_count', 0)}\n")
        report.write("Negative residential asset values are clipped to 0.\n")
    return df


if __name__ == "__main__":
    preprocess_dataset()
    print("Preprocessing completed successfully.")
