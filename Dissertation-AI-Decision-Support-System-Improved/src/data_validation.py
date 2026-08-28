"""Basic data-quality checks for the source dataset."""

import pandas as pd
from src.preprocess import load_and_clean_dataset
from src.config import DATASET_PATH
from src.config import NUMERICAL_COLUMNS


def validate() -> bool:
    raw = pd.read_csv(DATASET_PATH)
    raw.columns = raw.columns.str.strip()
    df = load_and_clean_dataset()
    print("DATA VALIDATION")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    for col in NUMERICAL_COLUMNS:
        source_col = raw[col] if col in raw.columns else df[col]
        print(f"Negative {col}: {int((source_col < 0).sum())}")
    return True


if __name__ == "__main__":
    validate()
