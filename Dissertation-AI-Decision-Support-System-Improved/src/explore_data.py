"""Generate a concise exploratory summary of the loan dataset."""

from src.config import DATASET_PATH, SUMMARIES_DIR
import pandas as pd


def explore():
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARIES_DIR / "dataset_exploration.txt", "w", encoding="utf-8") as f:
        f.write("FIRST 5 RECORDS\n" + df.head().to_string() + "\n\n")
        f.write(f"DATASET SHAPE\n{df.shape}\n\n")
        f.write(f"COLUMN NAMES\n{df.columns.tolist()}\n\n")
        f.write(f"DATA TYPES\n{df.dtypes.to_string()}\n\n")
        f.write(f"MISSING VALUES\n{df.isnull().sum().to_string()}\n\n")
        f.write(f"STATISTICAL SUMMARY\n{df.describe().to_string()}\n\n")
        f.write(f"LOAN STATUS DISTRIBUTION\n{df['loan_status'].value_counts().to_string()}\n")


if __name__ == "__main__":
    explore()
    print("Dataset exploration completed.")
