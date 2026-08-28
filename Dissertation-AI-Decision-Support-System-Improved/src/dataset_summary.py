"""Generate categorical and target distributions."""

import pandas as pd
from src.config import DATASET_PATH, SUMMARIES_DIR


def create_summary():
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARIES_DIR / "dataset_summary.txt", "w", encoding="utf-8") as f:
        for title, series in [
            ("LOAN STATUS DISTRIBUTION", df["loan_status"]),
            ("EDUCATION DISTRIBUTION", df["education"]),
            ("SELF EMPLOYED DISTRIBUTION", df["self_employed"]),
            ("DEPENDENTS DISTRIBUTION", df["no_of_dependents"]),
        ]:
            f.write(title + "\n")
            f.write(series.value_counts().sort_index().to_string() + "\n\n")


if __name__ == "__main__":
    create_summary()
    print("Dataset summary saved successfully.")
