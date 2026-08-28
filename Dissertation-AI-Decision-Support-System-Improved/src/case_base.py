"""Create the persistent CBR case base from the processed dataset."""

import pandas as pd
from src.config import CASE_BASE_PATH, PROCESSED_DIR


def create_case_base() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed = pd.read_csv(PROCESSED_DIR / "processed_loan_dataset.csv")
    processed.insert(0, "case_id", range(1, len(processed) + 1))
    processed.to_csv(CASE_BASE_PATH, index=False)
    return processed


if __name__ == "__main__":
    case_base = create_case_base()
    print(f"Case base created with {len(case_base):,} cases.")
