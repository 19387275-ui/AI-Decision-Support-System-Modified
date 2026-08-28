from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "loan_approval_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_data"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
RECOMMENDATIONS_DIR = PROJECT_ROOT / "outputs" / "recommendations"
SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"
CASE_BASE_PATH = PROCESSED_DIR / "case_base.csv"
SCALER_PATH = PROCESSED_DIR / "minmax_scaler.pkl"

FEATURE_COLUMNS = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
NUMERICAL_COLUMNS = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
