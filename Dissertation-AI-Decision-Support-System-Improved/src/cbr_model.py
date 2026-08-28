"""Case-Based Reasoning engine for loan decision support.

CBR cycle implemented here:
1. Retrieve: find the five nearest historical cases.
2. Reuse: inspect the decisions of those retrieved cases.
3. Adapt: use a majority vote (>= 3 approved neighbours => APPROVED).
4. Retain: the application/recommendation can be recorded by the UI.
"""

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import CASE_BASE_PATH, SCALER_PATH, FEATURE_COLUMNS, NUMERICAL_COLUMNS


def build_cbr_model(n_neighbors: int = 5):
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    df = pd.read_csv(CASE_BASE_PATH)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(df[FEATURE_COLUMNS])
    return df, knn


def scale_applicant_data(applicant_data: dict) -> list[float]:
    scaler = joblib.load(SCALER_PATH)
    numerical = pd.DataFrame([[applicant_data[col] for col in NUMERICAL_COLUMNS]], columns=NUMERICAL_COLUMNS)
    scaled = scaler.transform(numerical)[0]
    values = dict(zip(NUMERICAL_COLUMNS, scaled))
    return [values["no_of_dependents"], int(applicant_data["education"]), int(applicant_data["self_employed"]),
            values["income_annum"], values["loan_amount"], values["loan_term"], values["cibil_score"],
            values["residential_assets_value"], values["commercial_assets_value"],
            values["luxury_assets_value"], values["bank_asset_value"]]


def get_recommendation(applicant_data: dict, n_neighbors: int = 5) -> dict:
    df, knn = build_cbr_model(n_neighbors)
    scaled_input = scale_applicant_data(applicant_data)
    input_df = pd.DataFrame([scaled_input], columns=FEATURE_COLUMNS)
    distances, indices = knn.kneighbors(input_df)
    similar_cases = df.iloc[indices[0]].copy()
    similar_cases["distance"] = distances[0]
    similar_cases["similarity_percent"] = (1 / (1 + similar_cases["distance"])) * 100

    approved_count = int(similar_cases["loan_status"].sum())
    rejected_count = n_neighbors - approved_count
    recommendation = "APPROVED" if approved_count >= (n_neighbors // 2 + 1) else "REJECTED"
    confidence = max(approved_count, rejected_count) / n_neighbors * 100

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "approved_neighbours": approved_count,
        "rejected_neighbours": rejected_count,
        "similar_cases": similar_cases,
    }


if __name__ == "__main__":
    sample = {
        "no_of_dependents": 2, "education": 1, "self_employed": 0,
        "income_annum": 5_000_000, "loan_amount": 15_000_000, "loan_term": 10,
        "cibil_score": 750, "residential_assets_value": 5_000_000,
        "commercial_assets_value": 3_000_000, "luxury_assets_value": 10_000_000,
        "bank_asset_value": 4_000_000,
    }
    result = get_recommendation(sample)
    print(result["similar_cases"][["case_id", "loan_status", "distance", "similarity_percent"]])
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.2f}%")
