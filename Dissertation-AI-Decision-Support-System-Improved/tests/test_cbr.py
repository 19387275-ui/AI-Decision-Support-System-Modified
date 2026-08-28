import pytest
from src.cbr_model import get_recommendation

SAMPLE = {
    "no_of_dependents": 2,
    "education": 1,
    "self_employed": 0,
    "income_annum": 5_000_000,
    "loan_amount": 15_000_000,
    "loan_term": 10,
    "cibil_score": 750,
    "residential_assets_value": 5_000_000,
    "commercial_assets_value": 3_000_000,
    "luxury_assets_value": 10_000_000,
    "bank_asset_value": 4_000_000,
}


def test_recommendation_schema():
    result = get_recommendation(SAMPLE)
    assert result["recommendation"] in {"APPROVED", "REJECTED"}
    assert 0 <= result["confidence"] <= 100
    assert result["approved_neighbours"] + result["rejected_neighbours"] == 5
    assert len(result["similar_cases"]) == 5


def test_invalid_neighbour_count():
    with pytest.raises(ValueError):
        get_recommendation(SAMPLE, n_neighbors=0)
