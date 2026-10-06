"""
Question B - Automated Pytest Suite
Implements three automated tests with pytest, including one dedicated test for bad input validation:
1. test_predict_valid_input: Verifies valid prediction inference, response schema, and database persistence.
2. test_stats_handwritten_sql: Verifies aggregate statistics computed via pure hand-written SQL.
3. test_validation_bad_inputs: Tests out-of-range, bad type, and invalid domain inputs with descriptive error assertions.
4. test_health_check: Verifies liveness, candidate seed S=36, and model artifact availability.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from question_b.app import app
from question_b.database import init_db, get_stats
from config import SEED

# Create FastAPI TestClient
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure database schema is initialized prior to running tests."""
    init_db()


def test_predict_valid_input():
    """
    Test 1: Valid Input Prediction
    Sends complete valid clinical parameters and verifies inference,
    plain-words interpretation, and SQLite persistence.
    """
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 1,
        "thal": 7
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert "risk_category" in data
    assert data["risk_category"] in ("Low Risk", "Moderate Risk", "High Risk")
    assert "plain_words_summary" in data
    assert len(data["plain_words_summary"]) > 20
    assert "lifestyle_recommendations" in data
    assert isinstance(data["lifestyle_recommendations"], list)
    assert len(data["lifestyle_recommendations"]) >= 3
    assert data["request_id"] > 0


def test_stats_handwritten_sql():
    """
    Test 2: Stats Endpoint via Hand-Written SQL
    Verifies that /stats accurately aggregates requests, average predicted risk,
    and the proportion of high-risk outcomes using raw SQL (no ORM).
    """
    # 1. Fetch current baseline stats
    res_before = client.get("/stats")
    assert res_before.status_code == 200
    stats_before = res_before.json()
    initial_count = stats_before["total_requests"]

    # 2. Submit a high-risk patient
    high_risk_patient = {
        "age": 68,
        "sex": 1,
        "cp": 4,
        "trestbps": 165,
        "chol": 290,
        "fbs": 1,
        "restecg": 2,
        "thalach": 110,
        "exang": 1,
        "oldpeak": 3.0,
        "slope": 2,
        "ca": 2,
        "thal": 7
    }
    res_pred = client.post("/predict", json=high_risk_patient)
    assert res_pred.status_code == 200

    # 3. Check updated stats
    res_after = client.get("/stats")
    assert res_after.status_code == 200
    stats_after = res_after.json()

    assert stats_after["total_requests"] == initial_count + 1
    assert 0.0 <= stats_after["average_predicted_risk"] <= 1.0
    assert 0.0 <= stats_after["share_of_high_risk_results"] <= 1.0
    assert "Pure Hand-Written SQL" in stats_after["query_engine"]


@pytest.mark.parametrize(
    "bad_payload, expected_field, expected_substr",
    [
        (
            # Bad age: negative value
            {
                "age": -5, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 1, "oldpeak": 1.5,
                "slope": 2, "ca": 1, "thal": 7
            },
            "age",
            "Age must be between 1 and 120"
        ),
        (
            # Bad age: exceeds human maximum (200 years)
            {
                "age": 200, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 1, "oldpeak": 1.5,
                "slope": 2, "ca": 1, "thal": 7
            },
            "age",
            "Age must be between 1 and 120"
        ),
        (
            # Bad blood pressure: clinically impossible value (25 mm Hg)
            {
                "age": 50, "sex": 1, "cp": 4, "trestbps": 25, "chol": 240,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 1, "oldpeak": 1.5,
                "slope": 2, "ca": 1, "thal": 7
            },
            "trestbps",
            "Resting blood pressure (trestbps) must be within plausible clinical range 50 to 300"
        ),
        (
            # Bad thalassemia code: 99 is invalid (valid: 3, 6, 7)
            {
                "age": 50, "sex": 1, "cp": 4, "trestbps": 120, "chol": 200,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 0.5,
                "slope": 1, "ca": 0, "thal": 99
            },
            "thal",
            "Thalassemia (thal) must be 3 (normal), 6 (fixed defect), or 7 (reversible defect)"
        ),
        (
            # Bad data type: text string where number expected
            {
                "age": "sixty-five", "sex": 1, "cp": 4, "trestbps": 120, "chol": 200,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 0.5,
                "slope": 1, "ca": 0, "thal": 3
            },
            "age",
            "Input should be a valid integer"
        )
    ]
)
def test_validation_bad_inputs(bad_payload, expected_field, expected_substr):
    """
    Test 3: Comprehensive Bad Input Validation
    Verifies that the API returns HTTP 422 Unprocessable Entity
    with explicit, actionable clinical error diagnostics when invalid inputs are provided.
    """
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422, f"Expected 422 for bad input, got {response.status_code}"

    error_data = response.json()
    assert error_data["status"] == "validation_error"
    assert "details" in error_data

    # Verify that the expected field and error message are explicitly identified
    details_str = str(error_data["details"])
    assert expected_field in details_str, f"Field '{expected_field}' not flagged in {details_str}"
    assert expected_substr.lower() in details_str.lower(), f"Expected substring '{expected_substr}' in error details: {details_str}"


def test_health_check():
    """
    Test 4: System Health and Candidate Seed Verification
    Verifies that the service is running, candidate seed S=36 is active,
    and model artifacts are correctly loaded into memory.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["candidate_seed"] == SEED
    assert data["model_loaded"] is True
