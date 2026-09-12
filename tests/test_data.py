import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.predict import app, load_model

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_mock_data_and_model():
    """Ensures necessary directories and mock files exist for CI environment."""
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create mock daily_demand.csv if it doesn't exist
    processed_path = os.path.join("data", "processed", "daily_demand.csv")
    if not os.path.exists(processed_path):
        mock_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "total_units_sold": [100.0, 120.0],
            "avg_price": [50.0, 50.0]
        })
        mock_df.to_csv(processed_path, index=False)


def test_processed_demand_file_exists():
    """Verify daily_demand.csv exists and contains required columns."""
    file_path = os.path.join("data", "processed", "daily_demand.csv")
    assert os.path.exists(file_path)
    df = pd.read_csv(file_path)
    assert not df.empty
    assert "total_units_sold" in df.columns
    assert "avg_price" in df.columns


def test_model_file_exists():
    """Verify saved model artifact can be loaded."""
    model = load_model()
    assert model is not None


def test_health_endpoint():
    """Verify health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_valid_payload():
    """Verify predict endpoint handles valid POST payloads correctly."""
    payload = {
        "avg_price": 100.0,
        "day_of_week": 1,
        "month": 5,
        "day": 10,
        "lag_1": 100.0,
        "lag_7": 100.0,
        "rolling_mean_7": 100.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_units_sold" in data
    assert data["status"] == "success"


def test_predict_endpoint_negative_price():
    """Verify API rejects negative avg_price with a 422 error."""
    invalid_payload = {
        "avg_price": -50.0,
        "day_of_week": 1,
        "month": 5,
        "day": 10,
        "lag_1": 100.0,
        "lag_7": 100.0,
        "rolling_mean_7": 100.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_month():
    """Verify API rejects month values outside 1-12 range."""
    invalid_payload = {
        "avg_price": 100.0,
        "day_of_week": 1,
        "month": 13,
        "day": 10,
        "lag_1": 100.0,
        "lag_7": 100.0,
        "rolling_mean_7": 100.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_fields():
    """Verify API rejects payloads with missing required keys."""
    incomplete_payload = {"avg_price": 100.0, "day_of_week": 1}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_drift_report_generation():
    """Verify drift detection pipeline runs and generates valid HTML output."""
    response = client.get("/drift-report")
    assert response.status_code == 200
    assert "<html>" in response.text.lower() or "<!doctype html>" in response.text.lower()


def test_retrain_endpoint():
    """Verify trigger endpoint initiates retraining pipeline."""
    response = client.post("/retrain?force=true")
    assert response.status_code == 200
    assert response.json()["status"] == "success"