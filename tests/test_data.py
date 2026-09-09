import os
import pandas as pd
from fastapi.testclient import TestClient
from src.data_loader import validate_and_process_demand
from src.predict import app, load_model

client = TestClient(app)


def test_processed_demand_file_creation():
    """Verify daily_demand.csv is generated correctly."""
    validate_and_process_demand()
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
    """Verify API rejects negative avg_price with a 422 Unprocessable Entity error."""
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