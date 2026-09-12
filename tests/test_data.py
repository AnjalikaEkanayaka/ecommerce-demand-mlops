import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.predict import app, load_model

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_mock_environment():
    """Sets up mock datasets and required directories for CI environment."""
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    processed_path = os.path.join("data", "processed", "daily_demand.csv")
    
    # 30 days of synthetic data to ensure lag_7 and rolling_mean_7 yield valid rows
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    mock_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "total_units_sold": [100.0 + i for i in range(30)],
        "avg_price": [50.0] * 30
    })
    mock_df.to_csv(processed_path, index=False)

def test_processed_demand_file_exists():
    file_path = os.path.join("data", "processed", "daily_demand.csv")
    assert os.path.exists(file_path)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_endpoint_valid_payload():
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

def test_drift_report_endpoint():
    response = client.get("/drift-report")
    assert response.status_code == 200

def test_retrain_endpoint():
    response = client.post("/retrain?force=true")
    assert response.status_code == 200