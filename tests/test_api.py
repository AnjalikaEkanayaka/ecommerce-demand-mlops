import numpy as np
from fastapi.testclient import TestClient
from src.predict import app, load_model
from src.features import FEATURE_COLUMNS


PAYLOAD = {
    "day_of_week": 1,
    "month": 5,
    "day": 10,
    "lag_1": 100.0,
    "lag_7": 100.0,
    "rolling_mean_7": 100.0,
}


def test_missing_model(settings):
    assert load_model() is None


def test_health_and_prediction(model_artifact):
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.json() == {"status": "healthy", "model_loaded": True}
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 200
        prediction = response.json()["predicted_units_sold"]
        assert np.isfinite(prediction)
        assert abs(prediction - 100.0) < 1e-5


def test_invalid_request(model_artifact):
    with TestClient(app) as client:
        assert client.post("/predict", json={**PAYLOAD, "month": 13}).status_code == 422


def test_drift_endpoint(daily_data, settings):
    with TestClient(app) as client:
        response = client.get("/drift-report")
        assert response.status_code == 200
        assert "html" in response.headers["content-type"]
        assert (settings.report_dir / "drift_report.html").is_file()


def test_retrain_endpoint_uses_isolated_storage(daily_data, settings):
    with TestClient(app) as client:
        assert client.post("/retrain").json()["status"] == "skipped"
        assert not settings.model_path.exists()
        response = client.post("/retrain?force=true")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert load_model().n_features_in_ == len(FEATURE_COLUMNS)

def test_same_day_price_is_rejected(model_artifact):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={**PAYLOAD, "avg_price": 40.0},
        )

    assert response.status_code == 422


def test_prediction_without_model_returns_unavailable(settings):
    with TestClient(app) as client:
        response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 503