import numpy as np
import pytest
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
        assert response.json() == {"status": "healthy"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "model_loaded": True}
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 200
        prediction = response.json()["predicted_units_sold"]
        assert np.isfinite(prediction)
        assert abs(prediction - 100.0) < 1e-5


def test_invalid_request(model_artifact):
    with TestClient(app) as client:
        assert client.post("/predict", json={**PAYLOAD, "month": 13}).status_code == 422


def test_drift_endpoint(settings):
    run_id = "a" * 32
    report = settings.report_dir / "simulated" / run_id / "drift_report.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>Simulated drift</html>", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get(f"/drift-report?source=simulated&run_id={run_id}")
        assert response.status_code == 200
        assert "html" in response.headers["content-type"]
        assert "Simulated drift" in response.text


def test_missing_drift_report_does_not_generate_data(settings):
    with TestClient(app) as client:
        assert client.get("/drift-report", params={"source": "observed", "run_id": "b" * 32}).status_code == 404
        assert client.get("/drift-report", params={"source": "../", "run_id": "bad"}).status_code == 422
    assert not settings.report_dir.exists()

def test_retrain_without_trigger_is_skipped(settings):
    with TestClient(app) as client:
        response = client.post("/retrain")

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert not (
        settings.runtime_dir / "models" / "production.json"
    ).exists()


def test_retrain_endpoint_preserves_rejection(monkeypatch):
    def reject_candidate(drift_threshold_exceeded=False):
        assert drift_threshold_exceeded is True
        return {
            "status": "rejected",
            "reason": "Candidate did not improve forecasting error.",
        }

    monkeypatch.setattr(
        "src.predict.execute_retraining_pipeline",
        reject_candidate,
    )

    with TestClient(app) as client:
        response = client.post("/retrain?force=true")

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

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


def test_health_does_not_load_model(monkeypatch):
    def must_not_load():
        pytest.fail("Liveness must not load a model")

    monkeypatch.setattr("src.predict.load_model", must_not_load)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "healthy"}


def test_missing_model_is_not_ready(settings):
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503


@pytest.mark.parametrize("artifact", ["production.json", "metadata.json", "model.json"])
def test_corrupt_model_is_unavailable(model_artifact, settings, artifact):
    from src.model_store import ModelStore

    store = ModelStore(settings.runtime_dir / "models")
    directory = store.version_directory(store.production_version())
    path = store.production_path if artifact == "production.json" else directory / artifact
    path.write_text("invalid json", encoding="utf-8")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 503
        assert str(settings.runtime_dir) not in response.text


@pytest.mark.parametrize(
    "update",
    [{"lag_1": -1}, {"month": "5"}, {"day": True}, {"lag_7": "NaN"}],
)
def test_invalid_inputs_are_rejected_before_loading(update, monkeypatch):
    def must_not_load():
        pytest.fail("Invalid input must be rejected before loading")

    monkeypatch.setattr("src.predict.load_model", must_not_load)
    with TestClient(app) as client:
        assert client.post("/predict", json={**PAYLOAD, **update}).status_code == 422


@pytest.mark.parametrize("predictions", [[float("nan")], [float("inf")], [-1], [], [1, 2]])
def test_invalid_prediction_returns_unavailable(predictions, monkeypatch):
    class InvalidModel:
        def predict(self, features):
            assert list(features.columns) == FEATURE_COLUMNS
            return predictions

    monkeypatch.setattr("src.predict.load_model", lambda: InvalidModel())
    with TestClient(app) as client:
        assert client.post("/predict", json=PAYLOAD).status_code == 503


def test_prediction_failure_returns_unavailable(monkeypatch):
    class BrokenModel:
        def predict(self, features):
            raise ValueError("internal model detail")

    monkeypatch.setattr("src.predict.load_model", lambda: BrokenModel())
    with TestClient(app) as client:
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 503
        assert "internal model detail" not in response.text
