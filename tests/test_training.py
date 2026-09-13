import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor

from src.model_store import ModelStore
from src.retrain import execute_retraining_pipeline
from src.train import train_model


@pytest.fixture
def pipeline_data(settings):
    """Small dataset with a predictable seasonal-naive error."""
    demand = np.full(90, 100, dtype=int)

    # These seven days supply the first week's validation lag_7 values.
    demand[-21:-14] = 50

    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=90, freq="D"),
            "total_units_sold": demand,
        }
    )

    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(settings.processed_data_path, index=False)

    return data


def fixed_candidate(value):
    """Create controlled candidates to test the promotion decision."""
    def fit(features, target):
        model = XGBRegressor(
            n_estimators=2,
            max_depth=1,
            n_jobs=1,
            random_state=42,
        )
        model.fit(features, np.full(len(features), value, dtype=float))
        return model

    return fit


def test_good_candidate_is_promoted(
    pipeline_data, settings, monkeypatch
):
    monkeypatch.setattr("src.train.fit_candidate", fixed_candidate(100))

    result = train_model(settings)
    store = ModelStore(settings.runtime_dir / "models")

    assert result["status"] == "promoted"
    assert store.production_version() == result["candidate_version"]
    assert result["metrics"]["candidate"]["mae"] == pytest.approx(0)
    assert result["metrics"]["seasonal_naive"]["mae"] == pytest.approx(25)

    model, metadata = store.load_production()

    assert metadata["training_end"] == "2023-03-17"
    assert metadata["evaluation_end"] == "2023-03-31"
    assert model.n_features_in_ == 6

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    run = client.get_run(metadata["run_id"])

    assert run.data.metrics["candidate_mae"] == pytest.approx(0)
    assert run.data.metrics["seasonal_naive_mae"] == pytest.approx(25)


def test_worse_candidate_preserves_production(
    pipeline_data, settings, model_artifact, monkeypatch
):
    store = ModelStore(settings.runtime_dir / "models")
    previous_version = store.production_version()
    previous_pointer = store.production_path.read_bytes()

    monkeypatch.setattr("src.train.fit_candidate", fixed_candidate(200))

    result = train_model(settings)

    assert result["status"] == "rejected"
    assert store.production_version() == previous_version
    assert store.production_path.read_bytes() == previous_pointer

    # Production was evaluated on the current validation observations.
    assert result["metrics"]["production"]["mae"] == pytest.approx(0)
    assert result["metrics"]["candidate"]["mae"] == pytest.approx(100)

    # A rejected candidate remains available for inspection.
    assert result["candidate_version"] != previous_version
    assert (
        store.version_directory(result["candidate_version"]) / "model.json"
    ).exists()


def test_manual_trigger_cannot_reuse_previous_validation_window(
    pipeline_data, settings, monkeypatch
):
    monkeypatch.setattr("src.train.fit_candidate", fixed_candidate(100))
    first = train_model(settings)

    assert first["status"] == "promoted"

    store = ModelStore(settings.runtime_dir / "models")
    previous_pointer = store.production_path.read_bytes()

    def must_not_train(*args, **kwargs):
        pytest.fail("Training must not start without a fresh validation window.")

    monkeypatch.setattr("src.train.fit_candidate", must_not_train)

    result = execute_retraining_pipeline(
        drift_threshold_exceeded=True,
        settings=settings,
    )

    assert result["status"] == "skipped"
    assert store.production_path.read_bytes() == previous_pointer


def test_validation_cannot_overlap_production_training(
    pipeline_data, settings, model_artifact
):
    store = ModelStore(settings.runtime_dir / "models")
    previous_version = store.production_version()

    # Deliberately make production ineligible for this validation window.
    version = store.save_candidate(
        model_artifact,
        training_end="2023-03-20",
        evaluation_end="2023-03-31",
    )
    store.promote_candidate(
        version,
        expected_current_version=previous_version,
    )

    previous_pointer = store.production_path.read_bytes()

    with pytest.raises(ValueError, match="overlaps"):
        train_model(settings)

    assert store.production_path.read_bytes() == previous_pointer


def test_tracking_failure_preserves_production(
    pipeline_data, settings, model_artifact, monkeypatch
):
    # Candidate 120 would beat production 100 on this validation period.
    data = pipeline_data.copy()
    data.loc[data.index[-14:], "total_units_sold"] = 120
    data.to_csv(settings.processed_data_path, index=False)

    store = ModelStore(settings.runtime_dir / "models")
    previous_pointer = store.production_path.read_bytes()

    monkeypatch.setattr("src.train.fit_candidate", fixed_candidate(120))

    def fail_logging(*args, **kwargs):
        raise OSError("Simulated tracking failure")

    monkeypatch.setattr("src.train.mlflow.log_artifact", fail_logging)

    with pytest.raises(OSError, match="Simulated tracking failure"):
        train_model(settings)

    assert store.production_path.read_bytes() == previous_pointer


def test_real_candidate_training_completes(pipeline_data, settings):
    # This test uses the actual training function, without a controlled candidate.
    result = train_model(settings)
    store = ModelStore(settings.runtime_dir / "models")

    assert result["status"] in {"promoted", "rejected"}

    model, metadata = store.load_version(result["candidate_version"])
    assert model.n_features_in_ == 6
    assert metadata["run_id"] is not None

    if result["status"] == "promoted":
        assert store.production_version() == result["candidate_version"]
    else:
        assert store.production_version() is None


def test_legacy_pickle_is_not_silently_replaced(
    pipeline_data, settings
):
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    original_content = b"legacy model placeholder"
    settings.model_path.write_bytes(original_content)

    with pytest.raises(ValueError, match="legacy pickle"):
        train_model(settings)

    assert settings.model_path.read_bytes() == original_content
    assert not (
        settings.runtime_dir / "models" / "production.json"
    ).exists()