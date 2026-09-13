"""Every test owns its data, model, reports and tracking database."""

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.features import FEATURE_COLUMNS


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-demand")
    return Settings.from_env()


@pytest.fixture
def daily_data(settings):
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "total_units_sold": rng.integers(50, 150, len(dates)),
        "avg_price": rng.uniform(20, 60, len(dates)),
    })
    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.processed_data_path, index=False)
    return df


@pytest.fixture
def model_artifact(settings):
    from xgboost import XGBRegressor

    from src.model_store import ModelStore

    features = pd.DataFrame(
        np.ones((10, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    model = XGBRegressor(
        n_estimators=2,
        max_depth=1,
        n_jobs=1,
        random_state=42,
    )
    model.fit(features, np.full(10, 100.0))

    store = ModelStore(settings.runtime_dir / "models")
    version = store.save_candidate(
        model,
        training_end="2022-12-31",
        evaluation_end="2023-01-14",
    )

    # Install a known model only in this test's temporary directory.
    store.promote_candidate(
        version,
        expected_current_version=None,
    )

    return model
