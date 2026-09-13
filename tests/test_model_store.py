import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS
from src.model_store import ModelStore


@pytest.fixture
def candidate():
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

    return model, features


@pytest.fixture
def store(tmp_path):
    return ModelStore(tmp_path / "models")


def save_version(store, candidate):
    model, _ = candidate
    return store.save_candidate(
        model,
        training_end="2023-01-31",
        evaluation_end="2023-02-14",
        run_id="test-run",
    )


def test_saving_candidate_does_not_create_production(store, candidate):
    version = save_version(store, candidate)

    assert store.production_version() is None

    model, metadata = store.load_version(version)
    assert metadata["training_end"] == "2023-01-31"
    assert metadata["run_id"] == "test-run"

    np.testing.assert_allclose(
        model.predict(candidate[1]),
        candidate[0].predict(candidate[1]),
    )


def test_promotion_retains_previous_version(store, candidate):
    first = save_version(store, candidate)
    store.promote_candidate(first, expected_current_version=None)

    original_model = (
        store.version_directory(first) / "model.json"
    ).read_bytes()

    second = save_version(store, candidate)
    store.promote_candidate(second, expected_current_version=first)

    pointer = json.loads(store.production_path.read_text(encoding="utf-8"))

    assert pointer["version"] == second
    assert pointer["previous_version"] == first
    assert (
        store.version_directory(first) / "model.json"
    ).read_bytes() == original_model


def test_stale_promotion_is_rejected(store, candidate):
    first = save_version(store, candidate)
    second = save_version(store, candidate)

    store.promote_candidate(first, expected_current_version=None)

    with pytest.raises(RuntimeError, match="Production changed"):
        store.promote_candidate(second, expected_current_version=None)

    assert store.production_version() == first


def test_failed_pointer_update_preserves_production(
    store, candidate, monkeypatch
):
    first = save_version(store, candidate)
    store.promote_candidate(first, expected_current_version=None)
    second = save_version(store, candidate)

    def fail_replace(*args, **kwargs):
        raise OSError("Simulated write failure")

    monkeypatch.setattr("src.model_store.os.replace", fail_replace)

    with pytest.raises(OSError, match="Simulated write failure"):
        store.promote_candidate(second, expected_current_version=first)

    assert store.production_version() == first
    assert not (store.directory / ".promotion.lock").exists()


def test_invalid_metadata_blocks_loading(store, candidate):
    version = save_version(store, candidate)
    path = store.version_directory(version) / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["features"] = ["wrong_feature"]

    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata is incompatible"):
        store.load_version(version)


def test_overlapping_promotion_is_blocked(store, candidate):
    version = save_version(store, candidate)

    with store.promotion_lock():
        with pytest.raises(RuntimeError, match="Promotion is locked"):
            store.promote_candidate(
                version,
                expected_current_version=None,
            )

    assert store.production_version() is None