import json

import numpy as np
import pandas as pd
import pytest

from src.holdout import evaluate_holdout, prepare_holdout
from src.model_store import ModelStore


@pytest.fixture
def reserved(tmp_path, settings):
    path = tmp_path / 'input.csv'
    pd.DataFrame({'date': pd.date_range('2023-01-01', periods=90),
                  'total_units_sold': np.full(90, 100)}).to_csv(path, index=False)
    manifest = prepare_holdout(path, settings=settings)
    return path, manifest


def test_reserved_dates_excluded_from_training(reserved, settings):
    development = pd.read_csv(settings.processed_data_path)
    assert len(development) == 76
    assert development['date'].max() < reserved[1]['holdout_start']
    assert reserved[1]['holdout_days'] == 14


def test_prepare_refuses_existing_runtime(reserved, settings):
    original = settings.processed_data_path.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_holdout(reserved[0], settings=settings)
    assert settings.processed_data_path.read_bytes() == original


def test_holdout_report_does_not_change_production(reserved, settings, request):
    request.getfixturevalue('model_artifact')
    store = ModelStore(settings.runtime_dir / 'models')
    before = store.production_path.read_bytes()
    result = evaluate_holdout(settings)
    assert result['rows'] == 14
    assert result['model']['mae'] == pytest.approx(0)
    assert result['seasonal_naive']['mae'] == pytest.approx(0)
    assert store.production_path.read_bytes() == before
    with pytest.raises(FileExistsError):
        evaluate_holdout(settings)


def test_overlapping_model_cannot_use_holdout(reserved, settings, request):
    model = request.getfixturevalue('model_artifact')
    store = ModelStore(settings.runtime_dir / 'models')
    version = store.save_candidate(model, training_end='2023-03-17', evaluation_end='2023-03-31')
    store.promote_candidate(version, expected_current_version=store.production_version())
    with pytest.raises(ValueError, match='overlaps'):
        evaluate_holdout(settings)
    assert not (settings.runtime_dir / 'holdout/evaluation.json').exists()


def test_modified_reserved_data_is_rejected(reserved, settings):
    path = settings.runtime_dir / 'holdout/history.csv'
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        evaluate_holdout(settings)


def test_missing_selected_model(reserved, settings):
    with pytest.raises(ValueError, match='No selected'):
        evaluate_holdout(settings)
