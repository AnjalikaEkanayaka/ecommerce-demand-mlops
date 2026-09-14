import json

import numpy as np
import pandas as pd
import pytest

from src.drift_monitor import run_monitoring_pipeline


@pytest.fixture
def windows():
    reference = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=28),
                              'total_units_sold': np.arange(28) + 100})
    current = reference.copy()
    current['date'] += pd.Timedelta(days=28)
    return reference, current


@pytest.mark.parametrize('shift,expected', [(0, False), (1000, True)])
def test_real_evidently_report(windows, settings, shift, expected):
    reference, current = windows
    current['total_units_sold'] += shift
    result = run_monitoring_pipeline(reference, current, source='observed', settings=settings)
    assert result['dataset_drift'] is expected
    directory = settings.report_dir / 'observed' / result['run_id']
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary == result
    assert 'html' in (directory / 'drift_report.html').read_text(encoding='utf-8').lower()
    assert result['reference']['rows'] == 28


@pytest.mark.parametrize('drift', [True, False])
def test_drift_controls_retraining_request(windows, settings, monkeypatch, drift):
    reference, current = windows
    settings.processed_data_path.parent.mkdir(parents=True)
    pd.concat([reference, current]).to_csv(settings.processed_data_path, index=False)
    monkeypatch.setattr('src.drift_monitor.generate_drift_report', lambda *a, **k: drift)
    calls = []
    def retrain(**kwargs):
        calls.append(kwargs)
        return {'status': 'rejected', 'reason': 'Candidate did not improve'}
    monkeypatch.setattr('src.retrain.execute_retraining_pipeline', retrain)
    result = run_monitoring_pipeline(reference, current, source='observed', retrain=True, settings=settings)
    assert len(calls) == int(drift)
    assert result['retraining']['status'] == ('rejected' if drift else 'skipped')
    if drift:
        assert calls[0]['drift_threshold_exceeded'] is True
        assert calls[0]['settings'] == settings


def test_simulation_cannot_retrain(windows, settings):
    with pytest.raises(ValueError, match='Simulated'):
        run_monitoring_pipeline(*windows, source='simulated', retrain=True, settings=settings)
    assert not settings.report_dir.exists()


@pytest.mark.parametrize('problem', ['overlap', 'short', 'negative', 'gap'])
def test_invalid_windows_fail_before_report(windows, settings, problem):
    reference, current = windows
    if problem == 'overlap':
        current['date'] = reference['date']
    elif problem == 'short':
        current = current.iloc[:14]
    elif problem == 'negative':
        current.loc[0, 'total_units_sold'] = -1
    else:
        current = current.drop(index=10)
    with pytest.raises(ValueError):
        run_monitoring_pipeline(reference, current, source='observed', settings=settings)
    assert not settings.report_dir.exists()


def test_mismatched_training_data_blocks_trigger(windows, settings):
    reference, current = windows
    history = pd.concat([reference, current])
    history['total_units_sold'] += 1
    settings.processed_data_path.parent.mkdir(parents=True)
    history.to_csv(settings.processed_data_path, index=False)
    with pytest.raises(ValueError, match='unchanged'):
        run_monitoring_pipeline(reference, current, source='observed', retrain=True, settings=settings)
    assert not settings.report_dir.exists()


def test_failed_retraining_is_recorded(windows, settings, monkeypatch):
    settings.processed_data_path.parent.mkdir(parents=True)
    pd.concat(windows).to_csv(settings.processed_data_path, index=False)
    monkeypatch.setattr('src.drift_monitor.generate_drift_report', lambda *a, **k: True)
    def fail(**kwargs):
        raise OSError('training failed')
    monkeypatch.setattr('src.retrain.execute_retraining_pipeline', fail)
    with pytest.raises(OSError, match='training failed'):
        run_monitoring_pipeline(*windows, source='observed', retrain=True, settings=settings)
    summary = next(settings.report_dir.glob('observed/*/summary.json'))
    assert json.loads(summary.read_text())['retraining']['status'] == 'failed'
