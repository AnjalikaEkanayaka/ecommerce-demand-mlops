"""Reserve an offline holdout and report a selected model without promotion."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.drift_monitor import validated_daily
from src.evaluate import evaluate_predictions
from src.features import FEATURE_COLUMNS, TARGET_COLUMN, create_time_features
from src.model_store import ModelStore


def prepare_holdout(input_path, *, holdout_days=14, settings=None):
    settings = settings or Settings.from_env()
    if isinstance(holdout_days, bool) or not isinstance(holdout_days, int) or holdout_days < 14:
        raise ValueError('Reserve at least 14 holdout days.')
    data = validated_daily(pd.read_csv(input_path))
    if len(data) - holdout_days < 35:
        raise ValueError('Need at least 35 development days plus the holdout.')
    # A new directory prevents accidentally mixing this experiment with old models.
    settings.runtime_dir.mkdir(parents=True, exist_ok=False)
    directory = settings.runtime_dir / 'holdout'
    directory.mkdir()
    development = data.iloc[:-holdout_days]
    settings.processed_data_path.parent.mkdir(parents=True)
    development.to_csv(settings.processed_data_path, index=False)
    history_path = directory / 'history.csv'
    data.to_csv(history_path, index=False)
    manifest = {
        'holdout_start': data.iloc[-holdout_days]['date'].date().isoformat(),
        'holdout_days': holdout_days,
        'history_sha256': hashlib.sha256(history_path.read_bytes()).hexdigest(),
    }
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def evaluate_holdout(settings=None):
    settings = settings or Settings.from_env()
    directory = settings.runtime_dir / 'holdout'
    report_path = directory / 'evaluation.json'
    if report_path.exists():
        raise FileExistsError('Holdout already reported; do not tune against this result.')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    history_path = directory / 'history.csv'
    if hashlib.sha256(history_path.read_bytes()).hexdigest() != manifest['history_sha256']:
        raise ValueError('Reserved history changed.')
    production = ModelStore(settings.runtime_dir / 'models').load_production()
    if production is None:
        raise ValueError('No selected production model is available.')
    model, metadata = production
    start = pd.Timestamp(manifest['holdout_start'])
    if max(pd.Timestamp(metadata['training_end']), pd.Timestamp(metadata['evaluation_end'])) >= start:
        raise ValueError('Model training or selection overlaps the holdout.')
    featured = create_time_features(pd.read_csv(history_path))
    held = featured.loc[featured['date'] >= start]
    if len(held) != manifest['holdout_days']:
        raise ValueError('Holdout boundaries do not match the reservation.')
    result = {
        'model_version': metadata['version'],
        'holdout_start': manifest['holdout_start'],
        'holdout_end': held['date'].max().date().isoformat(),
        'rows': len(held),
        'history_sha256': manifest['history_sha256'],
        'model': evaluate_predictions(held[TARGET_COLUMN], model.predict(held[FEATURE_COLUMNS])),
        'seasonal_naive': evaluate_predictions(held[TARGET_COLUMN], held['lag_7']),
        'protocol': 'rolling one-day-ahead; reporting only, no selection or promotion',
    }
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--input', type=Path, required=True)
    prepare.add_argument('--days', type=int, default=14)
    sub.add_parser('evaluate')
    args = parser.parse_args()
    result = (prepare_holdout(args.input, holdout_days=args.days)
              if args.command == 'prepare' else evaluate_holdout())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
