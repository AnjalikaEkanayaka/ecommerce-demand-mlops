"""Explicit batch monitoring; simulated data never triggers retraining."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from src.config import Settings
from src.features import TARGET_COLUMN, create_time_features


def validated_daily(frame):
    create_time_features(frame)  # Reuse schema checks; retain history rows here.
    result = frame[['date', TARGET_COLUMN]].copy()
    result['date'] = pd.to_datetime(result['date'])
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN])
    return result.sort_values('date').reset_index(drop=True)


def describe_window(frame):
    return {
        'start': frame['date'].min().date().isoformat(),
        'end': frame['date'].max().date().isoformat(),
        'rows': len(frame),
        'sha256': hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
    }


def generate_drift_report(reference_df, current_df, *, output_dir):
    from evidently.metrics import DatasetDriftMetric
    from evidently.report import Report

    # A preset also includes DataDriftTable, which duplicates dataset_drift.
    # Request only the aggregate metric consumed by this monitoring workflow.
    report = Report(metrics=[DatasetDriftMetric(
        columns=[TARGET_COLUMN], stattest='ks', stattest_threshold=0.05,
    )])
    report.run(reference_data=reference_df[[TARGET_COLUMN]],
               current_data=current_df[[TARGET_COLUMN]])
    matches = [m['result']['dataset_drift'] for m in report.as_dict()['metrics']
               if 'dataset_drift' in m.get('result', {})]
    if len(matches) != 1 or not isinstance(matches[0], bool):
        raise ValueError('Evidently did not return an unambiguous drift decision.')
    report.save_html(str(output_dir / 'drift_report.html'))
    return matches[0]


def run_monitoring_pipeline(reference_df, current_df, *, source, retrain=False, settings=None):
    settings = settings or Settings.from_env()
    if source not in {'observed', 'simulated'}:
        raise ValueError('source must be observed or simulated')
    if source == 'simulated' and retrain:
        raise ValueError('Simulated drift cannot trigger retraining.')
    reference = validated_daily(reference_df)
    current = validated_daily(current_df)
    if min(len(reference), len(current)) < 28:
        raise ValueError('Each monitoring window requires at least 28 days.')
    if reference['date'].max() >= current['date'].min():
        raise ValueError('Reference dates must precede current dates without overlap.')
    if retrain:
        history = validated_daily(pd.read_csv(settings.processed_data_path))
        observed = pd.concat([reference, current]).set_index('date')[TARGET_COLUMN]
        matched = history.set_index('date')[TARGET_COLUMN].reindex(observed.index)
        if matched.isna().any() or not matched.eq(observed).all():
            raise ValueError('Training data must contain the monitored observations unchanged.')
        if history['date'].max() != current['date'].max():
            raise ValueError('Current monitoring must end on the latest training-data date.')
    run_id = uuid4().hex
    output_dir = settings.report_dir / source / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    drift = generate_drift_report(reference, current, output_dir=output_dir)
    result = {
        'run_id': run_id, 'source': source, 'dataset_drift': drift,
        'monitored_column': TARGET_COLUMN, 'test': 'ks', 'threshold': 0.05,
        'reference': describe_window(reference), 'current': describe_window(current),
        'retraining': {'status': 'skipped'},
    }
    def save_summary():
        (output_dir / 'summary.json').write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    save_summary()
    if retrain and drift:
        from src.retrain import execute_retraining_pipeline
        try:
            result['retraining'] = execute_retraining_pipeline(
                drift_threshold_exceeded=True, settings=settings)
        except Exception:
            result['retraining'] = {'status': 'failed'}
            save_summary()
            raise
        save_summary()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    current = pd.read_csv(args.current)
    if args.simulate:
        current = validated_daily(current)
        current[TARGET_COLUMN] = (current[TARGET_COLUMN] * 1.5).round().astype(int)
    print(json.dumps(run_monitoring_pipeline(
        pd.read_csv(args.reference), current,
        source='simulated' if args.simulate else 'observed', retrain=args.retrain,
    ), indent=2))


if __name__ == '__main__':
    main()
