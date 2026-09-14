# Batch demand drift monitoring

Monitoring compares recorded daily demand, not prediction errors or live request
telemetry. It cannot establish model degradation or concept drift. Reference and
current CSV files need `date` and `total_units_sold`, each with at least 28 complete,
consecutive days. Reference dates must precede current dates without overlap.
Choose a representative historical reference and comparable seasonal windows.
Neither window is silently filled with zeros or selected by a test fixture.

Observed-data check (paths refer to your own local, uncommitted CSV files):

```powershell
python -m src.drift_monitor --reference data/processed/reference.csv --current data/processed/current.csv
```

Add `--simulate` to multiply current counts by 1.5 for a labeled demonstration.
Simulation is not evidence of real production drift and cannot use `--retrain`.
The source label is caller-declared; software cannot verify that supplied CSVs
came from production. No data collector or scheduler is included.

Evidently uses a two-sample KS test with threshold 0.05 on the demand column.
This is a simple screening heuristic: daily observations are dependent and counts
are discrete, so the nominal p-value is not a calibrated production alarm.
Seasonality, small windows and repeated checks can mislead. Twenty-eight days is
an operational minimum, not a statistical guarantee. A drift signal alone never
justifies model promotion.

For observed data, `--retrain` requests the shared training/evaluation pipeline
only when drift is detected. First update the configured processed training CSV
with validated observations. It must contain both windows unchanged and end on
the current window's last date. Monitoring does not append or overwrite data.
The existing chronological validation, seasonal baseline, incumbent comparison,
MLflow logging, and atomic promotion safeguards still apply. The result can be
promoted, rejected, skipped (for example, insufficient fresh validation), or fail.
Run one batch at a time and do not change input files while it runs.

Every successful check creates an HTML report and `summary.json` under
`runtime/reports/<observed-or-simulated>/<run-id>/` (or APP_RUNTIME_DIR).
The summary records dates, row counts, input hashes, drift decision and retraining
outcome. Failed retraining is recorded and the command fails visibly. Failed
report generation cannot trigger training. Reports are ignored by Git.

View an existing report using `/drift-report?source=observed&run_id=<run-id>`.
The endpoint only reads the selected report; it does not generate data, monitor,
or retrain. There is no background or continuous production monitoring.
