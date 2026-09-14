# E-commerce demand forecasting MLOps

[![CI](https://github.com/AnjalikaEkanayaka/ecommerce-demand-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/AnjalikaEkanayaka/ecommerce-demand-mlops/actions/workflows/ci-cd.yml)

A CPU-only portfolio project for aggregate daily recorded order intake using
Python 3.12, XGBoost, MLflow, FastAPI and Evidently. It demonstrates chronological
forecast evaluation, candidate rejection/promotion and explicit batch monitoring.
It is a local demonstration, not a cloud deployment or continuously monitored service.

## Lifecycle

```text
Orders + order items → validated daily counts → historical features
  → candidate training → MLflow tracking → chronological validation
  → compare with seasonal naive and current production
  → promote only when qualified → FastAPI predictions

Separate observed daily batches → Evidently drift check
  → optional training request → same evaluation and promotion gates
```

Monitoring consumes supplied daily observations, not telemetry collected by the API.
Simulated monitoring is labeled separately and cannot trigger retraining.

## Data and forecasting method

The input is the Olist Brazilian E-Commerce Dataset, supplied separately by the
user. Put `olist_orders_dataset.csv` and `olist_order_items_dataset.csv` in
`data/raw/`. Datasets are not committed. Obtain them from the original dataset
publisher and respect its usage terms.

Each unique `(order_id, order_item_id)` contributes one recorded item to its
purchase date. The target column is `total_units_sold`, but it includes recorded
items from canceled orders: it is not fulfilled sales or unconstrained demand.
The model forecasts the marketplace total, not individual SKUs.

Features are day of week, month, day of month, lag 1, lag 7 and the mean of the
previous seven days. Same-day price is excluded. Missing calendar dates fail
validation unless zero filling is explicitly requested for a complete extract.

The latest 14 development days are validation; earlier days are training.
Evaluation is rolling one-day-ahead: observed demand from an earlier validation
day can supply a later day's lag. This is not a 14-day forecast from one origin.
See [data methodology](docs/data-methodology.md).

XGBoost uses 100 trees, depth 4, learning rate 0.05, a fixed seed, histogram
training and one CPU thread. A candidate must lower MAE by at least 1% against
both the lag-7 baseline and any incumbent, with no RMSE regression. All models
are compared on identical validation observations. Ties are rejected.

## Local setup

Docker is optional. On a 4 GB laptop, use one Python environment and avoid running
training, Docker and several development tools simultaneously. No GPU is needed.
From the project directory in PowerShell:

```powershell
# Create this only if you do not already have a working environment.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip check
```

Use `requirements.txt` for runtime-only installation. The `.in` files list direct
dependencies; `.txt` files pin the resolved dependency sets used by CI. Do not
install an unpinned MLflow version over this environment.

## Prepare data and train

Choose complete dates within your extract. Replace these example boundaries if
needed; the program validates them against the input:

```powershell
python -m src.data_loader --raw-dir data/raw --start-date 2017-01-01 --end-date 2018-08-31
```

Only add `--fill-missing-days` after confirming that absent days mean zero items,
not incomplete data. Output defaults to `runtime/data/daily_demand.csv`.

For an honest final portfolio assessment, follow [the holdout protocol](docs/holdout.md)
**before training or selecting a model**. It reserves a separate final period in
a fresh runtime. For the ordinary development/retraining demonstration:

```powershell
python -m src.train
```

The result is `promoted`, `rejected` or `skipped`. No production model is created
if the first candidate fails the baseline gate. Do not bypass the gate merely
to obtain predictions.

## Measured result

On the reserved 18–31 July 2018 holdout, XGBoost achieved MAE **62.03** versus
**67.50** for the seasonal baseline, but RMSE **118.80** versus **86.23**.
That is 8.1% lower MAE and 37.8% higher RMSE: a mixed result, not consistent
outperformance. The model passed the earlier validation gate; holdout reporting
did not alter promotion. See [results, dates and limitations](docs/results.md).

## MLflow and model storage

`APP_RUNTIME_DIR` defaults to `runtime`. `MLFLOW_TRACKING_URI` defaults to a new
SQLite store inside that directory; `MLFLOW_EXPERIMENT_NAME` selects the experiment.
Tracking is initialized only when training requests it. No committed database is
required. For the default local runtime:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///runtime/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 after training. MLflow records parameters, comparison
metrics, native candidate artifacts and the eligibility decision. The local
`runtime/models/production.json` pointer determines the served version; an
MLflow eligibility tag is not proof that publication succeeded.

Native XGBoost JSON versions are retained independently. Publication checks that
the incumbent has not changed and atomically replaces the pointer under a file
lock. Tracking failures prevent publication. A process crash can leave
`.promotion.lock`; remove it only after confirming no promotion process is running.
This is single-machine storage, not a distributed model registry. Legacy pickles
without evaluation metadata cannot be silently promoted.

## Prediction API

```powershell
python -m uvicorn src.predict:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for request examples.

- `/health`: process liveness, independent of model availability.
- `/ready`: 200 for a compatible production model; otherwise 503.
- `/predict`: six numeric features; invalid input returns 422, unavailable or
  invalid model output returns 503.
- `/drift-report`: serves a report selected by source and run ID, without executing monitoring.
- `/retrain?force=true`: requests evaluation, never unconditional promotion.

Administrative routes are synchronous and unauthenticated; keep this demo local.
See [API behavior and payload](docs/api.md).

## Drift and retraining

Provide separate reference/current daily CSVs with at least 28 days each, in
chronological non-overlapping windows:

```powershell
python -m src.drift_monitor --reference data/processed/reference.csv --current data/processed/current.csv
```

Add `--simulate` for an explicitly synthetic shift, or `--retrain` for an observed
batch that may request evaluation if drift is detected. These flags cannot be
combined. Observations must match the configured training history before a trigger
is allowed. A fresh validation window after the incumbent's prior evaluation is
required by the training pipeline.

Evidently uses a KS test at 0.05 on demand distribution. Seasonality, dependent
daily counts and repeated testing limit its interpretation; it does not prove
forecast degradation. No scheduler or data collector is included. Reports and
JSON summaries are stored under ignored runtime directories.
See [monitoring details](docs/monitoring.md).

## Tests, CI and optional Docker

```powershell
python -m pytest -v --junitxml=test-results/pytest.xml
```

Tests use small deterministic datasets and temporary stores. They cover schema
validation, historical features, prediction, actual Evidently reports, MLflow
round trips, candidate evaluation, promotion failure protection and holdout
boundaries. Synthetic test success does not establish real forecasting quality.
On the limited-memory laptop, prefer GitHub Actions for the full verification.

GitHub Actions installs pinned dependencies, runs `pip check` and pytest, uploads
JUnit results, builds Docker, validates Compose and checks empty-container startup.
It does not deploy or publish an image. The optional MLflow viewer shares an
explicit SQLite runtime volume with the API. See [Docker instructions](docs/docker.md).

## Project layout

- `src/`: processing, features, training, evaluation, model storage, API and monitoring.
- `tests/`: isolated unit and integration tests.
- `docs/`: methodology and operational instructions.
- `runtime/`: ignored data, models, SQLite tracking, reports and holdout outputs.
- `.github/workflows/`: CI verification.

See [real-data results](docs/results.md) and [verification status](docs/progress.md).
