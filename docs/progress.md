# Project cleanup checkpoints

Work is being verified in phases. A passing foundation checkpoint does not mean
that forecasting methodology or production promotion is already correct.

## Phase 1: reproducible runtime and isolated tests

- Centralized runtime paths in `src/config.py`; environment configuration is read
  when an operation starts. Imports do not initialize MLflow or create storage.
- Default tracking is a newly created SQLite database under ignored `runtime/`.
  An explicit HTTP tracking URI leaves artifact configuration to that server.
- Each test receives its own temporary data, models, reports and tracking database.
- Direct dependencies live in `requirements.in` and `requirements-dev.in`;
  generated `.txt` files also pin transitive dependencies.
- The CPU-only XGBoost distribution replaces the GPU-capable package.
- Docker no longer copies datasets or models from the build context.
- CI installs test dependencies, checks compatibility and uploads JUnit results.
- The old database, model and HTML report were removed from Git tracking; their
  local files were retained.

Checkpoint result on Python 3.12.14 / Windows:

- Full current suite: **9 passed** (first run: 270.07 seconds).
- Dependency compatibility: **125 packages checked, no conflicts**.
- CPU package check: `xgboost-cpu==2.1.4`, no NVIDIA distributions installed.
- Python source compilation and Git whitespace checks passed.
- Workflow and Compose YAML parsed; CI test-dependency wiring and Docker's lack
  of dataset/model COPY instructions were checked.
- Original model and database SHA-256 hashes remained unchanged.
- Third-party deprecation warnings remain visible. The existing root pytest
  cache was unwritable, so subsequent runs use `test-results/.pytest_cache`.

This is local verification, not a new GitHub Actions run or a Docker build.

Validation commands from the project root (Python 3.12):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m pytest -v
```

The working validation environment for this cleanup is `.venv-check`; the
pre-existing `.venv` could not launch its configured Python interpreter.

Default processed output is now `runtime/data/daily_demand.csv`, and the current
model path is `runtime/models/demand_model.pkl`. Old root-level data/model/database
files are not migrated or reused automatically. Raw input remains `data/raw/`.
Set `APP_RUNTIME_DIR` to select another output root, and `MLFLOW_TRACKING_URI` to
select another tracking store. Start with an empty store under the pinned MLflow
version; do not point it at the old committed database.

## Remaining phases

1. Shared features, schema validation, calendar continuity and leakage removal.
2. Candidate evaluation against the incumbent and seasonal-naive baseline;
   versioned artifacts, promotion/rejection and failure protection.
3. Inference lifecycle and drift-to-retraining orchestration, simulation isolation.
4. Final Docker/Compose, GitHub Actions, cleanup and README verification.

Training still uses the original overwrite policy during this first checkpoint.
Do not treat it as a safe production-promotion workflow yet. Tests exercise that
legacy path only inside temporary directories. Docker has only been statically
reviewed until an actual runner build completes.
