# Optional Docker setup

On a 4 GB laptop, prefer the normal Python environment. Docker is optional;
GitHub Actions builds and smoke-tests the image on its runner. No cloud service
or image publishing is configured. The image uses pinned Python dependencies,
CPU-only XGBoost, one API worker, and the OpenMP runtime rather than a compiler
suite. The Python slim tag and Debian packages can receive updates; builds are
not byte-for-byte locked.

## Start only the API

These commands require Docker on the machine where you choose to run them.
They are not required for ordinary local development.

```powershell
docker compose build api
docker compose up -d api
```

The API listens at http://127.0.0.1:8000. A fresh volume has no model: `/health`
returns 200, while `/ready` and valid prediction requests return 503. This is
expected. Building an image never trains a model or includes a dataset.

## Supply data and train explicitly

First create a validated daily CSV using the documented data processing steps.
The example below assumes it exists at `runtime/data/daily_demand.csv` locally.

```powershell
docker compose exec api mkdir -p /runtime/data
docker compose cp runtime/data/daily_demand.csv api:/runtime/data/daily_demand.csv
docker compose run --rm api python -m src.train
```

Training writes candidates and evaluation records to the named volume. It can
reject a candidate, leaving the API unready when no production model exists.
Never bypass promotion just to make readiness green. Run one training job at a
time. This optional training command consumes resources on the Docker host;
it is not a recommended extra workload for the 4 GB laptop.

## Optional MLflow viewer

After building the image:

```powershell
docker compose --profile tracking up -d mlflow
```

Open http://127.0.0.1:5000. The viewer uses the exact MLflow version already
installed in the application image; it does not install packages at startup.
Both services mount the same named volume at `/runtime`. The API explicitly
tracks directly to `sqlite:////runtime/mlflow.db`; the optional UI reads that
same store. It is not an HTTP tracking server, so no `localhost` server URI is
used between containers. Artifacts created inside Docker keep valid `/runtime`
paths in both services. Prediction does not need the viewer to be running.

Local Python development keeps its own runtime directory and SQLite database.
The Docker named volume is separate: do not copy a developer-local database
into it, since stored artifact paths may point to that developer's machine.
There is no dependency on a committed database or test-created directory.

## Stop

```powershell
docker compose --profile tracking down
```

The named volume persists across container recreation. `down --volumes` deletes
its data, models, reports and tracking history, so it is not a routine stop
command. Ports bind only to the Docker host's loopback address. The API's
administrative routes remain unauthenticated and are intended for this local demo.

CI runs the full test suite, builds the image, validates Compose syntax, and
starts an empty API container to check liveness and missing-model readiness.
It does not perform a complete two-service training deployment test.
