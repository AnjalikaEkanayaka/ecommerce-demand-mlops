# Prediction API

Run `python -m uvicorn src.predict:app --host 127.0.0.1 --port 8000`.
Local development does not require Docker. Interactive request documentation
is available at `http://127.0.0.1:8000/docs`.

- `GET /health` reports process liveness without loading a model.
- `GET /ready` returns 200 when the production artifact loads with the expected
  feature contract, or 503 when it is missing, corrupted, or incompatible.
- `POST /predict` accepts the six numeric features below. Unknown fields,
  numeric strings, booleans, and out-of-range values are rejected with 422.
  A missing model or invalid prediction produces 503.

```json
{
  "day_of_week": 1,
  "month": 5,
  "day": 10,
  "lag_1": 100.0,
  "lag_7": 100.0,
  "rolling_mean_7": 100.0
}
```

The caller supplies calendar features for the forecast date and demand history
available before that date. Monday is day 0. The API checks individual field
ranges, not calendar consistency or the provenance of caller-supplied history.
Forecasts are expected unit counts and may be fractional. Negative or nonfinite
outputs are rejected rather than silently changing the evaluated model's output.

The API reads the version selected by `production.json` under
`APP_RUNTIME_DIR/models`. It loads the artifact for each readiness or prediction
request, so subsequent requests see promoted versions without restarting. This
simple approach is intended for low request volume; there is no model cache or
background polling. Prediction does not contact MLflow.

The existing `/drift-report` route runs a simulated drift demonstration.
It is not live production monitoring. `/retrain?force=true` requests the shared
evaluation pipeline; it does not bypass promotion criteria. These synchronous
administrative routes have no authentication and this demo should run locally;
they will be reviewed in the monitoring phase.
