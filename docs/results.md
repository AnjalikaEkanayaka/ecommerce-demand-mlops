# Real-data evaluation

Experiment performed locally on 14 September 2026 with the fixed, CPU-only
XGBoost configuration in `src/train.py`. No tuning followed holdout evaluation.

## Protocol and provenance

- Input: Olist recorded order-item counts, 1 February 2017–31 July 2018,
  546 consecutive daily observations. No missing-date zero filling was used.
- Development: 532 days; first seven days provide lag history.
- Training targets: 8 February 2017–3 July 2018 (511 rows).
- Promotion validation: 4–17 July 2018 (14 rows).
- Reserved holdout: 18–31 July 2018 (14 rows).
- Model version: `cbdda609862e44d88e75ba41aba72688`.
- MLflow run: `8585033dd18e4b84bc71c7934f597e10`.
- Reserved history SHA-256:
  `840011fe7bc7bbfb08301aeab86d80d88735189b78ac9f8dd672f43421495361`.

Forecasts are rolling one-day-ahead with observed prior-day history. The weekly
seasonal-naive baseline uses the recorded count seven days earlier. These are
marketplace-level recorded items, including items on canceled orders, not SKU
forecasts or fulfilled sales. Calendar continuity does not prove extract completeness.

## Results

| Period | XGBoost MAE | Baseline MAE | XGBoost RMSE | Baseline RMSE |
| --- | ---: | ---: | ---: | ---: |
| Validation | 39.91 | 69.79 | 46.06 | 76.48 |
| Holdout | 62.03 | 67.50 | 118.80 | 86.23 |

Errors are in recorded items per day. The initial candidate qualified for
promotion against the seasonal baseline on validation; no incumbent existed.
The run therefore demonstrates initial promotion, not a real-data incumbent
replacement. Incumbent replacement and rejection are exercised in isolated tests.

On holdout, MAE was 8.1% lower but RMSE was 37.8% higher than the baseline.
This is mixed performance: average absolute error improved, while squared-error
performance worsened. It does not establish consistent superiority. If these
were promotion-validation metrics, the RMSE guard would reject the candidate.

Holdout evaluation is reporting-only and left the production pointer unchanged.
Its result is not a fresh promotion decision. The model remains the selected
local demonstration artifact, not a recommendation for production deployment.

## Limitations and reproducibility

Fourteen days is a short final window and gives no guarantee across seasons or
different business conditions. The reduction in validation MAE did not translate
into a similar holdout gain. No significance claim or confidence interval is made.
The larger RMSE alone does not identify the dates or causes of large errors.

Follow [the holdout instructions](holdout.md) with the same input period and pinned
dependencies to reproduce the protocol. The summary was checked against local
evaluation artifacts and model metadata. Generated data, SQLite stores and model
files remain ignored by Git; this small summary records the measured evidence.
Future changes informed by this result require new unseen evaluation data.
