# Final offline holdout

Reserve the final 14 days before selecting or tuning a model. This is an optional
portfolio evaluation protocol, separate from the ongoing batch retraining demo.
It uses a new runtime so existing models and data remain intact.

Starting with a validated daily CSV at `runtime/data/daily_demand.csv`, run:

```powershell
$env:APP_RUNTIME_DIR = 'runtime/holdout-experiment'
Remove-Item Env:MLFLOW_TRACKING_URI -ErrorAction SilentlyContinue
python -m src.holdout prepare --input runtime/data/daily_demand.csv
python -m src.train
```

These commands perform actual training; run them only when ready for that workload.
Preparation requires that the selected runtime directory does not already exist.
It writes only development dates to the path consumed by training. The complete
history is stored separately for holdout features, with a checksum and boundary
manifest. Training still reserves the final 14 development days for candidate
selection. At least 35 development days plus 14 holdout days are required; this
is a code minimum, not a recommendation for an adequate real-data sample.

Freeze the model configuration and selection procedure before reporting:

```powershell
python -m src.holdout evaluate
```

A selected production model must exist and both its training and selection dates
must precede the holdout. If no candidate qualified for promotion, report that
outcome; do not bypass the gate to obtain a holdout score.

The command writes `holdout/evaluation.json` in the selected runtime, with MAE,
RMSE, seasonal-naive comparison, model version, dates and data checksum. It does
not train, promote, reject or replace a model. Repeat reporting into the same
file is refused. This is a procedural safeguard, not tamper-proof isolation:
users can read the reserved CSV or create another runtime. Do not tune parameters
or choose another model after seeing holdout results and still call it untouched.
Use new future observations for another unbiased final assessment.

Forecasts are rolling one-day-ahead. Earlier holdout outcomes may supply lagged
inputs for later days, but are never used to refit the model. This is not a
multi-day forecast issued at a single origin. Report actual errors even when
worse than the baseline; no real-data performance result is claimed by tests.

For this offline experiment, keep reserved observations out of development data
and drift-triggered retraining until evaluation is finished. Return to normal
local settings afterward:

```powershell
Remove-Item Env:APP_RUNTIME_DIR
```

If you previously used a custom tracking URI, restore it explicitly as well.
