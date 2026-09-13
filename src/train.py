"""Train a candidate and promote it only after chronological evaluation."""

from datetime import date

import mlflow
import pandas as pd
from xgboost import XGBRegressor

from src.config import Settings, configure_tracking
from src.evaluate import evaluate_predictions, promotion_decision
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    create_time_features,
)
from src.model_store import ModelStore


MODEL_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
}


def fit_candidate(features, target):
    """Fit a small CPU-only model."""
    model = XGBRegressor(**MODEL_PARAMETERS)
    model.fit(features, target)
    return model


def train_model(settings=None):
    """Evaluate a candidate without blindly overwriting production."""
    settings = settings or Settings.from_env()

    if not settings.processed_data_path.exists():
        raise FileNotFoundError(
            "Processed data is missing. See "
            "'python -m src.data_loader --help' for processing options."
        )

    store = ModelStore(settings.runtime_dir / "models")
    production = store.load_production()

    # An old pickle does not contain the metadata needed for safe comparison.
    if production is None and settings.model_path.exists():
        raise ValueError(
            "A legacy pickle model exists without evaluation metadata. "
            "Preserve it and use a separate APP_RUNTIME_DIR for this lifecycle."
        )

    daily = pd.read_csv(settings.processed_data_path)
    featured = create_time_features(daily)

    training, validation = chronological_split(
        featured,
        validation_days=14,
    )

    training_end = training["date"].max().date().isoformat()
    validation_start = validation["date"].min().date()
    evaluation_end = validation["date"].max().date().isoformat()

    production_version = None

    if production is not None:
        production_model, production_metadata = production
        production_version = production_metadata["version"]

        previous_training_end = date.fromisoformat(
            production_metadata["training_end"]
        )
        previous_evaluation_end = date.fromisoformat(
            production_metadata["evaluation_end"]
        )

        if validation_start <= previous_training_end:
            raise ValueError(
                "Validation overlaps the production model's training period."
            )

        if validation_start <= previous_evaluation_end:
            return {
                "status": "skipped",
                "reason": (
                    "A completely new 14-day validation window is required."
                ),
            }

    X_train = training[FEATURE_COLUMNS]
    y_train = training[TARGET_COLUMN]
    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    configure_tracking(settings)

    with mlflow.start_run(run_name="demand_candidate") as run:
        candidate = fit_candidate(X_train, y_train)

        candidate_metrics = evaluate_predictions(
            y_validation,
            candidate.predict(X_validation),
        )

        # Seasonal-naive forecast: use demand from seven days earlier.
        baseline_metrics = evaluate_predictions(
            y_validation,
            validation["lag_7"],
        )

        production_metrics = None

        if production is not None:
            production_metrics = evaluate_predictions(
                y_validation,
                production_model.predict(X_validation),
            )

        decision = promotion_decision(
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            production_metrics=production_metrics,
        )

        candidate_version = store.save_candidate(
            candidate,
            training_end=training_end,
            evaluation_end=evaluation_end,
            run_id=run.info.run_id,
        )

        mlflow.log_params(candidate.get_params())
        mlflow.log_params(
            {
                "training_end": training_end,
                "validation_start": validation_start.isoformat(),
                "evaluation_end": evaluation_end,
                "min_mae_improvement": 0.01,
            }
        )

        comparisons = {
            "candidate": candidate_metrics,
            "seasonal_naive": baseline_metrics,
        }

        if production_metrics is not None:
            comparisons["production"] = production_metrics

        for name, metrics in comparisons.items():
            mlflow.log_metrics(
                {
                    f"{name}_mae": metrics["mae"],
                    f"{name}_rmse": metrics["rmse"],
                }
            )

        evaluation = {
            "candidate_version": candidate_version,
            "production_version_compared": production_version,
            "validation_start": validation_start.isoformat(),
            "evaluation_end": evaluation_end,
            "metrics": comparisons,
            "decision": decision,
        }

        mlflow.log_dict(evaluation, "evaluation.json")
        mlflow.set_tag(
            "promotion_eligible",
            str(decision["promote"]).lower(),
        )

        candidate_directory = store.version_directory(candidate_version)

        mlflow.log_artifact(
            str(candidate_directory / "model.json"),
            artifact_path="candidate",
        )
        mlflow.log_artifact(
            str(candidate_directory / "metadata.json"),
            artifact_path="candidate",
        )

    # Finish tracking successfully before attempting a production update.
    # The local production pointer is authoritative for what is being served.
    if decision["promote"]:
        store.promote_candidate(
            candidate_version,
            expected_current_version=production_version,
        )
        status = "promoted"
    else:
        status = "rejected"

    return {
        "status": status,
        "reason": decision["reason"],
        "candidate_version": candidate_version,
        "metrics": comparisons,
    }


if __name__ == "__main__":
    print(train_model())