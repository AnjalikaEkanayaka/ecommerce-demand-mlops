import os
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.config import Settings, configure_tracking
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_time_features,
    chronological_split,
)

def train_model(settings: Settings | None = None) -> None:
    """Trains an XGBoost regressor with MLflow tracking and registration."""
    settings = settings or Settings.from_env()
    if not settings.processed_data_path.exists():
        raise FileNotFoundError("Processed dataset missing. Run 'python -m src.data_loader' first.")

    df = pd.read_csv(settings.processed_data_path)
    featured_df = create_time_features(df)

    train_df, validation_df = chronological_split(
        featured_df,
        validation_days=14,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = validation_df[FEATURE_COLUMNS]
    y_test = validation_df[TARGET_COLUMN]

    # Set MLflow Experiment
    configure_tracking(settings)

    params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist"
    }

    with mlflow.start_run(run_name="xgboost_baseline") as run:
        # Log hyperparameters
        mlflow.log_params(params)

        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)

        # Log metrics to MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Save model locally
        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = settings.model_path
        joblib.dump(model, model_path)

        # Log and register model in MLflow Registry
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="ECommerceDemandXGBoost"
        )

        print("--- MLflow Experiment Logging Complete ---")
        print(f"Run ID: {run.info.run_id}")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train_model()
