import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.config import Settings, configure_tracking
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_time_features,
    chronological_split,
)

def execute_retraining_pipeline(drift_threshold_exceeded: bool = True, settings: Settings | None = None) -> bool:
    """
    Triggers automated model retraining if drift is detected or manual override is set.
    Logs updated metrics and overwrites model artifacts.
    """
    if not drift_threshold_exceeded:
        print("No significant drift detected. Skipping automated retraining.")
        return False

    print("--- Starting Automated Model Retraining Pipeline ---")

    settings = settings or Settings.from_env()
    if not settings.processed_data_path.exists():
        raise FileNotFoundError(f"Data not found at {settings.processed_data_path}. Run data processing first.")

    # Load data and build features
    raw_df = pd.read_csv(settings.processed_data_path)
    df = create_time_features(raw_df)

    train_df, validation_df = chronological_split(
        df,
        validation_days=14,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = validation_df[FEATURE_COLUMNS]
    y_val = validation_df[TARGET_COLUMN]

    # MLflow tracking
    configure_tracking(settings)

    with mlflow.start_run(run_name="retrained_xgboost_v2"):
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            random_state=42,
            n_jobs=1,
            tree_method="hist"
        )
        model.fit(X_train, y_train)

        # Predictions & Validation
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)

        # Log parameters & metrics to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, artifact_path="retrained_model")

        # Overwrite current production model artifact
        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, settings.model_path)

        print(f"Retraining Complete. Val MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"Updated production artifact saved to {settings.model_path}")

    return True


if __name__ == "__main__":
    execute_retraining_pipeline(drift_threshold_exceeded=True)
