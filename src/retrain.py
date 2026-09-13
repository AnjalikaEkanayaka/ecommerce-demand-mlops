import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.config import Settings, configure_tracking

def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates time-series lag and rolling window features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Date component features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Lag and rolling features
    df["lag_1"] = df["total_units_sold"].shift(1)
    df["lag_7"] = df["total_units_sold"].shift(7)
    df["rolling_mean_7"] = df["total_units_sold"].shift(1).rolling(window=7).mean()

    # Drop initial NaN rows caused by shifting
    df = df.dropna().reset_index(drop=True)
    return df


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
    df = create_time_series_features(raw_df)

    features = ["avg_price", "day_of_week", "month", "day", "lag_1", "lag_7", "rolling_mean_7"]
    target = "total_units_sold"

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

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
