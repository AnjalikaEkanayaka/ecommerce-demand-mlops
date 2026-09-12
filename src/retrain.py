import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PROCESSED_DATA_PATH = os.path.join("data", "processed", "daily_demand.csv")
MODEL_PATH = os.path.join("models", "demand_model.pkl")


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


def execute_retraining_pipeline(drift_threshold_exceeded: bool = True) -> bool:
    """
    Triggers automated model retraining if drift is detected or manual override is set.
    Logs updated metrics and overwrites model artifacts.
    """
    if not drift_threshold_exceeded:
        print("No significant drift detected. Skipping automated retraining.")
        return False

    print("--- Starting Automated Model Retraining Pipeline ---")

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Data not found at {PROCESSED_DATA_PATH}. Run data processing first.")

    # Load data and build features
    raw_df = pd.read_csv(PROCESSED_DATA_PATH)
    df = create_time_series_features(raw_df)

    features = ["avg_price", "day_of_week", "month", "day", "lag_1", "lag_7", "rolling_mean_7"]
    target = "total_units_sold"

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # MLflow tracking
    mlflow.set_experiment("ECommerce_Demand_Retraining")

    with mlflow.start_run(run_name="retrained_xgboost_v2"):
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
# ... inside execute_retraining_pipeline ...

        # Predictions & Validation
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)

        # Log parameters & metrics to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, name="retrained_model")

        # Overwrite current production model artifact
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Retraining Complete. Val MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"Updated production artifact saved to {MODEL_PATH}")

    return True


if __name__ == "__main__":
    execute_retraining_pipeline(drift_threshold_exceeded=True)