import os
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.config import Settings, configure_tracking


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates lag features and calendar indicators for time-series forecasting."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["lag_1"] = df["total_units_sold"].shift(1)
    df["lag_7"] = df["total_units_sold"].shift(7)
    df["rolling_mean_7"] = df["total_units_sold"].shift(1).rolling(window=7).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def train_model(settings: Settings | None = None) -> None:
    """Trains an XGBoost regressor with MLflow tracking and registration."""
    settings = settings or Settings.from_env()
    if not settings.processed_data_path.exists():
        raise FileNotFoundError("Processed dataset missing. Run 'python -m src.data_loader' first.")

    df = pd.read_csv(settings.processed_data_path)
    featured_df = create_time_features(df)

    features = ["avg_price", "day_of_week", "month", "day", "lag_1", "lag_7", "rolling_mean_7"]
    target = "total_units_sold"

    X = featured_df[features]
    y = featured_df[target]

    split_idx = int(len(featured_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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
