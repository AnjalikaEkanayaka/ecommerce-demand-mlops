import os
import joblib
import pandas as pd
from typing import Dict, Any


MODEL_PATH = os.path.join("models", "demand_model.pkl")


def load_model():
    """Loads saved model artifact."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run 'python -m src.train' first.")
    return joblib.load(MODEL_PATH)


def predict_demand(input_features: Dict[str, Any]) -> float:
    """Runs inference on a single dictionary input matching training schema."""
    model = load_model()
    
    # Required feature list order
    feature_order = ["avg_price", "day_of_week", "month", "day", "lag_1", "lag_7", "rolling_mean_7"]
    df = pd.DataFrame([input_features])[feature_order]

    prediction = model.predict(df)[0]
    return float(np.round(prediction, 2)) if 'np' in globals() else float(round(prediction, 2))


if __name__ == "__main__":
    # Test sample input
    sample_data = {
        "avg_price": 120.50,
        "day_of_week": 2,
        "month": 10,
        "day": 15,
        "lag_1": 150.0,
        "lag_7": 140.0,
        "rolling_mean_7": 145.2
    }
    predicted_value = predict_demand(sample_data)
    print(f"Predicted Daily Demand: {predicted_value} units")