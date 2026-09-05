import os
import pandas as pd
from src.data_loader import validate_and_process_demand
from src.predict import predict_demand, load_model

def test_processed_demand_file_creation():
    """Verify daily_demand.csv is generated correctly."""
    validate_and_process_demand()
    file_path = os.path.join("data", "processed", "daily_demand.csv")

    assert os.path.exists(file_path)
    df = pd.read_csv(file_path)
    assert not df.empty
    assert "total_units_sold" in df.columns
    assert "avg_price" in df.columns


def test_model_file_exists_and_predicts():
    """Verify saved model loads and generates valid float output."""
    model = load_model()
    assert model is not None

    sample_input = {
        "avg_price": 100.0,
        "day_of_week": 1,
        "month": 5,
        "day": 10,
        "lag_1": 100.0,
        "lag_7": 100.0,
        "rolling_mean_7": 100.0
    }
    prediction = predict_demand(sample_input)
    assert isinstance(prediction, float)
    assert prediction >= 0