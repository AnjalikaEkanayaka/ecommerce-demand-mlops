import os
import pandas as pd
from src.data_loader import validate_and_process_demand


def test_processed_demand_file_creation():
    """Verify daily_demand.csv is generated correctly."""
    validate_and_process_demand()
    file_path = os.path.join("data", "processed", "daily_demand.csv")

    assert os.path.exists(file_path)
    df = pd.read_csv(file_path)
    assert not df.empty
    assert "total_units_sold" in df.columns
    assert "avg_price" in df.columns