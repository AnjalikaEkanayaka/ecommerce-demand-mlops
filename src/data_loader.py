import os
import pandas as pd
from typing import Tuple
from src.utils import RawOrderItemSchema, ProcessedDemandSchema


RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw Olist orders and order items CSV files."""
    orders_path = os.path.join(RAW_DATA_DIR, "olist_orders_dataset.csv")
    items_path = os.path.join(RAW_DATA_DIR, "olist_order_items_dataset.csv")

    if not os.path.exists(orders_path) or not os.path.exists(items_path):
        raise FileNotFoundError("Raw dataset CSVs not found in data/raw/")

    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)
    return orders, items


def validate_and_process_demand() -> pd.DataFrame:
    """Cleans, validates, and aggregates daily demand metrics."""
    orders, items = load_raw_data()

    # Filter delivered orders
    delivered_orders = orders[orders["order_status"] == "delivered"].copy()

    # Merge with items
    df = pd.merge(items, delivered_orders, on="order_id", how="inner")

    # Extract date
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["date"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")

    # Aggregate daily metric summary
    daily_demand = (
        df.groupby(["date"])
        .agg(
            total_units_sold=("order_item_id", "count"),
            avg_price=("price", "mean"),
            total_revenue=("price", "sum"),
        )
        .reset_index()
    )

    # Assign category placeholder for unified time-series aggregation
    daily_demand["product_category"] = "all_products"

    # Validate schema row by row
    validated_records = []
    for record in daily_demand.to_dict(orient="records"):
        validated_record = ProcessedDemandSchema(**record)
        validated_records.append(validated_record.model_dump())

    processed_df = pd.DataFrame(validated_records)

    # Save output dataset
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "daily_demand.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"Data successfully processed and saved to {output_path}")

    return processed_df


if __name__ == "__main__":
    validate_and_process_demand()