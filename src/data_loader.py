"""Validate Olist input and prepare daily recorded order-item demand."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings
from src.features import TARGET_COLUMN


ORDER_COLUMNS = ["order_id", "order_purchase_timestamp"]
ITEM_COLUMNS = ["order_id", "order_item_id"]


def require_columns(df, required, label):
    if not df.columns.is_unique:
        raise ValueError(f"{label}: column names must be unique.")

    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{label}: dataset is empty.")


def parse_day(value):
    day = pd.Timestamp(value)

    if pd.isna(day) or day.tzinfo is not None:
        raise ValueError("Use a valid timezone-naive calendar date.")

    if day != day.normalize():
        raise ValueError("Date boundaries must not contain a time of day.")

    return day


def load_raw_data(raw_dir="data/raw"):
    """Read only the columns needed for counting recorded order items."""
    raw_dir = Path(raw_dir)

    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        usecols=ORDER_COLUMNS,
        dtype={"order_id": "string"},
    )

    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv",
        usecols=ITEM_COLUMNS,
        dtype={"order_id": "string"},
    )

    return orders, items


def aggregate_daily_demand(
    orders,
    items,
    *,
    start_date,
    end_date,
    fill_missing_days=False,
):
    """Return daily counts without writing files or modifying the inputs.

    Set fill_missing_days=True only when the input is known to be complete
    for the selected reporting period.
    """
    require_columns(orders, ORDER_COLUMNS, "Orders")
    require_columns(items, ITEM_COLUMNS, "Items")

    start = parse_day(start_date)
    end = parse_day(end_date)

    if start > end:
        raise ValueError("start_date must be on or before end_date.")

    orders = orders[ORDER_COLUMNS].copy()
    items = items[ITEM_COLUMNS].copy()

    for label, frame in [("Orders", orders), ("Items", items)]:
        ids = frame["order_id"].astype("string").str.strip()

        if ids.isna().any() or ids.eq("").any():
            raise ValueError(f"{label}: order_id cannot be missing or blank.")

        frame["order_id"] = ids

    if orders["order_id"].duplicated().any():
        raise ValueError("Orders: duplicate order_id values.")

    timestamps = pd.to_datetime(
        orders["order_purchase_timestamp"],
        format="ISO8601",
        errors="raise",
    )

    if timestamps.isna().any():
        raise ValueError("Purchase timestamps cannot be missing.")

    if timestamps.dt.tz is not None:
        raise ValueError("Purchase timestamps must use one local, naive timezone.")

    orders["date"] = timestamps.dt.normalize()

    if start < orders["date"].min() or end > orders["date"].max():
        raise ValueError(
            "Requested window is outside the observed order-date range."
        )

    if pd.api.types.is_bool_dtype(items["order_item_id"]):
        raise ValueError("order_item_id must contain positive whole numbers.")

    item_numbers = pd.to_numeric(items["order_item_id"], errors="raise")
    values = item_numbers.to_numpy(dtype=float)

    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or (values != np.floor(values)).any()
    ):
        raise ValueError("order_item_id must contain positive whole numbers.")

    items["order_item_id"] = item_numbers

    if items.duplicated(["order_id", "order_item_id"]).any():
        raise ValueError("Items: duplicate order/item keys.")

    if not items["order_id"].isin(orders["order_id"]).all():
        raise ValueError("Some items reference an unknown order_id.")

    selected_orders = orders.loc[
        orders["date"].between(start, end),
        ["order_id", "date"],
    ]

    joined = items.merge(
        selected_orders,
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    counts = joined.groupby("date").size()
    calendar = pd.date_range(start, end, freq="D")
    daily = counts.reindex(calendar)

    if daily.isna().any() and not fill_missing_days:
        raise ValueError(
            "Missing calendar dates. Confirm input completeness before "
            "using fill_missing_days=True."
        )

    return (
        daily.fillna(0)
        .astype("int64")
        .rename(TARGET_COLUMN)
        .rename_axis("date")
        .reset_index()
    )


def validate_and_process_demand(
    *,
    start_date,
    end_date,
    raw_dir="data/raw",
    fill_missing_days=False,
    settings=None,
):
    """Load explicitly selected input and save validated daily demand."""
    orders, items = load_raw_data(raw_dir)

    daily = aggregate_daily_demand(
        orders,
        items,
        start_date=start_date,
        end_date=end_date,
        fill_missing_days=fill_missing_days,
    )

    settings = settings or Settings.from_env()
    output = settings.processed_data_path
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output, index=False)

    print(f"Saved {len(daily)} daily observations to {output}")
    return daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare daily recorded order-item demand."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--fill-missing-days",
        action="store_true",
        help="Treat absent days as zero only for a confirmed complete extract.",
    )
    args = parser.parse_args()

    validate_and_process_demand(
        start_date=args.start_date,
        end_date=args.end_date,
        raw_dir=args.raw_dir,
        fill_missing_days=args.fill_missing_days,
    )