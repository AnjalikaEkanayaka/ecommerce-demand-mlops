import pandas as pd
import pytest

from src.data_loader import (
    aggregate_daily_demand,
    load_raw_data,
    validate_and_process_demand,
)


def sample_raw_data():
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "order_purchase_timestamp": [
                "2023-01-01 10:00:00",
                "2023-01-03 12:00:00",
            ],
            # Final status must not determine historical order intake.
            "order_status": ["canceled", "delivered"],
        }
    )

    items = pd.DataFrame(
        {
            "order_id": ["a", "a", "b"],
            "order_item_id": [1, 2, 1],
        }
    )

    return orders, items


def aggregate(orders, items, **kwargs):
    return aggregate_daily_demand(
        orders,
        items,
        start_date="2023-01-01",
        end_date="2023-01-03",
        **kwargs,
    )


def test_known_daily_counts_and_explicit_zero_day():
    orders, items = sample_raw_data()

    result = aggregate(orders, items, fill_missing_days=True)

    assert result["total_units_sold"].tolist() == [2, 0, 1]
    assert result["date"].tolist() == list(
        pd.date_range("2023-01-01", periods=3)
    )


def test_missing_days_require_confirmation():
    orders, items = sample_raw_data()

    with pytest.raises(ValueError, match="Missing calendar dates"):
        aggregate(orders, items)


def test_final_delivery_status_does_not_change_target():
    orders, items = sample_raw_data()
    original = aggregate(orders, items, fill_missing_days=True)

    orders["order_status"] = "delivered"
    changed = aggregate(orders, items, fill_missing_days=True)

    pd.testing.assert_frame_equal(original, changed)


def test_duplicate_order_is_rejected():
    orders, items = sample_raw_data()
    orders = pd.concat([orders, orders.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate order_id"):
        aggregate(orders, items)


def test_duplicate_item_is_rejected():
    orders, items = sample_raw_data()
    items = pd.concat([items, items.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate order/item"):
        aggregate(orders, items)


def test_unknown_order_reference_is_rejected():
    orders, items = sample_raw_data()
    items.loc[0, "order_id"] = "unknown"

    with pytest.raises(ValueError, match="unknown order_id"):
        aggregate(orders, items)


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_missing_order_id_is_rejected(bad_id):
    orders, items = sample_raw_data()
    orders.loc[0, "order_id"] = bad_id

    with pytest.raises(ValueError, match="missing or blank"):
        aggregate(orders, items)


@pytest.mark.parametrize("bad_number", [0, -1, 1.5, None])
def test_invalid_item_number_is_rejected(bad_number):
    orders, items = sample_raw_data()
    items["order_item_id"] = items["order_item_id"].astype(float)
    items.loc[0, "order_item_id"] = bad_number

    with pytest.raises(ValueError, match="positive whole numbers"):
        aggregate(orders, items)


def test_invalid_timestamp_is_rejected():
    orders, items = sample_raw_data()
    orders.loc[0, "order_purchase_timestamp"] = "not-a-date"

    with pytest.raises(ValueError):
        aggregate(orders, items)


def test_missing_required_column_is_rejected():
    orders, items = sample_raw_data()
    items = items.drop(columns="order_item_id")

    with pytest.raises(ValueError, match="missing columns"):
        aggregate(orders, items)


def test_input_frames_are_not_modified():
    orders, items = sample_raw_data()
    original_orders = orders.copy(deep=True)
    original_items = items.copy(deep=True)

    aggregate(orders, items, fill_missing_days=True)

    pd.testing.assert_frame_equal(orders, original_orders)
    pd.testing.assert_frame_equal(items, original_items)


def test_processing_round_trip(tmp_path, settings):
    orders, items = sample_raw_data()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    orders.to_csv(raw_dir / "olist_orders_dataset.csv", index=False)
    items.to_csv(raw_dir / "olist_order_items_dataset.csv", index=False)

    loaded_orders, loaded_items = load_raw_data(raw_dir)
    assert list(loaded_orders.columns) == [
        "order_id",
        "order_purchase_timestamp",
    ]
    assert list(loaded_items.columns) == ["order_id", "order_item_id"]

    expected = validate_and_process_demand(
        start_date="2023-01-01",
        end_date="2023-01-03",
        raw_dir=raw_dir,
        fill_missing_days=True,
        settings=settings,
    )

    saved = pd.read_csv(
        settings.processed_data_path,
        parse_dates=["date"],
    )

    pd.testing.assert_frame_equal(saved, expected)