import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    chronological_split,
    create_time_features,
)


def make_daily_data(days=40):
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=days, freq="D"),
            TARGET_COLUMN: np.arange(10, 10 + days),
        }
    )


def test_lag_and_rolling_values():
    result = create_time_features(make_daily_data())

    first = result.iloc[0]

    assert first["date"] == pd.Timestamp("2023-01-08")
    assert first["lag_1"] == 16
    assert first["lag_7"] == 10
    assert first["rolling_mean_7"] == 13
    assert len(result) == 33


def test_current_and_future_demand_do_not_change_available_features():
    original = make_daily_data()
    changed = original.copy()

    cutoff = original.loc[15, "date"]
    changed.loc[15:, TARGET_COLUMN] += 1000

    before = create_time_features(original)
    after = create_time_features(changed)

    pd.testing.assert_frame_equal(
        before.loc[before["date"] <= cutoff, FEATURE_COLUMNS],
        after.loc[after["date"] <= cutoff, FEATURE_COLUMNS],
    )


def test_same_day_price_is_not_used():
    data = make_daily_data()
    data["avg_price"] = 999.0

    result = create_time_features(data)

    assert "avg_price" not in FEATURE_COLUMNS
    assert "avg_price" not in result.columns


def test_missing_calendar_day_is_rejected():
    data = make_daily_data().drop(index=10)

    with pytest.raises(ValueError, match="Missing calendar dates"):
        create_time_features(data)


def test_duplicate_date_is_rejected():
    data = make_daily_data()
    data.loc[1, "date"] = data.loc[0, "date"]

    with pytest.raises(ValueError, match="exactly one"):
        create_time_features(data)


@pytest.mark.parametrize("bad_value", [-1, 1.5, np.nan, np.inf])
def test_invalid_demand_is_rejected(bad_value):
    data = make_daily_data()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(float)
    data.loc[10, TARGET_COLUMN] = bad_value

    with pytest.raises(ValueError, match="Demand"):
        create_time_features(data)


def test_missing_column_is_rejected():
    data = make_daily_data().drop(columns=TARGET_COLUMN)

    with pytest.raises(ValueError, match="Missing required columns"):
        create_time_features(data)


def test_short_history_is_rejected():
    with pytest.raises(ValueError, match="8 consecutive days"):
        create_time_features(make_daily_data(days=7))


def test_input_is_not_modified():
    data = make_daily_data()
    original = data.copy(deep=True)

    create_time_features(data)

    pd.testing.assert_frame_equal(data, original)


def test_unsorted_input_produces_the_same_features():
    data = make_daily_data()

    pd.testing.assert_frame_equal(
        create_time_features(data),
        create_time_features(data.iloc[::-1]),
    )


def test_split_is_chronological():
    featured = create_time_features(make_daily_data())
    train, validation = chronological_split(featured, validation_days=14)

    assert len(train) == 19
    assert len(validation) == 14
    assert train["date"].max() < validation["date"].min()


def test_split_rejects_insufficient_training_history():
    featured = create_time_features(make_daily_data(days=25))

    with pytest.raises(ValueError, match="14 training rows"):
        chronological_split(featured, validation_days=14)