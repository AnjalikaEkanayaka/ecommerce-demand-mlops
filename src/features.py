"""Historical features for rolling one-day-ahead demand forecasts."""

import numpy as np
import pandas as pd


TARGET_COLUMN = "total_units_sold"

FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "day",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
]


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using information available before each forecast day."""
    required = {"date", TARGET_COLUMN}

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Daily demand data is empty.")

    # Select only relevant columns. Realized same-day price is not a feature.
    result = df[["date", TARGET_COLUMN]].copy()

    if pd.api.types.is_numeric_dtype(result["date"]):
        raise ValueError("Dates must be calendar dates, not numbers.")

    result["date"] = pd.to_datetime(result["date"], errors="raise")

    if result["date"].isna().any():
        raise ValueError("Dates cannot be missing.")

    if result["date"].dt.tz is not None:
        raise ValueError("Use timezone-naive dates for the daily series.")

    if not result["date"].eq(result["date"].dt.normalize()).all():
        raise ValueError("Daily dates must not contain a time of day.")

    if result["date"].duplicated().any():
        raise ValueError("Each date must have exactly one demand observation.")

    if pd.api.types.is_bool_dtype(result[TARGET_COLUMN]):
        raise ValueError("Demand must contain counts, not booleans.")

    result[TARGET_COLUMN] = pd.to_numeric(
        result[TARGET_COLUMN],
        errors="raise",
    )

    demand = result[TARGET_COLUMN].to_numpy(dtype=float)

    if not np.isfinite(demand).all():
        raise ValueError("Demand cannot contain missing or infinite values.")

    if (demand < 0).any() or (demand != np.floor(demand)).any():
        raise ValueError("Demand must contain nonnegative whole-unit counts.")

    result = result.sort_values("date").reset_index(drop=True)

    gaps = result["date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Missing calendar dates. Resolve missing observations before "
            "creating features; do not automatically assume zero demand."
        )

    if len(result) < 8:
        raise ValueError("At least 8 consecutive days are required.")

    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day

    result["lag_1"] = result[TARGET_COLUMN].shift(1)
    result["lag_7"] = result[TARGET_COLUMN].shift(7)
    result["rolling_mean_7"] = (
        result[TARGET_COLUMN].shift(1).rolling(window=7).mean()
    )

    # Only the first seven rows lack sufficient historical context.
    return result.iloc[7:].reset_index(drop=True)


def chronological_split(
    featured_df: pd.DataFrame,
    validation_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest days for validation, keeping training earlier."""
    if isinstance(validation_days, bool) or not isinstance(validation_days, int):
        raise ValueError("validation_days must be a positive integer.")

    if validation_days < 1:
        raise ValueError("validation_days must be a positive integer.")

    required = {"date", TARGET_COLUMN, *FEATURE_COLUMNS}
    if not required.issubset(featured_df.columns):
        raise ValueError("Run create_time_features before splitting.")

    ordered = featured_df.sort_values("date").reset_index(drop=True)

    if ordered["date"].isna().any() or ordered["date"].duplicated().any():
        raise ValueError("Feature dates must be present and unique.")

    # A small minimum for this portfolio project, not a statistical guarantee.
    if len(ordered) - validation_days < 14:
        raise ValueError(
            "Need at least 14 training rows after feature generation, "
            "plus the requested validation days."
        )

    train = ordered.iloc[:-validation_days].copy()
    validation = ordered.iloc[-validation_days:].copy()

    return train, validation