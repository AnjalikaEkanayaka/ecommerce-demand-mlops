"""Forecast evaluation and explicit model-promotion decisions."""

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_predictions(actual, predicted):
    """Calculate errors for matching, finite, one-dimensional observations."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Actual values and predictions must be one-dimensional.")

    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Actual values and predictions must have equal nonzero length.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual values and predictions must be finite.")

    if (actual < 0).any():
        raise ValueError("Observed demand cannot be negative.")

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
    }


def valid_metrics(metrics):
    """Missing, negative, or nonfinite metrics cannot justify promotion."""
    try:
        return all(
            math.isfinite(float(metrics[name])) and float(metrics[name]) >= 0
            for name in ("mae", "rmse")
        )
    except (KeyError, TypeError, ValueError):
        return False


def promotion_decision(
    candidate_metrics,
    baseline_metrics,
    production_metrics=None,
    min_improvement=0.01,
):
    """Decide whether a candidate qualifies for promotion.

    All supplied metrics must come from the same validation observations.
    This function makes a decision only; it never modifies model files.

    min_improvement=0.01 requires at least a 1% MAE improvement.
    """
    if not math.isfinite(min_improvement) or not 0 <= min_improvement < 1:
        raise ValueError("min_improvement must be between 0 and 1, excluding 1.")

    comparisons = [("seasonal_naive", baseline_metrics)]

    if production_metrics is not None:
        comparisons.append(("production", production_metrics))

    if not valid_metrics(candidate_metrics):
        return {"promote": False, "reason": "Invalid candidate metrics."}

    for name, metrics in comparisons:
        if not valid_metrics(metrics):
            return {"promote": False, "reason": f"Invalid {name} metrics."}

    candidate_mae = float(candidate_metrics["mae"])
    candidate_rmse = float(candidate_metrics["rmse"])

    for name, metrics in comparisons:
        comparison_mae = float(metrics["mae"])
        comparison_rmse = float(metrics["rmse"])
        required_mae = comparison_mae * (1 - min_improvement)

        # Strictly better even when the requested improvement is zero.
        if candidate_mae >= comparison_mae or candidate_mae > required_mae:
            return {
                "promote": False,
                "reason": f"Insufficient MAE improvement against {name}.",
            }

        if candidate_rmse > comparison_rmse:
            return {
                "promote": False,
                "reason": f"RMSE worsened against {name}.",
            }

    return {
        "promote": True,
        "reason": "Candidate passed all MAE and RMSE comparisons.",
    }