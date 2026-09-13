import math

import pytest

from src.evaluate import evaluate_predictions, promotion_decision


def test_known_prediction_errors():
    metrics = evaluate_predictions([10, 20, 30], [12, 18, 30])

    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(8 / 3))


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([], []),
        ([10], [10, 20]),
        ([float("nan")], [10]),
        ([10], [float("inf")]),
        ([-1], [0]),
        ([[10]], [[10]]),
    ],
)
def test_invalid_observations_are_rejected(actual, predicted):
    with pytest.raises(ValueError):
        evaluate_predictions(actual, predicted)


def test_better_candidate_is_eligible():
    decision = promotion_decision(
        candidate_metrics={"mae": 8, "rmse": 10},
        baseline_metrics={"mae": 12, "rmse": 15},
        production_metrics={"mae": 10, "rmse": 12},
    )

    assert decision["promote"] is True


@pytest.mark.parametrize(
    "candidate",
    [
        {"mae": 11, "rmse": 11},
        {"mae": 10, "rmse": 11},
        {"mae": 9.95, "rmse": 11},
        {"mae": 8, "rmse": 13},
        {"mae": float("nan"), "rmse": 10},
        {"mae": -1, "rmse": 10},
    ],
)
def test_unqualified_candidate_is_rejected(candidate):
    decision = promotion_decision(
        candidate_metrics=candidate,
        baseline_metrics={"mae": 15, "rmse": 20},
        production_metrics={"mae": 10, "rmse": 12},
    )

    assert decision["promote"] is False


def test_candidate_must_also_beat_seasonal_baseline():
    decision = promotion_decision(
        candidate_metrics={"mae": 8, "rmse": 10},
        baseline_metrics={"mae": 7, "rmse": 9},
        production_metrics={"mae": 10, "rmse": 12},
    )

    assert decision["promote"] is False


def test_first_model_must_beat_baseline():
    decision = promotion_decision(
        candidate_metrics={"mae": 8, "rmse": 10},
        baseline_metrics={"mae": 10, "rmse": 12},
    )

    assert decision["promote"] is True


def test_tie_with_perfect_production_is_rejected():
    decision = promotion_decision(
        candidate_metrics={"mae": 0, "rmse": 0},
        baseline_metrics={"mae": 10, "rmse": 12},
        production_metrics={"mae": 0, "rmse": 0},
    )

    assert decision["promote"] is False


def test_invalid_production_metrics_block_promotion():
    decision = promotion_decision(
        candidate_metrics={"mae": 8, "rmse": 10},
        baseline_metrics={"mae": 10, "rmse": 12},
        production_metrics={"mae": float("nan"), "rmse": 12},
    )

    assert decision["promote"] is False