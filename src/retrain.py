"""Trigger the shared candidate-evaluation pipeline."""

import argparse

from src.config import Settings
from src.train import train_model


def execute_retraining_pipeline(
    drift_threshold_exceeded=False,
    settings: Settings | None = None,
):
    """A trigger permits evaluation, never unconditional promotion."""
    if not drift_threshold_exceeded:
        return {
            "status": "skipped",
            "reason": "No drift trigger or manual request was supplied.",
        }

    return train_model(settings=settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a new demand-forecasting candidate."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Request candidate training without a drift trigger.",
    )
    args = parser.parse_args()

    print(
        execute_retraining_pipeline(
            drift_threshold_exceeded=args.force,
        )
    )