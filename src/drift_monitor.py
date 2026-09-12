import os
import pandas as pd
from typing import Tuple

# Modern vs Legacy Evidently import handling
try:
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report
except ImportError:
    from evidently.presets import DataDriftPreset, TargetDriftPreset
    from evidently import Report


PROCESSED_DATA_PATH = os.path.join("data", "processed", "daily_demand.csv")
REPORT_DIR = os.path.join("reports")


def generate_drift_report(
    reference_df: pd.DataFrame, 
    current_df: pd.DataFrame
) -> Tuple[str, bool]:
    """
    Compares reference baseline data with current inference telemetry 
    using Evidently AI to evaluate data drift.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "drift_report.html")

    # Initialize Evidently Report with Data Drift Presets
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(report_path)

    # Parse drift summary to determine dataset drift status
    report_dict = report.as_dict()
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]

    return report_path, dataset_drift


def run_monitoring_pipeline() -> None:
    """Simulates production monitoring by comparing baseline data to drifted telemetry."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed dataset missing. Run 'python -m src.data_loader' first.")

    reference_df = pd.read_csv(PROCESSED_DATA_PATH)

    # Simulate production telemetry drift (increase prices by 30% and units sold)
    current_df = reference_df.copy()
    current_df["avg_price"] = current_df["avg_price"] * 1.30
    current_df["total_units_sold"] = current_df["total_units_sold"] * 1.50

    features = ["avg_price", "total_units_sold"]
    report_path, has_drifted = generate_drift_report(
        reference_df[features], 
        current_df[features]
    )

    print("--- Drift Monitoring Status ---")
    print(f"Report Generated: {report_path}")
    print(f"Dataset Drift Detected: {has_drifted}")


if __name__ == "__main__":
    run_monitoring_pipeline()