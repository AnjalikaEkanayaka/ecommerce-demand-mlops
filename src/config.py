"""Runtime configuration. Importing this module never creates files."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    runtime_dir: Path
    tracking_uri: str
    experiment_name: str

    @classmethod
    def from_env(cls):
        runtime = Path(os.environ.get("APP_RUNTIME_DIR", "runtime")).resolve()
        return cls(
            runtime_dir=runtime,
            tracking_uri=os.environ.get(
                "MLFLOW_TRACKING_URI", f"sqlite:///{(runtime / 'mlflow.db').as_posix()}"
            ),
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", "ecommerce-demand-forecasting"
            ),
        )

    @property
    def processed_data_path(self):
        return self.runtime_dir / "data" / "daily_demand.csv"

    @property
    def model_path(self):
        return self.runtime_dir / "models" / "demand_model.pkl"

    @property
    def report_dir(self):
        return self.runtime_dir / "reports"


def configure_tracking(settings: Settings):
    """Initialize an explicit store only when a training operation requests it.

    HTTP servers own their artifact configuration. Direct local stores use an
    absolute file URI so changing the working directory cannot misplace artifacts.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        artifact_location = None
        if not settings.tracking_uri.startswith(("http://", "https://")):
            artifacts = settings.runtime_dir / "mlartifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            artifact_location = artifacts.as_uri()
        client.create_experiment(settings.experiment_name, artifact_location=artifact_location)
    return mlflow.set_experiment(settings.experiment_name)
