import importlib

from src.config import Settings, configure_tracking


def test_configuration_has_no_filesystem_side_effects(settings):
    import src.config
    importlib.reload(src.config)
    assert not settings.runtime_dir.exists()


def test_environment_controls_paths(settings):
    assert Settings.from_env().model_path == settings.runtime_dir / "models" / "demand_model.pkl"
    assert "tracking.db" in settings.tracking_uri


def test_default_database_is_under_runtime(settings, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert Settings.from_env().tracking_uri == f"sqlite:///{(settings.runtime_dir / 'mlflow.db').as_posix()}"


def test_fresh_tracking_database_and_artifact_round_trip(settings):
    import mlflow
    from mlflow.tracking import MlflowClient

    configure_tracking(settings)
    with mlflow.start_run() as run:
        mlflow.log_metric("mae", 3.25)
        mlflow.log_dict({"source": "synthetic-test"}, "provenance.json")
        run_id = run.info.run_id
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    saved = client.get_run(run_id)
    assert saved.data.metrics["mae"] == 3.25
    assert saved.info.artifact_uri.startswith(settings.runtime_dir.as_uri())
    assert client.list_artifacts(run_id)[0].path == "provenance.json"
    assert mlflow.artifacts.load_dict(f"runs:/{run_id}/provenance.json") == {"source": "synthetic-test"}
