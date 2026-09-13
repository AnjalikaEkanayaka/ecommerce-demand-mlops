"""Versioned model artifacts and atomic production-pointer updates."""

from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from src.features import FEATURE_COLUMNS, TARGET_COLUMN


class ModelStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.production_path = self.directory / "production.json"

    def version_directory(self, version):
        if not isinstance(version, str) or not re.fullmatch(
            r"[0-9a-f]{32}", version
        ):
            raise ValueError("Invalid model version.")

        return self.directory / version

    @staticmethod
    def check_model(model):
        if model.n_features_in_ != len(FEATURE_COLUMNS):
            raise ValueError("Model has an incompatible feature count.")

        if model.get_booster().feature_names != FEATURE_COLUMNS:
            raise ValueError("Model has incompatible feature names or order.")

    def save_candidate(self, model, *, training_end, evaluation_end, run_id=None):
        """Save a new version without changing production.

        training_end is the final target date used to fit the model.
        """
        training_day = date.fromisoformat(training_end)
        evaluation_day = date.fromisoformat(evaluation_end)

        if evaluation_day <= training_day:
            raise ValueError("Evaluation must finish after training.")
        self.check_model(model)

        version = uuid4().hex
        directory = self.version_directory(version)
        directory.mkdir(parents=True, exist_ok=False)

        # Native XGBoost format, rather than a Python pickle.
        model.save_model(directory / "model.json")

        metadata = {
            "version": version,
            "schema_version": 1,
            "features": list(FEATURE_COLUMNS),
            "target": TARGET_COLUMN,
            "training_end": training_end,
            "evaluation_end": evaluation_end,
            "run_id": run_id,
        }

        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        return version

    def load_version(self, version):
        """Load an explicitly selected version and validate its contract."""
        from xgboost import XGBRegressor

        directory = self.version_directory(version)
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )

        if (
            metadata.get("version") != version
            or metadata.get("schema_version") != 1
            or metadata.get("features") != FEATURE_COLUMNS
            or metadata.get("target") != TARGET_COLUMN
        ):
            raise ValueError("Model metadata is incompatible.")

        training_day = date.fromisoformat(metadata["training_end"])
        evaluation_day = date.fromisoformat(metadata["evaluation_end"])

        if evaluation_day <= training_day:
            raise ValueError("Model metadata contains invalid date boundaries.")

        model = XGBRegressor(n_jobs=1)
        model.load_model(directory / "model.json")
        model.set_params(n_jobs=1)
        self.check_model(model)

        return model, metadata

    def production_version(self):
        """Return None only when no production pointer exists."""
        if not self.production_path.exists():
            return None

        pointer = json.loads(
            self.production_path.read_text(encoding="utf-8")
        )
        version = pointer["version"]
        self.version_directory(version)
        return version

    def load_production(self):
        version = self.production_version()

        if version is None:
            return None

        return self.load_version(version)

    @contextmanager
    def promotion_lock(self):
        """Prevent two processes from updating production simultaneously."""
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / ".promotion.lock"

        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "Promotion is locked. Another promotion may be running."
            ) from exc

        try:
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink()

    def promote_candidate(self, version, *, expected_current_version):
        """Publish a previously approved candidate.

        The caller must evaluate the candidate before calling this method.
        Reject stale decisions if production changed during evaluation.
        """
        temporary_path = (
            self.directory / f".production-{uuid4().hex}.tmp"
        )

        with self.promotion_lock():
            current = self.production_version()

            if current != expected_current_version:
                raise RuntimeError(
                    "Production changed during evaluation. Evaluate again."
                )

            # Verify that the complete candidate can load before publication.
            self.load_version(version)

            pointer = {
                "version": version,
                "previous_version": current,
            }

            try:
                with temporary_path.open("w", encoding="utf-8") as handle:
                    json.dump(pointer, handle, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())

                # Both paths are in the same directory/filesystem.
                os.replace(temporary_path, self.production_path)
            finally:
                temporary_path.unlink(missing_ok=True)