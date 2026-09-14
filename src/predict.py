import logging
import os
import numpy as np
import pandas as pd
from xgboost.core import XGBoostError
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from src.features import FEATURE_COLUMNS
from src.model_store import ModelStore

from src.drift_monitor import run_monitoring_pipeline
from src.retrain import execute_retraining_pipeline
from src.config import Settings

app = FastAPI(title="E-Commerce Demand Forecasting API")
logger = logging.getLogger(__name__)

def load_model():
    settings = Settings.from_env()
    store = ModelStore(settings.runtime_dir / "models")
    try:
        production = store.load_production()
    except (OSError, ValueError, TypeError, KeyError, AttributeError, XGBoostError):
        logger.exception("Production model could not be loaded")
        return None

    if production is None:
        return None

    model, metadata = production
    return model


class DemandPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    lag_1: float = Field(..., ge=0)
    lag_7: float = Field(..., ge=0)
    rolling_mean_7: float = Field(..., ge=0)


@app.get("/health")
def health_check():
    """Liveness does not read artifacts or contact MLflow."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Readiness requires a compatible production artifact."""
    require_model()
    return {"status": "ready", "model_loaded": True}


def require_model():
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="A compatible production model is not available.",
        )
    return model


@app.post("/predict")
def predict_demand(payload: DemandPayload):
    model = require_model()
    features = pd.DataFrame(
        [[getattr(payload, column) for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    try:
        predictions = np.asarray(model.predict(features), dtype=float)
        if (
            predictions.shape != (1,)
            or not np.isfinite(predictions).all()
            or (predictions < 0).any()
        ):
            raise ValueError("Invalid demand prediction")
    except (ValueError, TypeError, XGBoostError):
        logger.exception("Production prediction failed")
        raise HTTPException(
            status_code=503,
            detail="The production model could not produce a valid prediction.",
        ) from None
    return {
        "status": "success",
        "predicted_units_sold": float(predictions[0])
    }


@app.get("/drift-report", response_class=HTMLResponse)
def get_drift_report():
    """Triggers drift monitoring check and serves the interactive HTML report."""
    run_monitoring_pipeline()
    
    report_file = Settings.from_env().report_dir / "drift_report.html"
    if not os.path.exists(report_file):
         raise HTTPException(status_code=404, detail="Drift report not found.")
            
    with open(report_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/retrain")
def trigger_retrain(force: bool = False):
    """Request candidate evaluation and return its actual outcome."""
    return execute_retraining_pipeline(
        drift_threshold_exceeded=force,
    )
